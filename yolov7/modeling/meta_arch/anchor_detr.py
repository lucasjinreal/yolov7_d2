# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List, Dict, OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.utils import comm
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from yolov7.utils.detr_utils import HungarianMatcherAnchorDETR
from yolov7.utils.boxes import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, convert_coco_poly_to_mask, generalized_box_iou
from yolov7.utils.misc import NestedTensor, nested_tensor_from_tensor_list, accuracy

from alfred.utils.log import logger

from ..backbone.detr_backbone import Joiner, PositionEmbeddingSine
from ..backbone.anchordetr_backbone import Transformer
from .detr_seg import DETRsegm, PostProcessPanoptic, PostProcessSegm, sigmoid_focal_loss, dice_loss
from alfred.dl.torch.common import device
import pickle

__all__ = ["AnchorDetr"]


@META_ARCH_REGISTRY.register()
class AnchorDetr(nn.Module):
    """
    Implement AnchorDetr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_thresh = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.ignore_thresh = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        num_query_position = cfg.MODEL.DETR.NUM_QUERY_POSITION
        num_query_pattern = cfg.MODEL.DETR.NUM_QUERY_PATTERN
        spatial_prior = cfg.MODEL.DETR.SPATIAL_PRIOR

        backbone = MaskedBackboneTraceFriendly(cfg)

        transformer = Transformer(
            num_classes=self.num_classes+1,
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            num_feature_levels=num_feature_levels,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            activation="relu",
            num_query_position=num_query_position,
            num_query_pattern=num_query_pattern,
            spatial_prior=spatial_prior,
        )

        self.detr = AnchorDETR(backbone,
                               transformer,
                               num_feature_levels,
                               aux_loss=deep_supervision)
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(
                    frozen_weights,
                    map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherAnchorDETR(cost_class=1,
                                   cost_bbox=l1_weight,
                                   cost_giou=giou_weight)
        weight_dict = {"loss_ce": 2, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v
                     for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes+1,
            matcher=matcher,
            weight_dict=weight_dict,
            # eos_coef=no_object_weight,
            losses=losses,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.onnx_export = False

    def preprocess_input(self, x):
        # x = x.permute(0, 3, 1, 2)
        # x = F.interpolate(x, size=(640, 640))
        # x = F.interpolate(x, size=(512, 960))
        """
        x is N, CHW aleady permuted
        """
        x = [self.normalizer(i) for i in x]
        return x

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.onnx_export:
            logger.info('[WARN] exporting onnx...')
            assert isinstance(
                batched_inputs, (list, torch.Tensor)
            ) or isinstance(
                batched_inputs, list
            ), 'onnx export, batched_inputs only needs image tensor or list of tensors'
            images = self.preprocess_input(batched_inputs)
            # batched_inputs = batched_inputs.permute(0, 3, 1, 2)
            # image_ori_sizes = [batched_inputs.shape[1:3]]
        else:
            images = self.preprocess_image(batched_inputs)

        if self.onnx_export:
            self.detr.onnx_export = self.onnx_export
            self.detr.backbone.prepare_onnx_export()

        output = self.detr(images)

        if self.training:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            valid_loss_dict = {}
            for k in loss_dict.keys():
                if k in weight_dict:
                    valid_loss_dict[k] = loss_dict[k] * weight_dict[k]
                    # loss_dict[k] *= weight_dict[k]
            # print(loss_dict)
            # return loss_dict
            return valid_loss_dict
        else:
            if self.onnx_export:
                box_cls = output[0]
                box_pred = output[1]
                scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)
                box_pred = box_cxcywh_to_xyxy(box_pred)
                labels = labels.to(torch.float)
                print(scores.shape)
                # print(scores.unsqueeze(0).shape)
                a = torch.cat(
                    [box_pred,
                     scores.unsqueeze(-1),
                     labels.unsqueeze(-1)],
                    dim=-1)
                return a
            else:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                results = self.inference(box_cls, box_pred, mask_pred,
                                         images.image_sizes)
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                        results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h],
                                              dtype=torch.float,
                                              device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        # TODO make top-100 as an option for non-focal-loss as well
        scores, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), 100, dim=1
        )
        topk_boxes = topk_indexes // box_cls.shape[2]
        labels = topk_indexes % box_cls.shape[2]

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            boxes = box_cxcywh_to_xyxy(box_pred_per_image)
            boxes = torch.gather(
                boxes, 0, topk_boxes[i].unsqueeze(-1).repeat(1, 4))
            result.pred_boxes = Boxes(boxes)

            result.pred_boxes.scale(
                scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(
                    0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                # print('mask_pred shape: ', mask.shape)
                # mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                mask = BitMasks(mask.cpu())
                # result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)
                result.pred_bit_masks = mask.to(mask_pred[i].device)
            # print('box_pred_per_image: ', box_pred_per_image.shape)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def inference_onnx(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        appending indices as one of output for convinient select ??
        """
        pass

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [
            self.normalizer(x["image"].to(self.device)) for x in batched_inputs
        ]
        images = ImageList.from_tensors(images)
        return images


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskedBackboneTraceFriendly(nn.Module):
    """ 
    This is a thin wrapper around D2's backbone to provide padding masking.
    I change it into tracing friendly with this mask operation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS

        # if comm.is_main_process():
        #     a = torch.randn([1, 3, 256, 256])
        #     b = self.backbone(a)
        #     print('B: ', b)
        # self.backbone = torchvision.models.resnet50(pretrained=True)
        backbone_shape = self.backbone.output_shape()

        # pretrained_weights = cfg.MODEL.WEIGHTS
        # if pretrained_weights:
        #     logger.info(f'Loading pretrained weights from: {pretrained_weights}')
        #     with open(pretrained_weights, 'rb') as f:
        #         wgts = pickle.load(f, encoding='latin1')['model']
        #     # wgts = torch.load(pretrained_weights, map_location=lambda storage, loc: storage)
        #     new_weight = {}
        #     for k, v in wgts.items():
        #         v = torch.from_numpy(v)
        #         # new_weight['detr.' + k] = v
        #         new_weight[k] = v
        #     del wgts
        #     self.backbone.load_state_dict(new_weight, strict=False)
        #     del new_weight
        
        # if comm.is_main_process():
        #     c = self.backbone(a)
        #     print('C: ', c)

        if self.num_feature_levels > 1:
            self.num_channels = [512, 1024, 2048]
            self.return_interm_layers = ['res3', 'res4', 'res5']
            self.feature_strides = [8, 16, 32]
        else:
            self.num_channels = [2048]
            self.return_interm_layers = ['res5']
            self.feature_strides = [32]
            
        print(self.num_channels)
        self.onnx_export = False

    def forward(self, images):
        if isinstance(images, ImageList):
            features = self.backbone(images.tensor)
            device = images.tensor.device
        else:
            features = self.backbone(images.tensors)
            device = images.tensors.device

        if self.onnx_export:
            logger.info('[onnx export] in MaskedBackbone...')
            out: Dict[str, NestedTensor] = {}
            for name, x in features.items():
                m = images.mask
                print('m: ', m)
                print('m: ', m.shape)
                assert m is not None
                sp = x.shape[-2:]
                # mask = F.interpolate(m.to(torch.float), size=sp).to(torch.bool)[0]
                # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                m = m.unsqueeze(0).float()
                mask = F.interpolate(m, size=x.shape[-2:]).to(torch.bool)[0]
                print(mask.shape)
                out[name] = NestedTensor(x, mask)
            return out
        else:
            # features: res2, res3, res4, res5
            features_returned = OrderedDict()
            for l in self.return_interm_layers:
                features_returned[l] = features[l]

            masks = self.mask_out_padding(
                [
                    features_per_level.shape
                    for features_per_level in features_returned.values()
                ],
                images.image_sizes,
                device,
            )
            assert len(features_returned) == len(masks)
            out_nested_features = []
            
            for i, k in enumerate(self.return_interm_layers):
                out_nested_features.append(NestedTensor(features_returned[k], masks[i]))
            return out_nested_features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W),
                                                 dtype=torch.bool,
                                                 device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                # print('H', H, 'W', W, 'ceil: ', int(np.ceil(float(h) / self.feature_strides[idx])),)
                masks_per_feature_level[img_idx, :int(
                    np.ceil(float(h) / self.feature_strides[idx])
                ), :int(np.ceil(float(w) / self.feature_strides[idx])), ] = 0
            masks.append(masks_per_feature_level)
        return masks


class AnchorDETR(nn.Module):
    """ This is the AnchorDETR module that performs object detection """

    def __init__(self, backbone, transformer, num_feature_levels, aux_loss=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.num_feature_levels = num_feature_levels
        logger.info(f'{backbone.num_channels}')
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                if _ == 0:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
                else:
                    input_proj_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.onnx_export = False

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)
        # print(features)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src).unsqueeze(1))
            masks.append(mask)
            assert mask is not None

        srcs = torch.cat(srcs, dim=1)

        outputs_class, outputs_coord = self.transformer(srcs, masks)

        if self.onnx_export:
            return outputs_class[-1], outputs_coord[-1]
        else:
            out = {
                'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1]
            }
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(
                    outputs_class, outputs_coord)
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'pred_logits': a,
            'pred_boxes': b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if comm.get_world_size() > 1:
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / comm.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
        # print(losses)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{
            'scores': s,
            'labels': l,
            'boxes': b
        } for s, l, b in zip(scores, labels, boxes)]

        return results
