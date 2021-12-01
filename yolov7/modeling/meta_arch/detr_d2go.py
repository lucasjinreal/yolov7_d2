#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from yolov7.utils.boxes import convert_coco_poly_to_mask
from ..backbone.smcadetr_backbone import Joiner
from ..backbone.detr_backbone import Transformer as DetrTransformer
from ..backbone.smcadetr_backbone import Transformer as SMCATransformer
from ..backbone.detr_backbone import PositionEmbeddingSine, MLP
from yolov7.utils.detr_utils import HungarianMatcherD2go

from yolov7.modeling.loss.setcriterion import SetCriterion, FocalLossSetCriterion
from yolov7.utils.misc import NestedTensor, nested_tensor_from_tensor_list
from yolov7.utils.boxes import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .detr_seg import DETRsegm, PostProcessSegm
from .smca_detr import DETR as SMCADETR
from torch import nn
from alfred.utils.log import logger


__all__ = ["DetrD2go"]


class ResNetMaskedBackbone(nn.Module):
    """This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        if cfg.MODEL.DETR.NUM_FEATURE_LEVELS > 1:
            self.strides = [8, 16, 32]
        else:
            self.strides = [32]

        if cfg.MODEL.RESNETS.RES5_DILATION == 2:
            # fix dilation from d2
            self.backbone.stages[-1][0].conv2.dilation = (1, 1)
            self.backbone.stages[-1][0].conv2.padding = (1, 1)
            self.strides[-1] = self.strides[-1] // 2

        self.feature_strides = [
            backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = [
            backbone_shape[k].channels for k in backbone_shape.keys()]

    def forward(self, images):
        if isinstance(images, ImageList):
            features = self.backbone(images.tensor)
            device = images.tensor.device
        else:
            features = self.backbone(images.tensors)
            device = images.tensors.device

        if torch.onnx.is_in_onnx_export():
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
            features = self.backbone(images.tensor)
            # one tensor per feature level. Each tensor has shape (B, maxH, maxW)
            masks = self.mask_out_padding(
                [features_per_level.shape for features_per_level in features.values()],
                images.image_sizes,
                images.tensor.device,
            )
            assert len(features) == len(masks)
            for i, k in enumerate(features.keys()):
                features[k] = NestedTensor(features[k], masks[i])
            return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones(
                (N, H, W), dtype=torch.bool, device=device
            )
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


class FBNetMaskedBackbone(ResNetMaskedBackbone):
    """This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.backbone = build_backbone(cfg)
        self.out_features = cfg.MODEL.FBNET_V2.OUT_FEATURES
        self.feature_strides = list(
            self.backbone._out_feature_strides.values())
        self.num_channels = [
            self.backbone._out_feature_channels[k] for k in self.out_features
        ]
        self.strides = [
            self.backbone._out_feature_strides[k] for k in self.out_features
        ]

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        ret_features = {}
        for i, k in enumerate(features.keys()):
            if k in self.out_features:
                ret_features[k] = NestedTensor(features[k], masks[i])
        return ret_features


class SimpleSingleStageBackbone(ResNetMaskedBackbone):
    """This is a simple wrapper for single stage backbone,
    please set the required configs:
    cfg.MODEL.BACKBONE.SIMPLE == True,
    cfg.MODEL.BACKBONE.STRIDE, cfg.MODEL.BACKBONE.CHANNEL
    """

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.backbone = build_backbone(cfg)
        self.out_features = ["out"]
        assert cfg.MODEL.BACKBONE.SIMPLE is True
        self.feature_strides = [cfg.MODEL.BACKBONE.STRIDE]
        self.num_channels = [cfg.MODEL.BACKBONE.CHANNEL]
        self.strides = [cfg.MODEL.BACKBONE.STRIDE]

    def forward(self, images):
        # print(images.tensor.shape)
        y = self.backbone(images.tensor)
        if isinstance(y, Dict):
            y = list(y.values())[-1]
            # print(y.shape)
        masks = self.mask_out_padding(
            [y.shape],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(masks) == 1
        ret_features = {}
        ret_features[self.out_features[0]] = NestedTensor(y, masks[0])
        return ret_features


@META_ARCH_REGISTRY.register()
class DetrD2go(nn.Module):
    """
    Implement Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        cls_weight = cfg.MODEL.DETR.CLS_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        centered_position_encoding = cfg.MODEL.DETR.CENTERED_POSITION_ENCODIND
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS

        N_steps = hidden_dim // 2
        if "resnet" in cfg.MODEL.BACKBONE.NAME.lower():
            d2_backbone = ResNetMaskedBackbone(cfg)
        elif "fbnet" in cfg.MODEL.BACKBONE.NAME.lower():
            d2_backbone = FBNetMaskedBackbone(cfg)
        elif cfg.MODEL.BACKBONE.SIMPLE:
            d2_backbone = SimpleSingleStageBackbone(cfg)
        else:
            raise NotImplementedError

        backbone = Joiner(
            d2_backbone,
            PositionEmbeddingSine(
                N_steps, normalize=True, centered=centered_position_encoding
            ),
        )
        backbone.num_channels = d2_backbone.num_channels
        self.use_focal_loss = cfg.MODEL.DETR.USE_FOCAL_LOSS

        if cfg.MODEL.DETR.ATTENTION_TYPE == 'deformable':
            print('Deformable not supported now.')
        elif cfg.MODEL.DETR.ATTENTION_TYPE == 'DETR':
            transformer = DetrTransformer(
                d_model=hidden_dim,
                dropout=dropout,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                normalize_before=pre_norm,
                return_intermediate_dec=deep_supervision,
            )
            self.detr = DETR(
                backbone,
                transformer,
                num_classes=self.num_classes,
                num_queries=num_queries,
                aux_loss=deep_supervision,
                use_focal_loss=self.use_focal_loss,
            )
        elif cfg.MODEL.DETR.ATTENTION_TYPE == 'SMCA':
            transformer = SMCATransformer(
                d_model=hidden_dim,
                dropout=dropout,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                normalize_before=pre_norm,
                return_intermediate_dec=deep_supervision,
                dynamic_scale='type3'
            )
            self.detr = SMCADETR(
                backbone,
                transformer,
                num_classes=self.num_classes,
                num_queries=num_queries,
                aux_loss=deep_supervision,
                num_feature_levels=num_feature_levels,
                use_focal_loss=self.use_focal_loss,
            )
        else:
            logger.error(
                f'attention type: {cfg.MODEL.DETR.ATTENTION_TYPE} not support')
            exit(1)

        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != "":
                print("LOAD pre-trained weights")
                weight = torch.load(
                    frozen_weights, map_location=lambda storage, loc: storage
                )["model"]
                new_weight = {}
                for k, v in weight.items():
                    if "detr." in k:
                        new_weight[k.replace("detr.", "")] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ""))
            self.seg_postprocess = PostProcessSegm

        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcherD2go(
            cost_class=cls_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal_loss=self.use_focal_loss,
        )
        weight_dict = {"loss_ce": cls_weight, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        if self.use_focal_loss:
            self.criterion = FocalLossSetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
            )
        else:
            self.criterion = SetCriterion(
                self.num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                eos_coef=no_object_weight,
                losses=losses,
            )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
            self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
            self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

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
        if torch.onnx.is_in_onnx_export():
            logger.info('[WARN] exporting onnx...')
            assert isinstance(batched_inputs, (list, torch.Tensor)) or isinstance(
                batched_inputs, list), 'onnx export, batched_inputs only needs image tensor or list of tensors'
            images = self.preprocess_input(batched_inputs)
            # batched_inputs = batched_inputs.permute(0, 3, 1, 2)
            # image_ori_sizes = [batched_inputs.shape[1:3]]
        else:
            images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]

            # targets: List[Dict[str, torch.Tensor]]. Keys
            # "labels": [NUM_BOX,]
            # "boxes": [NUM_BOX, 4]
            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if not loss_dict[k].requires_grad:
                    loss_dict[k] = loss_dict[k].new_tensor(0)
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # print(loss_dict)
            return loss_dict
        else:
            if torch.onnx.is_in_onnx_export():
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                if self.use_focal_loss:
                    prob = box_cls.sigmoid()
                    # TODO make top-100 as an option for non-focal-loss as well
                    scores, topk_indexes = torch.topk(
                        prob.view(box_cls.shape[0], -1), 100, dim=1
                    )
                    topk_boxes = topk_indexes // box_cls.shape[2]
                    labels = topk_indexes % box_cls.shape[2]

                    print('topk_boxes: ', topk_boxes.shape)
                    print('box_pred: ', box_pred.shape)
                    topk_boxes = topk_boxes.to(torch.int64)
                    topk_boxes = topk_boxes.unsqueeze(-1).repeat(1, 1, 4)
                    print('topk_boxes: ', topk_boxes.shape)
                    box_pred = torch.gather(box_pred, 1, topk_boxes)
                else:
                    scores, labels = F.softmax(
                        box_cls, dim=-1)[:, :, :-1].max(-1)
                box_pred = box_cxcywh_to_xyxy(box_pred)
                labels = labels.to(torch.float)
                # print(scores.shape)
                # print(scores.unsqueeze(0).shape)
                a = torch.cat([box_pred, scores.unsqueeze(-1),
                              labels.unsqueeze(-1)], dim=-1)
                return a
            else:
                box_cls = output["pred_logits"]
                box_pred = output["pred_boxes"]
                mask_pred = output["pred_masks"] if self.mask_on else None
                results = self.inference(
                    box_cls, box_pred, mask_pred, images.image_sizes)
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes  # shape (NUM_BOX,)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)  # shape (NUM_BOX, 4)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, "gt_masks"):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({"masks": gt_masks})
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

        # For each box we assign the best class or the second best if the best on is `no_object`.
        if self.use_focal_loss:
            prob = box_cls.sigmoid()
            # TODO make top-100 as an option for non-focal-loss as well
            scores, topk_indexes = torch.topk(
                prob.view(box_cls.shape[0], -1), 100, dim=1
            )
            topk_boxes = topk_indexes // box_cls.shape[2]
            labels = topk_indexes % box_cls.shape[2]
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
            scores_per_image,
            labels_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            boxes = box_cxcywh_to_xyxy(box_pred_per_image)
            if self.use_focal_loss:
                boxes = torch.gather(
                    boxes, 0, topk_boxes[i].unsqueeze(-1).repeat(1, 4))

            result.pred_boxes = Boxes(boxes)
            result.pred_boxes.scale(
                scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(
                    mask_pred[i].unsqueeze(0),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(
                    result.pred_boxes.tensor.cpu(), 32
                )
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device))
                  for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DETR(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        use_focal_loss=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(
            hidden_dim, num_classes if use_focal_loss else num_classes + 1
        )
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(
            backbone.num_channels[-1], hidden_dim, kernel_size=1
        )
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
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
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        # src shape (B, C, H, W)
        # mask shape (B, H, W)
        src, mask = features[-1].decompose()
        assert mask is not None
        # hs shape (NUM_LAYER, B, S, hidden_dim)
        hs = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1]
        )[0]
        # shape (NUM_LAYER, B, S, NUM_CLASS + 1)
        outputs_class = self.class_embed(hs)
        # shape (NUM_LAYER, B, S, 4)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # pred_logits shape (B, S, NUM_CLASS + 1)
        # pred_boxes shape (B, S, 4)
        out = {"pred_logits": outputs_class[-1],
               "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
