import copy
import logging
import numpy as np
from typing import Dict, List, Tuple
import torch
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
from torch import Tensor, nn
import torch.distributed as dist
from torchvision.ops.boxes import box_iou

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.layers import batched_nms, cat, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

from ..head.encoder import DilatedEncoder
from ..head.decoder import Decoder
from ..head.box_regression import YOLOFBox2BoxTransform
from ..head.uniform_matcher import UniformMatcher

__all__ = ["YOLOF"]

logger = logging.getLogger(__name__)


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class YOLOF(nn.Module):
    """
    Implementation of YOLOF.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            encoder,
            decoder,
            anchor_generator,
            box2box_transform,
            anchor_matcher,
            num_classes,
            backbone_level="res5",
            pos_ignore_thresh=0.15,
            neg_ignore_thresh=0.7,
            focal_loss_alpha=0.25,
            focal_loss_gamma=2.0,
            box_reg_loss_type="giou",
            test_score_thresh=0.05,
            test_topk_candidates=1000,
            test_nms_thresh=0.6,
            max_detections_per_image=100,
            pixel_mean,
            pixel_std,
            vis_period=0,
            input_format="BGR"
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone
                interface
            encoder (nn.Module): a module that encodes feature from backbone
            decoder (nn.Module): a module to generate cls_score and box_reg
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of
                :class:`AnchorGenerator`
            box2box_transform (Box2BoxTransform): defines the transform from
                anchors boxes to instance boxes
            anchor_matcher (Matcher): label the anchors by matching them
                with ground truth.
            num_classes (int): number of classes. Used to label background
                proposals.
            # ignore thresholds
            pos_ignore_thresh (float): the threshold to ignore positive anchors
            neg_ignore_thresh (float): the threshold to ignore negative anchors
            # Loss parameters:
            focal_loss_alpha (float): focal_loss_alpha
            focal_loss_gamma (float): focal_loss_gamma
            box_reg_loss_type (str): Options are "smooth_l1", "giou"
            # Inference parameters:
            test_score_thresh (float): Inference cls score threshold, only
                anchors with score > INFERENCE_TH are considered for
                inference (to improve speed)
            test_topk_candidates (int): Select topk candidates before NMS
            test_nms_thresh (float): Overlap threshold used for non-maximum
                suppression (suppress boxes with IoU >= this threshold)
            max_detections_per_image (int):
                Maximum number of detections to return per image during
                inference (100 is based on the limit established for the
                COCO dataset).
            # Input parameters
            pixel_mean (Tuple[float]):
                Values to be used for image normalization (BGR order).
                To train on images of different number of channels, set
                different mean & std.
                Default values are the mean pixel value from ImageNet:
                [103.53, 116.28, 123.675]
            pixel_std (Tuple[float]):
                When using pre-trained models in Detectron1 or any MSRA models,
                std has been absorbed into its conv1 weights, so the std needs
                to be set 1. Otherwise, you can use [57.375, 57.120, 58.395]
                (ImageNet std)
            vis_period (int):
                The period (in terms of steps) for minibatch visualization at
                train time. Set to 0 to disable.
            input_format (str): Whether the model needs RGB, YUV, HSV etc.
        """
        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        # Anchors
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.anchor_matcher = anchor_matcher

        self.num_classes = num_classes
        self.backbone_level = backbone_level
        # Ignore thresholds:
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        # Loss parameters:
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.box_reg_loss_type = box_reg_loss_type
        assert self.box_reg_loss_type == 'giou', "Only support GIoU Loss."
        # Inference parameters:
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detections_per_image = max_detections_per_image
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(pixel_mean).view(-1, 1, 1)
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(pixel_std).view(-1, 1, 1)
        )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        backbone_level = cfg.MODEL.YOLOF.ENCODER.BACKBONE_LEVEL
        feature_shapes = [backbone_shape[backbone_level]]
        encoder = DilatedEncoder(cfg, backbone_shape)
        decoder = Decoder(cfg)
        anchor_generator = build_anchor_generator(cfg, feature_shapes)
        return {
            "backbone": backbone,
            "encoder": encoder,
            "decoder": decoder,
            "anchor_generator": anchor_generator,
            "box2box_transform": YOLOFBox2BoxTransform(
                weights=cfg.MODEL.YOLOF.BOX_TRANSFORM.BBOX_REG_WEIGHTS,
                add_ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.ADD_CTR_CLAMP,
                ctr_clamp=cfg.MODEL.YOLOF.BOX_TRANSFORM.CTR_CLAMP
            ),
            "anchor_matcher": UniformMatcher(cfg.MODEL.YOLOF.MATCHER.TOPK),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "num_classes": cfg.MODEL.YOLOF.DECODER.NUM_CLASSES,
            "backbone_level": backbone_level,
            # Ignore thresholds:
            "pos_ignore_thresh": cfg.MODEL.YOLOF.POS_IGNORE_THRESHOLD,
            "neg_ignore_thresh": cfg.MODEL.YOLOF.NEG_IGNORE_THRESHOLD,
            # Loss parameters:
            "focal_loss_alpha": cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_ALPHA,
            "focal_loss_gamma": cfg.MODEL.YOLOF.LOSSES.FOCAL_LOSS_GAMMA,
            "box_reg_loss_type": cfg.MODEL.YOLOF.LOSSES.BBOX_REG_LOSS_TYPE,
            # Inference parameters:
            "test_score_thresh": cfg.MODEL.YOLOF.SCORE_THRESH_TEST,
            "test_topk_candidates": cfg.MODEL.YOLOF.TOPK_CANDIDATES_TEST,
            "test_nms_thresh": cfg.MODEL.YOLOF.NMS_THRESH_TEST,
            "max_detections_per_image": cfg.MODEL.YOLOF.DETECTIONS_PER_IMAGE,
            # Vis parameters
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network
        predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.
        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(
            boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index],
                                                 img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach()
        predicted_boxes = predicted_boxes.cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; " \
                   f"Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts,
                such as:
                * "height", "width" (int): the output resolution of the model,
                  used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            in training, dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used
                during training only.
            in inference, the standard output format, described in
            :doc:`/tutorials/models`.
        """
        num_images = len(batched_inputs)
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[self.backbone_level]]

        anchors_image = self.anchor_generator(features)
        anchors = [copy.deepcopy(anchors_image) for _ in range(num_images)]
        pred_logits, pred_anchor_deltas = self.decoder(
            self.encoder(features[0]))
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(pred_logits, self.num_classes)]
        pred_anchor_deltas = [permute_to_N_HWA_K(pred_anchor_deltas, 4)]

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[
                0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in
                            batched_inputs]

            indices = self.get_ground_truth(
                anchors, pred_anchor_deltas, gt_instances)
            losses = self.losses(
                indices, gt_instances, anchors,
                pred_logits, pred_anchor_deltas)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors_image, pred_logits, pred_anchor_deltas,
                        images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(
                anchors_image,
                pred_logits,
                pred_anchor_deltas,
                images.image_sizes
            )
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self,
               indices,
               gt_instances,
               anchors,
               pred_class_logits,
               pred_anchor_deltas):
        pred_class_logits = cat(
            pred_class_logits, dim=1).view(-1, self.num_classes)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1).view(-1, 4)

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor
        # Boxes(Tensor(N*R, 4))
        predicted_boxes = self.box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors)
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)

        # We obtain positive anchors by choosing gt boxes' k nearest anchors
        # and leave the rest to be negative anchors. However, there may
        # exist negative anchors that have similar distances with the chosen
        # positives. These negatives may cause ambiguity for model training
        # if we just set them as negatives. Given that we want the model's
        # predict boxes on negative anchors to have low IoU with gt boxes,
        # we set a threshold on the IoU between predicted boxes and gt boxes
        # instead of the IoU between anchor boxes and gt boxes.
        ious = []
        pos_ious = []
        for i in range(N):
            src_idx, tgt_idx = indices[i]
            iou = box_iou(predicted_boxes[i, ...],
                          gt_instances[i].gt_boxes.tensor)
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            a_iou = box_iou(anchors[i].tensor,
                            gt_instances[i].gt_boxes.tensor)
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)
        ious = torch.cat(ious)
        ignore_idx = ious > self.neg_ignore_thresh
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.pos_ignore_thresh

        src_idx = torch.cat(
            [src + idx * anchors[0].tensor.shape[0] for idx, (src, _) in
             enumerate(indices)])
        gt_classes = torch.full(pred_class_logits.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=pred_class_logits.device)
        gt_classes[ignore_idx] = -1
        target_classes_o = torch.cat(
            [t.gt_classes[J] for t, (_, J) in zip(gt_instances, indices)])
        target_classes_o[pos_ignore_idx] = -1
        gt_classes[src_idx] = target_classes_o

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        if comm.get_world_size() > 1:
            dist.all_reduce(num_foreground)
        num_foreground = num_foreground * 1.0 / comm.get_world_size()

        # cls loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )
        # reg loss
        target_boxes = torch.cat(
            [t.gt_boxes.tensor[i] for t, (_, i) in zip(gt_instances, indices)],
            dim=0)
        target_boxes = target_boxes[~pos_ignore_idx]
        matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
            src_idx[~pos_ignore_idx]]
        loss_box_reg = giou_loss(
            matched_predicted_boxes, target_boxes, reduction="sum")

        return {
            "loss_cls": loss_cls / max(1, num_foreground),
            "loss_box_reg": loss_box_reg / max(1, num_foreground),
        }

    @torch.no_grad()
    def get_ground_truth(self, anchors, bbox_preds, targets):
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        N = len(anchors)
        # list[Tensor(R, 4)], one for each image
        all_anchors = Boxes.cat(anchors).tensor.reshape(N, -1, 4)
        # Boxes(Tensor(N*R, 4))
        box_delta = cat(bbox_preds, dim=1)
        # box_pred: xyxy; targets: xyxy
        box_pred = self.box2box_transform.apply_deltas(box_delta, all_anchors)
        indices = self.anchor_matcher(box_pred, all_anchors, targets)
        return indices

    def inference(
            self,
            anchors: List[Boxes],
            pred_logits: List[Tensor],
            pred_anchor_deltas: List[Tensor],
            image_sizes: List[Tuple[int, int]],
    ):
        """
        Arguments:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contain anchors of this image on the specific
                feature level.
            pred_logits, pred_anchor_deltas: list[Tensor], one per level. Each
                has shape (N, Hi * Wi * Ai, K or 4)
            image_sizes (List[(h, w)]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results: List[Instances] = []
        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, pred_logits_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
            self,
            anchors: List[Boxes],
            box_cls: List[Tensor],
            box_delta: List[Tensor],
            image_size: Tuple[int, int],
    ):
        """
        Single-image inference. Return bounding-box detection results by
        thresholdingon scores and applying non-maximum suppression (NMS).

        Args:
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors in that feature
                level.
            box_cls (list[Tensor]): list of #feature levels. Each entry
                contains tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K
                becomes 4.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
            # (HxWxAxK,)
            predicted_prob = box_cls_i.flatten().sigmoid_()

            # Apply two filtering below to make NMS faster.
            # 1. Keep boxes with confidence score higher than threshold
            keep_idxs = predicted_prob > self.test_score_thresh
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = nonzero_tuple(keep_idxs)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_topk_candidates, topk_idxs.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, idxs = predicted_prob.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[idxs[:num_topk]]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.test_nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images
