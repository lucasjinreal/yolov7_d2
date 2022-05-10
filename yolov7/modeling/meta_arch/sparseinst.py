# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from yolov7.modeling.transcoders.encoder_sparseinst import build_sparse_inst_encoder
from yolov7.modeling.transcoders.decoder_sparseinst import build_sparse_inst_decoder

from ..loss.sparseinst_loss import build_sparse_inst_criterion

# from .utils import nested_tensor_from_tensor_list
from yolov7.utils.misc import nested_tensor_from_tensor_list
from alfred.utils.log import logger
from alfred import print_shape

__all__ = ["SparseInst"]


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


def rescoring_mask_batch(scores, mask_pred, masks):
    # scores and masks contains batch
    print(f'mask_pred: {mask_pred.shape}, masks: {masks.shape}, scores: {scores.shape}')
    mask_pred_ = mask_pred.float()

    # # masks = (masks * mask_pred_).sum([2, 3])
    # mask_pred2 = torch.sum(mask_pred_, [2, 3])
    # mask_pred2 = mask_pred2 + 1e-6
    # # masks_to_m = masks / mask_pred
    # # print(masks_to_m.shape, scores.shape)
    # # return scores * masks_to_m
    # scores *= mask_pred2
    # return scores

    return scores * ((masks * mask_pred_).sum([2, 3]) / (mask_pred_.sum([2, 3]) + 1e-6))

def batched_index_select(input, dim, index):
    views = [1 if i != dim else -1 for i in range(len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    # making the first dim of output be B
    return torch.cat(torch.chunk(torch.gather(input, dim, index), chunks=index.shape[0], dim=dim), dim=0)

@META_ARCH_REGISTRY.register()
class SparseInst(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()

        # encoder & decoder
        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        # matcher & loss (matcher is built in loss)
        self.criterion = build_sparse_inst_criterion(cfg)

        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self.pixel_mean = (
            torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        )
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # only for onnx export
        self.normalizer_trans = lambda x: (x - self.pixel_mean.unsqueeze(0)) / self.pixel_std.unsqueeze(0)

        # inference
        # self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.cls_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS

    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image

    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has("gt_masks"):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(gt_masks.polygons, h, w)

            target["masks"] = gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets

    def preprocess_inputs_onnx(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.normalizer_trans(x)
        return x

    def forward(self, batched_inputs):
        if torch.onnx.is_in_onnx_export():
            logger.info("[WARN] exporting onnx...")
            assert isinstance(batched_inputs, (list, torch.Tensor)) or isinstance(
                batched_inputs, list
            ), "onnx export, batched_inputs only needs image tensor or list of tensors"
            images = self.preprocess_inputs_onnx(batched_inputs)
            logger.info(f'images onnx input: {images.shape}')
        else:
            images = self.preprocess_inputs(batched_inputs)

        # if isinstance(images, (list, torch.Tensor)):
        #     images = nested_tensor_from_tensor_list(images)

        if isinstance(images, ImageList):
            max_shape = images.tensor.shape[2:]
            features = self.backbone(images.tensor)
        else:
            # onnx trace
            max_shape = images.shape[2:]
            features = self.backbone(images)

        features = self.encoder(features)
        output = self.decoder(features)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            losses = self.criterion(output, targets, max_shape)
            return losses
        else:
            if torch.onnx.is_in_onnx_export():
                results = self.inference_onnx(
                    output, batched_inputs, max_shape
                )
                return results
            else:
                results = self.inference(
                    output, batched_inputs, max_shape, images.image_sizes
                )
            processed_results = [{"instances": r} for r in results]
            return processed_results

    def forward_test(self, images):
        pass

    def inference(self, output, batched_inputs, max_shape, image_sizes):
        # max_detections = self.max_detections
        results = []
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)

        for _, (
            scores_per_image,
            mask_pred_per_image,
            batched_input,
            img_shape,
        ) in enumerate(zip(pred_scores, pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = labels
                results.append(result)
                continue

            h, w = img_shape
            # rescoring mask using maskness
            scores = rescoring_mask(
                scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image
            )

            # upsample the masks to the original resolution:
            # (1) upsampling the masks to the padded inputs, remove the padding area
            # (2) upsampling/downsampling the masks to the original sizes
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1),
                size=max_shape,
                mode="bilinear",
                align_corners=False,
            )[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image,
                size=ori_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            # mask_pred = BitMasks(mask_pred)

            # using Detectron2 Instances to store the final results
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = labels
            results.append(result)
        return results

    def inference_onnx(self, output, batched_inputs, max_shape):
        from alfred import print_shape
        # max_detections = self.max_detections
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        # solve Nan problems with a minimal epsilon
        pred_scores = torch.sqrt(pred_scores * pred_objectness + 1e-3)
        print(f'pred_scores: {pred_scores.shape}, perd_masks: {pred_masks.shape}')

        # all_scores = []
        # all_labels = []
        # all_masks = []
        # print('max_shape: ', max_shape)

        # for _, (
        #     scores_per_image,
        #     mask_pred_per_image,
        #     batched_input,
        # ) in enumerate(zip(pred_scores, pred_masks, batched_inputs)):

        #     # max/argmax
        #     scores, labels = torch.max(scores_per_image, dim=scores_per_image.dim()-1)
        #     # cls threshold
        #     # keep = scores > self.cls_threshold
        #     _, keep = torch.topk(scores, k=50)
        #     print(keep.shape, scores.shape)
        #     scores = scores[keep]
        #     labels = labels[keep]
        #     # print(scores, labels)
        #     mask_pred_per_image = mask_pred_per_image[keep]
            
        #     h, w = max_shape
        #     # rescoring mask using maskness
        #     scores = rescoring_mask(
        #         scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image
        #     )

        #     # upsample the masks to the original resolution:
        #     # (1) upsampling the masks to the padded inputs, remove the padding area
        #     # (2) upsampling/downsampling the masks to the original sizes
        #     print('mask_pred_per_image: ', mask_pred_per_image.shape)
        #     mask_pred_per_image = F.interpolate(
        #         mask_pred_per_image.unsqueeze(1),
        #         size=max_shape,
        #         mode="bilinear",
        #         align_corners=False,
        #     )[:, :h, :w]

        #     mask_pred = mask_pred_per_image > self.mask_threshold

        #     all_masks.append(mask_pred)
        #     all_scores.append(scores)
        #     all_labels.append(labels)

        # do it in batch
        # max/argmax
        scores, labels = torch.max(pred_scores, dim=pred_scores.dim()-1)
        K_ = min(50, self.max_detections)
        _, keep = torch.topk(scores, k=K_)
        print(keep.shape, scores.shape)
        keep_flt = keep.view(-1, K_)
        scores = scores.view(-1)
        labels = labels.view(-1)
        scores = scores[keep_flt]
        labels = labels[keep_flt]

        # advanced select
        pred_masks = pred_masks.view(-1, pred_masks.shape[-2], pred_masks.shape[-1])
        print_shape(keep_flt, pred_masks)
        mask_pred_batch = pred_masks[keep_flt] # 1, 100, 160, 160
        
        h, w = max_shape
        # rescoring mask using maskness
        scores = rescoring_mask_batch(
            scores, mask_pred_batch > self.mask_threshold, mask_pred_batch
        )



        # upsample the masks to the original resolution:
        # (1) upsampling the masks to the padded inputs, remove the padding area
        # (2) upsampling/downsampling the masks to the original sizes
        print('mask_pred_per_image: ', mask_pred_batch.shape)
        mask_pred_batch = F.interpolate(
            mask_pred_batch,
            size=max_shape,
            mode="bilinear",
            align_corners=False,
        )[:, :h, :w]
        mask_pred = mask_pred_batch > self.mask_threshold

    
        # do scores here
        # masks_values = mask_pred.float() * mask_pred_batch
        # masks_values = torch.sum(masks_values, [-2, -1])
        # scores = scores * masks_values
        # print(scores, labels)
   
        # all_masks = torch.stack(all_masks).to(torch.long)
        # all_scores = torch.stack(all_scores)
        # all_labels = torch.stack(all_labels)
        # # logger.info(f'all_scores: {all_scores.shape}')
        # # logger.info(f'all_labels: {all_labels.shape}')
        # logger.info(f'all_masks: {all_masks.shape}')
        # return all_masks, all_scores, all_labels
        return mask_pred, scores, labels
