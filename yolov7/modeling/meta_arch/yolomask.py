
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Copyright (c) Lucas Jin. telegram: lucasjin
from typing import List

import logging
from cv2 import KeyPoint
from numpy.core.numeric import ones_like
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torchvision.ops import batched_nms

from detectron2.modeling.meta_arch import build
from detectron2.layers import ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY, ResNet, ResNetBlockBase, META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, image_list
from detectron2.utils import comm
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone

from alfred.dl.metrics.iou_loss import ciou_loss, ciou
from alfred.utils.log import logger

from alfred.dl.torch.common import print_tensor, device
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from alfred.vis.image.mask import label2color_mask, vis_bitmasks

from ..neck.yolo_pafpn import YOLOPAFPN
from yolov7.utils.boxes import postprocess, bbox_ious2, BoxModeMy, postprocessv5, anchor_ious, postprocess_yolomask
from ..backbone.layers.wrappers import BaseConv, NearestUpsample, ConvBNRelu
import time

__all__ = ["YOLOV7", "YOLOHead"]

supported_backbones = ['resnet', 'res2net',
                       'swin', 'efficient', 'darknet', 'pvt']


def conv_bn_leaky(*args, **kwargs):
    return ConvBNRelu(*args, **kwargs, activation='leaky')


@META_ARCH_REGISTRY.register()
class YOLOMask(nn.Module):

    def __init__(self, cfg):
        super(YOLOMask, self).__init__()
        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.ignore_threshold = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.depth_mul = cfg.MODEL.YOLO.DEPTH_MUL
        self.width_mul = cfg.MODEL.YOLO.WIDTH_MUL

        self.num_anchors = len(cfg.MODEL.YOLO.ANCHORS) // 3
        self.anchors = cfg.MODEL.YOLO.ANCHORS
        self.anchor_mask = cfg.MODEL.YOLO.ANCHOR_MASK

        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.max_boxes_num = cfg.MODEL.YOLO.MAX_BOXES_NUM
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES

        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.change_iter = 10
        self.iter = 0

        self.use_l1 = False

        assert len([i for i in supported_backbones if i in cfg.MODEL.BACKBONE.NAME]
                   ) > 0, 'Only {} supported.'.format(supported_backbones)

        self.backbone = build_backbone(cfg)
        backbone_shape_out = self.backbone.output_shape()
        self.size_divisibility = 32 if self.backbone.size_divisibility == 0 else self.backbone.size_divisibility
        logger.info('YOLO.ANCHORS: {}'.format(cfg.MODEL.YOLO.ANCHORS))
        backbone_shape = [
            backbone_shape_out[i].channels for i in self.in_features]
        logger.info('backboneshape: {}, size_divisibility: {}'.format(
            backbone_shape, self.size_divisibility))

        # FPN only need P3, P4, P5
        self.neck = YOLOPAFPN(
            depth=self.depth_mul, width=self.width_mul, in_features=self.in_features[1:])

        # prepare for OrienHead, need P2, P3, P4, P5
        orien_dim = self.num_anchors * 6
        self.orien_head = OrienHead(orien_dim=orien_dim, up_channels=cfg.MODEL.YOLO.ORIEN_HEAD.UP_CHANNELS,
                                    in_features=self.in_features, input_shape=backbone_shape_out)

        ch = backbone_shape[1:][::-1]
        self.m = nn.ModuleList(
            nn.Conv2d(x,  self.num_anchors * (5 + self.num_classes), 1) for x in ch)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x / 255. - pixel_mean) / pixel_std
        self.padded_value = cfg.MODEL.PADDED_VALUE

        # ANCHORS_MASK =
        # self.anchor_mask = ANCHORS_MASK
        print(self.anchors)
        multi_losses = OrienMaskYOLOMultiScaleLoss(grid_size=[[17, 17], [34, 34], [68, 68]],
                                                   image_size=None,
                                                   anchors=self.anchors,
                                                   anchor_mask=self.anchor_mask,
                                                   num_classes=self.num_classes,
                                                   center_region=0.6,
                                                   valid_region=0.6,
                                                   label_smooth=False,
                                                   obj_ignore_threshold=self.ignore_threshold,
                                                   weight=[
                                                       1, 1, 1, 1, 1, 20, 20],
                                                   scales_weight=[1, 1, 1])
        self.loss_evaluators = multi_losses.get_losses()
        self.to(self.device)

    def update_iter(self, i):
        self.iter = i

    def preprocess_image(self, batched_inputs, training):
        # for a in batched_inputs:
        #     img = a["image"].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        #     ins = a['instances']
        #     bboxes = ins.gt_boxes.tensor.cpu().numpy().astype(int)
        #     clss = ins.gt_classes.cpu().numpy()
        #     im = img.copy()
        #     bit_masks = ins.gt_masks.tensor.cpu().numpy()
        #     print(bit_masks.shape)
        #     # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        #     im = vis_bitmasks(im, bit_masks)
        #     im = visualize_det_cv2_part(im, None, clss, bboxes, is_show=True)

        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        images = [self.normalizer(x) for x in images]
        images = [x.type(torch.float) for x in images]

        images = ImageList.from_tensors(
            images, size_divisibility=self.size_divisibility, pad_value=self.padded_value/255.)
        # logger.info('images ori shape: {}'.format(images.tensor.shape))
        # logger.info('images ori shape: {}'.format(images.image_sizes))

        # sync image size for all gpus
        comm.synchronize()
        if training and self.iter - 89999 > 0 and not self.use_l1:
            meg = torch.BoolTensor(1).to(self.device)
            comm.synchronize()
            if comm.is_main_process():
                logger.info(
                    '[master] enable l1 loss now at iter: {}'.format(self.iter))
                # enable l1 loss at last 50000 iterations
                meg.fill_(True)

            if comm.get_world_size() > 1:
                comm.synchronize()
                dist.broadcast(meg, 0)
            # self.head.use_l1 = meg.item()
            self.use_l1 = meg.item()
            comm.synchronize()
            logger.info(
                'check head l1: {}'.format(self.use_l1))

        if training:
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(self.device) for x in batched_inputs
                ]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10)
                gt_instances = [
                    x["targets"].to(self.device) for x in batched_inputs
                ]
            else:
                gt_instances = None

            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(self.device) for x in batched_inputs
                ]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10)
                gt_instances = [
                    x["targets"].to(self.device) for x in batched_inputs
                ]
            else:
                gt_instances = None

            if gt_instances:
                _, _, h, w = images.tensor.shape
                for i, instance in enumerate(gt_instances):
                    # convert x1y1x2y2 default to cxcywh
                    instance.gt_boxes.tensor = BoxModeMy.convert(
                        instance.gt_boxes.tensor, from_mode=BoxModeMy.XYXY_ABS, to_mode=BoxModeMy.XYWH_ABS)
            labels = gt_instances
        else:
            labels = None

        self.iter += 1
        return images, labels, images.image_sizes

    def preprocess_input(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = [F.interpolate(x, size=(640, 640)) for x in images]
        images = torch.cat(images, dim=0)
        x = images.permute(0, 3, 1, 2)
        # x = F.interpolate(x, size=(640, 640))
        # x = F.interpolate(x, size=(512, 960))
        # x = self.normalizer(x)
        return x

    def forward(self, batched_inputs):
        images, labels, image_ori_sizes = self.preprocess_image(
            batched_inputs, self.training)

        # batched_inputs[0]['image'] = images.tensor[0].cpu() * 255
        # self.visualize_data(batched_inputs[0])
        if self.eval:
            t0 = time.time()

        x = images.tensor
        img_size = x.shape[-2:]
        # logger.info('img size: {}'.format(img_size))

        out_features = self.backbone(x)
        # 32, 16, 8, l, m, s -> change to normal order, 8, 16, 32
        outputs = self.neck(out_features)
        # for i, a in enumerate(outputs):
        #     print(i, a.shape)
        # print(out_features[self.in_features[0]].shape)

        outs_boxes = []
        for i, x in enumerate(outputs[::-1]):
            outs_boxes.append(self.m[i](x))

        # reverse outputs from PAN to 32, 16, 8
        outs_oriens = self.orien_head(
            [out_features[self.in_features[0]], *outputs])
        # 32, 16, 8
        outs = list(zip(outs_boxes, outs_oriens))

        if self.training:
            if self.use_l1:
                for l in self.loss_evaluators:
                    l.use_l1 = True
            losses = [
                loss_evaluator(out, labels, img_size) for out, loss_evaluator in zip(
                    outs, self.loss_evaluators)
            ]
            losses_dict = {}
            for key in losses[0].keys():
                losses_dict[key] = sum([loss[key] for loss in losses])
            return losses_dict
        else:
            preds_boxes = []
            preds_oriens = []
            xys = []
            whs = []
            dets_anchor_idxes = []
            for out, loss_evaluator in zip(outs, self.loss_evaluators):
                pred_boxes_obj_cls, pixel_orien, xy, wh, dets_anchor_idx = loss_evaluator(
                    out, labels, img_size)
                preds_boxes.append(pred_boxes_obj_cls)
                preds_oriens.append(pixel_orien)
                # preds_oriens.insert(0, pixel_orien)
                xys.append(xy)
                whs.append(wh)
                dets_anchor_idxes.append(dets_anchor_idx)

            predictions = torch.cat(preds_boxes, 1)
            preds_oriens = torch.cat(preds_oriens, 1)
            xys = torch.cat(xys, 1)
            whs = torch.cat(whs, 1)
            dets_anchor_idxes = torch.cat(dets_anchor_idxes, 1)

            t1 = time.time()

            detections, pred_masks = postprocess_yolomask(predictions,
                                                          preds_oriens, xys, whs, dets_anchor_idxes,
                                                          self.num_classes,
                                                          self.conf_threshold,
                                                          #   0.2,
                                                          self.nms_threshold, orien_thre=0.3)
            results = []
            for idx, out in enumerate(detections):
                image_size = image_ori_sizes[idx]

                if out is None:
                    out = x.new_zeros((0, 7))
                    pred_mask = torch.tensor([])
                else:
                    pred_mask = pred_masks[idx]
                    pred_mask = pred_mask[:, :image_size[0], :image_size[1]]

                result = Instances(image_size)
                result.pred_boxes = Boxes(out[:, :4])
                # result.scores = out[:, 5] * out[:, 4]
                result.scores = out[:, 4]
                result.pred_classes = out[:, -1]
                # clip mask to ori image sizes since mask added padding
                result.pred_bit_masks = pred_mask
                results.append(result)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            # return processed_results, t1-t0
            return processed_results


class OrienHead(nn.Module):

    def __init__(self, orien_dim=18, up_channels=64, in_features: List[str] = None, input_shape: List[ShapeSpec] = None):
        super(OrienHead, self).__init__()

        # orien_dim must be 3 * anchor_num

        # (todo) make OrienHead dynamic with different backbone
        # input_shape should be 4d, 8d, 16d, 32d -> P2, P3, P4, P5
        assert len(
            input_shape) == 4, 'current support P2, P3, P4, P5 as input of OrienHead.'
        # self.up32 = self._build_up(512, 64, 8)
        # self.up16 = self._build_up(256, 64, 4)
        # self.up8 = self._build_up(128, 64, 2)
        # self.up4 = conv_bn_leaky(64, 64, 1)

        # might expanded to 128?
        # self.upped_channels = 64
        self.upped_channels = up_channels
        # self.upped_channels = 90

        self.up_levels_2to5 = nn.ModuleList()
        for i, s in enumerate(in_features):
            if i == 0:
                l = conv_bn_leaky(
                    input_shape[in_features[i]].channels, self.upped_channels, 1)
            else:
                l = self._build_up(
                    input_shape[in_features[i]].channels, self.upped_channels, 2**i)
            self.up_levels_2to5.append(l)

        self.neck_orien = self._build_neck(len(
            input_shape) * self.upped_channels, len(input_shape) * self.upped_channels // 2)

        self.orien_dim = orien_dim
        self.orien_m = self._build_orien_m(
            len(input_shape) * self.upped_channels // 2, orien_dim)
        # self.to(device)

    @classmethod
    def _build_neck(cls, in_channels, out_channels):
        return nn.Sequential(
            conv_bn_leaky(in_channels, out_channels, 1),
            conv_bn_leaky(out_channels, out_channels * 2, 3, padding=1),
            conv_bn_leaky(out_channels * 2, out_channels, 1),
            conv_bn_leaky(out_channels, out_channels * 2, 3, padding=1),
            conv_bn_leaky(out_channels * 2, out_channels, 1)
        )

    @classmethod
    def _build_up(cls, in_channels, out_channels, upsample=2):
        return nn.Sequential(
            conv_bn_leaky(in_channels, out_channels, 1),
            NearestUpsample(scale_factor=upsample)
        )

    @classmethod
    def _build_orien_m(cls, in_channels, out_channels):
        return nn.Sequential(
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            conv_bn_leaky(in_channels * 2, in_channels, 1),
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            conv_bn_leaky(in_channels * 2, in_channels, 1),
            conv_bn_leaky(in_channels, in_channels * 2, 3, padding=1),
            nn.Conv2d(in_channels * 2, out_channels, 1)
        )

    def forward(self, x):
        # out3 from backbone biggest fea
        # fpn_out32, fpn_out16, fpn_out8, out4 = x
        # print('fpn_out32: ', fpn_out32.shape)
        # print('fpn_out16: ', fpn_out16.shape)
        # print('fpn_out8: ', fpn_out8.shape)
        # print('out4: ', out4.shape)

        # x in order of: P2, P3, P4, P5
        outs = []
        for up, a in zip(self.up_levels_2to5, x):
            o = up(a)
            outs.append(o)
        outs = torch.cat(outs, dim=1)
        # 64*4 -> 256 | 128*4 -> 512
        oriens = self.neck_orien(outs)

        # oriens = self.neck4(torch.cat([
        #     self.up32(fpn_out32), self.up16(
        #         fpn_out16), self.up8(fpn_out8), self.up4(out4)
        # ], dim=1))

        oriens = self.orien_m(oriens)
        # orien32, orien16, orien8 = torch.split(
        #     oriens, self.orien_dim // 3, dim=1)  # na * 2
        orien8, orien16, orien32 = torch.split(
            oriens, self.orien_dim // 3, dim=1)  # na * 2
        return orien32, orien16, orien8


class OrienMaskYOLOLoss(nn.Module):
    def __init__(self, grid_size, image_size, anchors, anchor_mask, num_classes, center_region=0.6, valid_region=0.6,
                 label_smooth=False, obj_ignore_threshold=0.5, weight=None):
        super(OrienMaskYOLOLoss, self).__init__()
        self.device = None
        self.grid_h, self.grid_w = _pair(grid_size)
        self.image_h = None
        self.image_w = None

        self.num_anchors = len(anchor_mask)
        self.anchor_mask = anchor_mask if anchor_mask else list(
            range(self.num_anchors))
        self.anchors = anchors
        self.num_classes = num_classes
        self.dets_anchor_idx = None
        self.base_xy = None

        # coordinate ratio between grid-level and pixel-level
        self.grid_wh = torch.tensor([self.grid_w, self.grid_h]).float()
        self.image_wh = None
        self.scale_wh = None
        # store pixel-level / grid-level / normalized all anchors
        self.pixel_all_anchors = torch.tensor(anchors).float()
        self.grid_all_anchors = None
        # store selected anchors with anchor_mask
        self.pixel_anchors = self.pixel_all_anchors[anchor_mask]
        self.grid_anchors = None
        # store mesh indices
        self.grid_mesh_y, self.grid_mesh_x = torch.meshgrid([
            torch.arange(self.grid_h, dtype=torch.float32),
            torch.arange(self.grid_w, dtype=torch.float32)
        ])
        self.grid_mesh_xy = torch.stack(
            [self.grid_mesh_x, self.grid_mesh_y], dim=-1)
        self.pixel_mesh_y = None
        self.pixel_mesh_x = None
        self.pixel_mesh_xy = None

        # loss type
        self.center_region = center_region
        self.valid_region = valid_region
        self.label_smooth = 1.0 / max(num_classes, 40) if label_smooth else 0
        self.obj_ignore_threshold = obj_ignore_threshold

        self.l1 = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCELoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')
        self.use_l1 = False
        self.eps = 1e-8

    def _set_device(self, device):
        # update device of tensors for convenience
        self.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))

    def _adapt_dynamic_hw(self, predict, image_size):
        # pred_bbox (nB, nA * (coord4 + obj1 + cls), grid_h, grid_w)
        b, _, h, w = predict.shape
        device = predict.device
        self.grid_h = h
        self.grid_w = w

        self.image_h, self.image_w = image_size

        # coordinate ratio between grid-level and pixel-level
        self.grid_wh = torch.tensor(
            [self.grid_w, self.grid_h]).float().to(device)
        self.image_wh = torch.tensor(
            [self.image_w, self.image_h]).float().to(device)
        self.scale_wh = self.image_wh / self.grid_wh
        # store pixel-level / grid-level / normalized all anchors
        self.pixel_all_anchors = torch.tensor(self.anchors).float().to(device)
        self.grid_all_anchors = self.pixel_all_anchors / self.scale_wh
        # store selected anchors with anchor_mask
        self.pixel_anchors = self.pixel_all_anchors[self.anchor_mask]
        self.grid_anchors = self.grid_all_anchors[self.anchor_mask]
        self.normalized_anchors = self.pixel_anchors / self.image_wh
        # store mesh indices
        self.grid_mesh_y, self.grid_mesh_x = torch.meshgrid([
            torch.arange(self.grid_h, dtype=torch.float32),
            torch.arange(self.grid_w, dtype=torch.float32)
        ])
        self.grid_mesh_xy = torch.stack(
            [self.grid_mesh_x, self.grid_mesh_y], dim=-1).to(device)
        self.pixel_mesh_y, self.pixel_mesh_x = torch.meshgrid([
            torch.arange(self.image_h, dtype=torch.float32).to(device),
            torch.arange(self.image_w, dtype=torch.float32).to(device)
        ])
        self.pixel_mesh_xy = torch.stack(
            (self.pixel_mesh_x, self.pixel_mesh_y), dim=-1).to(device)

        # for orien
        # since anchor_mask is oppsite, you must opposite pred_orien later
        anchor_idx = torch.tensor(self.anchor_mask, device=self.device).view(
            self.num_anchors, 1, 1)
        anchor_idx = anchor_idx.expand(b, self.num_anchors, h, w).contiguous()
        self.dets_anchor_idx = anchor_idx.view(b, -1)

        self.base_xy = torch.zeros(self.pixel_anchors.size(
            0), 2, self.image_h, self.image_w, device=self.device)
        base_y, base_x = torch.meshgrid(
            [torch.arange(self.image_h, device=self.device, dtype=torch.float) / self.image_h * self.grid_h,
             torch.arange(self.image_w, device=self.device, dtype=torch.float) / self.image_w * self.grid_w])
        self.base_xy = torch.stack([base_x, base_y], dim=0)
        self.base_xy = self.base_xy.expand(
            self.pixel_anchors.size(0), 2, self.image_h, self.image_w)

    def forward(self, predict, target=None, image_size=None):
        # pred_bbox (nB, nA * (coord4 + obj1 + cls), grid_h, grid_w)
        # pred_orien (nB, nA * (x1 + y1), image_h, image_w)
        pred_bbox, pred_orien = predict
        if pred_bbox.device != self.device:
            self._set_device(pred_bbox.device)

        self._adapt_dynamic_hw(pred_bbox, image_size)
        # print('forward loss. ', pred_bbox.shape)
        # print('forward loss. ', pred_orien.shape)

        nB = pred_bbox.size(0)
        nA = self.num_anchors
        nH, nW = self.grid_h, self.grid_w

        # pred_bbox (nB, nA, grid_h, grid_w, coord4 + obj1 + cls)
        # pred_orien (nB, nA, image_h, image_w, x1 + y1)
        pred_bbox = pred_bbox.view(
            nB, nA, -1, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        pred_orien = F.interpolate(
            pred_orien, scale_factor=4, mode='bilinear', align_corners=False)
        pred_orien = pred_orien.view(nB, nA, 2, self.image_h, self.image_w)

        # predict x, y, w, h, obj, cls, orien
        pred_xy = pred_bbox[..., 0:2].sigmoid()
        pred_wh = pred_bbox[..., 2:4]
        pred_obj = pred_bbox[..., 4].sigmoid()
        pred_cls = pred_bbox[..., 5:].sigmoid()

        # predict boxes
        # xy = sigmoid(xy) + grid_xy
        # wh = exp(wh) * anchor_wh
        pred_boxes = torch.zeros(nB, nA * nH * nW, 4, device=self.device)

        if target is not None:
            pred_boxes[..., 0:2] = (
                pred_xy.detach() + self.grid_mesh_xy).view(nB, -1, 2)
            pred_boxes[..., 2:4] = (pred_wh.detach().exp(
            ) * self.grid_anchors.view(1, -1, 1, 1, 2)).view(nB, -1, 2)
            # nb, 3, 2, h, w -> nb, 3, h, w, 2
            pred_orien = pred_orien.permute(0, 1, 3, 4, 2).contiguous()

            # build target
            bbox_pos_mask, bbox_neg_mask, bbox_pos_scale, txy, twh, tiou, tcls, \
                orien_pos_mask, orien_neg_mask, torien \
                = self.build_targets(pred_boxes, target, image_size)

            if not torch.isfinite(pred_wh).all():
                logger.error('pred_wh not finite')
                exit()

            # (todo) add ciou loss here
            # calculate bbox loss
            if self.use_l1:
                # loss_xy = (self.bce(pred_xy, txy) *
                #        bbox_pos_scale.unsqueeze(-1)).sum() / nB
                loss_xy = (self.mse(pred_xy, txy) *
                           bbox_pos_scale.unsqueeze(-1)).sum() / nB
                loss_wh = (self.mse(pred_wh, twh) *
                           bbox_pos_scale.unsqueeze(-1)).sum() / 2 / nB

            # ciou
            mask_viewed = bbox_pos_mask.view(nB, -1).to(torch.bool)
            tgt_scale = bbox_pos_scale.view(nB, -1)

            pboxes = torch.cat([pred_xy, pred_wh], axis=-1)
            pboxes = pboxes.view(nB, -1, 4)
            tboxes = torch.cat([txy, twh], axis=-1)
            tboxes = tboxes.view(nB, -1, 4).to(self.device)

            tboxes = tboxes[mask_viewed]
            pboxes = pboxes[mask_viewed]
            tgt_scale = tgt_scale[mask_viewed]

            if pboxes.shape[0] > 0:
                lbox = ciou(pboxes, tboxes, sum=False).to(pboxes.device)
                lbox = tgt_scale*lbox.T
                lbox = lbox.sum()
            else:
                lbox = torch.tensor(self.eps).to(pboxes.device)
            # ciou end

            loss_obj_all = self.bce(pred_obj, bbox_pos_mask)
            loss_obj_pos = (loss_obj_all * bbox_pos_mask).sum() / nB
            loss_obj_neg = (loss_obj_all * bbox_neg_mask).sum() / nB
            loss_cls = (self.bce(pred_cls, tcls) *
                        bbox_pos_mask.unsqueeze(-1)).sum() / nB

            # calculate orien loss
            num_orien_pos = orien_pos_mask.sum()
            num_orien_neg = orien_neg_mask.sum()
            loss_orien_all = self.smoothl1(pred_orien, torien)
            loss_orien_pos = (loss_orien_all * orien_pos_mask.unsqueeze(-1)).sum() \
                / num_orien_pos * bbox_pos_mask.sum() / nB \
                if num_orien_pos > 0 else pred_orien.new_zeros([])
            loss_orien_neg = (loss_orien_all * orien_neg_mask.unsqueeze(-1)).sum() \
                / num_orien_neg * bbox_pos_mask.sum() / nB \
                if num_orien_neg > 0 else pred_orien.new_zeros([])

            loss_items = {
                "loss_box": (lbox / nB) * 1.2,
                "loss_obj_pos": loss_obj_pos,
                "loss_obj_neg": loss_obj_neg,
                "loss_cls": loss_cls,
                "loss_orien_pos": loss_orien_pos * 1.1,
                "loss_orien_neg": loss_orien_neg,
            }
            if self.use_l1:
                loss_items['loss_xy'] = loss_xy
                loss_items['loss_wh'] = loss_wh
            return loss_items
        else:
            pred_wh = pred_wh.detach().exp()
            pred_boxes[..., 0:2] = (
                pred_xy.detach() + self.grid_mesh_xy).view(nB, -1, 2)
            pred_boxes[..., 2:4] = (
                pred_wh * self.grid_anchors.view(1, -1, 1, 1, 2)).view(nB, -1, 2)

            # for orien mask recover
            orien_xy = pred_boxes[..., 0:2].clone()
            orien_wh = (pred_wh * self.grid_anchors.view(1, -
                                                         1, 1, 1, 2)).view(nB, -1, 2)

            # recover boxes
            pred_boxes[..., 0:2] = pred_boxes[..., 0:2] * self.scale_wh
            pred_boxes[..., 2:4] = pred_boxes[..., 2:4] * self.scale_wh
            pred_obj = pred_obj.view(nB, -1, 1)
            pred_cls = pred_cls.view(nB, -1, self.num_classes)
            pred_boxes_obj_cls = torch.cat(
                [pred_boxes, pred_obj, pred_cls], dim=-1)

            pixel_orien = pred_orien * self.grid_anchors.view(-1, 2, 1, 1) / 2
            pixel_orien += self.base_xy
            return pred_boxes_obj_cls, pixel_orien, orien_xy, orien_wh, self.dets_anchor_idx

    def build_targets(self, pred_boxes, target, image_size):
        device = pred_boxes.device
        bbc = [
            torch.cat(
                [instance.gt_boxes.tensor,
                 instance.gt_classes.float().unsqueeze(-1),
                 ], dim=-1
            )
            for i, instance in enumerate(target)
        ]
        bbc = torch.cat(bbc, dim=0).to(device)
        indexes = torch.tensor(
            [0] + [i.gt_boxes.tensor.shape[0] for i in target])
        gt_bbox = bbc[:, :4]
        gt_cls = bbc[:, 4].type(torch.int64)
        gt_index = torch.cumsum(indexes, dim=0).type(torch.int).to(device)
        img_h, img_w = image_size
        gt_mask = []
        for ins in target:
            a_masks = ins.gt_masks.tensor
            n, h, w = a_masks.shape
            a_masks_t = torch.zeros([n, img_h, img_w]).to(device)
            a_masks_t[:, :h, :w] = a_masks
            gt_mask.append(a_masks_t)
        gt_mask = torch.cat(gt_mask, dim=0).to(device)

        nB = len(gt_index) - 1
        nA = self.num_anchors
        nH, nW = self.grid_h, self.grid_w

        bbox_pos_mask = torch.zeros(nB, nA, nH, nW, device=self.device)
        bbox_neg_mask = torch.ones(nB, nA, nH, nW, device=self.device)
        bbox_pos_scale = torch.zeros(nB, nA, nH, nW, device=self.device)
        txy = torch.zeros(nB, nA, nH, nW, 2, device=self.device)
        twh = torch.zeros(nB, nA, nH, nW, 2, device=self.device)
        tiou = torch.zeros(nB, nA, nH, nW, device=self.device)
        tcls = torch.full((nB, nA, nH, nW, self.num_classes),
                          self.label_smooth, device=self.device, dtype=torch.float)
        orien_mask = torch.zeros(
            nB, nA, self.image_h, self.image_w, device=self.device, dtype=torch.long)
        torien = torch.zeros(nB, nA, self.image_h,
                             self.image_w, 2, device=self.device)

        # use grid as unit size
        gt_bbox = gt_bbox * \
            torch.tensor([nW/img_w, nH/img_h, nW/img_w, nH/img_h], device=self.device,
                         dtype=torch.float32)

        for b in range(nB):
            # skip sample with no instance
            gt_index_current, gt_index_next = gt_index[b], gt_index[b + 1]
            if gt_index_current == gt_index_next:
                continue

            # take ground truth of b-th sample
            # refer to collate_fn for details
            gt_bbox_b = gt_bbox[gt_index_current:gt_index_next]
            gt_cls_b = gt_cls[gt_index_current:gt_index_next]
            gt_mask_b = gt_mask[gt_index_current:gt_index_next]

            # ignore predictions if iou(pred, ground_true) is larger than threshold
            iou_pred_gt = bbox_ious2(pred_boxes[b], gt_bbox_b)
            is_ignore = (iou_pred_gt > self.obj_ignore_threshold).any(dim=1)
            bbox_neg_mask[b].masked_fill_(
                is_ignore.view_as(bbox_neg_mask[b]), 0)

            # match ground truth with anchor according to iou
            # remove anchors with argmax(iou) belonging to other scale
            iou_gt_anchors = anchor_ious(
                gt_bbox_b[:, 2:], self.grid_all_anchors)
            match_index = iou_gt_anchors.argmax(dim=1)
            match_mask = torch.tensor(
                [best_n in self.anchor_mask for best_n in match_index], device=self.device)
            match_index = torch.masked_select(match_index, match_mask)
            if match_index.numel() == 0:
                continue

            # positive indices
            match_anchor = torch.zeros_like(match_index)
            for idx, mask_id in enumerate(self.anchor_mask):
                match_anchor[match_index == mask_id] = idx
            gt_xy, gt_wh = gt_bbox_b[match_mask].split(2, dim=-1)
            grid_x = torch.clamp(torch.floor(gt_xy[:, 0]), 0, nW - 1).long()
            grid_y = torch.clamp(torch.floor(gt_xy[:, 1]), 0, nH - 1).long()
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)

            # bbox targets
            # take grid as unit size
            bbox_pos_mask[b, match_anchor, grid_y, grid_x] = 1
            bbox_neg_mask[b, match_anchor, grid_y, grid_x] = 0
            bbox_pos_scale[b, match_anchor, grid_y, grid_x] = 2 - \
                torch.prod(gt_wh, dim=-1) / (nW * nH)
            txy[b, match_anchor, grid_y, grid_x] = gt_xy - grid_xy.float()
            twh[b, match_anchor, grid_y, grid_x] = torch.log(
                gt_wh / self.grid_anchors[match_anchor])

            tcls[b, match_anchor, grid_y, grid_x,
                 gt_cls_b[match_mask]] = 1 - self.label_smooth
            match_gt = torch.arange(gt_bbox_b.size(
                0), device=self.device)[match_mask]
            tiou[b, match_anchor, grid_y, grid_x] = \
                iou_pred_gt.view(
                    *tiou.shape[1:], -1)[match_anchor, grid_y, grid_x, match_gt]

            # orientation targets
            # take pixel as coordinate unit
            gt_mask_b = gt_mask_b[match_mask]
            pixel_x = (gt_xy[:, 0] * self.scale_wh[0]).flatten()
            pixel_y = (gt_xy[:, 1] * self.scale_wh[1]).flatten()
            # extend box region
            valid_x = (gt_xy[:, 0] * self.scale_wh[0]).flatten()
            valid_y = (gt_xy[:, 1] * self.scale_wh[1]).flatten()
            valid_w = ((gt_wh[:, 0] * self.valid_region + 0.5)
                       * self.scale_wh[0]).flatten()
            valid_h = ((gt_wh[:, 1] * self.valid_region + 0.5)
                       * self.scale_wh[1]).flatten()
            center_wh = torch.stack(
                [valid_w, valid_h], dim=-1) / self.valid_region * self.center_region
            region_x1 = (valid_x - valid_w).clamp(min=0,
                                                  max=self.image_w - 1).round().long()
            region_x2 = (valid_x + valid_w).clamp(min=0,
                                                  max=self.image_w - 1).round().long() + 1
            region_y1 = (valid_y - valid_h).clamp(min=0,
                                                  max=self.image_h - 1).round().long()
            region_y2 = (valid_y + valid_h).clamp(min=0,
                                                  max=self.image_h - 1).round().long() + 1

            for gt_inst_mask, a, x1, x2, y1, y2, x, y, wh in zip(
                    gt_mask_b, match_anchor, region_x1, region_x2,
                    region_y1, region_y2, pixel_x, pixel_y, center_wh):
                # relative position to box center
                offset_xy = self.pixel_mesh_xy.clone()
                offset_xy[..., 0] -= x
                offset_xy[..., 1] -= y

                # clone data to avoid ambiguity
                orien_mask_inst = orien_mask[b, a].clone()
                torien_inst = torien[b, a].clone()

                # roi region
                is_roi = (self.pixel_mesh_x >= float(x1)) & (self.pixel_mesh_x < float(x2)) & \
                         (self.pixel_mesh_y >= float(y1)) & (
                             self.pixel_mesh_y < float(y2))

                # current instance region in roi
                # set orien_mask = -1 and set torien pointing to the base position
                is_inst = (is_roi & (gt_inst_mask > 0))
                orien_mask_inst.masked_fill_(is_inst, -1)
                torien_inst = torch.where(
                    is_inst.unsqueeze(-1).expand_as(offset_xy), offset_xy, torien_inst)

                # no instance region in roi
                # set orien_mask += 1 and set torien pointing to the boarder of extended box
                not_inst = (is_roi & (gt_inst_mask == 0)
                            & (orien_mask_inst >= 0))
                orien_mask_inst += not_inst.long()
                offset_xy_length = offset_xy.abs().clamp(min=1e-8)
                neg_offset_scale = (
                    wh / offset_xy_length).clamp(min=1).min(dim=-1)[0] - 1
                neg_offset = neg_offset_scale.unsqueeze(
                    -1) * offset_xy.sign() * offset_xy_length
                torien_inst = torch.where(not_inst.unsqueeze(-1).expand_as(offset_xy),
                                          torien_inst + neg_offset, torien_inst)

                # update orien_mask and torien
                orien_mask[b, a] = orien_mask_inst.clone()
                torien[b, a] = torien_inst.clone()

        # set negative ones as the average of their torien sums
        orien_pos_mask = (orien_mask < 0).float()
        orien_neg_mask = (orien_mask > 0).float()
        is_invalid = (orien_mask == 0)
        torien = torien / (self.pixel_anchors.view(1, nA, 1, 1, 2) / 2)
        orien_mask.masked_fill_(is_invalid, 1000)
        torien = torien / orien_mask.unsqueeze(-1).float()
        return (bbox_pos_mask, bbox_neg_mask, bbox_pos_scale, txy, twh, tiou, tcls,
                orien_pos_mask, orien_neg_mask, torien)


class OrienMaskYOLOMultiScaleLoss:
    def __init__(self, grid_size, image_size, anchors, anchor_mask, num_classes,
                 center_region=0.6, valid_region=0.7, label_smooth=False,
                 obj_ignore_threshold=0.5, weight=None, scales_weight=None):
        assert len(grid_size) == len(anchor_mask)
        self.grid_size = grid_size
        self.image_size = image_size
        self.anchors = anchors
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.center_region = center_region
        self.valid_region = valid_region
        self.label_smooth = label_smooth
        self.obj_ignore_threshold = obj_ignore_threshold
        self.weight = weight
        self.num_scales = len(grid_size)
        super(OrienMaskYOLOMultiScaleLoss, self).__init__()

    def get_losses(self):
        loss = []
        for i in range(self.num_scales):
            loss.append(
                OrienMaskYOLOLoss(
                    self.grid_size[i], self.image_size, self.anchors, self.anchor_mask[i],
                    self.num_classes, self.center_region, self.valid_region,
                    self.label_smooth, self.obj_ignore_threshold
                )
            )
        return loss
