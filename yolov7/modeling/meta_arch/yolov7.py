#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Copyright (c) Lucas Jin. telegram: lucasjin


import logging
import random
from collections import OrderedDict
import cv2
from torch._C import device
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling.meta_arch import build
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    BACKBONE_REGISTRY,
    ResNet,
    ResNetBlockBase,
    META_ARCH_REGISTRY,
)
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, boxes, image_list
from detectron2.utils import comm
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone

from alfred.dl.metrics.iou_loss import bboxes_iou as bboxes_iou2
from alfred.dl.metrics.iou_loss import ciou_loss, ciou
from alfred.utils.log import logger
from alfred.dl.torch.common import device
from .utils import generalized_batched_nms

from yolov7.utils.boxes import postprocess, bboxes_iou
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy

from yolov7.modeling.neck.yolo_fpn import YOLOFPN
from yolov7.modeling.neck.yolo_pafpn import YOLOPAFPN


__all__ = ["YOLOV7", "YOLOHead"]
supported_backbones = [
    "resnet",
    "res2net",
    "regnet",
    "swin",
    "efficient",
    "darknet",
    "pvt",
]


@META_ARCH_REGISTRY.register()
class YOLOV7(nn.Module):
    """
    YOLO model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, cfg):
        super(YOLOV7, self).__init__()

        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.neck_type = cfg.MODEL.YOLO.NECK.TYPE
        self.with_spp = cfg.MODEL.YOLO.NECK.WITH_SPP
        self.depth_mul = cfg.MODEL.YOLO.DEPTH_MUL
        self.width_mul = cfg.MODEL.YOLO.WIDTH_MUL

        self.max_iter = cfg.SOLVER.MAX_ITER
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.max_boxes_num = cfg.MODEL.YOLO.MAX_BOXES_NUM
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES

        self.change_iter = 10
        self.iter = 0

        assert (
            len([i for i in supported_backbones if i in cfg.MODEL.BACKBONE.NAME]) > 0
        ), "Only {} supported.".format(supported_backbones)

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.size_divisibility = (
            32
            if self.backbone.size_divisibility == 0
            else self.backbone.size_divisibility
        )
        backbone_shape = [backbone_shape[i].channels for i in self.in_features]

        if comm.is_main_process():
            logger.info("YOLO.ANCHORS: {}".format(cfg.MODEL.YOLO.ANCHORS))
            logger.info("backboneshape: {}".format(backbone_shape))

        # todo: wrap this to neck, support SPP , DarkNeck, PAN

        # # out 0
        # out_filter_0 = len(
        #     cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES)
        # self.out0 = self._make_embedding(
        #     [512, 1024], backbone_shape[-1], out_filter_0)

        # # out 1
        # out_filter_1 = len(
        #     cfg.MODEL.YOLO.ANCHORS[1]) * (5 + cfg.MODEL.YOLO.CLASSES)
        # self.out1_cbl = self._make_cbl(512, 256, 1)
        # self.out1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.out1 = self._make_embedding(
        #     [256, 512], backbone_shape[-2] + 256, out_filter_1)

        # # out 2
        # out_filter_2 = len(
        #     cfg.MODEL.YOLO.ANCHORS[2]) * (5 + cfg.MODEL.YOLO.CLASSES)
        # self.out2_cbl = self._make_cbl(256, 128, 1)
        # self.out2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.out2 = self._make_embedding(
        #     [128, 256], backbone_shape[-3] + 128, out_filter_2)

        if self.neck_type == "fpn":
            self.neck = YOLOFPN(
                width=self.width_mul,
                in_channels=backbone_shape,
                in_features=self.in_features,
                with_spp=self.with_spp,
            )
            # 256, 512, 1024 -> 1024, 512, 256
            self.m = nn.ModuleList(
                nn.Conv2d(
                    x, len(cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES), 1
                )
                for x in self.neck.out_channels
            )
        elif self.neck_type == "pafpn":
            width_mul = backbone_shape[0] / 256
            self.neck = YOLOPAFPN(
                depth=self.depth_mul, width=width_mul, in_features=self.in_features
            )
            self.m = nn.ModuleList(
                nn.Conv2d(
                    x, len(cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES), 1
                )
                for x in backbone_shape
            )
        else:
            logger.info(f"type: {self.neck_type} not valid, using default FPN neck.")
            self.neck = YOLOFPN(
                width=self.width_mul,
                in_channels=backbone_shape,
                in_features=self.in_features,
                with_spp=self.with_spp,
            )
            # 256, 512, 1024 -> 1024, 512, 256
            self.m = nn.ModuleList(
                nn.Conv2d(
                    x, len(cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES), 1
                )
                for x in self.neck.out_channels
            )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        self.padded_value = cfg.MODEL.PADDED_VALUE
        self.loss_evaluators = [
            YOLOHead(cfg, anchor, level)
            for level, anchor in enumerate(cfg.MODEL.YOLO.ANCHORS)
        ]
        self.to(self.device)

    def update_iter(self, i):
        self.iter = i

    def _make_cbl(self, _in, _out, ks):
        """cbl = conv + batch_norm + leaky_relu"""
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            _in, _out, kernel_size=ks, stride=1, padding=pad, bias=False
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(_out)),
                    ("relu", nn.LeakyReLU(0.1)),
                ]
            )
        )

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList(
            [
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
            ]
        )
        m.add_module(
            "conv_out",
            nn.Conv2d(
                filters_list[1],
                out_filter,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )
        return m

    def preprocess_image(self, batched_inputs, training):
        # for a in batched_inputs:
        #     img = a["image"].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        #     ins = a['instances']
        #     bboxes = ins.gt_boxes.tensor.cpu().numpy().astype(int)
        #     clss = ins.gt_classes.cpu().numpy()
        #     im = img.copy()
        #     im = visualize_det_cv2_part(im, None, clss, bboxes, is_show=True)

        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        images = [self.normalizer(x) for x in images]

        images = ImageList.from_tensors(
            images,
            size_divisibility=self.size_divisibility,
            pad_value=self.padded_value / 255.0,
        )
        # logger.info('images ori shape: {}'.format(images.tensor.shape))
        # logger.info('images ori shape: {}'.format(images.image_sizes))

        # sync image size for all gpus
        comm.synchronize()
        if training and self.iter == self.max_iter - 49990:
            meg = torch.BoolTensor(1).to(self.device)
            comm.synchronize()
            if comm.is_main_process():
                logger.info("[master] enable l1 loss now at iter: {}".format(self.iter))
                # enable l1 loss at last 50000 iterations
                meg.fill_(True)

            if comm.get_world_size() > 1:
                comm.synchronize()
                dist.broadcast(meg, 0)
            # self.head.use_l1 = meg.item()
            self.use_l1 = meg.item()
            comm.synchronize()

        if training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN,
                    "'targets' in the model inputs is now renamed to 'instances'!",
                    n=10,
                )
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            targets = [
                torch.cat(
                    [
                        instance.gt_classes.float().unsqueeze(-1),
                        instance.gt_boxes.tensor,
                    ],
                    dim=-1,
                )
                for instance in gt_instances
            ]
            labels = torch.zeros((bs, self.max_boxes_num, 5))
            # todo: what if targets more than max_boxes_num?
            for i, target in enumerate(targets):
                if target.shape[0] > self.max_boxes_num:
                    target = target[: self.max_boxes_num, :]
                labels[i][: target.shape[0]] = target
            labels[:, :, 1:] = labels[:, :, 1:]
        else:
            labels = None
        self.iter += 1
        return images, labels, images.image_sizes

    def forward(self, batched_inputs):
        images, labels, image_ori_sizes = self.preprocess_image(
            batched_inputs, self.training
        )

        # batched_inputs[0]['image'] = images.tensor[0].cpu() * 255
        # self.visualize_data(batched_inputs[0])

        x = images.tensor
        img_size = x.shape[-2:]
        # logger.info('img size: {}'.format(img_size))

        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 4:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        out_features = self.backbone(x)
        outputs = self.neck(out_features)

        # for a in outputs:
        #     print(a.shape)

        outs = []
        for i, x in enumerate(outputs):
            outs.append(self.m[i](x))
        # in large, medium, small order

        # # yolo branch 0
        # out0, out0_branch = _branch(self.out0, x0)  # large

        # #  yolo branch 1
        # x1_in = self.out1_cbl(out0_branch)
        # x1_in = self.out1_upsample(x1_in)
        # x1_in = torch.cat([x1_in, x1], 1)
        # out1, out1_branch = _branch(self.out1, x1_in)  # medium

        # #  yolo branch 2
        # x2_in = self.out2_cbl(out1_branch)
        # x2_in = self.out2_upsample(x2_in)
        # x2_in = torch.cat([x2_in, x2], 1)
        # out2, out2_branch = _branch(self.out2, x2_in)  # small
        # outputs = [out0, out1, out2]

        if self.training:
            losses = [
                loss_evaluator(out, labels, img_size)
                for out, loss_evaluator in zip(outs, self.loss_evaluators)
            ]
            if self.loss_type == "v7":
                keys = ["loss_iou", "loss_xy", "loss_wh", "loss_conf", "loss_cls"]
            else:
                keys = ["loss_x", "loss_y", "loss_w", "loss_h", "loss_conf", "loss_cls"]
            losses_dict = {}
            for key in keys:
                losses_dict[key] = sum([loss[key] for loss in losses])
            return losses_dict
        else:
            predictions_list = [
                loss_evaluator(out, labels, img_size)
                for out, loss_evaluator in zip(outs, self.loss_evaluators)
            ]

            predictions = torch.cat(predictions_list, 1)
            detections = postprocess(
                predictions, self.num_classes, self.conf_threshold, self.nms_threshold
            )

            results = []
            for idx, out in enumerate(detections):
                if out is None:
                    out = x.new_zeros((0, 7))
                # image_size = images.image_sizes[idx]
                image_size = image_ori_sizes[idx]
                result = Instances(image_size)
                result.pred_boxes = Boxes(out[:, :4])
                result.scores = out[:, 5] * out[:, 4]
                result.pred_classes = out[:, -1]
                results.append(result)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            # return processed_results, None
            return processed_results


class YOLOHead(nn.Module):
    def __init__(self, cfg, anchors, level):
        super(YOLOHead, self).__init__()
        self.level = level
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.all_anchors = np.array(cfg.MODEL.YOLO.ANCHORS).reshape([-1, 2])
        self.anchors = anchors
        self.ref_anchors = np.zeros((len(self.all_anchors), 4))
        self.ref_anchors[:, 2:] = self.all_anchors
        self.ref_anchors = torch.from_numpy(self.ref_anchors)
        self.anchor_ratio_thresh = cfg.MODEL.YOLO.LOSS.ANCHOR_RATIO_THRESH

        self.num_anchors = len(anchors)
        # self.num_anchors = self.ref_anchors.shape[0]
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.bbox_attrs = 5 + self.num_classes

        self.ignore_threshold = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        self.lambda_xy = cfg.MODEL.YOLO.LOSS.LAMBDA_XY
        self.lambda_wh = cfg.MODEL.YOLO.LOSS.LAMBDA_WH
        self.lambda_conf = cfg.MODEL.YOLO.LOSS.LAMBDA_CONF
        self.lambda_cls = cfg.MODEL.YOLO.LOSS.LAMBDA_CLS
        self.lambda_iou = cfg.MODEL.YOLO.LOSS.LAMBDA_IOU

        self.build_target_type = cfg.MODEL.YOLO.LOSS.BUILD_TARGET_TYPE  # v5 or default

        self.eps = 1e-8

        self.mse_loss = nn.MSELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")

        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input, targets=None, image_size=(416, 416)):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # image_size is input tensor size, we need convert anchor to this rel.
        stride_h = image_size[0] / in_h
        stride_w = image_size[1] / in_w

        scaled_anchors = [(a_w, a_h) for a_w, a_h in self.anchors]

        prediction = (
            input.view(bs, self.num_anchors, self.bbox_attrs, in_h, in_w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # place bbox_attr to last order

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # conf = torch.sigmoid(prediction[..., 4])       # Conf
        # pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        conf = prediction[..., 4]  # Conf
        pred_cls = prediction[..., 5:]  # Cls pred.

        def FloatTensor(x):
            return torch.FloatTensor(x).to(pred_cls.device)  # noqa

        def LongTensor(x):
            return torch.LongTensor(x).to(pred_cls.device)  # noqa

        # Calculate offsets for each grid
        grid_x = FloatTensor(
            torch.linspace(0, in_w - 1, in_w)
            .repeat(in_h, 1)
            .repeat(bs * self.num_anchors, 1, 1)
            .view(x.shape)
        )
        grid_y = FloatTensor(
            torch.linspace(0, in_h - 1, in_h)
            .repeat(in_w, 1)
            .t()
            .repeat(bs * self.num_anchors, 1, 1)
            .view(y.shape)
        )
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = prediction[..., :4].clone()

        # (todo) modified to adopt YOLOv5 style offsets
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        pred_boxes[..., 0] *= stride_w
        pred_boxes[..., 1] *= stride_h

        # check if is training
        if targets is not None:
            #  build target
            if self.build_target_type == "v5":
                (
                    mask,
                    obj_mask,
                    tx,
                    ty,
                    tw,
                    th,
                    tgt_scale,
                    tcls,
                    nlabel,
                ) = self.get_target_yolov5(
                    targets,
                    pred_boxes,
                    image_size,
                    in_w,
                    in_h,
                    stride_w,
                    stride_h,
                    self.ignore_threshold,
                )
            else:
                (
                    mask,
                    obj_mask,
                    tx,
                    ty,
                    tw,
                    th,
                    tgt_scale,
                    tcls,
                    nlabel,
                    num_fg,
                ) = self.get_target(
                    targets,
                    pred_boxes,
                    image_size,
                    in_w,
                    in_h,
                    stride_w,
                    stride_h,
                    self.ignore_threshold,
                )

            mask, obj_mask = mask.cuda(), obj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tgt_scale, tcls = tgt_scale.cuda(), tcls.cuda()

            if self.loss_type == "v7":
                # loss_conf = (obj_mask * self.bce_obj(conf, mask)).sum() / bs
                # mask is positive samples
                loss_obj = self.bce_obj(conf, mask)
                loss_obj = obj_mask * loss_obj
                # loss_obj_neg = (loss_obj * (1-mask)*obj_mask)
                # loss_obj = loss_obj_pos + loss_obj_neg
                loss_obj = loss_obj.sum()

                loss_cls = self.bce_cls(pred_cls[mask == 1], tcls[mask == 1]).sum()

                x = x.unsqueeze(-1)
                y = y.unsqueeze(-1)
                w = w.unsqueeze(-1)
                h = h.unsqueeze(-1)
                tx = tx.unsqueeze(-1)
                ty = ty.unsqueeze(-1)
                tw = tw.unsqueeze(-1)
                th = th.unsqueeze(-1)

                # loss_xy and loss_wh
                loss_x = torch.abs(x - tx)
                loss_y = torch.abs(y - ty)
                loss_xy = tgt_scale.unsqueeze(-1) * (loss_x + loss_y)
                loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

                loss_w = torch.abs(w - tw)
                loss_h = torch.abs(h - th)
                loss_wh = tgt_scale.unsqueeze(-1) * (loss_w + loss_h)
                loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

                # replace with iou loss
                mask_viewed = mask.view(bs, -1).to(torch.bool)
                tgt_scale = tgt_scale.view(bs, -1)

                pboxes = torch.cat([x, y, w, h], axis=-1)
                pboxes = pboxes.view(bs, -1, 4)

                tboxes = torch.cat([tx, ty, tw, th], axis=-1)
                tboxes = tboxes.view(bs, -1, 4).to(pred_boxes.device)

                tboxes = tboxes[mask_viewed]
                pboxes = pboxes[mask_viewed]
                tgt_scale = tgt_scale[mask_viewed]
                # print('tboxes: ', tboxes.shape)
                # print('pboxes: ', pboxes.shape)
                # print('tgt_scale: ', tgt_scale.shape)

                if pboxes.shape[0] > 0:
                    lbox = ciou(pboxes, tboxes, sum=False).to(pboxes.device)
                    lbox = tgt_scale * lbox.T
                    lbox = lbox.sum()
                else:
                    lbox = torch.tensor(self.eps).to(pboxes.device)

                loss = {
                    "loss_xy": loss_xy / bs,
                    "loss_wh": loss_wh / bs,
                    "loss_iou": (lbox / bs) * self.lambda_iou,
                    "loss_conf": (loss_obj / bs) * self.lambda_conf,
                    "loss_cls": (loss_cls / bs) * self.lambda_cls,
                }
            else:
                loss_conf = (obj_mask * self.bce_obj(conf, mask)).sum() / bs
                loss_cls = self.bce_cls(pred_cls[mask == 1], tcls[mask == 1]).sum() / bs

                loss_x = (
                    mask * tgt_scale * self.bce_loss(x * mask, tx * mask)
                ).sum() / bs
                loss_y = (
                    mask * tgt_scale * self.bce_loss(y * mask, ty * mask)
                ).sum() / bs
                loss_w = (
                    mask * tgt_scale * self.l1_loss(w * mask, tw * mask)
                ).sum() / bs
                loss_h = (
                    mask * tgt_scale * self.l1_loss(h * mask, th * mask)
                ).sum() / bs

                # we are not using loss_x, loss_y here, just using a simple ciou loss
                loss = {
                    "loss_x": loss_x * self.lambda_xy,
                    "loss_y": loss_y * self.lambda_xy,
                    "loss_w": loss_w * self.lambda_wh,
                    "loss_h": loss_h * self.lambda_wh,
                    "loss_conf": loss_conf * self.lambda_conf,
                    "loss_cls": loss_cls * self.lambda_cls,
                }
            return loss
        else:
            conf = torch.sigmoid(conf)
            pred_cls = torch.sigmoid(pred_cls)
            # Results
            output = torch.cat(
                (
                    pred_boxes.view(bs, -1, 4),
                    conf.view(bs, -1, 1),
                    pred_cls.view(bs, -1, self.num_classes),
                ),
                -1,
            )
            return output.data

    def get_target(
        self,
        target,
        pred_boxes,
        img_size,
        in_w,
        in_h,
        stride_w,
        stride_h,
        ignore_threshold,
    ):
        def FloatTensor(x):
            return torch.FloatTensor(x).to(pred_boxes.device)  # noqa

        bs = target.size(0)

        # logger.info('in_h, {}, in_w: {}'.format(in_h, in_w))
        # logger.info('stride_h, {}, stride_w: {}'.format(stride_h, stride_w))
        # logger.info('target shape: {}'.format(target.shape))

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        obj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tgt_scale = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        tcls = torch.zeros(
            bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False
        )
        nlabel = (target.sum(dim=2) > 0).sum(dim=1)
        gx_all = (target[:, :, 1] + target[:, :, 3]) / 2.0  # center x
        gy_all = (target[:, :, 2] + target[:, :, 4]) / 2.0  # center y
        gw_all = target[:, :, 3] - target[:, :, 1]  # width
        gh_all = target[:, :, 4] - target[:, :, 2]  # height
        gi_all = (gx_all / stride_w).to(torch.int16)
        gj_all = (gy_all / stride_h).to(torch.int16)

        num_fg = 0
        for b in range(bs):
            n = int(nlabel[b])
            if n == 0:
                continue

            truth_box = FloatTensor(np.zeros((n, 4)))
            truth_box[:, 2] = gw_all[b, :n]
            truth_box[:, 3] = gh_all[b, :n]
            truth_i = gi_all[b, :n]
            truth_j = gj_all[b, :n]

            # change match strategy, by not using IoU maxium
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(), self.ref_anchors.type_as(truth_box.cpu()), xyxy=False
            )
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            # so we know which level it belongs to, 3 might be len(anchors)
            best_n = best_n_all % 3
            best_n_mask = (best_n_all // 3) == self.level

            truth_box[:n, 0] = gx_all[b, :n]
            truth_box[:n, 1] = gy_all[b, :n]
            pred_box = pred_boxes[b]

            pred_ious = bboxes_iou(pred_box.view(-1, 4), truth_box, xyxy=False)
            # print(pred_box.shape)
            # pred_ious = bboxes_iou2(pred_box.view(-1, 4),
            #                         truth_box, x1y1x2y2=False, CIoU=True)

            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > ignore_threshold
            pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for t in range(best_n.shape[0]):
                if best_n_mask[t] == 1:
                    # belong's to current level
                    gi, gj = truth_i[t], truth_j[t]
                    gx, gy = gx_all[b, t], gy_all[b, t]
                    gw, gh = gw_all[b, t], gh_all[b, t]

                    a = best_n[t]

                    # Masks
                    mask[b, a, gj, gi] = 1  # 17, 17
                    obj_mask[b, a, gj, gi] = 1
                    num_fg += 1

                    # Coordinates
                    tx[b, a, gj, gi] = gx / stride_w - gi
                    ty[b, a, gj, gi] = gy / stride_h - gj
                    # Width and height
                    tw[b, a, gj, gi] = torch.log(gw / self.anchors[a][0] + 1e-16)
                    th[b, a, gj, gi] = torch.log(gh / self.anchors[a][1] + 1e-16)

                    tgt_scale[b, a, gj, gi] = 2.0 - gw * gh / (
                        img_size[0] * img_size[1]
                    )
                    # One-hot encoding of label
                    tcls[b, a, gj, gi, int(target[b, t, 0])] = 1

        num_fg = max(num_fg, 1)
        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls, nlabel, num_fg

    def get_target_yolov5(
        self,
        target,
        pred_boxes,
        img_size,
        in_w,
        in_h,
        stride_w,
        stride_h,
        ignore_threshold,
    ):
        def FloatTensor(x):
            return torch.FloatTensor(x).to(pred_boxes.device)  # noqa

        bs = target.size(0)

        # logger.info('in_h, {}, in_w: {}'.format(in_h, in_w))
        # logger.info('stride_h, {}, stride_w: {}'.format(stride_h, stride_w))
        # logger.info('target shape: {}'.format(target.shape))

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        obj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tgt_scale = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        tcls = torch.zeros(
            bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False
        )
        nlabel = (target.sum(dim=2) > 0).sum(dim=1)
        gx_all = (target[:, :, 1] + target[:, :, 3]) / 2.0  # center x
        gy_all = (target[:, :, 2] + target[:, :, 4]) / 2.0  # center y
        gw_all = target[:, :, 3] - target[:, :, 1]  # width
        gh_all = target[:, :, 4] - target[:, :, 2]  # height
        gi_all = (gx_all / stride_w).to(torch.int16)
        gj_all = (gy_all / stride_h).to(torch.int16)

        for b in range(bs):
            n = int(nlabel[b])
            if n == 0:
                continue

            truth_box = FloatTensor(np.zeros((n, 4)))
            truth_box[:, 2] = gw_all[b, :n]
            truth_box[:, 3] = gh_all[b, :n]
            truth_i = gi_all[b, :n]
            truth_j = gj_all[b, :n]

            # change match strategy, by not using IoU maxium
            # anchor_ious_all = bboxes_iou(truth_box.cpu(),
            #                              self.ref_anchors.type_as(truth_box.cpu()), xyxy=False)
            # print('anchor_ious_all: ', anchor_ious_all.shape) # [1, 9], [6, 9]
            # print(anchor_ious_all)
            # best_n_all = np.argmax(anchor_ious_all, axis=1)
            # print('best_n_all: ', best_n_all) # tensor([3, 3, 4, 4, 4, 4, 4, 3, 4, 3, 2, 3]) , 12 boxes
            # # so we know which level it belongs to, 3 might be len(anchors)
            # best_n = best_n_all % 3
            # # print(best_n)  # tensor([0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 2, 0])
            # best_n_mask = ((best_n_all // 3) == self.level)
            # print(best_n_mask) # tensor([False, False, False, False, False, False, False, False, False, False, False, False])

            # (todo) this strategy not work, find why
            anchor_indices_mask = get_matching_anchors(
                truth_box.cpu(),
                self.ref_anchors.type_as(truth_box.cpu()),
                xyxy=False,
                anchor_ratio_thresh=self.anchor_ratio_thresh,
            )
            # [[False, False, False, False,  True,  True, False, False, False],
            # [False, False, False, False, False, False, False,  True, False],
            # [False, False, False,  True,  True,  True, False, False, False],
            # [False,  True,  True,  True,  True, False, False, False, False]]  N x anchor_num
            # one box, might have more than one anchor in all 9 anchors
            # select mask of current level
            anchor_indices_mask = anchor_indices_mask[
                :,
                self.level * self.num_anchors : self.level * self.num_anchors
                + self.num_anchors,
            ]
            # now we get boxes anchor indices, of current level

            truth_box[:n, 0] = gx_all[b, :n]
            truth_box[:n, 1] = gy_all[b, :n]
            pred_box = pred_boxes[b]

            pred_ious = bboxes_iou(pred_box.view(-1, 4), truth_box, xyxy=False)

            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > ignore_threshold
            pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
            obj_mask[b] = ~pred_best_iou

            if anchor_indices_mask.shape[0] == 0:
                continue

            # best_n.shape[0] is GT nums
            for t in range(anchor_indices_mask.shape[0]):
                # if best_n_mask[t] == 1:
                # belong's to current level

                # we already filtered in mask
                gi, gj = truth_i[t], truth_j[t]
                gx, gy = gx_all[b, t], gy_all[b, t]
                gw, gh = gw_all[b, t], gh_all[b, t]

                # a = best_n[t]
                anchor_mask = anchor_indices_mask[t].to(torch.int)
                a = torch.argmax(anchor_mask)
                # a can not bigger than 3

                # Masks
                mask[b, a, gj, gi] = 1  # 17, 17
                obj_mask[b, a, gj, gi] = 1

                # Coordinates
                tx[b, a, gj, gi] = gx / stride_w - gi
                ty[b, a, gj, gi] = gy / stride_h - gj
                # Width and height
                tw[b, a, gj, gi] = torch.log(gw / self.anchors[a][0] + 1e-16)
                th[b, a, gj, gi] = torch.log(gh / self.anchors[a][1] + 1e-16)

                tgt_scale[b, a, gj, gi] = 2.0 - gw * gh / (img_size[0] * img_size[1])
                # One-hot encoding of label
                tcls[b, a, gj, gi, int(target[b, t, 0])] = 1

        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls, nlabel


def get_matching_anchors(gt_boxes, anchors, anchor_ratio_thresh=2.1, xyxy=True):
    """
    using YOLOv5 style choose anchors by given refanchors and gt_boxes
    we select anchors by comparing ratios rather than IoU
    bboxes_a: gt_boxes
    bboxes_b: anchors
    """
    if xyxy:
        t_wh = gt_boxes[:, None, 2:] - gt_boxes[:, None, :2]
    else:
        t_wh = gt_boxes[:, None, 2:]

    r = t_wh[:, None, :] / anchors[:, 2:]  # wh ratio
    # print('r', r)
    # print(r.shape)
    r = r.squeeze(1)
    j = torch.max(r, 1.0 / r).max(-1)[0] < anchor_ratio_thresh
    # print('j shape: ', j.shape)
    # j can be used for best_n_all
    return j
