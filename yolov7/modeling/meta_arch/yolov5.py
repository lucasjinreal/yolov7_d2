
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Copyright (c) Lucas Jin. telegram: lucasjin


import enum
import logging
import random
from collections import OrderedDict
import math
from numpy.core.numeric import ones_like
from yolov7.modeling.meta_arch.yolo import YOLO

from torch.nn.modules.conv import Conv2d
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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

from ..neck.yolo_pafpn import YOLOPAFPN
from yolov7.utils.boxes import postprocess, bboxes_iou, BoxModeMy, postprocessv5
import time

__all__ = ["YOLOV7", "YOLOHead"]

supported_backbones = ['resnet', 'res2net',
                       'swin', 'efficient', 'darknet', 'pvt']


@META_ARCH_REGISTRY.register()
class YOLOV5(nn.Module):

    def __init__(self, cfg):
        super(YOLOV5, self).__init__()
        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE

        self.depth_mul = cfg.MODEL.YOLO.DEPTH_MUL
        self.width_mul = cfg.MODEL.YOLO.WIDTH_MUL

        self.size = 512
        self.multi_size = [320, 352, 384, 416, 448, 480,
                           512, 544, 576, 608]  # actually we disabled it
        self.change_iter = 10
        self.iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES

        assert len([i for i in supported_backbones if i in cfg.MODEL.BACKBONE.NAME]
                   ) > 0, 'Only {} supported.'.format(supported_backbones)

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.size_divisibility = 32 if self.backbone.size_divisibility == 0 else self.backbone.size_divisibility
        logger.info('YOLO.ANCHORS: {}'.format(cfg.MODEL.YOLO.ANCHORS))
        backbone_shape = [backbone_shape[i].channels for i in self.in_features]
        logger.info('backboneshape: {}, size_divisibility: {}'.format(
            backbone_shape, self.size_divisibility))

        # don't specific in_channels, let it calculate
        self.neck = YOLOPAFPN(
            depth=self.depth_mul, width=self.width_mul, in_features=self.in_features)

        ch = backbone_shape
        self.m = nn.ModuleList(nn.Conv2d(x, len(
            cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES), 1) for x in ch)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x / 255. - pixel_mean) / pixel_std
        self.padded_value = cfg.MODEL.PADDED_VALUE
        self.loss_evaluator = YOLOV5Head(cfg, cfg.MODEL.YOLO.ANCHORS)
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
        #     im = visualize_det_cv2_part(im, None, clss, bboxes, is_show=True)

        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        images = [self.normalizer(x) for x in images]

        images = ImageList.from_tensors(
            images, size_divisibility=self.size_divisibility, pad_value=self.padded_value/255.)
        # logger.info('images ori shape: {}'.format(images.tensor.shape))
        # logger.info('images ori shape: {}'.format(images.image_sizes))

        # sync image size for all gpus
        comm.synchronize()
        if training and self.iter % self.change_iter == 0:
            if self.iter < self.max_iter - 20000:
                meg = torch.LongTensor(1).to(self.device)
                comm.synchronize()
                if comm.is_main_process():
                    size = np.random.choice(self.multi_size)
                    meg.fill_(size)

                if comm.get_world_size() > 1:
                    comm.synchronize()
                    dist.broadcast(meg, 0)
                self.size = meg.item()

                comm.synchronize()
            else:
                self.size = 608

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

            if gt_instances:
                _, _, h, w = images.tensor.shape
                for i, instance in enumerate(gt_instances):
                    # yolov5 using normalized input
                    # print(instance.image_size)
                    # ori_w, ori_h = instance.image_size
                    instance.gt_boxes.tensor = BoxModeMy.convert_and_normalize(
                        instance.gt_boxes.tensor, from_mode=BoxModeMy.XYXY_ABS, to_mode=BoxModeMy.XYWH_ABS, ori_w=w, ori_h=h)

            targets = [
                torch.cat(
                    [torch.ones_like(instance.gt_classes.float().unsqueeze(-1)) * i,
                     instance.gt_classes.float().unsqueeze(-1), instance.gt_boxes.tensor], dim=-1
                )
                for i, instance in enumerate(gt_instances)
            ]
            targets = torch.cat(targets, dim=0)
            nL = targets.shape[0]
            labels = torch.zeros((nL, 6)).to(self.device)
            if nL:
                labels = targets

            # labels = torch.zeros((bs, 100, 5))
            # for i, target in enumerate(targets):
            #     labels[i][:target.shape[0]] = target
            # labels[:, :, 1:] = labels[:, :, 1:] / 512. * self.size
            # labels[:, :, 1:] = labels[:, :, 1:]
        else:
            labels = None

        self.iter += 1
        return images, labels, images.image_sizes

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
        c = 0
        if self.eval:
            t0 = time.time()

        images, labels, image_ori_sizes = self.preprocess_image(
            batched_inputs, self.training)

        # batched_inputs[0]['image'] = images.tensor[0].cpu() * 255
        # self.visualize_data(batched_inputs[0])

        x = images.tensor
        img_size = x.shape[-2:]
        # logger.info('img size: {}'.format(img_size))

        out_features = self.backbone(x)
        outputs = self.neck(out_features)  # 512, 1024, 2048, s, m, l
        # for i, a in enumerate(outputs):
        #     print(i, a.shape)

        outs = []
        for i, x in enumerate(outputs):
            outs.append(self.m[i](x))

        # for i, a in enumerate(outs):
        #     print(a.shape)

        if self.training:
            losses = self.loss_evaluator(outs, labels, img_size)
            return losses
        else:
            predictions = self.loss_evaluator(outs, image_size=img_size)
            # p = predictions[predictions[:, :, 4] > 0.1]
            # print(p)
            # print(predictions[:, :, 4])
            detections = postprocessv5(predictions,
                                       self.num_classes,
                                       self.conf_threshold,
                                       self.nms_threshold)
            results = []
            for idx, out in enumerate(detections):
                if out is None:
                    out = x.new_zeros((0, 7))
                image_size = image_ori_sizes[idx]
                result = Instances(image_size)
                result.pred_boxes = Boxes(out[:, :4])
                result.scores = out[:, 5] * out[:, 4]
                result.pred_classes = out[:, -1]
                results.append(result)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


def smooth_BCE(eps=0.1):
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YOLOV5Head(nn.Module):
    def __init__(self, cfg, anchors):
        super(YOLOV5Head, self).__init__()
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.all_anchors = np.array(cfg.MODEL.YOLO.ANCHORS).reshape([-1, 2])
        # self.anchors = anchors

        self.anchors = np.array(cfg.MODEL.YOLO.ANCHORS)
        # self.ref_anchors = torch.from_numpy(self.ref_anchors)
        # ref_anchors divided by [8, 16, 32] in forward
        self.ref_anchors = None

        self.num_anchors = len(anchors)
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.bbox_attrs = 5 + self.num_classes

        self.ignore_threshold = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        self.lambda_xy = cfg.MODEL.YOLO.LOSS.LAMBDA_XY
        self.lambda_wh = cfg.MODEL.YOLO.LOSS.LAMBDA_WH
        self.lambda_conf = cfg.MODEL.YOLO.LOSS.LAMBDA_CONF
        self.lambda_cls = cfg.MODEL.YOLO.LOSS.LAMBDA_CLS
        self.lambda_iou = cfg.MODEL.YOLO.LOSS.LAMBDA_IOU

        self.anchor_thresh = cfg.MODEL.YOLO.LOSS.ANCHOR_RATIO_THRESH

        self.eps = 1e-8

        self.inplace = True

        # BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        # BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCEcls = nn.BCEWithLogitsLoss()
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.label_smoothing = 0.0
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=self.label_smoothing)

        # Focal loss
        self.fl_gamma = 0.0
        g = self.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.nl = len(anchors)
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.ssi = 0  # stride 16 index
        self.gr = 1.0
        self.autobalance = False
        self.BCEcls, self.BCEobj = BCEcls, BCEobj

        # used for inference
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchor_grid = a.clone().view(self.nl, 1, -1, 1, 1, 2)
        self.na = self.num_anchors
        self.no = self.bbox_attrs

        self.nc = self.num_classes
        self.stride = []  # should be [8, 16, 32]
        # why using registered buffer?

    def forward(self, preds, targets=None, image_size=(416, 416)):
        device = preds[0].device
        self.anchor_grid = self.anchor_grid.to(device)

        for i in range(len(preds)):
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx = preds[i].shape
            preds[i] = preds[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()
            if len(self.stride) < len(preds):
                stride = int(image_size[0] / ny)
                self.stride.append(stride)

        if self.ref_anchors is None:
            self.ref_anchors = self.anchors / \
                torch.tensor(self.stride).view(-1, 1, 1)
            # print(self.anchor_grid)
            # print_tensor(self.ref_anchors, 'ref_anchors')

        if targets != None:
            self.ref_anchors = self.ref_anchors.to(device)
            lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(
                1, device=device), torch.zeros(1, device=device)
            lxy, lwh = torch.zeros(
                1, device=device), torch.zeros(1, device=device)
            # print_tensor(targets, 'targets')
            tcls, tbox, indices, anchors = self.build_target(
                preds, targets)  # targets

            # print(anchors)
            # print(tbox[0])

            # Losses
            for i, pi in enumerate(preds):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(
                    pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    # prediction subset corresponding to targets
                    ps = pi[b, a, gj, gi]

                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box

                    txy = tbox[i][:, :2]
                    twh = tbox[i][:, 2:]

                    lxy += torch.abs(pxy - txy).mean([0, 1]).sum()
                    # lwh += torch.abs(pwh - twh).sum([0, 1]).mean()

                    # print_tensor(pbox, 'pbox')
                    # print_tensor(tbox[i], 'tbox')
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                    # print(iou)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    tobj[b, a, gj, gi] = (
                        1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                    # Classification
                    if self.nc > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(
                            ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                obji = self.BCEobj(pi[..., 4], tobj)
                # print(obji)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * \
                        0.9999 + 0.0001 / obji.detach().item()

            if self.autobalance:
                self.balance = [x / self.balance[self.ssi]
                                for x in self.balance]
            lbox *= self.lambda_iou
            lobj *= (self.lambda_conf * (image_size[0]/640)**2)
            lcls *= self.lambda_cls
            lxy *= self.lambda_iou
            lwh *= self.lambda_iou
            bs = tobj.shape[0]  # batch size

            # loss = lbox + lobj + lcls
            # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

            return {
                # "total_loss:": loss * bs,
                "loss_xy": lxy*bs,
                # "loss_wh": lwh*bs,
                "loss_box": lbox * bs,
                "loss_obj": lobj * bs,
                "loss_cls": lcls * bs
            }
        else:
            # Inference stage
            x = preds
            z = []
            for i in range(self.nl):
                bs, _, ny, nx, _ = x[i].shape  # bs,a,y,x,o
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                                   self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                        self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * \
                        self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * \
                        self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def build_target(self, preds, targets):
        # print_tensor(targets, 'targets', ignore_value=False)
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(
            na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.ref_anchors[i]
            # print_tensor(anchors, 'anchors')
            # print(preds[i].shape)
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # print_tensor(targets, 'targets-before gain')
            # Match targets to anchors
            t = targets * gain
            # print_tensor(t, 'targets-after gain')
            if nt:
                # Matches
                # print('t[:, :, 4:6]: ', t[:, :, 4:6].shape)  # [3, 101, 2]
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # print_tensor(r, 'r', ignore_value=False)  # [3, 101, 2]
                # print_tensor(t[:, :, 4:6], 't46',
                #              ignore_value=False)  # [3, 101, 2]
                # print(anchors[:, None])
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.anchor_thresh  # compare
                # print_tensor(j, 'j')

                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter
                # print_tensor(t, 't')

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # image, anchor, grid indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
