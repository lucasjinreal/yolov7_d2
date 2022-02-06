
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.
# Copyright (c) Lucas Jin. telegram: lucasjin


import logging
import random
from collections import OrderedDict
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

from alfred.utils.log import logger
from .utils import generalized_batched_nms

__all__ = ["YOLO", "YOLOHead"]


@META_ARCH_REGISTRY.register()
class YOLO(nn.Module):
    """
    YOLO model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, cfg):
        super(YOLO, self).__init__()
        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.nms_type = cfg.MODEL.NMS_TYPE

        self.size = 512
        self.multi_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] # actually we disabled it
        self.change_iter = 10
        self.iter = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape
        self.size_divisibility = self.backbone.size_divisibility
        logger.info('backboneshape: {}, {}'.format(
            backbone_shape, self.size_divisibility))
        logger.info('YOLO.ANCHORS: {}'.format(cfg.MODEL.YOLO.ANCHORS))

        # out 0
        out_filter_0 = len(
            cfg.MODEL.YOLO.ANCHORS[0]) * (5 + cfg.MODEL.YOLO.CLASSES)
        self.out0 = self._make_embedding(
            [512, 1024], backbone_shape[-1], out_filter_0)

        # out 1
        out_filter_1 = len(
            cfg.MODEL.YOLO.ANCHORS[1]) * (5 + cfg.MODEL.YOLO.CLASSES)
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out1 = self._make_embedding(
            [256, 512], backbone_shape[-2] + 256, out_filter_1)

        # out 2
        out_filter_2 = len(
            cfg.MODEL.YOLO.ANCHORS[2]) * (5 + cfg.MODEL.YOLO.CLASSES)
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.out2 = self._make_embedding(
            [128, 256], backbone_shape[-3] + 128, out_filter_2)

        # todo: remove std and mean, all divid 255.
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x / 255. - pixel_mean) / pixel_std

        self.loss_evaluators = [
            YOLOHead(cfg, anchor, level) for level, anchor in enumerate(cfg.MODEL.YOLO.ANCHORS)]
        self.to(self.device)

    def _make_cbl(self, _in, _out, ks):
        ''' cbl = conv + batch_norm + leaky_relu
        '''
        pad = (ks - 1) // 2 if ks else 0
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(_in, _out, kernel_size=ks,
                               stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

    def _make_embedding(self, filters_list, in_filters, out_filter):
        m = nn.ModuleList([
            self._make_cbl(in_filters, filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3),
            self._make_cbl(filters_list[1], filters_list[0], 1),
            self._make_cbl(filters_list[0], filters_list[1], 3)])
        m.add_module("conv_out", nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                           stride=1, padding=0, bias=True))
        return m

    def preprocess_image(self, batched_inputs, training):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        images = [self.normalizer(x) for x in images]

        images = ImageList.from_tensors(
            images, size_divisibility=self.size_divisibility, pad_value=144/255.)
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

            targets = [
                torch.cat(
                    [instance.gt_classes.float().unsqueeze(-1), instance.gt_boxes.tensor], dim=-1
                )
                for instance in gt_instances
            ]
            labels = torch.zeros((bs, 100, 5))
            for i, target in enumerate(targets):
                labels[i][:target.shape[0]] = target
            # labels[:, :, 1:] = labels[:, :, 1:] / 512. * self.size
            labels[:, :, 1:] = labels[:, :, 1:]
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
        images, labels, image_ori_sizes = self.preprocess_image(batched_inputs, self.training)
        # if self.training:
            
        # else:
            # images = batched_inputs
            # labels = None

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
        # x2, x1, x0 = self.backbone(x)
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        #  yolo branch 0
        out0, out0_branch = _branch(self.out0, x0)
        #  yolo branch 1
        x1_in = self.out1_cbl(out0_branch)
        x1_in = self.out1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.out1, x1_in)
        #  yolo branch 2
        x2_in = self.out2_cbl(out1_branch)
        x2_in = self.out2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, out2_branch = _branch(self.out2, x2_in)

        outputs = [out0, out1, out2]

        if self.training:
            losses = [
                loss_evaluator(out, labels, img_size) for out, loss_evaluator in zip(
                    outputs, self.loss_evaluators)
            ]
            keys = ["loss_x", "loss_y", "loss_w",
                    "loss_h", "loss_conf", "loss_cls"]
            losses_dict = {}
            for key in keys:
                losses_dict[key] = sum([loss[key] for loss in losses])
            return losses_dict
        else:
            predictions_list = [loss_evaluator(out, labels, img_size) for
                                out, loss_evaluator in zip(outputs, self.loss_evaluators)]

            predictions = torch.cat(predictions_list, 1)
            detections = postprocess(predictions,
                                     self.num_classes,
                                     self.conf_threshold,
                                     self.nms_threshold,
                                     nms_type=self.nms_type)

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
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})

            return processed_results
            # return results


class YOLOHead(nn.Module):
    def __init__(self, cfg, anchors, level):
        super(YOLOHead, self).__init__()
        self.level = level
        self.all_anchors = np.array(cfg.MODEL.YOLO.ANCHORS).reshape([-1, 2])
        self.anchors = anchors
        self.ref_anchors = np.zeros((len(self.all_anchors), 4))
        self.ref_anchors[:, 2:] = self.all_anchors
        self.ref_anchors = torch.from_numpy(self.ref_anchors)

        self.num_anchors = len(anchors)
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.bbox_attrs = 5 + self.num_classes

        self.ignore_threshold = cfg.MODEL.YOLO.IGNORE_THRESHOLD
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, input, targets=None, image_size=(416, 416)):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        # stride_h = image_size[1] / in_h
        # stride_w = image_size[0] / in_w
        stride_h = image_size[0] / in_h
        stride_w = image_size[1] / in_w

        # scaled_anchors = [(a_w / stride_w, a_h / stride_h)
        #                  for a_w, a_h in self.anchors]

        scaled_anchors = [(a_w, a_h)
                          for a_w, a_h in self.anchors]

        prediction = input.view(bs, self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                         # Width
        h = prediction[..., 3]                         # Height
        conf = torch.sigmoid(prediction[..., 4])       # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        def FloatTensor(x): return torch.FloatTensor(x).to(pred_cls.device)  # noqa
        def LongTensor(x): return torch.LongTensor(x).to(pred_cls.device)  # noqa

        # Calculate offsets for each grid
        grid_x = FloatTensor(torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            bs * self.num_anchors, 1, 1).view(x.shape))
        grid_y = FloatTensor(torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            bs * self.num_anchors, 1, 1).view(y.shape))
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(
            1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(
            1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = prediction[..., :4].clone()
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        pred_boxes[..., 0] *= stride_w
        pred_boxes[..., 1] *= stride_h
        pred_boxes = pred_boxes.data

        if targets is not None:
            #  build target
            mask, obj_mask, \
                tx, ty, tw, th, \
                tgt_scale, tcls = self.get_target(targets, pred_boxes, image_size,
                                                  in_w, in_h,
                                                  stride_w, stride_h,
                                                  self.ignore_threshold)

            mask, obj_mask = mask.cuda(), obj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tgt_scale, tcls = tgt_scale.cuda(), tcls.cuda()

            loss_x = (mask * tgt_scale *
                      self.bce_loss(x * mask, tx * mask)).sum() / bs
            loss_y = (mask * tgt_scale *
                      self.bce_loss(y * mask, ty * mask)).sum() / bs
            loss_w = (mask * tgt_scale *
                      self.l1_loss(w * mask, tw * mask)).sum() / bs
            loss_h = (mask * tgt_scale *
                      self.l1_loss(h * mask, th * mask)).sum() / bs

            loss_conf = (obj_mask * self.bce_loss(conf, mask)).sum() / bs

            loss_cls = self.bce_loss(
                pred_cls[mask == 1], tcls[mask == 1]).sum() / bs

            #  total loss = losses * weight
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
            # Results
            output = torch.cat((pred_boxes.view(bs, -1, 4),
                                conf.view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, pred_boxes, img_size,
                   in_w, in_h, stride_w, stride_h, ignore_threshold):

        def FloatTensor(x): return torch.FloatTensor(x).to(pred_boxes.device)  # noqa

        bs = target.size(0)

        # logger.info('in_h, {}, in_w: {}'.format(in_h, in_w))
        # logger.info('stride_h, {}, stride_w: {}'.format(stride_h, stride_w))
        # logger.info('target shape: {}'.format(target.shape))

        mask = torch.zeros(bs, self.num_anchors, in_h,
                           in_w, requires_grad=False)
        obj_mask = torch.ones(bs, self.num_anchors,
                              in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tgt_scale = torch.zeros(bs, self.num_anchors,
                                in_h, in_w, requires_grad=False)

        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w,
                           self.num_classes, requires_grad=False)
        nlabel = (target.sum(dim=2) > 0).sum(dim=1)
        gx_all = (target[:, :, 1] + target[:, :, 3]) / 2.0   # center x
        gy_all = (target[:, :, 2] + target[:, :, 4]) / 2.0  # center y
        gw_all = (target[:, :, 3] - target[:, :, 1])        # width
        gh_all = (target[:, :, 4] - target[:, :, 2])        # height
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

            anchor_ious_all = bboxes_iou(truth_box.cpu(),
                                         self.ref_anchors.type_as(truth_box.cpu()), xyxy=False)
            best_n_all = np.argmax(anchor_ious_all, axis=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all // 3) == self.level)

            truth_box[:n, 0] = gx_all[b, :n]
            truth_box[:n, 1] = gy_all[b, :n]
            pred_box = pred_boxes[b]

            pred_ious = bboxes_iou(pred_box.view(-1, 4),
                                   truth_box, xyxy=False)

            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > ignore_threshold)
            pred_best_iou = pred_best_iou.view(pred_box.shape[:3])
            obj_mask[b] = ~pred_best_iou

            if sum(best_n_mask) == 0:
                continue

            for t in range(best_n.shape[0]):
                if best_n_mask[t] == 1:
                    gi, gj = truth_i[t], truth_j[t]
                    gx, gy = gx_all[b, t], gy_all[b, t]
                    gw, gh = gw_all[b, t], gh_all[b, t]

                    a = best_n[t]

                    # Masks
                    mask[b, a, gj, gi] = 1  # 17, 17
                    obj_mask[b, a, gj, gi] = 1

                    # Coordinates
                    tx[b, a, gj, gi] = gx / stride_w - gi
                    ty[b, a, gj, gi] = gy / stride_h - gj
                    # Width and height
                    tw[b, a, gj, gi] = torch.log(
                        gw / self.anchors[a][0] + 1e-16)
                    th[b, a, gj, gi] = torch.log(
                        gh / self.anchors[a][1] + 1e-16)

                    tgt_scale[b, a, gj, gi] = 2.0 - gw * \
                        gh / (img_size[0] * img_size[1])
                    # One-hot encoding of label
                    tcls[b, a, gj, gi, int(target[b, t, 0])] = 1

        return mask, obj_mask, tx, ty, tw, th, tgt_scale, tcls


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    return area_i / (area_a[:, None] + area_b - area_i)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, nms_type='normal'):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.
    Returns:
        output (list of torch tensor):
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] *
                     class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        confidence = detections[:, 4] * detections[:, 5]
        nms_out_index = generalized_batched_nms(detections[:, :4], confidence,
                                                detections[:, -1], nms_thre,
                                                nms_type=nms_type)
        detections[:, 4] = confidence / detections[:, 5]

        detections = detections[nms_out_index]

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].unique()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            if output[i] is None:
                output[i] = detections_class
            else:
                output[i] = torch.cat((output[i], detections_class))

    return output
