import torch.nn as nn
import torch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone

from detectron2.structures import Boxes, ImageList, Instances, image_list
from detectron2.utils import comm
from detectron2.utils.logger import log_first_n
from detectron2.modeling.postprocessing import detector_postprocess

import torch.distributed as dist

import numpy as np
import time
import logging
from alfred.utils.log import logger

from ..head.yolox_head import YOLOXHead
from ..head.yolov6_head import YOLOv6Head
from ..neck.yolo_pafpn import YOLOPAFPN
from ..neck.reppan import RepPANNeck

from yolov7.utils.boxes import postprocess, BoxModeMy


"""
Implementation of YOLOv6

"""


@META_ARCH_REGISTRY.register()
class YOLOV6(nn.Module):
    def __init__(self, cfg):
        super(YOLOV6, self).__init__()
        # configurations
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.conf_threshold = cfg.MODEL.YOLO.CONF_THRESHOLD
        self.nms_threshold = cfg.MODEL.YOLO.NMS_THRESHOLD
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.loss_type = cfg.MODEL.YOLO.LOSS_TYPE
        self.head_type = cfg.MODEL.YOLO.HEAD.TYPE

        # l1 loss will open at last 15 epochs
        self.use_l1 = False

        self.depth_mul = cfg.MODEL.YOLO.DEPTH_MUL
        self.width_mul = cfg.MODEL.YOLO.WIDTH_MUL

        self.iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.enable_l1_loss_at = cfg.INPUT.MOSAIC_AND_MIXUP.DISABLE_AT_ITER
        self.num_classes = cfg.MODEL.YOLO.CLASSES
        self.max_boxes_num = cfg.MODEL.YOLO.MAX_BOXES_NUM
        self.in_features = cfg.MODEL.YOLO.IN_FEATURES
        self.neck_type = cfg.MODEL.YOLO.NECK.TYPE

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.size_divisibility = (
            32
            if self.backbone.size_divisibility == 0
            else self.backbone.size_divisibility
        )
        backbone_shape = [backbone_shape[i].channels for i in self.in_features]
        logger.info(
            "backboneshape: {}, size_divisibility: {}".format(
                backbone_shape, self.size_divisibility
            )
        )

        # don't specific in_channels, let it calculate

        if self.neck_type == "reppan":
            self.neck = RepPANNeck(
                channels_list=self.backbone.channels_list,
                num_repeats=self.backbone.num_repeats,
                in_features=self.in_features,
            )
            logger.warning("Using YOLOv6 RepPAN neck!")
        else:
            self.neck = YOLOPAFPN(
                depth=self.depth_mul, width=self.width_mul, in_features=self.in_features
            )

        if self.head_type == "yolov6":
            self.head = YOLOv6Head(
                self.num_classes, channels_list=self.backbone.channels_list
            )
        else:
            self.head = YOLOXHead(
                self.num_classes, width=self.width_mul, in_channels=backbone_shape
            )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.padded_value = cfg.MODEL.PADDED_VALUE
        self.normalizer = lambda x: (x / 255.0 - pixel_mean) / pixel_std
        self.to(self.device)
        self.onnx_export = False
        self.onnx_vis = False

        self.apply(self._init_model)
        self.head.initialize_biases(1e-2)

    @staticmethod
    def _init_model(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def update_iter(self, i):
        self.iter = i

    def preprocess_image(self, batched_inputs, training):
        images = [x["image"].to(self.device) for x in batched_inputs]
        bs = len(images)
        # images = [self.normalizer(x) for x in images]
        images = [x.type(torch.float) for x in images]

        images = ImageList.from_tensors(
            images,
            size_divisibility=self.size_divisibility,
            pad_value=self.padded_value,
        )
        # logger.info('images ori shape: {}'.format(images.tensor.shape))

        if training and self.iter > self.enable_l1_loss_at and not self.use_l1:
            meg = torch.BoolTensor(1).to(self.device)
            if comm.is_main_process():
                logger.info("[master] enable l1 loss now at iter: {}".format(self.iter))
                # enable l1 loss at last 50000 iterations
                meg.fill_(True)

            if comm.get_world_size() > 1:
                comm.synchronize()
                if comm.is_main_process():
                    dist.broadcast(meg, 0)
            self.head.use_l1 = meg.item()
            self.use_l1 = meg.item()
            comm.synchronize()
            logger.info("check head l1: {}".format(self.head.use_l1))

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

            if gt_instances:
                for i in gt_instances:
                    i.gt_boxes.tensor = BoxModeMy.convert(
                        i.gt_boxes.tensor,
                        from_mode=BoxModeMy.XYXY_ABS,
                        to_mode=BoxModeMy.XYWH_ABS,
                    )

            targets = [
                torch.cat(
                    # YOLOX using [cls, box], box is cx cy w h
                    [
                        instance.gt_classes.float().unsqueeze(-1),
                        instance.gt_boxes.tensor,
                    ],
                    dim=-1
                    # [instance.gt_boxes.tensor, instance.gt_classes.float().unsqueeze(-1), ], dim=-1
                )
                for instance in gt_instances
            ]

            labels = torch.zeros((bs, self.max_boxes_num, 5))
            # first dim assign -1 for none-classes
            labels[:, :, 0] = -1
            for i, target in enumerate(targets):
                if target.shape[0] > self.max_boxes_num:
                    target = target[: self.max_boxes_num, :]
                labels[i][: target.shape[0]] = target
        else:
            labels = None

        # self.iter += 1
        return images, labels, images.image_sizes

    def preprocess_input(self, x):
        x = x.permute(0, 3, 1, 2)
        # x = F.interpolate(x, size=(640, 640))
        # x = F.interpolate(x, size=(512, 960))
        # x = self.normalizer(x)
        return x

    def forward(self, batched_inputs):
        if self.onnx_export:
            logger.info("[WARN] exporting onnx...")
            assert isinstance(batched_inputs, torch.Tensor) or isinstance(
                batched_inputs, list
            ), "onnx export, batched_inputs only needs image tensor"
            x = self.preprocess_input(batched_inputs)
            # batched_inputs = batched_inputs.permute(0, 3, 1, 2)
            image_ori_sizes = [batched_inputs.shape[1:3]]
        else:
            images, labels, image_ori_sizes = self.preprocess_image(
                batched_inputs, self.training
            )
            if labels is not None:
                labels = labels.to(images.device)

            x = images.tensor
            img_size = x.shape[-2:]
            # logger.info('img size: {}'.format(img_size))

        if self.eval:
            t0 = time.time()

        out_features = self.backbone(x)
        # for k, v in out_features.items():
        #     print(k, v.shape)
        fpn_outs = self.neck(out_features)  # 512, 1024, 2048, s, m, l
        # for i in fpn_outs:
        #     print(i.shape)

        if self.training:
            # print(labels)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, labels, x
            )

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
            }
            if self.use_l1:
                outputs["l1_loss"] = l1_loss
            return outputs
        else:
            if self.onnx_export:
                if not self.onnx_vis:
                    # self.head.decode_in_inference = False
                    self.head.decode_in_inference = True
                    self.head.onnx_export = True
                    # we wrap box decode into onnx model as well
                    outputs = self.head(fpn_outs)
                    return outputs
                else:
                    self.head.decode_in_inference = True
                    outputs = self.head(fpn_outs)
                    detections = postprocess(
                        outputs,
                        self.num_classes,
                        self.conf_threshold,
                        self.nms_threshold,
                    )
                    return detections
            else:
                outputs = self.head(fpn_outs)

                t1 = time.time()

                detections = postprocess(
                    outputs, self.num_classes, self.conf_threshold, self.nms_threshold
                )

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
                    results, batched_inputs, images.image_sizes
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                # return processed_results, t1 - t0
                return processed_results
