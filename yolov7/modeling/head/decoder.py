import math
from typing import Tuple

import torch
import torch.nn as nn

from ..backbone.layers.utils import get_activation, get_norm


class Decoder(nn.Module):
    """
    Head Decoder for YOLOF.

    This module contains two types of components:
        - A classification head with two 3x3 convolutions and one
            classification 3x3 convolution
        - A regression head with four 3x3 convolutions, one regression 3x3
          convolution, and one implicit objectness 3x3 convolution
    """

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        # fmt: off
        self.in_channels = cfg.MODEL.YOLOF.DECODER.IN_CHANNELS
        self.num_classes = cfg.MODEL.YOLOF.DECODER.NUM_CLASSES
        self.num_anchors = cfg.MODEL.YOLOF.DECODER.NUM_ANCHORS
        self.cls_num_convs = cfg.MODEL.YOLOF.DECODER.CLS_NUM_CONVS
        self.reg_num_convs = cfg.MODEL.YOLOF.DECODER.REG_NUM_CONVS
        self.norm_type = cfg.MODEL.YOLOF.DECODER.NORM
        self.act_type = cfg.MODEL.YOLOF.DECODER.ACTIVATION
        self.prior_prob = cfg.MODEL.YOLOF.DECODER.PRIOR_PROB
        # fmt: on

        self.INF = 1e8
        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                nn.Conv2d(self.in_channels,
                          self.in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(get_norm(self.norm_type, self.in_channels))
            cls_subnet.append(get_activation(self.act_type))
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                nn.Conv2d(self.in_channels,
                          self.in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(get_norm(self.norm_type, self.in_channels))
            bbox_subnet.append(get_activation(self.act_type))
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(self.in_channels,
                                   self.num_anchors * self.num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels,
                                   self.num_anchors * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.object_pred = nn.Conv2d(self.in_channels,
                                     self.num_anchors,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self,
                feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) + torch.clamp(
                objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg
