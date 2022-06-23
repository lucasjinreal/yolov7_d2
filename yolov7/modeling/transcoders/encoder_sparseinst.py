# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d
from alfred.utils.log import logger

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class MyAdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        self.sz = sz

    def forward(self, x):
        inp_size = x.size()
        kernel_width, kernel_height = inp_size[2], inp_size[3]
        if self.sz is not None:
            if isinstance(self.sz, int):
                kernel_width = math.ceil(inp_size[2] / self.sz)
                kernel_height = math.ceil(inp_size[3] / self.sz)
            elif isinstance(self.sz, list) or isinstance(self.sz, tuple):
                assert len(self.sz) == 2
                kernel_width = math.ceil(inp_size[2] / self.sz[0])
                kernel_height = math.ceil(inp_size[3] / self.sz[1])
        if torch.is_tensor(kernel_width):
            kernel_width = kernel_width.item()
            kernel_height = kernel_height.item()
        return F.avg_pool2d(
            input=x, ceil_mode=False, kernel_size=(kernel_width, kernel_height)
        )


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = MyAdaptiveAvgPool2d((size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=F.relu_(stage(feats)),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            for stage in self.stages
        ] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        # self.norm = cfg.MODEL.SPARSE_INST.ENCODER.NORM
        # depthwise = cfg.MODEL.SPARSE_INST.ENCODER.DEPTHWISE
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        # self.using_bias = self.norm == ""
        fpn_laterals = []
        fpn_outputs = []
        # groups = self.num_channels if depthwise else 1
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        features = features[::-1]
        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(
            features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]
        ):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(
                prev_features, scale_factor=2.0, mode="nearest"
            )
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [outputs[0]] + [
            F.interpolate(x, size, mode="bilinear", align_corners=False)
            for x in outputs[1:]
        ]
        features = self.fusion(torch.cat(features, dim=1))
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
