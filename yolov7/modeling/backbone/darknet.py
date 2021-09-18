#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from collections import OrderedDict

import numpy as np

import torch
from torch import nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from detectron2.utils.file_io import PathManager
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec

import logging
import os


__all__ = []

logger = logging.getLogger(__name__)


def parse_darknet_conv_weights(module, weights, ptr):
    """
    Utility function to parse official darknet weights into torch.
    """
    conv_layer = module[0]
    try:
        batch_normalize = isinstance(module[1], _BatchNorm)
    except Exception:
        batch_normalize = False
    if batch_normalize:
        # Load BN bias, weights, running mean and running variance
        bn_layer = module[1]
        num_b = bn_layer.bias.numel()  # Number of biases
        # Bias
        bn_b = torch.from_numpy(
            weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_b)
        ptr += num_b
        # Weight
        bn_w = torch.from_numpy(
            weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_w)
        ptr += num_b
        # Running Mean
        bn_rm = torch.from_numpy(
            weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_rm)
        ptr += num_b
        # Running Var
        bn_rv = torch.from_numpy(
            weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_rv)
        ptr += num_b
    else:
        # Load conv. bias
        num_b = conv_layer.bias.numel()
        conv_b = torch.from_numpy(
            weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
        conv_layer.bias.data.copy_(conv_b)
        ptr += num_b
    # Load conv. weights
    num_w = conv_layer.weight.numel()
    conv_w = torch.from_numpy(
        weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
    conv_layer.weight.data.copy_(conv_w)
    ptr += num_w

    return ptr


def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1) -> nn.Sequential:
    "Create a seuence Conv2d->BatchNorm2d->LeakyReLu layer."
    return nn.Sequential(
        OrderedDict([
            ("conv", nn.Conv2d(ni, nf, kernel_size=ks,
                               bias=False, stride=stride, padding=ks // 2)),
            ("bn", nn.BatchNorm2d(nf)),
            ("relu", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
        ]))


class Flatten(Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full: bool = False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class ResLayer(Module):
    "Resnet style layer with `ni` inputs."

    def __init__(self, ni: int):
        super(ResLayer, self).__init__()
        self.layer1 = conv_bn_lrelu(ni, ni // 2, ks=1)
        self.layer2 = conv_bn_lrelu(ni // 2, ni, ks=3)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class Darknet(Backbone):
    "https://github.com/pjreddie/darknet"
    depth2blocks = {
        21: [1, 1, 2, 2, 1],
        53: [1, 2, 8, 8, 4],
    }

    def make_group_layer(self, ch_in: int, num_blocks: int, stride: int = 1):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        return [conv_bn_lrelu(ch_in, ch_in * 2, stride=stride)] \
            + [(ResLayer(ch_in * 2)) for i in range(num_blocks)]

    def __init__(self, depth, ch_in=3, nf=32, out_features=None, num_classes=None):
        """
        depth (int): depth of darknet used in model, usually use [21, 53] for this param
        ch_in (int): input channels, for example, ch_in of RGB image is 3
        nf (int): number of filters output in stem.
        out_features (List[str]): desired output layer name.
        num_classes (int): For ImageNet, num_classes is 1000. If None, no linear layer will be
            added.
        """
        super(Darknet, self).__init__()
        self.stem = conv_bn_lrelu(ch_in, nf, ks=3, stride=1)
        self.num_classes = num_classes

        current_stride = 1
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": nf}

        "create darknet with `nf` and `num_blocks` layers"
        self.stages_and_names = []
        num_blocks = Darknet.depth2blocks[depth]
        # out_idx = [0]
        # for nb in num_blocks:
        #     out_idx.append(out_idx[-1] + 1 + nb)
        # out_idx.pop(0)
        self._output_shape = dict()

        for i, nb in enumerate(num_blocks):
            stage = nn.Sequential(
                *self.make_group_layer(nf, nb, stride=2))
            name = 'dark' + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride
            current_stride *= 2
            nf *= 2
            self._out_feature_channels[name] = nf
            self._output_shape[name] = ShapeSpec(
                channels=nf, stride=16
            )
        if num_classes is not None:
            name = "linear"
            self.add_module(name, nn.Sequential([
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(nf, num_classes)]))

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert(len(self._out_features))
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    # @property
    def output_shape(self):
        return self._output_shape

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 32

    def load_darknet_weights(self, weights):
        # Parses and loads the weights stored in 'weights'

        # Read weights file
        with open(weights, 'rb') as f:
            # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
            # (int32) version info: major, minor, revision
            self.version = np.fromfile(f, dtype=np.int32, count=3)
            # (int64) number of images seen during training
            self.seen = np.fromfile(f, dtype=np.int64, count=1)

            weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

        ptr = 0
        for i, (mdef, module) in enumerate(self.named_children()):
            if mdef == "stem":
                ptr = parse_darknet_conv_weights(module, weights, ptr)
            elif mdef.startswith("dark"):
                for j, (sub_mdef, sub_module) in enumerate(module.named_children()):
                    if isinstance(sub_module, nn.Sequential):
                        ptr = parse_darknet_conv_weights(
                            sub_module, weights, ptr)
                    elif isinstance(sub_module, ResLayer):
                        for sub_sub_mdef, sub_sub_module in sub_module.named_children():
                            if isinstance(sub_sub_module, nn.Sequential):
                                ptr = parse_darknet_conv_weights(
                                    sub_sub_module, weights, ptr)


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, input_shape):
    depth = cfg.MODEL.DARKNET.DEPTH
    stem_channels = cfg.MODEL.DARKNET.STEM_OUT_CHANNELS
    output_features = cfg.MODEL.DARKNET.OUT_FEATURES

    model = Darknet(depth, input_shape.channels,
                    stem_channels, output_features)
    filename = cfg.MODEL.DARKNET.WEIGHTS
    if filename.startswith("s3://"):
        with PathManager.open(filename, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        model.load_state_dict(state_dict)
    elif os.path.exists(filename):
        with PathManager.open(filename, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        logger.info("Pass load pretrained model for darknet")
    return model


if __name__ == "__main__":
    model = Darknet(53, 32)
    print(model)
