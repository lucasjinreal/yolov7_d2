#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

# from .darknet import Darknet
from ..backbone.layers.wrappers import BaseConv
from ..backbone.layers.wrappers import SPPBottleneck


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self, depth=53, width=1.0, in_channels=[256, 512, 1024], in_features=["dark3", "dark4", "dark5"], with_spp=False
    ):
        super().__init__()

        # self.backbone = Darknet(depth)
        self.in_features = in_features

        base_ch = int(512 * width)  # width can be 0.5 1 2 -> 256, 512, 1024

        # out 0
        self.out0 = self._make_embedding([base_ch, base_ch*2], in_channels[2])

        # out 1
        self.out1_cbl = self._make_cbl(base_ch, base_ch//2, 1)
        self.out1 = self._make_embedding(
            [base_ch//2, base_ch], in_channels[1] + base_ch//2)

        # out 2
        self.out2_cbl = self._make_cbl(base_ch//2, base_ch//4, 1)
        self.out2 = self._make_embedding(
            [base_ch//4, base_ch//2], in_channels[0] + base_ch//4)

        # for output usage
        self.out_channels = [base_ch, base_ch//2, base_ch//4]

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.with_spp = with_spp
        if self.with_spp:
            self.spp = SPPBottleneck(in_channels[-1], in_channels[-1])

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),

                self._make_cbl(filters_list[1], filters_list[0], 1),

                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                # self._make_cbl(filters_list[0], filters_list[1], 3)
            ]
        )
        return m

    def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
        with open(filename, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        print("loading pretrained weights...")
        self.backbone.load_state_dict(state_dict)

    def forward(self, out_features):
        """
        Args:
            out_features: dict of backbone output features

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        #  backbone
        # out_features = self.backbone(inputs)S
        x2, x1, x0 = [out_features[f] for f in self.in_features]

        if self.with_spp:
            x0 = self.spp(x0)

        # yolo branch 0
        out0 = self.out0(x0)

        #  yolo branch 1
        x1_in = self.out1_cbl(out0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out1)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2 = self.out2(x2_in)

        # if self.with_spp:
        #     out0 = self.spp(out0)
        # outputs = (out2, out1, out0) # s, m, l
        outputs = (out0, out1, out2)  # l, m, s: 1024, 512, 256
        return outputs
