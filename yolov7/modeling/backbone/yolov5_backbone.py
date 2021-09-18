import torch.nn as nn
from nb.torch.blocks.common import Focus, Conv, C3, SPP, BottleneckCSP, C3TR
from nb.torch.utils.common import make_divisible

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY



"""

Currently not used
"""

class YOLOv5BackBone(Backbone):
    def __init__(self, focus=True, version='L', with_C3TR=False):
        super(YOLOv5BackBone, self).__init__()
        self.version = version
        self.with_focus = focus
        self.with_c3tr = with_C3TR
        gains = {'s': {'gd': 0.33, 'gw': 0.5},
                 'm': {'gd': 0.67, 'gw': 0.75},
                 'l': {'gd': 1, 'gw': 1},
                 'x': {'gd': 1.33, 'gw': 1.25}}
        self.gd = gains[self.version.lower()]['gd']  # depth gain
        self.gw = gains[self.version.lower()]['gw']  # width gain

        self.channels_out = {
            'stage1': 64,
            'stage2_1': 128,
            'stage2_2': 128,
            'stage3_1': 256,
            'stage3_2': 256,
            'stage4_1': 512,
            'stage4_2': 512,
            'stage5': 1024,
            'spp': 1024,
            'csp1': 1024,
            'conv1': 512
        }
        self.re_channels_out()

        if self.with_focus:
            self.stage1 = Focus(3, self.channels_out['stage1'])
        else:
            self.stage1 = Conv(3, self.channels_out['stage1'], 3, 2)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(
            self.channels_out['stage1'], self.channels_out['stage2_1'], k=3, s=2)
        self.stage2_2 = C3(
            self.channels_out['stage2_1'], self.channels_out['stage2_2'], self.get_depth(3))
        self.stage3_1 = Conv(
            self.channels_out['stage2_2'], self.channels_out['stage3_1'], 3, 2)
        self.stage3_2 = C3(
            self.channels_out['stage3_1'], self.channels_out['stage3_2'], self.get_depth(9))
        self.stage4_1 = Conv(
            self.channels_out['stage3_2'], self.channels_out['stage4_1'], 3, 2)
        self.stage4_2 = C3(
            self.channels_out['stage4_1'], self.channels_out['stage4_2'], self.get_depth(9))
        self.stage5 = Conv(
            self.channels_out['stage4_2'], self.channels_out['stage5'], 3, 2)
        self.spp = SPP(self.channels_out['stage5'],
                       self.channels_out['spp'], [5, 9, 13])
        if self.with_c3tr:
            self.c3tr = C3TR(
                self.channels_out['spp'], self.channels_out['csp1'], self.get_depth(3), False)
        else:
            self.csp1 = C3(
                self.channels_out['spp'], self.channels_out['csp1'], self.get_depth(3), False)
        self.conv1 = Conv(
            self.channels_out['csp1'], self.channels_out['conv1'], 1, 1)
        self.out_shape = {'C3_size': self.channels_out['stage3_2'],
                          'C4_size': self.channels_out['stage4_2'],
                          'C5_size': self.channels_out['conv1']}
        print("backbone output channel: C3 {}, C4 {}, C5 {}".format(self.channels_out['stage3_2'],
                                                                    self.channels_out['stage4_2'],
                                                                    self.channels_out['conv1']))

    def forward(self, x):
        x = self.stage1(x)
        x21 = self.stage2_1(x)
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22)
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3)
        c4 = self.stage4_2(x41)
        x5 = self.stage5(c4)
        spp = self.spp(x5)
        if not self.with_c3tr:
            csp1 = self.csp1(spp)
            c5 = self.conv1(csp1)
        else:
            c3tr = self.c3tr(spp)
            c5 = self.conv1(c3tr)
        return c3, c4, c5

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
