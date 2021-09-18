import math


import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# for detectron2
import fvcore.nn.weight_init as weight_init

from detectron2.layers import FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from torchvision import models

from .layers import Conv2d, SeparableConv2d, MaxPool2d, MemoryEfficientSwish, Swish


"""
Original source: https://github.com/sxhxliang/detectron2_backbone
with some self-defined modifications
"""


# https://github.com/lukemelas/EfficientNet-PyTorch/releases
model_urls = {
    'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

params = {
    # stride=2:  ----> block 1 ,3, 5, 11
    'efficientnet_b0': (1.0, 1.0, 224, 0.2),
    # stride=2:  ----> block 2, 5, 8, 16
    'efficientnet_b1': (1.0, 1.1, 240, 0.2),
    'efficientnet_b2': (1.1, 1.2, 260, 0.3),
    'efficientnet_b3': (1.2, 1.4, 300, 0.3),
    'efficientnet_b4': (1.4, 1.8, 380, 0.4),
    'efficientnet_b5': (1.6, 2.2, 456, 0.4),
    'efficientnet_b6': (1.8, 2.6, 528, 0.5),
    'efficientnet_b7': (2.0, 3.1, 600, 0.5),
}


class MBConvBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2,
                 use_se=True):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        self.use_se = use_se
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        bn_mom = 0.99  # tensorflow bn_mom
        bn_mom = round(1 - bn_mom, 3)  # pytorch = 1 - tensorflow
        bn_eps = 1e-3
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []

        self.expand_ratio = in_planes != hidden_dim
        if self.expand_ratio:
            self._expand_conv = Conv2d(
                in_planes, hidden_dim, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(hidden_dim, momentum=bn_mom, eps=bn_eps)
        # dw
        self._depthwise_conv = Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, groups=hidden_dim, bias=False, padding_mode='static_same')
        self._bn1 = nn.BatchNorm2d(hidden_dim, momentum=bn_mom, eps=bn_eps)
        # se
        if self.use_se:
            self._se_reduce = Conv2d(hidden_dim, reduced_dim, kernel_size=1)
            self._se_expand = Conv2d(reduced_dim, hidden_dim, kernel_size=1)

        self._project_conv = Conv2d(
            hidden_dim, out_planes, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_planes, momentum=bn_mom, eps=bn_eps)
        self._swish = MemoryEfficientSwish()

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish(x)
        # dw
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        # se
        if self.use_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        # project
        x = self._project_conv(x)
        x = self._bn2(x)

        if self.use_residual:
            return inputs + self._drop_connect(x)
        else:
            return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

    def freeze_at(self, stage):
        pass


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

# class EfficientNet(nn.Module):


class EfficientNet(Backbone):
    def __init__(self,
                 width_mult=1.0,
                 depth_mult=1.0,
                 dropout_rate=0.2,
                 num_classes=1000,
                 features_indices=[1, 4, 10, 15],
                 bn_mom=0.99,
                 bn_eps=1e-3
                 ):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.extract_features = num_classes <= 0
        # stride=2:  ----> block 1 ,3, 5 ,11
        self.return_features_indices = features_indices
        out_feature_channels = []
        out_feature_strides = [4, 8, 16, 32]
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7
        ]

        out_channels = _round_filters(32, width_mult)
        self._conv_stem = Conv2d(
            3, out_channels, 3, stride=2, bias=False, padding_mode='static_same')
        self._bn0 = nn.BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish = MemoryEfficientSwish()

        _blocks = nn.ModuleList([])
        in_channels = out_channels
        num_block = 0
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            for i in range(repeats):
                stride = s if i == 0 else 1
                _blocks.append(MBConvBlock(in_channels, out_channels,
                                           expand_ratio=t, stride=stride, kernel_size=k))
                in_channels = out_channels
                if num_block in self.return_features_indices:
                    out_feature_channels.append(out_channels)
                num_block += 1

        self._blocks = _blocks

        # for classification
        if self.num_classes > 0:
            last_channels = _round_filters(1280, width_mult)
            self._conv_head = Conv2d(
                in_channels, last_channels, 1, stride=1, bias=False, padding_mode='static_same')
            self._bn1 = nn.BatchNorm2d(
                last_channels, momentum=bn_mom, eps=bn_eps)
            self._swish = Swish()

            # self._avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
            self._dropout = nn.Dropout(dropout_rate, inplace=False)
            self._fc = nn.Linear(in_features=last_channels,
                                 out_features=num_classes, bias=True)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self._out_feature_strides = {
            "stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32}
        self._out_feature_channels = {k: c for k, c in zip(
            self._out_feature_strides.keys(), out_feature_channels)}

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)
        features = []
        for i, block in enumerate(self._blocks):
            x = block(x)
            if self.extract_features and i in self.return_features_indices:
                features.append(x)

        if self.num_classes > 0:
            x = self._swish(self._bn1(self._conv_head(x)))
            x = x.mean([2, 3])
            x = self._dropout(x)
            x = self._fc(x)
            return x
        assert len(self._out_feature_strides.keys()) == len(features)
        for i, ii in enumerate(features):
            print('EEE: ', i,  ii.shape)
        return dict(zip(self._out_feature_strides.keys(), features))

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def freeze_at(self, stage):
        pass

    def output_shape(self):
        return {f"stride{s}":
                ShapeSpec(channels=self._out_feature_channels[k], stride=s)
                for k, s in self._out_feature_strides.items()}

    @property
    def size_divisibility(self) -> int:
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 8


def _efficientnet(arch, pretrained, progress, **kwargs):
    width_mult, depth_mult, _, dropout_rate = params[arch]
    model = EfficientNet(width_mult, depth_mult, dropout_rate, **kwargs)
    # model.set_swish(memory_efficient=True)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)

        if 'num_classes' in kwargs and kwargs['num_classes'] != 1000:
            del state_dict['classifier.1.weight']
            del state_dict['classifier.1.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, input_shape):
    """
    Create a EfficientNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    arch = cfg.MODEL.EFFICIENTNET.NAME
    features_indices = cfg.MODEL.EFFICIENTNET.FEATURE_INDICES
    _out_features = cfg.MODEL.EFFICIENTNET.OUT_FEATURES
    width_mult, depth_mult, _, dropout_rate = params[arch]
    assert arch in params.keys(), '{} not in supported keys. {}'.format(arch, params.keys())
    backbone = EfficientNet(width_mult, depth_mult, dropout_rate,
                            num_classes=0, features_indices=features_indices)

    pretrained = cfg.MODEL.EFFICIENTNET.PRETRAINED
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch.replace('_', '-')])
        # del state_dict['classifier.1.weight']
        # del state_dict['classifier.1.bias']
        backbone.load_state_dict(state_dict, strict=False)
    backbone._out_features = _out_features
    return backbone


@BACKBONE_REGISTRY.register()
def build_efficientnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone




if __name__ == "__main__":
    x = torch.ones(1, 3, 112, 112)
    # model = efficientnet_b0(num_classes=0)
    # print(model)
    # print(model._out_feature_channels)
    # # print(model(x).shape)
    # # # state_dict = torch.load('efficientnet-b0-4cfa50.pth')
    # # # model.load_state_dict(state_dict)
    # print(model.output_shape())
    # for o in model(x):
    #     print(o.shape)
    # model = SeparableConv2d(3,3,3, padding_mode='static_same')
    # print(model(x).shape)
