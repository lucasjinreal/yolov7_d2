import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.utils import comm

from .layers.utils import get_norm

try:
    from mish_cuda import MishCuda as Mish
except Exception:
    if comm.is_main_process():
        logger = logging.getLogger(__name__)
        logger.warning(
            "Install mish-cuda to speed up training and inference. More "
            "importantly, replace the naive Mish with MishCuda will give a "
            "~1.5G memory saving during training."
        )

    def mish(x):
        return x.mul(F.softplus(x).tanh())

    class Mish(nn.Module):
        def __init__(self):
            super(Mish, self).__init__()

        def forward(self, x):
            return mish(x)


"""
Code originally come from YOLOF repo.
"""


def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1,
                       norm_type="BN"):
    """
    A help function to build a 'conv-bn-activation' module
    """
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    layers.append(get_norm(norm_type, planes, eps=1e-4, momentum=0.03))
    layers.append(Mish())
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None,
                 norm_type="BN"):
        """Residual Block for DarkNet.

        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        self.bn1 = get_norm(norm_type, inplanes, eps=1e-4, momentum=0.03)
        self.bn2 = get_norm(norm_type, planes, eps=1e-4, momentum=0.03)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = Mish()

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out


class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.
    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concat into fuse transition layer.

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        dilation (int): conv dilation
        stride (int): stride for the base layer
        norm_type (str): normalization layer type.
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2,
                 norm_type="BN"):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_type=norm_type
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_type=norm_type
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


def make_dark_layer(block,
                    inplanes,
                    planes,
                    num_blocks,
                    dilation=1,
                    stride=2,
                    norm_type="BN"):
    downsample = ConvNormActivation(
        inplanes=inplanes,
        planes=planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        norm_type=norm_type
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                downsample=downsample if i == 0 else None,
                dilation=dilation,
                norm_type=norm_type
            )
        )
    return nn.Sequential(*layers)


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1,
                       norm_type="BN"):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0,
        norm_type=norm_type
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation,
                norm_type=norm_type
            )
        )
    return nn.Sequential(*layers)


class DarkNet(Backbone):
    """DarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767

    Args:
        depth (int): Depth of Darknet, from {53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_features (List[str]): Output features.
        norm_type (str): type of normalization layer.
        res5_dilation (int): dilation for the last stage
    """

    arch_settings = {
        53: (DarkBlock, (1, 2, 8, 8, 4))
    }

    def __init__(self,
                 depth,
                 with_csp=False,
                 out_features=["res5"],
                 norm_type="BN",
                 res5_dilation=1):
        super(DarkNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.with_csp = with_csp
        self._out_features = out_features
        self.norm_type = norm_type
        self.res5_dilation = res5_dilation

        self.block, self.stage_blocks = self.arch_settings[depth]
        self.inplanes = 32

        self.output_shape_dict = dict()

        self._make_stem_layer()

        self.dark_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            if i == 4 and self.res5_dilation == 2:
                dilation = self.res5_dilation
                stride = 1
            if not self.with_csp:
                layer = make_dark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    dilation=dilation,
                    stride=stride,
                    norm_type=self.norm_type
                )
            else:
                layer = make_cspdark_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    is_csp_first_stage=True if i == 0 else False,
                    dilation=dilation,
                    norm_type=self.norm_type
                )
                layer = CrossStagePartialBlock(
                    self.inplanes,
                    planes,
                    stage_layers=layer,
                    is_csp_first_stage=True if i == 0 else False,
                    dilation=dilation,
                    stride=stride,
                    norm_type=self.norm_type
                )
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            # layer_name = 'dark{}'.format(i + 1)
            self.add_module(layer_name, layer)
            self.dark_layers.append(layer_name)
            self.output_shape_dict[layer_name] = ShapeSpec(channels=planes)

        # freeze stage<=2
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False
        for p in self.layer2.parameters():
            p.requires_grad = False

    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = get_norm(
            self.norm_type, self.inplanes, eps=1e-4, momentum=0.03
        )
        self.act1 = Mish()

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        for i, layer_name in enumerate(self.dark_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if layer_name in self._out_features:
                outputs[layer_name] = x
        outputs[self._out_features[-1]] = x
        return outputs

    def output_shape(self):
        self.output_shape_dict["res5"] = ShapeSpec(
            channels=1024, stride=16 if self.res5_dilation == 2 else 32
        )
        return self.output_shape_dict

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


@BACKBONE_REGISTRY.register()
def build_cspdarknet_backbone(cfg, input_shape=None):
    """
    Create a DarkNet/CSPDarkNet instance from config.

    Returns:
        DarkNet: a :class:`DarkNet` instance.
    """
    depth = cfg.MODEL.DARKNET.DEPTH
    with_csp = cfg.MODEL.DARKNET.WITH_CSP
    out_features = cfg.MODEL.DARKNET.OUT_FEATURES
    norm_type = cfg.MODEL.DARKNET.NORM
    res5_dilation = cfg.MODEL.DARKNET.RES5_DILATION
    return DarkNet(
        depth=depth,
        with_csp=with_csp,
        out_features=out_features,
        norm_type=norm_type,
        res5_dilation=res5_dilation
    )
