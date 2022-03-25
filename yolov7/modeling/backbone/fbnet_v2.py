#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import copy
import itertools
import logging
from typing import List
import copy

import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    BACKBONE_REGISTRY,
    RPN_HEAD_REGISTRY,
    Backbone,
    build_anchor_generator,
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7
from detectron2.modeling.roi_heads import box_head, keypoint_head, mask_head
from detectron2.utils.logger import log_first_n
try:
    from mobile_cv.arch.fbnet_v2 import fbnet_builder as mbuilder
    from mobile_cv.arch.utils.helper import format_dict_expanding_list_values
except Exception as e:
    pass


logger = logging.getLogger(__name__)

FBNET_BUILDER_IDENTIFIER = "fbnetv2"


class FBNetV2ModelArch(object):
    _MODEL_ARCH = {}

    @staticmethod
    def add(name, arch):
        assert (
            name not in FBNetV2ModelArch._MODEL_ARCH
        ), "Arch name '{}' is already existed".format(name)
        FBNetV2ModelArch._MODEL_ARCH[name] = arch

    @staticmethod
    def add_archs(archs):
        for name, arch in archs.items():
            FBNetV2ModelArch.add(name, arch)

    @staticmethod
    def get(name):
        return copy.deepcopy(FBNetV2ModelArch._MODEL_ARCH[name])


def _get_builder_norm_args(cfg):
    norm_name = cfg.MODEL.FBNET_V2.NORM
    norm_args = {"name": norm_name}
    assert all(isinstance(x, dict) for x in cfg.MODEL.FBNET_V2.NORM_ARGS)
    for dic in cfg.MODEL.FBNET_V2.NORM_ARGS:
        norm_args.update(dic)
    return norm_args


def _merge_fbnetv2_arch_def(cfg):
    arch_def = {}
    assert all(
        isinstance(x, dict) for x in cfg.MODEL.FBNET_V2.ARCH_DEF
    ), cfg.MODEL.FBNET_V2.ARCH_DEF
    for dic in cfg.MODEL.FBNET_V2.ARCH_DEF:
        arch_def.update(dic)
    return arch_def


def _parse_arch_def(cfg):
    arch = cfg.MODEL.FBNET_V2.ARCH
    arch_def = cfg.MODEL.FBNET_V2.ARCH_DEF
    assert (arch != "" and not arch_def) ^ (
        not arch and arch_def != []
    ), "Only allow one unset node between MODEL.FBNET_V2.ARCH ({}) and MODEL.FBNET_V2.ARCH_DEF ({})".format(
        arch, arch_def
    )
    arch_def = FBNetV2ModelArch.get(
        arch) if arch else _merge_fbnetv2_arch_def(cfg)
    # NOTE: arch_def is a dictionary describing the CNN architecture for creating
    # the detection model. It can describe a wide range of models including the
    # original FBNet. Each key-value pair expresses either a sub part of the model
    # like trunk or head, or stores other meta information.
    message = 'Using un-unified arch_def for ARCH "{}" (without scaling):\n{}'.format(
        arch, format_dict_expanding_list_values(arch_def)
    )
    log_first_n(logging.INFO, message, n=1, key="message")
    return arch_def


def _get_fbnet_builder_and_arch_def(cfg):
    arch_def = _parse_arch_def(cfg)

    # NOTE: one can store extra information in arch_def to configurate FBNetBuilder,
    # after this point, builder and arch_def will become independent.
    basic_args = arch_def.pop("basic_args", {})

    builder = mbuilder.FBNetBuilder(
        width_ratio=cfg.MODEL.FBNET_V2.SCALE_FACTOR,
        width_divisor=cfg.MODEL.FBNET_V2.WIDTH_DIVISOR,
        bn_args=_get_builder_norm_args(cfg),
    )
    builder.add_basic_args(**basic_args)

    return builder, arch_def


def _get_stride_per_stage(blocks):
    """
    Count the accummulated stride per stage given a list of blocks. The mbuilder
    provides API for counting per-block accumulated stride, this function leverages
    it to count per-stage accumulated stride.

    Input: a list of blocks from the unified arch_def. Note that the stage_idx
        must be contiguous (not necessarily starting from 0), and can be
        non-ascending (not tested).
    Output: a list of accumulated stride per stage, starting from lowest stage_idx.
    """
    stride_per_block = mbuilder.count_stride_each_block(blocks)

    assert len(stride_per_block) == len(blocks)
    stage_idx_set = {s["stage_idx"] for s in blocks}
    # assume stage idx are contiguous, eg. 1, 2, 3, ...
    assert max(stage_idx_set) - min(stage_idx_set) + 1 == len(stage_idx_set)
    start_stage_id = min(stage_idx_set)
    ids_per_stage = [
        [i for i, s in enumerate(blocks) if s["stage_idx"] == stage_idx]
        for stage_idx in range(start_stage_id, start_stage_id + len(stage_idx_set))
    ]  # eg. [[0], [1, 2], [3, 4, 5, 6], ...]
    block_stride_per_stage = [
        [stride_per_block[i] for i in ids] for ids in ids_per_stage
    ]  # eg. [[1], [2, 1], [2, 1, 1, 1], ...]
    stride_per_stage = [
        list(itertools.accumulate(s, lambda x, y: x * y))[-1]
        for s in block_stride_per_stage
    ]  # eg. [1, 2, 2, ...]
    accum_stride_per_stage = list(
        itertools.accumulate(stride_per_stage, lambda x, y: x * y)
    )  # eg. [first*1, first*2, first*4, ...]

    assert accum_stride_per_stage[-1] == mbuilder.count_strides(blocks)
    return accum_stride_per_stage


def fbnet_identifier_checker(func):
    """Can be used to decorate _load_from_state_dict"""

    def wrapper(self, state_dict, prefix, *args, **kwargs):
        possible_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
        if not all(FBNET_BUILDER_IDENTIFIER in k for k in possible_keys):
            logger.warning(
                "Couldn't match FBNetV2 pattern given prefix {}, possible keys: \n{}".format(
                    prefix, "\n".join(possible_keys)
                )
            )
            if any("xif" in k for k in possible_keys):
                raise RuntimeError(
                    "Seems a FBNetV1 trained checkpoint is loaded by FBNetV2 model,"
                    " which is not supported. Please consider re-train your model"
                    " using the same setup as before (it will be FBNetV2). If you"
                    " need to run the old FBNetV1 models, those configs can be"
                    " still found, see D19477651 as example."
                )
        return func(self, state_dict, prefix, *args, **kwargs)

    return wrapper


# pyre-fixme[11]: Annotation `Sequential` is not defined as a type.
class FBNetModule(nn.Sequential):
    @fbnet_identifier_checker
    def _load_from_state_dict(self, *args, **kwargs):
        return super()._load_from_state_dict(*args, **kwargs)


def build_fbnet(cfg, name, in_channels):
    """
    Create a FBNet module using FBNet V2 builder.
    Args:
        cfg (CfgNode): the config that contains MODEL.FBNET_V2.
        name (str): the key in arch_def that represents a subpart of network
        in_channels (int): input channel size
    Returns:
        nn.Sequential: the first return is a nn.Sequential, each element
            corresponds a stage in arch_def.
        List[ShapeSpec]: the second return is a list of ShapeSpec containing the
            output channels and accumulated strides for that stage.
    """
    builder, raw_arch_def = _get_fbnet_builder_and_arch_def(cfg)
    # Reset the last_depth for this builder (might have been cached), this is
    # the only mutable member variable.
    builder.last_depth = in_channels

    # NOTE: Each sub part of the model consists of several stages and each stage
    # has several blocks. "Raw" arch_def (Dict[str, List[List[Tuple]]]) uses a
    # list of stages to describe the architecture, which is more compact and
    # thus written as builtin metadata (inside FBNetV2ModelArch) or config
    # (MODEL.FBNET_V2.ARCH_DEF). "Unified" arch_def (Dict[str, List[Dict]])
    # uses a list blocks from all stages instead, which is recognized by builder.
    arch_def = mbuilder.unify_arch_def(raw_arch_def, [name])
    arch_def = {name: arch_def[name]}
    logger.info(
        "Build FBNet using unified arch_def:\n{}".format(
            format_dict_expanding_list_values(arch_def)
        )
    )
    arch_def_blocks = arch_def[name]

    stages = []
    trunk_stride_per_stage = _get_stride_per_stage(arch_def_blocks)
    shape_spec_per_stage = []
    for i, stride_i in enumerate(trunk_stride_per_stage):
        stages.append(
            builder.build_blocks(
                arch_def_blocks,
                stage_indices=[i],
                prefix_name=FBNET_BUILDER_IDENTIFIER + "_",
            )
        )
        shape_spec_per_stage.append(
            ShapeSpec(
                channels=builder.last_depth,
                stride=stride_i,
            )
        )
    return FBNetModule(*stages), shape_spec_per_stage


class FBNetV2Backbone(Backbone):
    """
    Backbone (bottom-up) for FBNet.

    Hierarchy:
        trunk0:
            xif0_0
            xif0_1
            ...
        trunk1:
            xif1_0
            xif1_1
            ...
        ...

    Output features:
        The outputs from each "stage", i.e. trunkX.
    """

    def __init__(self, cfg):
        super(FBNetV2Backbone, self).__init__()
        stages, shape_specs = build_fbnet(
            cfg, name="trunk", in_channels=cfg.MODEL.FBNET_V2.STEM_IN_CHANNELS
        )

        self._trunk_stage_names = []
        self._trunk_stages = []

        self._out_feature_channels = {}
        self._out_feature_strides = {}
        for i, (stage, shape_spec) in enumerate(zip(stages, shape_specs)):
            name = "trunk{}".format(i)
            self.add_module(name, stage)
            self._trunk_stage_names.append(name)
            self._trunk_stages.append(stage)
            self._out_feature_channels[name] = shape_spec.channels
            self._out_feature_strides[name] = shape_spec.stride

        # returned features are the final output of each stage
        self._out_features = self._trunk_stage_names
        self._trunk_stage_names = tuple(self._trunk_stage_names)

    def __prepare_scriptable__(self):
        ret = copy.deepcopy(self)
        ret._trunk_stages = nn.ModuleList(ret._trunk_stages)
        for k in self._trunk_stage_names:
            delattr(ret, k)
        return ret

    @fbnet_identifier_checker
    def _load_from_state_dict(self, *args, **kwargs):
        return super()._load_from_state_dict(*args, **kwargs)

    # return features for each stage
    def forward(self, x):
        features = {}
        for name, stage in zip(self._trunk_stage_names, self._trunk_stages):
            x = stage(x)
            features[name] = x
        return features


class FBNetV2FPN(FPN):
    """
    FPN module for FBNet.
    """

    pass


def build_fbnet_backbone(cfg):
    return FBNetV2Backbone(cfg)


@BACKBONE_REGISTRY.register()
class FBNetV2C4Backbone(Backbone):
    def __init__(self, cfg, _):
        super(FBNetV2C4Backbone, self).__init__()
        self.body = build_fbnet_backbone(cfg)
        self._out_features = self.body._out_features
        self._out_feature_strides = self.body._out_feature_strides
        self._out_feature_channels = self.body._out_feature_channels

    def forward(self, x):
        return self.body(x)


@BACKBONE_REGISTRY.register()
def FBNetV2FpnBackbone(cfg, _):
    backbone = FBNetV2FPN(
        bottom_up=build_fbnet_backbone(cfg),
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )

    return backbone


@BACKBONE_REGISTRY.register()
def FBNetV2RetinaNetBackbone(cfg, _):
    bottom_up = build_fbnet_backbone(cfg)
    in_channels_p6p7 = bottom_up.output_shape(
    )[cfg.MODEL.FPN.IN_FEATURES[-1]].channels
    top_block = LastLevelP6P7(in_channels_p6p7, cfg.MODEL.FPN.OUT_CHANNELS)
    top_block.in_feature = cfg.MODEL.FPN.IN_FEATURES[-1]
    backbone = FBNetV2FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=top_block,
    )

    return backbone


@RPN_HEAD_REGISTRY.register()
class FBNetV2RpnHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(FBNetV2RpnHead, self).__init__()

        in_channels = [x.channels for x in input_shape]
        assert len(set(in_channels)) == 1
        in_channels = in_channels[0]
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert len(set(num_cell_anchors)) == 1
        num_cell_anchors = num_cell_anchors[0]

        self.rpn_feature, shape_specs = build_fbnet(
            cfg, name="rpn", in_channels=in_channels
        )
        self.rpn_regressor = RPNHeadConvRegressor(
            in_channels=shape_specs[-1].channels,
            num_anchors=num_cell_anchors,
            box_dim=box_dim,
        )

    def forward(self, x: List[torch.Tensor]):
        x = [self.rpn_feature(y) for y in x]
        return self.rpn_regressor(x)


@box_head.ROI_BOX_HEAD_REGISTRY.register()
class FBNetV2RoIBoxHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetV2RoIBoxHead, self).__init__()

        self.roi_box_conv, shape_specs = build_fbnet(
            cfg, name="bbox", in_channels=input_shape.channels
        )
        self._out_channels = shape_specs[-1].channels

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.roi_box_conv(x)
        if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
            x = self.avgpool(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        return ShapeSpec(channels=self._out_channels)


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetV2RoIKeypointHead(keypoint_head.BaseKeypointRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetV2RoIKeypointHead, self).__init__(
            cfg=cfg,
            input_shape=input_shape,
        )

        self.feature_extractor, shape_specs = build_fbnet(
            cfg, name="kpts", in_channels=input_shape.channels
        )

        self.predictor = KeypointRCNNPredictor(
            in_channels=shape_specs[-1].channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetV2RoIKeypointHeadKRCNNPredictorNoUpscale(keypoint_head.BaseKeypointRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetV2RoIKeypointHeadKRCNNPredictorNoUpscale, self).__init__(
            cfg=cfg,
            input_shape=input_shape,
        )

        self.feature_extractor, shape_specs = build_fbnet(
            cfg,
            name="kpts",
            in_channels=input_shape.channels,
        )

        self.predictor = KeypointRCNNPredictorNoUpscale(
            in_channels=shape_specs[-1].channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetV2RoIKeypointHeadKPRCNNIRFPredictorNoUpscale(
    keypoint_head.BaseKeypointRCNNHead,
):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetV2RoIKeypointHeadKPRCNNIRFPredictorNoUpscale, self).__init__(
            cfg=cfg,
            input_shape=input_shape,
        )

        self.feature_extractor, shape_specs = build_fbnet(
            cfg,
            name="kpts",
            in_channels=input_shape.channels,
        )

        self.predictor = KeypointRCNNIRFPredictorNoUpscale(
            cfg,
            in_channels=shape_specs[-1].channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@keypoint_head.ROI_KEYPOINT_HEAD_REGISTRY.register()
class FBNetV2RoIKeypointHeadKPRCNNConvUpsamplePredictorNoUpscale(
    keypoint_head.BaseKeypointRCNNHead,
):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(
            FBNetV2RoIKeypointHeadKPRCNNConvUpsamplePredictorNoUpscale, self
        ).__init__(
            cfg=cfg,
            input_shape=input_shape,
        )

        self.feature_extractor, shape_specs = build_fbnet(
            cfg,
            name="kpts",
            in_channels=input_shape.channels,
        )

        self.predictor = KeypointRCNNConvUpsamplePredictorNoUpscale(
            cfg,
            in_channels=shape_specs[-1].channels,
            num_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        )

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x


@mask_head.ROI_MASK_HEAD_REGISTRY.register()
class FBNetV2RoIMaskHead(mask_head.BaseMaskRCNNHead):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(FBNetV2RoIMaskHead, self).__init__(
            cfg=cfg,
            input_shape=input_shape,
        )

        self.feature_extractor, shape_specs = build_fbnet(
            cfg,
            name="mask",
            in_channels=input_shape.channels,
        )

        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.predictor = MaskRCNNConv1x1Predictor(
            shape_specs[-1].channels, num_classes)

    def layers(self, x):
        x = self.feature_extractor(x)
        x = self.predictor(x)
        return x
