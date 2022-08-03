from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone

import torch
from timm.models.cspnet import cspresnet50d


@BACKBONE_REGISTRY.register()
def build_cspresnet50d_backbone(cfg, input_shape):
    """
    Create a EfficientNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    arch = cfg.MODEL.EFFICIENTNET.NAME
    features_indices = cfg.MODEL.EFFICIENTNET.FEATURE_INDICES
    _out_features = cfg.MODEL.EFFICIENTNET.OUT_FEATURES
    backbone = cspresnet50d(pretrained=True)
    backbone._out_features = _out_features
    return backbone

