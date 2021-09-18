
from detectron2.modeling.backbone import RegNet
from detectron2.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock

from detectron2.modeling.backbone import BACKBONE_REGISTRY


Regnet_types_config = {
    "RegNetX_200MF": {
        "stem_width": 32,
        "w_a": 36.44,
        "w_0": 24,
        "w_m": 2.49,
        "group_width": 8,
        "depth": 13,
    },
    "RegNetX_400MF": {
        "stem_width": 32,
        "w_a": 24.48,
        "w_0": 24,
        "w_m": 2.54,
        "group_width": 16,
        "depth": 22,
    },
    "RegNetX_600MF": {
        "stem_width": 32,
        "w_a": 36.97,
        "w_0": 48,
        "w_m": 2.24,
        "group_width": 24,
        "depth": 16,
    },
    # regnety series
    'RegNetY_200MF': {
        "stem_width": 32,
        "w_a": 36.44,
        "w_0": 24,
        "w_m": 2.49,
        "group_width": 8,
        "depth": 13,
        "se_ratio": 0.25,
    },
    'RegNetY_400MF': {
        "stem_width": 32,
        "w_a": 27.89,
        "w_0": 48,
        "w_m": 2.09,
        "group_width": 8,
        "depth": 16,
        "se_ratio": 0.25,
    },
    'RegNetY_600MF': {
        "stem_width": 32,
        "w_a": 32.54,
        "w_0": 48,
        "w_m": 2.32,
        "group_width": 16,
        "depth": 15,
        "se_ratio": 0.25,
    },
    'RegNetY_800MF': {
        "stem_width": 32,
        "w_a": 38.84,
        "w_0": 56,
        "w_m": 2.4,
        "group_width": 16,
        "depth": 14,
        "se_ratio": 0.25,
    },
    'RegNetY_1_6GF': {
        "stem_width": 32,
        "w_a": 20.71,
        "w_0": 48,
        "w_m": 2.65,
        "group_width": 24,
        "depth": 27,
        "se_ratio": 0.25,
    },
    'RegNetY_3_2GF': {
        "stem_width": 32,
        "w_a": 42.63,
        "w_0": 80,
        "w_m": 2.66,
        "group_width": 24,
        "depth": 21,
        "se_ratio": 0.25,
    },
    'RegNetY_4_0GF': {
        "stem_width": 32,
        "w_a": 31.41,
        "w_0": 96,
        "w_m": 2.24,
        "group_width": 64,
        "depth": 22,
        "se_ratio": 0.25,
    },
    'RegNetY_6_4GF': {
        "stem_width": 32,
        "w_a": 33.22,
        "w_0": 112,
        "w_m": 2.27,
        "group_width": 72,
        "depth": 25,
        "se_ratio": 0.25,
    },
    'RegNetY_8_0GF': {
        "stem_width": 32,
        "w_a": 76.82,
        "w_0": 192,
        "w_m": 2.19,
        "group_width": 56,
        "depth": 17,
        "se_ratio": 0.25,
    },
    'RegNetY_12GF': {
        "stem_width": 32,
        "w_a": 73.36,
        "w_0": 168,
        "w_m": 2.37,
        "group_width": 112,
        "depth": 19,
        "se_ratio": 0.25,
    },
    'RegNetY_16GF': {
        "stem_width": 32,
        "w_a": 106.23,
        "w_0": 200,
        "w_m": 2.48,
        "group_width": 112,
        "depth": 18,
        "se_ratio": 0.25,
    },
    'RegNetY_32GF': {
        "stem_width": 32,
        "w_a": 115.89,
        "w_0": 232,
        "w_m": 2.53,
        "group_width": 232,
        "depth": 20,
        "se_ratio": 0.25,
    },
}


@BACKBONE_REGISTRY.register()
def build_regnet_backbone(cfg, input_shape=None):
    t = cfg.MODEL.REGNETS.TYPE  # "x" or "y"
    out_features = cfg.MODEL.REGNETS.OUT_FEATURES  # ["s1", "s2", "s3", "s4"]
    assert t in Regnet_types_config.keys(), 'type must be one of: {}'.format(
        Regnet_types_config.keys())

    args = Regnet_types_config[t]
    return RegNet(
        stem_class=SimpleStem,
        block_class=ResBottleneckBlock,
        norm="SyncBN",
        out_features=out_features,
        **args
    )