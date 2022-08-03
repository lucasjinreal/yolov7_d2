from .res2net_v1b import res2net50_v1b, res2net50_v1b_26w_4s, res2net101_v1b, res2net101_v1b_26w_4s, res2net152_v1b_26w_4s
from .res2next import res2next50
from detectron2.modeling.backbone import build_backbone, BACKBONE_REGISTRY


@BACKBONE_REGISTRY.register()
def build_res2net_backbone(cfg, input_shape):
    """
    Create a Res2Net instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    r2type = cfg.MODEL.RESNETS.R2TYPE
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    if r2type == "res2net50_v1b":
        model = res2net50_v1b(pretrained=True, out_features=out_features)
    elif r2type == "res2net50_v1b_26w_4s":
        model = res2net50_v1b_26w_4s(pretrained=True, out_features=out_features)
    elif r2type == "res2net101_v1b":
        model = res2net101_v1b(pretrained=True, out_features=out_features)
    elif r2type == "res2net101_v1b_26w_4s":
        model = res2net101_v1b_26w_4s(pretrained=True, out_features=out_features)
    elif r2type == "res2next50":
        model = res2next50(pretrained=True, out_features=out_features)
    return model