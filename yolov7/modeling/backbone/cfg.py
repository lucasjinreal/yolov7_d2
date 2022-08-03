from detectron2.config import CfgNode as CN


def add_fbnet_v2_default_configs(_C):
    _C.MODEL.FBNET_V2 = CN()

    _C.MODEL.FBNET_V2.ARCH = "default"
    _C.MODEL.FBNET_V2.ARCH_DEF = []
    # number of channels input to trunk
    _C.MODEL.FBNET_V2.STEM_IN_CHANNELS = 3
    _C.MODEL.FBNET_V2.SCALE_FACTOR = 1.0
    # the output channels will be divisible by WIDTH_DIVISOR
    _C.MODEL.FBNET_V2.WIDTH_DIVISOR = 1

    # normalization configs
    # name of norm such as "bn", "sync_bn", "gn"
    _C.MODEL.FBNET_V2.NORM = "bn"
    # for advanced use case that requries extra arguments, passing a list of
    # dict such as [{"num_groups": 8}, {"momentum": 0.1}] (merged in given order).
    # Note that string written it in .yaml will be evaluated by yacs, thus this
    # node will become normal python object.
    # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L410
    _C.MODEL.FBNET_V2.NORM_ARGS = []

    _C.MODEL.VT_FPN = CN()

    _C.MODEL.VT_FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    _C.MODEL.VT_FPN.OUT_CHANNELS = 256
    _C.MODEL.VT_FPN.LAYERS = 3
    _C.MODEL.VT_FPN.TOKEN_LS = [16, 16, 8, 8]
    _C.MODEL.VT_FPN.TOKEN_C = 1024
    _C.MODEL.VT_FPN.HEADS = 16
    _C.MODEL.VT_FPN.MIN_GROUP_PLANES = 64
    _C.MODEL.VT_FPN.NORM = "BN"
    _C.MODEL.VT_FPN.POS_HWS = []
    _C.MODEL.VT_FPN.POS_N_DOWNSAMPLE = []



def add_convnext_default_configs(_C):
    _C.MODEL.CONVNEXT = CN()

    _C.MODEL.CONVNEXT.OUT_FEATURES = ["dark3", "dark4", "dark5"]
    _C.MODEL.CONVNEXT.WEIGHTS = ""
    _C.MODEL.CONVNEXT.DEPTH_WISE = False

