
from detectron2.config import CfgNode as CN

def add_convnext_default_configs(_C):
    _C.MODEL.CONVNEXT = CN()

    _C.MODEL.CONVNEXT.OUT_FEATURES = ["dark3", "dark4", "dark5"]
    _C.MODEL.CONVNEXT.WEIGHTS = ""
    _C.MODEL.CONVNEXT.DEPTH_WISE = False
