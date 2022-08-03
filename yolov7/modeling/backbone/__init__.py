from .darknet import build_darknet_backbone
from .swin_transformer import build_swin_transformer_backbone
from .efficientnet import build_efficientnet_backbone, build_efficientnet_fpn_backbone
from .cspdarknet import build_cspdarknet_backbone
from .pvt_v2 import build_pvt_v2_backbone

from .res2nets.wrapper import build_res2net_backbone

from .darknetx import build_cspdarknetx_backbone
from .regnet import build_regnet_backbone
from .fbnet_v3 import *
from .fbnet_v2 import FBNetV2C4Backbone, build_fbnet
from .resnetvd import build_resnet_vd_backbone

from .convnext import build_convnext_backbone
from .efficientrep import build_efficientrep_backbone