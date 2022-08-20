# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN
from .utils.get_default_cfg import get_default_solver_configs
from .modeling.backbone.cfg import add_fbnet_v2_default_configs
from .configs.config_sparseinst import add_sparse_inst_config
from .configs.config_convnext import add_convnext_default_configs


def add_yolo_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    get_default_solver_configs(_C)
    add_fbnet_v2_default_configs(_C)
    add_sparse_inst_config(_C)
    add_convnext_default_configs(_C)

    _C.DATASETS.CLASS_NAMES = []

    # Allowed values are 'normal', 'softnms-linear', 'softnms-gaussian', 'cluster'
    _C.MODEL.NMS_TYPE = "normal"
    _C.MODEL.ONNX_EXPORT = False
    _C.MODEL.PADDED_VALUE = 114.0
    _C.MODEL.FPN.REPEAT = 2
    _C.MODEL.FPN.OUT_CHANNELS_LIST = [256, 512, 1024]
    # _C.MODEL.BACKBONE.STRIDE = []
    # _C.MODEL.BACKBONE.CHANNEL = []

    # Add Bi-FPN support
    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = "GN"
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.REGNETS = CN()
    _C.MODEL.REGNETS.TYPE = "x"
    _C.MODEL.REGNETS.OUT_FEATURES = ["s2", "s3", "s4"]

    # Append some LR_SCHEDULER
    _C.SOLVER.LR_SCHEDULER = CN()
    _C.SOLVER.LR_SCHEDULER.NAME = "WarmupMultiStepLR"
    _C.SOLVER.LR_SCHEDULER.MAX_ITER = 40000
    _C.SOLVER.LR_SCHEDULER.MAX_EPOCH = 500
    _C.SOLVER.LR_SCHEDULER.STEPS = (30000,)
    _C.SOLVER.LR_SCHEDULER.WARMUP_FACTOR = 1.0 / 1000
    _C.SOLVER.LR_SCHEDULER.WARMUP_ITERS = 1000
    _C.SOLVER.LR_SCHEDULER.WARMUP_METHOD = "linear"
    _C.SOLVER.LR_SCHEDULER.GAMMA = 0.1

    # Add Input
    _C.INPUT.INPUT_SIZE = [640, 640]  # h,w order

    # Add yolo config
    _C.MODEL.YOLO = CN()
    _C.MODEL.YOLO.NUM_BRANCH = 3
    _C.MODEL.YOLO.BRANCH_DILATIONS = [1, 2, 3]
    _C.MODEL.YOLO.TEST_BRANCH_IDX = 1
    _C.MODEL.YOLO.VARIANT = "yolov3"  # can be yolov5 yolov7 as well
    _C.MODEL.YOLO.ANCHORS = [
        [[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [42, 119]],
        [[10, 13], [16, 30], [33, 23]],
    ]
    _C.MODEL.YOLO.ANCHOR_MASK = []
    _C.MODEL.YOLO.CLASSES = 80
    _C.MODEL.YOLO.MAX_BOXES_NUM = 100
    _C.MODEL.YOLO.IN_FEATURES = ["dark3", "dark4", "dark5"]
    _C.MODEL.YOLO.CONF_THRESHOLD = 0.01
    _C.MODEL.YOLO.NMS_THRESHOLD = 0.5
    _C.MODEL.YOLO.IGNORE_THRESHOLD = 0.07
    _C.MODEL.YOLO.NORMALIZE_INPUT = False

    _C.MODEL.YOLO.WIDTH_MUL = 1.0
    _C.MODEL.YOLO.DEPTH_MUL = 1.0

    _C.MODEL.YOLO.IOU_TYPE = "ciou"  # diou or iou
    _C.MODEL.YOLO.LOSS_TYPE = "v4"

    _C.MODEL.YOLO.LOSS = CN()
    _C.MODEL.YOLO.LOSS.LAMBDA_XY = 1.0
    _C.MODEL.YOLO.LOSS.LAMBDA_WH = 1.0
    _C.MODEL.YOLO.LOSS.LAMBDA_CLS = 1.0
    _C.MODEL.YOLO.LOSS.LAMBDA_CONF = 1.0
    _C.MODEL.YOLO.LOSS.LAMBDA_IOU = 1.1
    _C.MODEL.YOLO.LOSS.USE_L1 = True
    _C.MODEL.YOLO.LOSS.ANCHOR_RATIO_THRESH = 4.0
    _C.MODEL.YOLO.LOSS.BUILD_TARGET_TYPE = "default"

    _C.MODEL.YOLO.NECK = CN()
    _C.MODEL.YOLO.NECK.TYPE = "yolov3"  # default is FPN, can be pafpn as well
    _C.MODEL.YOLO.NECK.WITH_SPP = False  #

    _C.MODEL.YOLO.HEAD = CN()
    _C.MODEL.YOLO.HEAD.TYPE = "yolox"

    _C.MODEL.YOLO.ORIEN_HEAD = CN()
    _C.MODEL.YOLO.ORIEN_HEAD.UP_CHANNELS = 64

    # add backbone configs
    _C.MODEL.DARKNET = CN()
    _C.MODEL.DARKNET.DEPTH = 53
    _C.MODEL.DARKNET.WITH_CSP = True
    _C.MODEL.DARKNET.RES5_DILATION = 1
    _C.MODEL.DARKNET.NORM = "BN"
    _C.MODEL.DARKNET.STEM_OUT_CHANNELS = 32
    _C.MODEL.DARKNET.OUT_FEATURES = ["dark3", "dark4", "dark5"]
    _C.MODEL.DARKNET.WEIGHTS = ""
    _C.MODEL.DARKNET.DEPTH_WISE = False

    # add for res2nets
    _C.MODEL.RESNETS.R2TYPE = "res2net50_v1d"

    # add backbone swin-transformer configs
    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.TYPE = "tiny"  # could be tiny, small, big
    _C.MODEL.SWIN.WEIGHTS = ""
    _C.MODEL.SWIN.PATCH = 4
    _C.MODEL.SWIN.WINDOW = 7
    _C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    # using index in swin backbone as output
    _C.MODEL.SWIN.OUT_FEATURES = [1, 2, 3]

    # add backbone EfficientNet configs
    _C.MODEL.EFFICIENTNET = CN()
    _C.MODEL.EFFICIENTNET.NAME = "efficientnet_b0"
    _C.MODEL.EFFICIENTNET.PRETRAINED = True
    _C.MODEL.EFFICIENTNET.FEATURE_INDICES = [1, 4, 10, 15]
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]

    # _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.SUBTYPE = "s"
    _C.MODEL.BACKBONE.PRETRAINED = True
    _C.MODEL.BACKBONE.WEIGHTS = ""
    _C.MODEL.BACKBONE.FEATURE_INDICES = [1, 4, 10, 15]
    _C.MODEL.BACKBONE.OUT_FEATURES = ["stride8", "stride16", "stride32"]

    # add SOLOv2 options
    _C.MODEL.SOLOV2 = CN()

    # Instance hyper-parameters
    _C.MODEL.SOLOV2.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    _C.MODEL.SOLOV2.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
    _C.MODEL.SOLOV2.FPN_SCALE_RANGES = (
        (1, 96),
        (48, 192),
        (96, 384),
        (192, 768),
        (384, 2048),
    )
    _C.MODEL.SOLOV2.SIGMA = 0.2
    # Channel size for the instance head.
    _C.MODEL.SOLOV2.INSTANCE_IN_CHANNELS = 256
    _C.MODEL.SOLOV2.INSTANCE_CHANNELS = 512
    # Convolutions to use in the instance head.
    _C.MODEL.SOLOV2.NUM_INSTANCE_CONVS = 4
    _C.MODEL.SOLOV2.USE_DCN_IN_INSTANCE = False
    _C.MODEL.SOLOV2.TYPE_DCN = "DCN"
    _C.MODEL.SOLOV2.NUM_GRIDS = [40, 36, 24, 16, 12]
    # Number of foreground classes.
    _C.MODEL.SOLOV2.NUM_CLASSES = 80
    _C.MODEL.SOLOV2.NUM_KERNELS = 256
    _C.MODEL.SOLOV2.NORM = "GN"
    _C.MODEL.SOLOV2.USE_COORD_CONV = True
    _C.MODEL.SOLOV2.PRIOR_PROB = 0.01

    # Mask hyper-parameters.
    # Channel size for the mask tower.
    _C.MODEL.SOLOV2.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
    _C.MODEL.SOLOV2.MASK_IN_CHANNELS = 256
    _C.MODEL.SOLOV2.MASK_CHANNELS = 128
    _C.MODEL.SOLOV2.NUM_MASKS = 256

    # Test cfg.
    _C.MODEL.SOLOV2.NMS_PRE = 500
    _C.MODEL.SOLOV2.SCORE_THR = 0.1
    _C.MODEL.SOLOV2.UPDATE_THR = 0.05
    _C.MODEL.SOLOV2.MASK_THR = 0.5
    _C.MODEL.SOLOV2.MAX_PER_IMG = 100
    # NMS type: matrix OR mask.
    _C.MODEL.SOLOV2.NMS_TYPE = "matrix"
    # Matrix NMS kernel type: gaussian OR linear.
    _C.MODEL.SOLOV2.NMS_KERNEL = "gaussian"
    _C.MODEL.SOLOV2.NMS_SIGMA = 2

    # Loss cfg.
    _C.MODEL.SOLOV2.LOSS = CN()
    _C.MODEL.SOLOV2.LOSS.FOCAL_USE_SIGMOID = True
    _C.MODEL.SOLOV2.LOSS.FOCAL_ALPHA = 0.25
    _C.MODEL.SOLOV2.LOSS.FOCAL_GAMMA = 2.0
    _C.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT = 1.0
    _C.MODEL.SOLOV2.LOSS.DICE_WEIGHT = 3.0

    # DETR config
    cfg.MODEL.DETR = CN()
    cfg.MODEL.DETR.NUM_CLASSES = 80
    cfg.MODEL.BACKBONE.SIMPLE = False
    cfg.MODEL.BACKBONE.STRIDE = 1
    cfg.MODEL.BACKBONE.CHANNEL = 0

    # FBNet
    cfg.MODEL.FBNET_V2.OUT_FEATURES = ["trunk3"]

    # For Segmentation
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ""
    # LOSS
    cfg.MODEL.DETR.DEFORMABLE = False
    cfg.MODEL.DETR.USE_FOCAL_LOSS = False
    cfg.MODEL.DETR.CENTERED_POSITION_ENCODIND = False
    cfg.MODEL.DETR.CLS_WEIGHT = 1.0
    cfg.MODEL.DETR.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.DETR.DEEP_SUPERVISION = True
    cfg.MODEL.DETR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DETR.WITH_BOX_REFINE = False
    cfg.MODEL.DETR.TWO_STAGE = False
    cfg.MODEL.DETR.DECODER_BLOCK_GRAD = True

    # TRANSFORMER
    cfg.MODEL.DETR.ATTENTION_TYPE = "DETR"  # can be SMCA, RCDA
    cfg.MODEL.DETR.NHEADS = 8
    cfg.MODEL.DETR.DROPOUT = 0.1
    cfg.MODEL.DETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DETR.ENC_LAYERS = 6
    cfg.MODEL.DETR.DEC_LAYERS = 6
    cfg.MODEL.DETR.PRE_NORM = False
    cfg.MODEL.DETR.BBOX_EMBED_NUM_LAYERS = 3
    cfg.MODEL.DETR.HIDDEN_DIM = 256
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DETR.FROZEN_WEIGHTS = ""
    cfg.MODEL.DETR.NUM_FEATURE_LEVELS = 1  # can be 3 tambien
    # for AnchorDETR
    cfg.MODEL.DETR.NUM_QUERY_POSITION = 300
    cfg.MODEL.DETR.NUM_QUERY_PATTERN = 3
    cfg.MODEL.DETR.SPATIAL_PRIOR = "learned"

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # Input Configs
    # Mosaic part
    _C.INPUT.MOSAIC = CN()
    _C.INPUT.MOSAIC.ENABLED = False
    _C.INPUT.MOSAIC.DEBUG_VIS = False
    _C.INPUT.MOSAIC.POOL_CAPACITY = 1000
    _C.INPUT.MOSAIC.NUM_IMAGES = 4
    _C.INPUT.MOSAIC.MIN_OFFSET = 0.2
    _C.INPUT.MOSAIC.MOSAIC_WIDTH = 640
    _C.INPUT.MOSAIC.MOSAIC_HEIGHT = 640

    # mosaic and mixup
    _C.INPUT.MOSAIC_AND_MIXUP = CN()
    _C.INPUT.MOSAIC_AND_MIXUP.ENABLED = False
    _C.INPUT.MOSAIC_AND_MIXUP.DEBUG_VIS = False
    _C.INPUT.MOSAIC_AND_MIXUP.POOL_CAPACITY = 1000
    _C.INPUT.MOSAIC_AND_MIXUP.NUM_IMAGES = 4
    _C.INPUT.MOSAIC_AND_MIXUP.DEGREES = 10.0
    _C.INPUT.MOSAIC_AND_MIXUP.TRANSLATE = 0.1
    _C.INPUT.MOSAIC_AND_MIXUP.SCALE = [0.5, 1.5]
    _C.INPUT.MOSAIC_AND_MIXUP.MSCALE = [0.5, 1.5]
    _C.INPUT.MOSAIC_AND_MIXUP.SHEAR = 2.0
    _C.INPUT.MOSAIC_AND_MIXUP.PERSPECTIVE = 0.0
    _C.INPUT.MOSAIC_AND_MIXUP.ENABLE_MIXUP = True
    # when doing mosaic, wh will random range (even with rectangle mosiac)
    _C.INPUT.MOSAIC_AND_MIXUP.MOSAIC_WIDTH_RANGE = (512, 800)
    _C.INPUT.MOSAIC_AND_MIXUP.MOSAIC_HEIGHT_RANGE = (512, 800)
    _C.INPUT.MOSAIC_AND_MIXUP.DISABLE_AT_ITER = 120000

    # Shift transformation
    _C.INPUT.SHIFT = CN()
    _C.INPUT.SHIFT.SHIFT_PIXELS = 32

    # Color jitter
    _C.INPUT.COLOR_JITTER = CN()
    _C.INPUT.COLOR_JITTER.BRIGHTNESS = False
    _C.INPUT.COLOR_JITTER.SATURATION = False
    _C.INPUT.COLOR_JITTER.LIGHTING = False

    # Distortion transformation
    _C.INPUT.DISTORTION = CN()
    _C.INPUT.DISTORTION.ENABLED = False
    _C.INPUT.DISTORTION.HUE = 0.1
    _C.INPUT.DISTORTION.SATURATION = 1.5
    _C.INPUT.DISTORTION.EXPOSURE = 1.5

    # Resize transformation
    _C.INPUT.RESIZE = CN()
    _C.INPUT.RESIZE.ENABLED = False
    _C.INPUT.RESIZE.SHAPE = (640, 640)
    _C.INPUT.RESIZE.SCALE_JITTER = (0.8, 1.2)
    _C.INPUT.RESIZE.TEST_SHAPE = (608, 608)

    # Jitter crop transformation
    _C.INPUT.JITTER_CROP = CN()
    _C.INPUT.JITTER_CROP.ENABLED = False
    _C.INPUT.JITTER_CROP.JITTER_RATIO = 0.3

    # GridMask part
    _C.INPUT.GRID_MASK = CN()
    _C.INPUT.GRID_MASK.ENABLED = False
    _C.INPUT.GRID_MASK.MODE = 1
    _C.INPUT.GRID_MASK.PROB = 0.3
    _C.INPUT.GRID_MASK.USE_HEIGHT = True
    _C.INPUT.GRID_MASK.USE_WIDTH = True

    # Wandb Part
    _C.WANDB = CN()
    _C.WANDB.ENABLED = False
    _C.WANDB.PROJECT_NAME = "yolov7"
