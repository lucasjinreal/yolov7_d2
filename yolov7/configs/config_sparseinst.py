# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_sparse_inst_config(cfg):

    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MASK_ON = True
    # [SparseInst]
    cfg.MODEL.SPARSE_INST = CN()

    # parameters for inference
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005
    cfg.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.45
    cfg.MODEL.SPARSE_INST.MAX_DETECTIONS = 100

    # [Encoder]
    cfg.MODEL.SPARSE_INST.ENCODER = CN()
    cfg.MODEL.SPARSE_INST.ENCODER.NAME = "FPNPPMEncoder"
    cfg.MODEL.SPARSE_INST.ENCODER.NORM = ""
    cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS = 256

    # [Decoder]
    cfg.MODEL.SPARSE_INST.DECODER = CN()
    cfg.MODEL.SPARSE_INST.DECODER.NAME = "BaseIAMDecoder"
    cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 80
    # kernels for mask features
    cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128
    # upsample factor for output masks
    cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2.0
    cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False
    cfg.MODEL.SPARSE_INST.DECODER.GROUPS = 4
    # decoder.inst_branch
    cfg.MODEL.SPARSE_INST.DECODER.INST = CN()
    cfg.MODEL.SPARSE_INST.DECODER.INST.DIM = 256
    cfg.MODEL.SPARSE_INST.DECODER.INST.CONVS = 4
    # decoder.mask_branch
    cfg.MODEL.SPARSE_INST.DECODER.MASK = CN()
    cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256
    cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 4

    # [Loss]
    cfg.MODEL.SPARSE_INST.LOSS = CN()
    cfg.MODEL.SPARSE_INST.LOSS.NAME = "SparseInstCriterion"
    cfg.MODEL.SPARSE_INST.LOSS.ITEMS = ("labels", "masks")
    # loss weight
    cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT = 5.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT = 2.0
    # iou-aware objectness loss weight
    cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT = 1.0

    # [Matcher]
    cfg.MODEL.SPARSE_INST.MATCHER = CN()
    cfg.MODEL.SPARSE_INST.MATCHER.NAME = "SparseInstMatcher"
    cfg.MODEL.SPARSE_INST.MATCHER.ALPHA = 0.8
    cfg.MODEL.SPARSE_INST.MATCHER.BETA = 0.2

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = False

    # [Dataset mapper]
    cfg.MODEL.SPARSE_INST.DATASET_MAPPER = "SparseInstDatasetMapper"
