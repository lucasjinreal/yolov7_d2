#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from detectron2.config import CfgNode as CN


def add_d2go_data_default_configs(_C):
    _C.D2GO_DATA = CN()

    # Config for "detectron2go.data.extended_coco.extended_coco_load"
    _C.D2GO_DATA.DATASETS = CN()
    # List of class names to use when loading the data, this applies to train
    # and test separately. Default value means using all classes, otherwise it'll create
    # new json file containing only given categories.
    _C.D2GO_DATA.DATASETS.TRAIN_CATEGORIES = ()
    _C.D2GO_DATA.DATASETS.TEST_CATEGORIES = ()

    # Register a list of COCO datasets in config
    # The following specifies additional coco data to inject. The required is the
    # name (NAMES), image root (IM_DIRS), coco json file (JSON_FILES) while keypoint
    # metadata (KEYPOINT_METADATA) is optional. The keypoint metadata name provided
    # here is used to lookup the metadata specified within the KEYPOINT_METADATA
    # metadata registry specified in "data/keypoint_metadata_registry.py". For adding
    # new use cases, simply register new metadata to that registry.
    _C.D2GO_DATA.DATASETS.COCO_INJECTION = CN()
    _C.D2GO_DATA.DATASETS.COCO_INJECTION.NAMES = []
    _C.D2GO_DATA.DATASETS.COCO_INJECTION.IM_DIRS = []
    _C.D2GO_DATA.DATASETS.COCO_INJECTION.JSON_FILES = []
    _C.D2GO_DATA.DATASETS.COCO_INJECTION.KEYPOINT_METADATA = []

    # On-the-fly register a list of datasets located under detectron2go/datasets
    # by specifying the filename (without .py).
    _C.D2GO_DATA.DATASETS.DYNAMIC_DATASETS = []

    # TODO: potentially add this config
    # # List of extra keys in annotation, the item will be forwarded by
    # # extended_coco_load.
    # _C.D2GO_DATA.DATASETS.ANNOTATION_FIELDS_TO_FORWARD = ()

    # Config for D2GoDatasetMapper
    _C.D2GO_DATA.MAPPER = CN()
    # dataset mapper name
    _C.D2GO_DATA.MAPPER.NAME = "D2GoDatasetMapper"
    # When enabled, image item from json dataset doesn't need to have width/hegiht,
    # they will be backfilled once image is loaded. This may cause issue when
    # width/hegiht is acutally been used by extended_coco_load, eg. grouping
    # by aspect ratio.
    _C.D2GO_DATA.MAPPER.BACKFILL_SIZE = False
    _C.D2GO_DATA.MAPPER.RETRY = 3
    _C.D2GO_DATA.MAPPER.CATCH_EXCEPTION = True

    _C.D2GO_DATA.AUG_OPS = CN()
    # List of transforms that are represented by string. Each string starts with
    # a registered name in TRANSFORM_OP_REGISTRY, optionally followed by a string
    # argument (separated by "::") which can be used for initializing the
    # transform object. See build_transform_gen for the detail.
    # Some examples are:
    # example 1: RandomFlipOp
    # example 2: RandomFlipOp::{}
    # example 3: RandomFlipOp::{"prob":0.5}
    # example 4: RandomBrightnessOp::{"intensity_min":1.0, "intensity_max":2.0}
    # NOTE: search "example repr:" in fbcode for examples.
    _C.D2GO_DATA.AUG_OPS.TRAIN = ["ResizeShortestEdgeOp", "RandomFlipOp"]
    _C.D2GO_DATA.AUG_OPS.TEST = ["ResizeShortestEdgeOp"]

    _C.D2GO_DATA.TEST = CN()
    # Evaluate on the first specified number of images for each datset during
    # testing, default value 0 means using all images.
    # NOTE: See maybe_subsample_n_images for details.
    _C.D2GO_DATA.TEST.MAX_IMAGES = 0
    _C.D2GO_DATA.TEST.SUBSET_SAMPLING = "frontmost"  # one of {"frontmost", "random"}

    return _C
