#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""

Training script using custom coco format dataset

what you need to do is simply change the img_dir and annotation path here
Also define your own categories.

"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from train_det import Trainer, setup


def register_custom_datasets():
    # facemask dataset
    DATASET_ROOT = "./datasets/facemask"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
    VAL_PATH = os.path.join(DATASET_ROOT, "val")
    TRAIN_JSON = os.path.join(ANN_ROOT, "instances_train2017.json")
    VAL_JSON = os.path.join(ANN_ROOT, "instances_val2017.json")
    register_coco_instances("facemask_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("facemask_val", {}, VAL_JSON, VAL_PATH)

    # tl dataset
    DATASET_ROOT = "./datasets/tl"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    VAL_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    TRAIN_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_val_val.json")
    register_coco_instances("tl_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("tl_val", {}, VAL_JSON, VAL_PATH)

    # visdrone dataset
    DATASET_ROOT = "./datasets/visdrone"
    ANN_ROOT = os.path.join(DATASET_ROOT, "visdrone_coco_anno")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "VisDrone2019-DET-train/images")
    VAL_PATH = os.path.join(DATASET_ROOT, "VisDrone2019-DET-val/images")
    TRAIN_JSON = os.path.join(ANN_ROOT, "VisDrone2019-DET_train_coco.json")
    VAL_JSON = os.path.join(ANN_ROOT, "VisDrone2019-DET_val_coco.json")
    register_coco_instances("visdrone_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("visdrone_val", {}, VAL_JSON, VAL_PATH)

    # wearmask dataset
    DATASET_ROOT = "./datasets/wearmask"
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "images/train2017")
    VAL_PATH = os.path.join(DATASET_ROOT, "images/val2017")
    TRAIN_JSON = os.path.join(ANN_ROOT, "train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "val.json")
    register_coco_instances("mask_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("mask_val", {}, VAL_JSON, VAL_PATH)

    # VOC dataset in coco format
    DATASET_ROOT = "./datasets/voc"
    ANN_ROOT = DATASET_ROOT
    TRAIN_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    VAL_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
    TRAIN_JSON = os.path.join(ANN_ROOT, "annotations_coco_train_2012.json")
    VAL_JSON = os.path.join(ANN_ROOT, "annotations_coco_val_2012.json")

    register_coco_instances("voc_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("voc_val", {}, VAL_JSON, VAL_PATH)

    # ADD YOUR DATASET CONFIG HERE
    # dataset names registered must be unique, different than any of above


register_custom_datasets()


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
