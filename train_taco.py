#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""

Training script using custom coco format dataset

what you need to do is simply change the img_dir and annotation path here
Also define your own categories.

"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.dataset_mapper import DatasetMapper

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper


# here is your dataset config
CLASS_NAMES = ['Aluminium foil',
               'Battery',
               'Aluminium blister pack',
               'Carded blister pack',
               'Other plastic bottle',
               'Clear plastic bottle',
               'Glass bottle',
               'Plastic bottle cap',
               'Metal bottle cap',
               'Broken glass',
               'Food Can',
               'Aerosol',
               'Drink can',
               'Toilet tube',
               'Other carton',
               'Egg carton',
               'Drink carton',
               'Corrugated carton',
               'Meal carton',
               'Pizza box',
               'Paper cup',
               'Disposable plastic cup',
               'Foam cup',
               'Glass cup',
               'Other plastic cup',
               'Food waste',
               'Glass jar',
               'Plastic lid',
               'Metal lid',
               'Other plastic',
               'Magazine paper',
               'Tissues',
               'Wrapping paper',
               'Normal paper',
               'Paper bag',
               'Plastified paper bag',
               'Plastic film',
               'Six pack rings',
               'Garbage bag',
               'Other plastic wrapper',
               'Single-use carrier bag',
               'Polypropylene bag',
               'Crisp packet',
               'Spread tub',
               'Tupperware',
               'Disposable food container',
               'Foam food container',
               'Other plastic container',
               'Plastic glooves',
               'Plastic utensils',
               'Pop tab',
               'Rope & strings',
               'Scrap metal',
               'Shoe',
               'Squeezable tube',
               'Plastic straw',
               'Paper straw',
               'Styrofoam piece',
               'Unlabeled litter',
               'Cigarette']
DATASET_ROOT = './datasets/taco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')
VAL_PATH = os.path.join(DATASET_ROOT, 'images')
TRAIN_JSON = os.path.join(ANN_ROOT, 'train_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'train_val.json')
PREDEFINED_SPLITS_DATASET = {
    "taco_train": (TRAIN_PATH, TRAIN_JSON),
    "taco_val": (VAL_PATH, VAL_JSON),
}


def plain_register_dataset():
    for k, v in PREDEFINED_SPLITS_DATASET.items():
        DatasetCatalog.register(
            k, lambda: load_coco_json(v[1], v[0]))
        MetadataCatalog.get(k).set(thing_classes=CLASS_NAMES,
                                   evaluator_type='coco',
                                   json_file=v[1],
                                   image_root=v[0])


plain_register_dataset()


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        # test our own dataset mapper to add more augmentations
        # return build_detection_train_loader(cfg, mapper=MyDatasetMapper(cfg, True))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


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
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
