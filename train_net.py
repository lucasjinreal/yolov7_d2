#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.modeling import build_model
from detectron2.utils import comm
import logging
from detectron2.solver import build_lr_scheduler, LRMultiplier, WarmupParamScheduler
from fvcore.common.param_scheduler import CosineParamScheduler

from yolov7.data.dataset_mapper import MyDatasetMapper, MyDatasetMapper2
from yolov7.config import add_yolo_config
from yolov7.utils.allreduce_norm import all_reduce_norm


class Trainer(DefaultTrainer):

    custom_mapper = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        # test our own dataset mapper to add more augmentations
        cls.custom_mapper = MyDatasetMapper2(cfg, True)
        return build_detection_train_loader(cfg, mapper=cls.custom_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        # logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))
        return model

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()
        if comm.get_world_size() == 1:
            self.model.update_iter(self.iter)
        else:
            self.model.module.update_iter(self.iter)

        if self.iter > self.cfg.INPUT.MOSAIC_AND_MIXUP.DISABLE_AT_ITER and self.cfg.INPUT.MOSAIC_AND_MIXUP.ENABLED:
            # disable augmentation
            self.cfg.defrost()
            self.cfg.INPUT.MOSAIC_AND_MIXUP.ENABLED = False
            self.cfg.freeze()
            self.custom_mapper.disable_aug()

        # if comm.is_main_process():
        #     # when eval period, apply all_reduce_norm as in https://github.com/Megvii-BaseDetection/YOLOX/issues/547#issuecomment-903220346
        #     interval = self.cfg.SOLVER.CHECKPOINT_PERIOD if self.cfg.TEST.EVAL_PERIOD == 0 else self.cfg.TEST.EVAL_PERIOD
        #     if self.iter % interval == 0 and self.iter != 0:
        #         all_reduce_norm(self.model)
        #         self.checkpointer.save('latest')


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
    # print('trainer.start: ', trainer.start_iter)
    # trainer.model.iter = trainer.start_iter
    # print('trainer.start: ', trainer.model.iter)
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
