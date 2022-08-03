# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.engine import hooks
from detectron2.modeling import build_model
from detectron2.solver.build import maybe_add_gradient_clipping

from yolov7.data.dataset_mapper import DetrDatasetMapper
from yolov7.config import add_yolo_config
from yolov7.optimizer import build_optimizer_mapper


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if "detr" in cfg.MODEL.META_ARCHITECTURE.lower():
            mapper = DetrDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        # params: List[Dict[str, Any]] = []
        # memo: Set[torch.nn.parameter.Parameter] = set()
        # for key, value in model.named_parameters(recurse=True):
        #     if not value.requires_grad:
        #         continue
        #     # Avoid duplicating parameters
        #     if value in memo:
        #         continue
        #     memo.add(value)
        #     lr = cfg.SOLVER.BASE_LR
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
        #     if "backbone" in key:
        #         lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        #     params += [{"params": [value], "lr": lr,
        #                 "weight_decay": weight_decay}]

        # # optim: the optimizer class
        # def maybe_add_full_model_gradient_clipping(optim):
        #     # detectron2 doesn't have full model gradient clipping now
        #     clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        #     enable = (
        #         cfg.SOLVER.CLIP_GRADIENTS.ENABLED
        #         and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
        #         and clip_norm_val > 0.0
        #     )
        #     class FullModelGradientClippingOptimizer(optim):
        #         def step(self, closure=None):
        #             all_params = itertools.chain(
        #                 *[x["params"] for x in self.param_groups])
        #             torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
        #             super().step(closure=closure)

        #     return FullModelGradientClippingOptimizer if enable else optim

        # optimizer_type = cfg.SOLVER.OPTIMIZER
        # if optimizer_type == "SGD":
        #     optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
        #         params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        #     )
        # elif optimizer_type == "ADAMW":
        #     optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
        #         params, cfg.SOLVER.BASE_LR
        #     )
        # else:
        #     raise NotImplementedError(f"no optimizer type {optimizer_type}")
        # if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        #     optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        # return optimizer
        return build_optimizer_mapper(cfg, model)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=200))
        return ret

    @classmethod
    def build_model(cls, cfg):
        # remove print model
        model = build_model(cfg)
        return model

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
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
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
