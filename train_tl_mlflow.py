#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""

Training script using custom coco format dataset

what you need to do is simply change the img_dir and annotation path here
Also define your own categories.

"""

import os
from datetime import timedelta
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils import comm
from detectron2.engine import hooks, HookBase

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper2

import mlflow

DATASET_ROOT = "./datasets/tl"
ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
TRAIN_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
VAL_PATH = os.path.join(DATASET_ROOT, "JPEGImages")
TRAIN_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_train.json")
VAL_JSON = os.path.join(ANN_ROOT, "annotations_coco_tls_val.json")

register_coco_instances("tl_train", {}, TRAIN_JSON, TRAIN_PATH)
register_coco_instances("tl_val", {}, VAL_JSON, VAL_PATH)


class MLFlowSnapshotHook(HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def after_train(self):
        final_model_path = f"{self.trainer.cfg.OUTPUT_DIR}/model_final.pth"
        mlflow.log_artifact(final_model_path, "model")


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        # test our own dataset mapper to add more augmentations
        return build_detection_train_loader(cfg, mapper=MyDatasetMapper2(cfg, True))

    # @classmethod
    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )
            ret.append(MLFlowSnapshotHook())

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            if comm.is_main_process():
                results = self._last_eval_results["bbox"]
                for k in results:
                    mlflow.log_metric(k, results[k], self.iter)
                if results["AP"] > self.best_ap:
                    self.best_ap = results["AP"]
                    self.best_iter = self.iter
                    mlflow.log_metric("best_AP", self.best_ap, self.iter)
                    best_iter = (7 - len(str(self.best_iter))) * "0" + str(
                        self.best_iter
                    )
                    best_model_path = f"{self.cfg.OUTPUT_DIR}/model_{best_iter}.pth"
                    new_path = f"{self.cfg.OUTPUT_DIR}/model_best.pth"
                    os.rename(best_model_path, new_path)
                    mlflow.log_artifact(new_path, "model")
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


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
    if comm.is_main_process():
        mlflow.set_experiment("traffic_light")
        run = mlflow.start_run(run_name="yolox_s_tl")
        print(f"mlflow run_id: {run.info.run_id}")
        mlflow.log_param("max_iter", cfg.SOLVER.MAX_ITER)
        mlflow.log_param("images_per_batch", cfg.SOLVER.IMS_PER_BATCH)
        trainer.best_ap = 0
        trainer.best_iter = 0
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
        timeout=timedelta(50),
        args=(args,),
    )
