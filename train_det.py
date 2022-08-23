"""
train detection entrance

Copyright @2022 YOLOv7 authors

"""
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.utils import comm
from yolov7.data.dataset_mapper import MyDatasetMapper, MyDatasetMapper2
from yolov7.config import add_yolo_config
from yolov7.utils.d2overrides import default_setup
from yolov7.utils.wandb.wandb_logger import is_wandb_available


class Trainer(DefaultTrainer):

    custom_mapper = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        cls.custom_mapper = MyDatasetMapper2(cfg, True)
        return build_detection_train_loader(cfg, mapper=cls.custom_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

    def build_writers(self):
        if self.cfg.WANDB.ENABLED is is_wandb_available():
            from yolov7.utils.wandb.wandb_logger import WandbWriter

            writers = super().build_writers() + [
                WandbWriter(self.cfg.WANDB.PROJECT_NAME)
            ]
        else:
            writers = super().build_writers()
        return writers


def setup(args):
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
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
