
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from numpy.core.fromnumeric import sort
import tqdm
import torch
import time
import random
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import numpy as np
from detectron2.data.catalog import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper2

from alfred.vis.image.mask import label2color_mask, vis_bitmasks
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy

# constants
WINDOW_NAME = "COCO detections"


def register_all_known_datasets():
    """
    Please change your own path here
    """

    r = './datasets/coco'
    register_coco_instances("coco_2017_train_mini", {}, os.path.join(
        r, 'annotations/instances_minitrain2017.json'), os.path.join(r, 'train2017'))
    register_coco_instances("coco_2014_val_mini", {}, os.path.join(
        r, 'annotations/instances_minival2014.json'), os.path.join(r, 'val2014'))

    r = './datasets/tl'
    register_coco_instances("tl_train", {}, os.path.join(
        r, 'annotations/annotations_coco_tls_train.json'), os.path.join(r, 'JPEGImages'))
    register_coco_instances("tl_val", {}, os.path.join(
        r, 'annotations/annotations_coco_tls_val.json'), os.path.join(r, 'JPEGImages'))

    r = './datasets/visdrone'
    ANN_ROOT = os.path.join(r, 'visdrone_coco_anno')
    register_coco_instances("visdrone_train", {}, os.path.join(
        ANN_ROOT, 'VisDrone2019-DET_train_coco.json'), os.path.join(r, 'VisDrone2019-DET-train/images'))
    register_coco_instances("visdrone_val", {}, os.path.join(
        ANN_ROOT, 'VisDrone2019-DET_val_coco.json'), os.path.join(r, 'VisDrone2019-DET-val/images'))


register_all_known_datasets()


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.YOLO.CONF_THRESHOLD = 0.3
    # cfg.MODEL.YOLO.NMS_THRESHOLD = 0.6
    # cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1

    cfg.freeze()
    return cfg


class DefaultPredictor:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(
                original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            predictions = self.model([inputs])[0]
            c = time.time() - tic
            print('cost: {}, fps: {}'.format(c, 1/c))
            return predictions


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    val_dataset_name = cfg.DATASETS.TEST[0]

    model = DefaultPredictor(cfg).model

    evaluator = COCOEvaluator(val_dataset_name, ("bbox",),
                              False, output_dir="./output_val/")
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    res = inference_on_dataset(model, val_loader, evaluator)
    print(res)
