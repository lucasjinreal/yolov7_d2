"""
Using Atom to quantize d2 models

such as YOLOX

this is WIP, not full work now.
"""
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
from yolov7.config import add_yolo_config
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    DatasetCatalog,
)
from detectron2.data import build_detection_test_loader

from alfred.vis.image.mask import label2color_mask, vis_bitmasks
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from alfred.dl.torch.common import device
from detectron2.data.dataset_mapper import DatasetMapper
from yolov7.data.dataset_mapper import MyDatasetMapper

from atomquant.atom.prepare_by_platform import prepare_by_platform, BackendType
from atomquant.atom.convert_deploy import convert_deploy
from torchvision import transforms
import torchvision
import torch
import yaml
from easydict import EasyDict

backend_dict = {
    "Academic": BackendType.Academic,
    "Tensorrt": BackendType.Tensorrt,
    "SNPE": BackendType.SNPE,
    "PPLW8A16": BackendType.PPLW8A16,
    "NNIE": BackendType.NNIE,
    "Vitis": BackendType.Vitis,
    "ONNX_QNN": BackendType.ONNX_QNN,
    "PPLCUDA": BackendType.PPLCUDA,
}


"""
WIP.

"""


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while "root" in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config["root"])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


torch.set_grad_enabled(False)


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
            image = self.aug.get_transform(original_image).apply_image(original_image)
            print("image after transform: ", image.shape)
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # do not do transpose here
            image = torch.as_tensor(image.astype("float32"))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.YOLO.CONF_THRESHOLD = 0.3
    cfg.MODEL.YOLO.NMS_THRESHOLD = 0.6
    cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1

    cfg.INPUT.MIN_SIZE_TEST = 672  # 90ms
    # cfg.INPUT.MIN_SIZE_TEST = 512 # 70ms
    # cfg.INPUT.MIN_SIZE_TEST = 1080  # 40ms
    # cfg.INPUT.MAX_SIZE_TEST = 640 # 40ms
    # cfg.INPUT.MAX_SIZE_TEST = 768 # 70ms
    cfg.INPUT.MAX_SIZE_TEST = 1080  # 70ms
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "-qc",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="quantize config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_test_image(f, h, w, bs=1):
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = torch.tensor(a.astype(np.float32)).to(device).unsqueeze(0).repeat(bs, 1, 1, 1)
    return a_t, a


def load_test_image_detr(f, h, w):
    """
    detr do not using
    """
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = torch.tensor(a.astype(np.float32)).permute(2, 0, 1).to(device)
    return (
        torch.stack(
            [
                a_t,
            ]
        ),
        a,
    )


def get_model_infos(config_file):
    if "sparse_inst" in config_file:
        # output_names = ["masks", "scores", "labels"]
        output_names = ["masks", "scores"]
        input_names = ["images"]
        dynamic_axes = {"images": {0: "batch"}}
        return input_names, output_names, dynamic_axes
    elif "detr" in config_file:
        return ["boxes", "scores", "labels"]
    else:
        return ["outs"]


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        imgs = batch["images"]
        print(imgs)
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data


def get_quantize_model(model, config):
    backend_type = (
        BackendType.Academic
        if not hasattr(config.quantize, "backend")
        else backend_dict[config.quantize.backend]
    )
    extra_prepare_dict = (
        {} if not hasattr(config, "extra_prepare_dict") else config.extra_prepare_dict
    )
    return prepare_by_platform(model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = (
        BackendType.Academic
        if not hasattr(config.quantize, "backend")
        else backend_dict[config.quantize.backend]
    )
    output_path = (
        "./"
        if not hasattr(config.quantize, "deploy")
        else config.quantize.deploy.output_path
    )
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = (
        False
        if not hasattr(config.quantize.deploy, "deploy_to_qlinear")
        else config.quantize.deploy.deploy_to_qlinear
    )

    convert_deploy(
        model,
        backend_type,
        {"input": [1, 3, 224, 224]},
        output_path=output_path,
        model_name=model_name,
        deploy_to_qlinear=deploy_to_qlinear,
    )


def evaluate_model(model, test_loader, criterion=None):
    t0 = time.time()
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    t1 = time.time()
    print(f"eval loss: {eval_loss}, eval acc: {eval_accuracy}, cost: {t1 - t0}")
    return eval_loss, eval_accuracy


def prepare_dataloader(cfg):
    test_loader = build_detection_test_loader(
        cfg, "coco_2017_val", mapper=MyDatasetMapper(cfg, True)
    )
    return test_loader


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    model = predictor.model
    # must in onnx export for PTQ, since we need export onnx later.
    model.onnx_export = True

    onnx_f = os.path.join(
        "weights", os.path.basename(cfg.MODEL.WEIGHTS).split(".")[0] + ".onnx"
    )
    test_loader = prepare_dataloader(cfg)

    config_f = args.qc
    config = parse_config(config_f)
    print(config)
    model.to(device)
    model.eval()

    if hasattr(config, "quantize"):
        model = get_quantize_model(model, config)
        print("now model in quantized mode.")

    model.to(device)
    evaluate_model(model, test_loader)

    # evaluate
    if not hasattr(config, "quantize"):
        evaluate_model(model, test_loader)
    elif config.quantize.quantize_type == "advanced_ptq":
        print("begin calibration now!")
        cali_data = load_calibrate_data(
            test_loader, cali_batchsize=config.quantize.cali_batchsize
        )
        from mqbench.utils.state import (
            enable_quantization,
            enable_calibration_woquantization,
        )

        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
        model(cali_data[0].cuda())
        print("begin advanced PTQ now!")
        if hasattr(config.quantize, "reconstruction"):
            model = ptq_reconstruction(model, cali_data, config.quantize.reconstruction)
        enable_quantization(model)
        evaluate_model(model, test_loader)
        if hasattr(config.quantize, "deploy"):
            deploy(model, config)
    elif config.quantize.quantize_type == "naive_ptq":
        print("begin calibration now!")
        cali_data = load_calibrate_data(
            test_loader, cali_batchsize=config.quantize.cali_batchsize
        )
        from atomquant.atom.utils.state import (
            enable_quantization,
            enable_calibration_woquantization,
        )

        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type="act_fake_quant")
        for batch in cali_data:
            model(batch.to(device))
        enable_calibration_woquantization(model, quantizer_type="weight_fake_quant")
        model(cali_data[0].to(device))
        print("begin quantization now!")
        enable_quantization(model)
        # print(model)
        evaluate_model(model, test_loader)
        if hasattr(config.quantize, "deploy"):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")
