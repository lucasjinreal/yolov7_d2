# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
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
import onnx
from alfred.vis.image.mask import label2color_mask, vis_bitmasks
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from alfred.dl.torch.common import device


"""
this script used as export onnx only.

Not all models support onnx export. Once it exported, you can using onnx for 
deployment or TVM accelerate.

Command:

python3 export_onnx.py --config-file configs/coco/yolox_s.yaml --input ./images/COCO_val2014_000000002153.jpg --opts MODEL.WEIGHTS ./output/coco_yolox_s/model_final.pth

"""

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
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # nargs="+",
        default="./images/COCO_val2014_000000001722.jpg",
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
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="verbose when onnx export",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def change_detr_onnx(onnx_path):
    """
    Fix default detr onnx model output all 0
    """
    node_configs = [
        (1660, 1662),
        (2775, 2777),
        (2961, 2963),
        (3333, 3335),
        (4077, 4079),
    ]
    if "batch_2" in onnx_path:
        node_number = node_configs[1]
    elif "batch_4" in onnx_path:
        node_number = node_configs[2]
    elif "batch_8" in onnx_path:
        node_number = node_configs[3]
    elif "batch_16" in onnx_path:
        node_number = node_configs[4]
    else:
        node_number = node_configs[0]

    graph = gs.import_onnx(onnx.load(onnx_path))
    for node in graph.nodes:
        if node.name == f"Gather_{node_number[0]}":
            print(node.inputs[1])
            node.inputs[1].values = np.int64(5)
            print(node.inputs[1])
        elif node.name == f"Gather_{node_number[1]}":
            print(node.inputs[1])
            node.inputs[1].values = np.int64(5)
            print(node.inputs[1])

    onnx.save(gs.export_onnx(graph), onnx_path + "_changed.onnx")
    print(f"[INFO] onnx修改完成, 保存在{onnx_path + '_changed.onnx'}.")


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
    # return torch.stack([a_t, a_t]), a


def detr_postprocess(out_boxes, ori_img):
    """
    normalized xyxy output
    """
    h, w, _ = ori_img.shape
    out_boxes[..., 0] *= w
    out_boxes[..., 1] *= h
    out_boxes[..., 2] *= w
    out_boxes[..., 3] *= h
    return out_boxes


def vis_res_fast(res, img, colors):
    res = res[0].cpu().numpy()
    scores = res[:, -2]
    clss = res[:, -1]
    bboxes = res[:, :4]

    indices = scores > 0.6
    bboxes = bboxes[indices]
    scores = scores[indices]
    clss = clss[indices]

    img = visualize_det_cv2_part(
        img, scores, clss, bboxes, force_color=colors, is_show=True
    )
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


def get_model_infos(config_file):
    if "sparse_inst" in config_file:
        output_names = ["masks", "scores", "labels"]
        # output_names = ["masks", "scores"]
        input_names = ["images"]
        dynamic_axes = {"images": {0: "batch"}}
        return input_names, output_names, dynamic_axes
    elif "detr" in config_file:
        return ["boxes", "scores", "labels"]
    else:
        return ["outs"]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    assert os.path.isfile(args.input), "onnx export only support send a image file."

    cfg = setup_cfg(args)
    colors = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in range(cfg.MODEL.YOLO.CLASSES)
    ]

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    # h = 1056
    # w = 1920
    h = 640
    w = 640
    inp, ori_img = load_test_image(args.input, h, w)
    # TODO: remove hard coded for detr
    # inp, ori_img = load_test_image_detr(args.input, h, w)
    logger.info(f"input shape: {inp.shape}")

    model = predictor.model
    model = model.float()
    model.onnx_export = True

    onnx_f = os.path.join(
        "weights", os.path.basename(cfg.MODEL.WEIGHTS).split(".")[0] + ".onnx"
    )

    input_names, output_names, dynamic_axes = get_model_infos(args.config_file)
    torch.onnx.export(
        model,
        inp,
        onnx_f,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        do_constant_folding=True,
        verbose=args.verbose,
        dynamic_axes=dynamic_axes,
    )
    logger.info("Model saved into: {}".format(onnx_f))

    # use onnxsimplify to reduce reduent model.
    sim_onnx = onnx_f.replace(".onnx", "_sim.onnx")
    os.system(
        f"python3 -m onnxsim {onnx_f} {sim_onnx} --dynamic-input-shape --input-shape 1,{h},{w},3"
    )
    logger.info("generate simplify onnx to: {}".format(sim_onnx))
    if "detr" in sim_onnx:
        # this is need for detr onnx model
        change_detr_onnx(sim_onnx)

    logger.info("test if onnx export logic is right...")
    model.onnx_vis = True
    out = model(inp)
    out = detr_postprocess(out, ori_img)
    # detr postprocess
    vis_res_fast(out, ori_img, colors=colors)

    logger.info('Now tracing model into torchscript.. If this failed, just ignore it.')
    ts_f = os.path.join(
        'weights', os.path.basename(cfg.MODEL.WEIGHTS).split('.')[0] + '.pt')
    traced = torch.jit.trace(model, inp)
    torch.jit.save(traced, ts_f)
    logger.info('Model saved into: {}'.format(ts_f))
