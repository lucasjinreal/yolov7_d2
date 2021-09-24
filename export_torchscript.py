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


from alfred.vis.image.mask import label2color_mask, vis_bitmasks
from alfred.vis.image.det import visualize_det_cv2_part, visualize_det_cv2_fancy
from alfred.dl.torch.common import device


"""
this script used as export torchscript only.

Not all models support torchscript export. Once it exported, you can using torchscript for 
deployment or TVM accelerate.

Command:

python3 export_torchscript.py --config-file configs/coco/yolox_s.yaml --input ./images/COCO_val2014_000000002153.jpg --opts MODEL.WEIGHTS ./output/coco_yolox_s/model_final.pth

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
            image = self.aug.get_transform(
                original_image).apply_image(original_image)
            print('image after transform: ', image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            predictions = self.model([inputs])[0]
            c = time.time() - tic
            print('cost: {}, fps: {}'.format(c, 1/c))
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
        default='./images/COCO_val2014_000000001722.jpg',
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
        action='store_true',
        help="verbose when onnx export",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def load_test_image(f, h, w):
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = torch.tensor(a.astype(np.float32)).unsqueeze(0)
    return a_t, a


def load_test_image_detr(f, h, w):
    """
    detr do not using 
    """
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = torch.tensor(a.astype(np.float32)).permute(2, 0, 1).to(device)
    return torch.stack([a_t,]), a
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
        img, scores, clss, bboxes, force_color=colors, is_show=True)
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    assert os.path.isfile(
        args.input), 'onnx export only support send a image file.'

    cfg = setup_cfg(args)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(cfg.MODEL.YOLO.CLASSES)]

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    h = 768
    w = 960
    # h = 640
    # w = 640
    # inp, ori_img = load_test_image(args.input, h, w)
    inp, ori_img = load_test_image_detr(args.input, h, w)
    print('input shape: ', inp.shape)
    # inp = inp.to(torch.device('cuda'))

    model = predictor.model
    model = model.float()
    model.onnx_export = True

    ts_f = os.path.join(
        'weights', os.path.basename(cfg.MODEL.WEIGHTS).split('.')[0] + '.torchscript.pt')
    traced = torch.jit.trace(model, inp)
    torch.jit.save(traced, ts_f)
    logger.info('Model saved into: {}'.format(ts_f))

    logger.info('test if torchscript export logic is right...')
    model.onnx_vis = True
    out = model(inp)
    out = detr_postprocess(out, ori_img)
    # detr postprocess
    vis_res_fast(out, ori_img, colors=colors)
