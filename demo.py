import argparse
import multiprocessing as mp
import pathlib
import random
import time

import cv2
import detectron2.data.transforms as T
import torch
from alfred.utils.file_io import ImageSourceIter
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.mask import vis_bitmasks_with_classes
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures.masks import BitMasks
from detectron2.utils.logger import setup_logger
from tqdm import trange

from yolov7.config import add_yolo_config

# constants
WINDOW_NAME = "COCO detections"


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
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            # predictions, pure_t = self.model([inputs])
            predictions = self.model([inputs])
            predictions = predictions[0]
            c = time.time() - tic
            print("cost: {}, fps: {}".format(c, 1 / c))
            return predictions


def setup_cfg(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.YOLO.CONF_THRESHOLD = args.confidence_threshold
    cfg.MODEL.YOLO.NMS_THRESHOLD = args.nms_threshold
    cfg.MODEL.YOLO.IGNORE_THRESHOLD = 0.1
    # force devices based on user device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.INPUT.MAX_SIZE_TEST = 600  # 90ms
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
    parser.add_argument(
        "-i",
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "-c",
        "--confidence-threshold",
        type=float,
        default=0.21,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "-n",
        "--nms-threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Name of Weights & Biases Project.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Name of Weights & Biases Entity.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def vis_res_fast(res, img, class_names, colors, thresh):
    ins = res["instances"]
    bboxes = None
    if ins.has("pred_boxes"):
        bboxes = ins.pred_boxes.tensor.cpu().numpy()
    scores = ins.scores.cpu().numpy()
    clss = ins.pred_classes.cpu().numpy().astype(int)
    if ins.has("pred_bit_masks"):
        bit_masks = ins.pred_bit_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        # img = vis_bitmasks_with_classes(img, clss, bit_masks)
        # img = vis_bitmasks_with_classes(img, clss, bit_masks, force_colors=colors, mask_border_color=(255, 255, 255), thickness=2)
        img = vis_bitmasks_with_classes(
            img, 
            clss, 
            bit_masks,
            class_names=class_names,
            force_colors=None, 
            draw_contours=True, 
            alpha=0.8
        )

    if ins.has("pred_masks"):
        bit_masks = ins.pred_masks
        if isinstance(bit_masks, BitMasks):
            bit_masks = bit_masks.tensor.cpu().numpy()
        img = vis_bitmasks_with_classes(
            img,
            clss,
            bit_masks,
            class_names=class_names,
            force_colors=None,
            draw_contours=True,
            alpha=0.6,
            thickness=2,
        )
    thickness = 1 if ins.has("pred_bit_masks") else 2
    font_scale = 0.3 if ins.has("pred_bit_masks") else 0.4
    if bboxes is not None:
        img = visualize_det_cv2_part(
            img,
            scores,
            clss,
            bboxes,
            class_names=class_names,
            force_color=colors,
            line_thickness=thickness,
            font_scale=font_scale,
            thresh=thresh,
        )
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = cfg.DATASETS.CLASS_NAMES
    predictor = DefaultPredictor(cfg)

    print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    colors = [
        [random.randint(0, 255) for _ in range(3)]
        for _ in range(cfg.MODEL.YOLO.CLASSES)
    ]
    conf_thresh = cfg.MODEL.YOLO.CONF_THRESHOLD
    print("confidence thresh: ", conf_thresh)

    iter = ImageSourceIter(args.input)
    if args.wandb_project is not None:
        from yolov7.utils.wandb.wandb_logger import WandbInferenceLogger

        inference_logger = WandbInferenceLogger(
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            conf_threshold=args.confidence_threshold,
            config=cfg,
        )
    else:
        inference_logger = None

    while iter.ok:
        im = next(iter)
        if isinstance(im, str):
            image_path = im
            im = cv2.imread(im)
            res = predictor(im)
            if inference_logger:
                inference_logger.log_inference(image_path, res)

            res = vis_res_fast(res, im, class_names, colors, conf_thresh)
        # cv2.imshow('frame', res)
        if args.output:
            if pathlib.Path(args.output).is_dir():
                out_path = pathlib.Path(args.output) / pathlib.Path(image_path).name
            else:
                out_path = args.output
        else:
            out_path = "frame"
        cv2.imshow(out_path, res)

        if iter.video_mode:
            cv2.waitKey(1)
        else:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                continue
    if inference_logger:
        inference_logger.finish_run()
