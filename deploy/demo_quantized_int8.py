from torch import Tensor
from wanwu.core.backends.trt import TensorRTInferencer
import os
import cv2
import argparse
import numpy as np
import onnxruntime
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.mask import vis_bitmasks_with_classes
from alfred.utils.file_io import ImageSourceIter


def vis_res_fast(img, boxes, masks, scores, labels):
    if masks is not None:
        # masks shape, might not same as img, resize contours if so
        img = vis_bitmasks_with_classes(
            img,
            labels,
            masks,
            force_colors=None,
            draw_contours=True,
            mask_border_color=[255, 255, 255],
        )
    thickness = 1 if masks is None else 2
    font_scale = 0.3 if masks is None else 0.4
    if boxes:
        img = visualize_det_cv2_part(
            img,
            scores,
            labels,
            boxes,
            line_thickness=thickness,
            font_scale=font_scale,
        )
    return img


def load_test_image(f, h, w):
    a = cv2.imread(f)
    a = cv2.resize(a, (w, h))
    a_t = np.expand_dims(np.array(a).astype(np.float32), axis=0)
    return a_t, a


def preprocess_image(img, h, w):
    a = cv2.resize(img, (w, h))
    a_t = np.expand_dims(np.array(a).astype(np.float32), axis=0)
    return a_t, img


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default="test_image.png",
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="demo_output",
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-t",
        "--type",
        default='sparseinst',
        help="model type.",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    
    engine_f = args.model
    trt_model = TensorRTInferencer(engine_f)
    input_shape = trt_model.ori_input_shape
    print('input shape: ', input_shape)

    iter = ImageSourceIter(args.image_path)
    while True:
        im = next(iter)
        if isinstance(im, str):
            im = cv2.imread(im)

        inp, ori_img = preprocess_image(im, h=input_shape[0], w=input_shape[1])
        output = trt_model.infer(inp)

        print(output)

        if "sparse" in args.type:
            masks, scores, labels = None, None, None
            for o in output:
                if o.dtype == np.float32:
                    scores = o
                if o.dtype == np.int32 or o.dtype == np.int64:
                    labels = o
                if o.dtype == bool:
                    masks = o
            masks = masks[0]
            print(masks.shape)
            if len(masks.shape) > 3:
                masks = np.squeeze(masks, axis=1)
            scores = scores[0]
            labels = labels[0]
            # keep = scores > 0.15
            keep = scores > 0.06
            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]
            print(scores)
            print(labels)
            print(masks.shape)
            img = vis_res_fast(im, None, masks, scores, labels)
        else:
            predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
            # boxes_xyxy /= ratio
            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.1)
            final_boxes, final_scores, final_cls_inds = (
                dets[:, :4],
                dets[:, 4],
                dets[:, 5],
            )
            img = visualize_det_cv2_part(
                ori_img, final_scores, final_cls_inds, final_boxes
            )
            cv2.imshow("aa", img)
            cv2.waitKey(0)

        cv2.imshow("YOLOv7 SparseInst CPU int8", img)
        if iter.video_mode:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv2.waitKey(0)
