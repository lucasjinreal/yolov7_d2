import argparse
from cProfile import label
import os
import cv2
import numpy as np
import onnxruntime
from alfred.vis.image.det import visualize_det_cv2_part
from alfred.vis.image.mask import vis_bitmasks_with_classes
from alfred.utils.file_io import ImageSourceIter

"""

this script using for testing onnx model exported from YOLOv7
inference via onnxruntime

"""


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        # xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    image = padded_img

    cv2.imshow("aad", image.astype(np.uint8))
    # cv2.waitKey()

    image = image.astype(np.float32)
    image = image[:, :, ::-1]
    # image /= 255.0
    # if mean is not None:
    #     image -= mean
    # if std is not None:
    #     image /= std
    # image = image.transpose(swap)
    image = np.ascontiguousarray(image, dtype=np.float32)
    return image, r


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
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    parser.add_argument(
        "-int8",
        '--int8',
        action="store_true",
        help="Whether your model uses int8.",
    )
    return parser


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


if __name__ == "__main__":
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(",")))
    session = onnxruntime.InferenceSession(args.model)

    iter = ImageSourceIter(args.image_path)
    while True:
        im = next(iter)
        if isinstance(im, str):
            im = cv2.imread(im)

        inp, ori_img = preprocess_image(im, h=input_shape[0], w=input_shape[1])

        ort_inputs = {session.get_inputs()[0].name: inp}
        output = session.run(None, ort_inputs)

        if "sparse" in args.model:
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
            keep = scores > (0.13 if args.int8 else 0.32)
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
