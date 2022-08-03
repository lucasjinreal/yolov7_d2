"""
Using atomquant to quant SparseInst model
"""
from atomquant.onnx.ptq_cpu import quantize_static_onnx
from atomquant.onnx.dataloader import get_calib_dataloader_coco
from torchvision import transforms
import cv2
import numpy as np
import sys
import os
import onnxruntime as ort


def preprocess_func(img, target):
    w = 640
    h = 640
    a = cv2.resize(img, (w, h))
    a_t = np.array(a).astype(np.float32)
    boxes = []
    for t in target:
        boxes.append(t["bbox"])
    target = np.array(boxes)
    return a_t, target


def pqt(onnx_f):
    coco_root = os.path.expanduser("~/data/coco/images/val2017")
    anno_f = os.path.expanduser("~/data/coco/annotations/instances_val2017_val_val_train.json")

    session = ort.InferenceSession(onnx_f)
    input_name = session.get_inputs()[0].name

    calib_dataloader = get_calib_dataloader_coco(
        coco_root, anno_f, preprocess_func=preprocess_func, input_names=input_name, bs=1, max_step=50
    )
    quantize_static_onnx(onnx_f, calib_dataloader=calib_dataloader)


if __name__ == "__main__":
    onnx_f = sys.argv[1]
    pqt(onnx_f)
