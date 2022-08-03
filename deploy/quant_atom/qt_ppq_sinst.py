"""

Examples on how to quantize with PPQ

I dont suggest you using PPQ, it has a lot of bugs.

"""
from typing import Iterable

from loguru import logger
import torch
from torch.utils.data import DataLoader
from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq import graphwise_error_analyse, layerwise_error_analyse
from ppq.api import (
    export_ppq_graph,
    quantize_onnx_model
)
import sys
from torchvision import transforms
import torchvision
import torch
from atomquant.onnx.dataloader import get_calib_dataloader_coco
import os
import cv2
import numpy as np
import onnxruntime as ort
from torchvision.datasets.coco import CocoDetection
from alfred.dl.torch.common import device


def preprocess_func(img, target):
    w = 640
    h = 640
    a = cv2.resize(img, (w, h))
    a_t = np.array(a).astype(np.float32)
    boxes = []
    for t in target:
        boxes.append(t["bbox"])
    target = np.array(boxes)
    a_t = torch.as_tensor(a_t)
    target = torch.as_tensor(target)
    return a_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
        targets = torch.stack(targets)
    else:
        images = np.array(images)
    return images


if __name__ == "__main__":
    ONNX_PATH = sys.argv[1]

    coco_root = os.path.expanduser("~/data/coco/images/val2017")
    anno_f = os.path.expanduser(
        "~/data/coco/annotations/instances_val2017_val_val_train.json"
    )

    # coco_ds = CocoDetection(coco_root, anno_f, )

    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name

    calib_dataloader = get_calib_dataloader_coco(
        coco_root,
        anno_f,
        preprocess_func=preprocess_func,
        input_names=input_name,
        bs=1,
        max_step=50,
        collate_fn=collate_fn
    )

    REQUIRE_ANALYSE = False
    BATCHSIZE = 1
    # INPUT_SHAPE = [3, 224, 224]
    INPUT_SHAPE = [640, 640, 3]
    DEVICE = "cuda"  
    PLATFORM = (
        # TargetPlatform.ORT_OOS_INT8
        TargetPlatform.TRT_INT8
    ) 
    EXECUTING_DEVICE = "cpu"  # 'cuda' or 'cpu'.

    # create a setting for quantizing your network with PPL CUDA.
    # quant_setting = QuantizationSettingFactory.pplcuda_setting()
    quant_setting = QuantizationSettingFactory.default_setting()
    # quant_setting.equalization = True  # use layerwise equalization algorithm.
    quant_setting.equalization = False  # tensorrt false
    quant_setting.dispatcher = (
        "conservative"  # dispatch this network in conservertive way.
    )

    
    # quantize your model.
    quantized = quantize_onnx_model(
        onnx_import_file=ONNX_PATH,
        calib_dataloader=calib_dataloader.dataloader_holder,
        calib_steps=120,
        input_shape=[BATCHSIZE] + INPUT_SHAPE,
        setting=quant_setting,
        # collate_fn=collate_fn,
        platform=PLATFORM,
        device=DEVICE,
        verbose=0,
    )

    # Quantization Result is a PPQ BaseGraph instance.
    assert isinstance(quantized, BaseGraph)

    try:
        if REQUIRE_ANALYSE:
            print("正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:")
            reports = graphwise_error_analyse(
                graph=quantized,
                running_device=EXECUTING_DEVICE,
                steps=32,
                dataloader=calib_dataloader.dataloader_holder,
                collate_fn=lambda x: x.to(EXECUTING_DEVICE),
            )
            for op, snr in reports.items():
                if snr > 0.1:
                    logger.warning(f"层 {op} 的累计量化误差显著，请考虑进行优化")
            print("正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:")
            layerwise_error_analyse(
                graph=quantized,
                running_device=EXECUTING_DEVICE,
                interested_outputs=None,
                dataloader=calib_dataloader.dataloader_holder,
                collate_fn=lambda x: x.to(EXECUTING_DEVICE),
            )
    except Exception as e:
        logger.warning('analyse got some error, but that is OK, pass it.')


    # EXPORT_TARGET = TargetPlatform.ORT_OOS_INT8
    EXPORT_TARGET = TargetPlatform.TRT_INT8
    # EXPORT_TARGET = TargetPlatform.TRT_INT8
    os.makedirs('Output/', exist_ok=True)
    # export quantized graph.
    export_ppq_graph(
        graph=quantized,
        platform=EXPORT_TARGET,
        graph_save_to=f"Output/quantized_{EXPORT_TARGET}.onnx",
        config_save_to=f"Output/quantized_{EXPORT_TARGET}.json",
    )

    