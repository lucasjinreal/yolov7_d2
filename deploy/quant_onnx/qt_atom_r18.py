"""
Using atomquant to quant SparseInst model
"""
from atomquant.onnx.ptq_cpu import quantize_static_onnx
from atomquant.onnx.dataloader import (
    get_calib_dataloader_coco,
    get_calib_dataloader_from_dataset,
)
from torchvision import transforms
import cv2
import numpy as np
import sys
import os
import onnxruntime as ort
import torchvision


if __name__ == "__main__":
    model_p = sys.argv[1]
    model_qp = os.path.join(
        os.path.dirname(model_p),
        os.path.basename(model_p).replace(".onnx", "_int8.onnx"),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )

    session = ort.InferenceSession(model_p)
    input_name = session.get_inputs()[0].name

    calib_dataloader = get_calib_dataloader_from_dataset(
        test_set, input_names=input_name, bs=1, max_step=50
    )
    quantize_static_onnx(model_p, calib_dataloader=calib_dataloader)
