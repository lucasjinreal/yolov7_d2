"""
Test using onnxruntime to quantization a int8 CPU model
"""
from cv2 import calibrationMatrixValues
from onnxruntime.quantization import quantize_static, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType

import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import glob
import time
import sys
from torchvision import models
from torchvision import transforms
import torchvision
import torch


def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
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
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )
    return train_loader, test_loader


class CalibDataLoaderFromDataLoader(CalibrationDataReader):
    def __init__(self, test_loader) -> None:
        super().__init__()
        self.test_loader = iter(test_loader)

    def get_next(self) -> dict:
        res = next(self.test_loader, None)
        if res:
            images, labels = res
            if isinstance(images, torch.Tensor):
                images = images.cpu().numpy()
            return {"data": images}
        else:
            return None


def evaluate_onnx_model(model_p, test_loader, criterion=None):
    running_loss = 0
    running_corrects = 0

    session = ort.InferenceSession(model_p)
    input_name = session.get_inputs()[0].name

    total = 0.0
    for inputs, labels in test_loader:
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()

        start = time.perf_counter()
        outputs = session.run([], {input_name: inputs})
        end = (time.perf_counter() - start) * 1000
        total += end

        outputs = outputs[0]
        preds = np.argmax(outputs, 1)
        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0
        # statistics
        running_corrects += np.sum(preds == labels)

    # eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    total /= len(test_loader)
    print(f"eval loss: {0}, eval acc: {eval_accuracy}, cost: {total}")
    return 0, eval_accuracy


def run_time(model_p):
    session = ort.InferenceSession(model_p)
    input_name = session.get_inputs()[0].name
    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, 224, 224), np.float32)
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


if __name__ == "__main__":
    model_p = sys.argv[1]
    model_qp = os.path.join(
        os.path.dirname(model_p),
        os.path.basename(model_p).replace(".onnx", "_int8.onnx"),
    )

    train_loader, test_loader = prepare_dataloader(eval_batch_size=2)
    dr = CalibDataLoaderFromDataLoader(test_loader)
    quantize_static(
        model_p,
        model_qp,
        dr,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print("Calibrated and quantied.")

    run_time(model_p)
    evaluate_onnx_model(model_p, test_loader)
    run_time(model_qp)
    evaluate_onnx_model(model_qp, test_loader)
