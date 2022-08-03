"""
Using atomquant to quant SparseInst model
"""
from atomquant.onnx.ptq_cpu import quantize_static_onnx
from atomquant.onnx.dataloader import (
    get_calib_dataloader_from_dataset,
)
from torchvision import transforms
import cv2
import numpy as np
import sys
import os
import onnxruntime as ort
import torchvision
import time


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
        test_set, input_names=input_name, bs=1, max_step=100
    )
    quantize_static_onnx(model_p, calib_dataloader=calib_dataloader)

    evaluate_onnx_model(model_qp, calib_dataloader.dataloader_holder)
    evaluate_onnx_model(model_p, calib_dataloader.dataloader_holder)
