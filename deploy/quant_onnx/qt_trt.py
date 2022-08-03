from multiprocessing.spawn import prepare
import torch
import torch.utils.data
from torch import nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from tqdm import tqdm
import torchvision
from torchvision import models
from pytorch_quantization import quant_modules
from torchvision import transforms
import os
import time

quant_modules.initialize()


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


def evaluate_model(model, test_loader, device=torch.device("cpu"), criterion=None):
    t0 = time.time()
    model.eval()
    model.to(device)
    running_loss = 0
    running_corrects = 0
    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)
    t1 = time.time()
    print(f"eval loss: {eval_loss}, eval acc: {eval_accuracy}, cost: {t1 - t0}")
    return eval_loss, eval_accuracy


def calib_and_quant(model, data_loader, test_loader):
    def collect_stats(model, data_loader, num_batches):
        """Feed data to the network and collect statistic"""

        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
            model(image.cuda())
            if i >= num_batches:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    def compute_amax(model, **kwargs):
        # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                print(f"{name:40}: {module}")
        model.cuda()

    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    model.cuda()
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=2)
        compute_amax(model, method="percentile", percentile=99.99)

    evaluate_model(model, test_loader)

    # Save the model
    torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")
    return model


def quant():
    train_loader, test_loader = prepare_dataloader()
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(512, 10)
    if os.path.exists("r18_raw.pth"):
        model.load_state_dict(torch.load("r18_raw.pth", map_location="cpu"))
    else:
        # train_model(model, train_loader, test_loader, torch.device("cuda"))
        print("train finished.")
        torch.save(model.state_dict(), "r18_row.pth")

    with torch.no_grad():
        model_quant = calib_and_quant(model, train_loader, test_loader)
        model_quant.cuda()
        dummy_input = torch.randn(128, 3, 224, 224, device="cuda")

        input_names = ["actual_input_1"]
        output_names = ["output1"]

        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        print(model_quant)
        # enable_onnx_checker needs to be disabled. See notes below.
        torch.onnx.export(
            model_quant,
            dummy_input,
            "quant_r18.onnx",
            verbose=False,
            opset_version=13,
        )


if __name__ == "__main__":
    quant()
