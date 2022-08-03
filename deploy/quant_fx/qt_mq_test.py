
from statistics import mode
import numpy as np
import argparse
from torchvision.models.resnet import resnet50, resnet18
import torch.nn as nn
import os
import time
from easydict import EasyDict
import yaml
import sys
from alfred.dl.torch.common import device
from alfred.utils.log import logger
from atomquant.atom.prepare_by_platform import prepare_by_platform, BackendType
from atomquant.atom.convert_deploy import convert_deploy
from torchvision import transforms
import torchvision
import torch

backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)


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


def evaluate_model(model, test_loader, criterion=None):
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


if __name__ == '__main__':
    train_loader, test_loader = prepare_dataloader()

    config_f = sys.argv[1]
    config = parse_config(config_f)
    print(config)
    # first finetune model on cifar, we don't have imagnet so using cifar as test
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    if os.path.exists("r18_raw.pth"):
        model.load_state_dict(torch.load("r18_raw.pth", map_location="cpu"))
    else:
        # train_model(model, train_loader, test_loader, device)
        print("train finished.")
        # torch.save(model.state_dict(), "r18_raw.pth")
    model.to(device)
    model.eval()

    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
        print('now model in quantized mode.')
    
    model.to(device)
    evaluate_model(model, test_loader)

    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate_model(model, test_loader)
    elif config.quantize.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(test_loader, cali_batchsize=config.quantize.cali_batchsize)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin advanced PTQ now!')
        if hasattr(config.quantize, 'reconstruction'):
            model = ptq_reconstruction(
                model, cali_data, config.quantize.reconstruction)
        enable_quantization(model)
        evaluate_model(model, test_loader)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    elif config.quantize.quantize_type == 'naive_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(test_loader, cali_batchsize=config.quantize.cali_batchsize)
        from atomquant.atom.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.to(device))
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].to(device))
        print('begin quantization now!')
        enable_quantization(model)
        # print(model)
        evaluate_model(model, test_loader)
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")