import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms
from torchvision.models.resnet import resnet50, resnet18
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import (
    get_default_qconfig,
)
from torch import optim
from torch.onnx import OperatorExportTypes
import os
import time


def train_model(model, train_loader, test_loader, device):
    # The training configurations were not carefully selected.
    learning_rate = 1e-2
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(num_epochs):
        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(
            model=model, test_loader=test_loader, device=device, criterion=criterion
        )
        print(
            "Epoch: {:02d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
                epoch, train_loss, train_accuracy, eval_loss, eval_accuracy
            )
        )
    return model


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


def get_output_from_logits(logits):
    probs = F.softmax(logits)
    label, prob = torch.max(probs, dim=-1)
    print(label, prob)
    return


def calib_quant_model(model, calib_dataloader):
    assert isinstance(
        model, ObservedGraphModule
    ), "model must be a perpared fx ObservedGraphModule."
    model.eval()
    with torch.inference_mode():
        for inputs, labels in calib_dataloader:
            model(inputs)
    print("calib done.")


def quant_fx(model):
    model.eval()
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
        # 'object_type': []
    }
    model_to_quantize = copy.deepcopy(model)
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    print("prepared model: ", prepared_model)

    quantized_model = convert_fx(prepared_model)
    print("quantized model: ", quantized_model)
    torch.save(model.state_dict(), "r18.pth")
    torch.save(quantized_model.state_dict(), "r18_quant.pth")


def quant_calib_and_eval(model):
    # test only on CPU
    model.to(torch.device("cpu"))
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
        # 'object_type': []
    }

    model2 = copy.deepcopy(model)
    model_prepared = prepare_fx(model2, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    model_int8.load_state_dict(torch.load("r18_quant.pth"))
    model_int8.eval()

    a = torch.randn([1, 3, 224, 224])
    o1 = model(a)
    o2 = model_int8(a)

    diff = torch.allclose(o1, o2, 1e-4)
    print(diff)
    print(o1.shape, o2.shape)
    print(o1, o2)
    get_output_from_logits(o1)
    get_output_from_logits(o2)

    train_loader, test_loader = prepare_dataloader()
    evaluate_model(model, test_loader)
    evaluate_model(model_int8, test_loader)

    # calib quant model
    model2 = copy.deepcopy(model)
    model_prepared = prepare_fx(model2, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    torch.save(model_int8.state_dict(), "r18.pth")
    model_int8.eval()

    model_prepared = prepare_fx(model2, qconfig_dict)
    calib_quant_model(model_prepared, test_loader)
    model_int8 = convert_fx(model_prepared)
    torch.save(model_int8.state_dict(), "r18_quant_calib.pth")
    evaluate_model(model_int8, test_loader)


def export_quant_onnx(model):
    model.to(torch.device("cpu"))
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
        # 'object_type': []
    }

    model2 = copy.deepcopy(model)
    model_prepared = prepare_fx(model2, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    model_int8.load_state_dict(torch.load("r18_quant_calib.pth"))
    model_int8.eval()

    a = torch.randn([1, 3, 224, 224])
    torch.onnx.export(model_int8, a, "r18_int8.onnx")
    print("int8 onnx saved.")


def export_quant_torchscript(model):
    model.to(torch.device("cpu"))
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        "": qconfig,
        # 'object_type': []
    }
    print(qconfig_dict)
    with torch.no_grad():
        model2 = copy.deepcopy(model)
        model_prepared = prepare_fx(model2, qconfig_dict)
        model_int8 = convert_fx(model_prepared)
        model_int8.load_state_dict(torch.load("r18_quant_calib.pth"))
        model_int8.eval()

        a = torch.randn([1, 3, 224, 224])
        # torch.jit.save.export(model_int8, a, "r18_int8.onnx")
        sm = torch.jit.trace(model_int8, a)
        sm.save("r18_int8.torchscript")

        dm = torch.jit.load("r18_int8.torchscript")

        print(model_int8)
        torch.onnx.export(
            model_int8,
            a,
            "r18_int8.onnx",
            opset_version=13,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        print("int8 onnx saved.")
        evaluate_model(dm, test_loader)


if __name__ == "__main__":
    train_loader, test_loader = prepare_dataloader()

    # first finetune model on cifar, we don't have imagnet so using cifar as test
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)
    if os.path.exists("r18_raw.pth"):
        model.load_state_dict(torch.load("r18_raw.pth", map_location="cpu"))
    else:
        from alfred.dl.torch.common import device
        train_model(model, train_loader, test_loader, device)
        print("train finished.")
        torch.save(model.state_dict(), "r18_raw.pth")

    with torch.no_grad():
        quant_fx(model)
        quant_calib_and_eval(model)
        export_quant_torchscript(model)
