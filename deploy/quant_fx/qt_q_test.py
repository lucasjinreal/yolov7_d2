import torch
from torch import quantization
from torchvision import models

qat_resnet18 = models.resnet18(pretrained=True).cuda()

qat_resnet18.qconfig = quantization.QConfig(
    activation=quantization.default_fake_quant,
    weight=quantization.default_per_channel_weight_fake_quant,
)
quantization.prepare_qat(qat_resnet18, inplace=True)
qat_resnet18.apply(quantization.enable_observer)
qat_resnet18.apply(quantization.enable_fake_quant)

dummy_input = torch.randn(16, 3, 224, 224).cuda()
_ = qat_resnet18(dummy_input)
for module in qat_resnet18.modules():
    if isinstance(module, quantization.FakeQuantize):
        module.calculate_qparams()
qat_resnet18.apply(quantization.disable_observer)

qat_resnet18.cuda()

input_names = ["actual_input_1"]
output_names = ["output1"]


torch.onnx.export(
    qat_resnet18, dummy_input, "quant_model.onnx", verbose=True, opset_version=13
)
