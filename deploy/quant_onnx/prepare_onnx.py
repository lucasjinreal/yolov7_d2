from torchvision.models.resnet import resnet18
from torch import nn
import os
import torch


model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)
if os.path.exists("r18_raw.pth"):
    model.load_state_dict(torch.load("r18_raw.pth", map_location="cpu"))
else:
    pass

model.eval()

a = torch.randn([1, 3, 224, 224])
torch.onnx.export(
    model,
    a,
    "r18.onnx",
    input_names=["data"],
    dynamic_axes={"data": {0: "batch", 2: "h", 3: "w"}},
    opset_version=13
)
