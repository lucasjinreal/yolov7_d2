import torch
import torch.nn as nn
import copy
from torchvision.models.resnet import resnet50
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, get_default_qconfig


def quant_fx():
    model = resnet50(pretrained=True)
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        '': qconfig,
        # 'object_type': []
    }
    model_to_quantize = copy.deepcopy(model)
    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
    print('prepared model: ', prepared_model)

    quantized_model = convert_fx(prepared_model)
    print('quantized model: ', quantized_model)

    torch.save(model.state_dict(), 'r18.pth')
    torch.save(quantized_model.state_dict(), 'r18_quant.pth')


def quant2():
    model = resnet50(pretrained=True)
    model.eval()

    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {
        '': qconfig,
        # 'object_type': []
    }
    
    model2 = copy.deepcopy(model)
    model_prepared = prepare_fx(model2, qconfig_dict)
    model_int8 = convert_fx(model_prepared)
    model_int8.load_state_dict(torch.load('r18_quant.pth'))

    a = torch.randn([1, 3, 224, 224])
    o1 = model(a)
    o2 = model_int8(a)

    diff = torch.allclose(o1, o2, 1e-4)
    print(diff)
    print(o1, o2)


if __name__ == '__main__':
    quant_fx()
    quant2()
