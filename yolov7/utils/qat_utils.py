#!/usr/bin/env python3
import logging
from functools import partial

import torch
import torch.distributed as dist
try:
    from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
except ImportError:
    print('QAT disabled.')


logger = logging.getLogger(__name__)


def mixin_with_subclass(module, mix_class):
    """Create a subclass of type(module) and mix_class while using all the data
    from the `module` object
    """
    ModuleType = type(module)

    class SubClass(mix_class, ModuleType):
        def __init__(self, module):
            assert isinstance(module, ModuleType)
            # initialize the parent by copying the dict directly
            self.__dict__ = module.__dict__.copy()

    ret = SubClass(module)
    return ret


def _has_module(model, module_type):
    for x in model.modules():
        if isinstance(x, module_type):
            return True
    return False


def check_for_learnable_fake_quant_ops(qat_method, model):
    """Make sure learnable observers are used if qat method is `learnable`"""
    if qat_method == "learnable":
        if not _has_module(model, _LearnableFakeQuantize):
            raise Exception(
                "No learnable fake quant is used for learnable quantzation, please use d2go.utils.qat_utils.get_qat_qconfig() to get proper qconfig"
            )


def iterate_module_named_parameters(model, check_requires_grad=True):
    """Iterate over all parameters for the model"""
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if check_requires_grad and not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            yield module_name, module, module_param_name, value


def get_qat_qconfig(backend, qat_method="default"):
    assert backend in ["qnnpack", "fbgemm"]
    assert qat_method in ["default", "learnable"]
    if qat_method == "default":
        return torch.quantization.get_default_qat_qconfig(backend)

    ACT_CONFIGS = {
        # follow `get_default_qat_qconfig()`
        # fbcode/caffe2/torch/quantization/qconfig.py
        "fbgemm": {
            "reduce_range": True,
        },
        "qnnpack": {
            "reduce_range": False,
        },
    }

    WEIGHT_CONFIGS = {
        # follow `default_per_channel_weight_fake_quant`
        # fbcode/caffe2/torch/quantization/fake_quantize.py
        "fbgemm": {
            "observer": torch.quantization.MovingAveragePerChannelMinMaxObserver,
            "qscheme": torch.per_channel_symmetric,
            "reduce_range": False,
            "ch_axis": 0,
        },
        # follow `default_weight_fake_quant`
        # fbcode/caffe2/torch/quantization/fake_quantize.py
        "qnnpack": {
            "observer": torch.quantization.MovingAverageMinMaxObserver,
            "qscheme": torch.per_tensor_symmetric,
            "reduce_range": False,
        },
    }

    act = _LearnableFakeQuantize.with_args(
        observer=torch.quantization.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        use_grad_scaling=True,
        **ACT_CONFIGS[backend],
    )
    weight = _LearnableFakeQuantize.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        use_grad_scaling=True,
        **WEIGHT_CONFIGS[backend],
    )
    return torch.quantization.QConfig(activation=act, weight=weight)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def sync_tensor(data):
    world_size = get_world_size()
    if world_size > 1:
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        data /= world_size


def toggle_lqat_fake_quant(mod, enable):
    """Toggle fake quantization for learnable qat"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_fake_quant(enable)


# enable/disable fake quantization for learnable qat
enable_lqat_fake_quant = partial(toggle_lqat_fake_quant, enable=True)
disable_lqat_fake_quant = partial(toggle_lqat_fake_quant, enable=False)


def toggle_lqat_static_observer(mod, enable):
    """Toggle static observers for learnable qat"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_observer_update(enable)


# enable/disable static observer for learnable qat
enable_lqat_static_observer = partial(toggle_lqat_static_observer, enable=True)
disable_lqat_static_observer = partial(
    toggle_lqat_static_observer, enable=False)


def enable_lqat_learnable_observer(mod):
    """Enable learning observers, will disable static observer updates"""
    if type(mod) == _LearnableFakeQuantize:
        sync_tensor(mod.scale.data)
        sync_tensor(mod.zero_point.data)
        mod.toggle_qparam_learning(
            enabled=True).toggle_observer_update(enabled=False)


def disable_lqat_learnable_observer(mod):
    """Disable learning observers"""
    if type(mod) == _LearnableFakeQuantize:
        mod.toggle_qparam_learning(enabled=False)


def get_optimizer_param_groups_learnable_qat(model, _):
    """Set the weight decay for scale/zero_point for learnable_fake_quant to 0"""
    params = []
    for (
        _module_name,
        module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model, check_requires_grad=False):
        if isinstance(module, _LearnableFakeQuantize):
            if module_param_name in ("scale", "zero_point"):
                params += [
                    {
                        "params": [value],
                        "weight_decay": 0.0,
                    }
                ]

    return params


def _is_observer_key(state_dict_key):
    observer_keys = ["activation_post_process", "weight_fake_quant"]
    return any(x in state_dict_key for x in observer_keys)


def _is_q_state_dict(state_dict):
    return any(_is_observer_key(k) for k in state_dict)


class ModelGetOptimizerParamGroupLearnableQATMixin:
    def get_optimizer_param_groups(self, opts):
        ret = []
        if hasattr(super(), "get_optimizer_param_groups"):
            ret = super().get_optimizer_param_groups(opts)
        ret += get_optimizer_param_groups_learnable_qat(self, opts)
        return ret


def setup_qat_get_optimizer_param_groups(model, qat_method):
    """Add a function `get_optimizer_param_groups` to the model so that it could
    return proper weight decay for learnable qat
    """
    if qat_method != "learnable":
        return model

    assert _is_q_state_dict(model.state_dict())

    model = mixin_with_subclass(
        model, ModelGetOptimizerParamGroupLearnableQATMixin)
    assert hasattr(model, "get_optimizer_param_groups")
    return model
