#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
from typing import Any, Dict, List, Optional, Union

import torch
from ..utils.qat_utils import iterate_module_named_parameters
from detectron2.solver.build import (
    maybe_add_gradient_clipping as d2_maybe_add_gradient_clipping,
    reduce_param_groups,
)
from detectron2.utils.registry import Registry


D2GO_OPTIM_MAPPER_REGISTRY = Registry("D2GO_OPTIM_MAPPER")

logger = logging.getLogger(__name__)


OptimizerModelsType = Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]


def get_optimizer_param_groups(model: OptimizerModelsType, cfg):
    """
    Get override optimizer parameter groups
       * Get all default parameters
       # Get parameter groups for normalization and bias
       # Get parameter groups from model if the model implements `get_optimizer_param_groups()`
    Parameters appear later will override parameters appear earlier
    """
    # get all parameters that requires gradient
    params = get_optimizer_param_groups_default(model)

    # parameter groups for lr
    params += get_optimizer_param_groups_lr(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        lr_multipliers_overwrite=_merge_dict(cfg.SOLVER.LR_MULTIPLIER_OVERWRITE),
    )

    # parameter groups for normalization, bias, and embedding
    params += get_optimizer_param_groups_weight_decay(
        model,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
    )

    # parameter groups from model function `model.get_optimizer_param_groups(opts)`
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if hasattr(model, "get_optimizer_param_groups"):
        logger.info(
            "Getting optimizer parameter groups from model.get_optimizer_param_groups()"
        )
        params += model.get_optimizer_param_groups(cfg)

    return reduce_param_groups(params)


def get_optimizer_param_groups_default(model: OptimizerModelsType):
    ret = [
        {
            "params": list(
                filter(
                    lambda x: x.requires_grad,
                    model.parameters(),
                )
            )
        }
    ]
    return ret


def get_optimizer_param_groups_lr(
    model: OptimizerModelsType,
    base_lr: float,
    bias_lr_factor: float = 1.0,
    lr_multipliers_overwrite: Optional[Dict[str, float]] = None,
):
    """
    Allow setting up lr for modules
    base_lr: lr for all modules
    bias_lr_factor: scale factor for lr for bias term
    lr_multipliers_overwrite (dict: str-> float):
        Applying different lr multiplier to a set of parameters whose names
        containing certain keys. For example, if lr_multipliers_overwrite={'backbone': 0.1},
        the LR for the parameters whose names containing 'backbone' will be scaled to 0.1x.
        Set lr_multipliers_overwrite=None if no multipliers required.
    """
    params: List[Dict[str, Any]] = []
    for (
        module_name,
        _module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        cur_lr = base_lr
        if module_param_name == "bias":
            cur_lr = base_lr * bias_lr_factor
        if lr_multipliers_overwrite is not None:
            for kname, mult in lr_multipliers_overwrite.items():
                if kname in module_name:
                    # apply multiplier for the params containing kname, e.g. backbone
                    cur_lr = cur_lr * mult

        params += [
            {
                "params": [value],
                "lr": cur_lr,
            }
        ]

    return params


def get_optimizer_param_groups_weight_decay(
    model: OptimizerModelsType,
    weight_decay: Optional[float],
    weight_decay_norm: Optional[float] = None,
    weight_decay_bias: Optional[float] = None,
    weight_decay_embed: Optional[float] = None,
):
    """
    Allow setting up weight decay for normalization, embedding and bias
    """
    if weight_decay_norm is None:
        weight_decay_norm = weight_decay
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    if weight_decay_embed is None:
        weight_decay_embed = weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    for (
        _module_name,
        module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        cur_wd = weight_decay
        if isinstance(module, norm_module_types):
            cur_wd = weight_decay_norm
        elif isinstance(module, torch.nn.Embedding):
            cur_wd = weight_decay_embed
        elif module_param_name == "bias":
            cur_wd = weight_decay_bias
        if cur_wd is not None:
            params += [
                {
                    "params": [value],
                    "weight_decay": cur_wd,
                }
            ]

    return params


def get_optimizer_param_groups_override(
    model: OptimizerModelsType,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Allow setting up overrides for parameter groups
    overrides (dict: str -> (dict: str -> float)):
        if not `None`, provides values for optimizer hyperparameters
        (LR, weight decay) for module parameters with a given name; e.g.
        {"embedding": {"lr": 0.01, "weight_decay": 0.1}} will set the LR and
        weight decay values for all module parameters named `embedding` (default: None)
    """

    params: List[Dict[str, Any]] = []

    if overrides is None:
        return params

    for (
        _module_name,
        _module,
        module_param_name,
        value,
    ) in iterate_module_named_parameters(model):
        schedule_params = {}
        if module_param_name in overrides:
            schedule_params.update(overrides[module_param_name])
            params += [{"params": [value], **schedule_params}]

    return params


def maybe_add_gradient_clipping(cfg, optim):  # optim: the optimizer class
    # detectron2 doesn't have full model gradient clipping now
    clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
    enable = (
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED
        and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
        and clip_norm_val > 0.0
    )

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    if enable:
        return FullModelGradientClippingOptimizer
    return d2_maybe_add_gradient_clipping(cfg, optim)


def _merge_dict(in_dict):
    ret_dict = {}
    assert all(isinstance(x, dict) for x in in_dict)
    for dic in in_dict:
        ret_dict.update(dic)
    return ret_dict


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def sgd(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_optimizer_param_groups(model, cfg)
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_optimizer_param_groups(model, cfg)
    return maybe_add_gradient_clipping(cfg, torch.optim.AdamW)(
        params, cfg.SOLVER.BASE_LR
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def sgd_mt(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build a multi_tensor SGD optimizer that works significantly faster.
    This version is expected to be the default implementation for SGD
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    params = get_optimizer_param_groups(model, cfg)
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.SGD)(
        params,
        cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
        nesterov=cfg.SOLVER.NESTEROV,
    )


@D2GO_OPTIM_MAPPER_REGISTRY.register()
def adamw_mt(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build a multi_tensor adamw optimizer that works significantly faster.
    This version is expected to be the default implementation for adamw
    optimizer by end of H1'21. To benefit from the speedup, the number
    of parameter groups needs to be reduced using `reduce_param_groups`.
    """
    params = get_optimizer_param_groups(model, cfg)
    return maybe_add_gradient_clipping(cfg, torch.optim._multi_tensor.AdamW)(
        params, cfg.SOLVER.BASE_LR
    )


def build_optimizer_mapper(cfg, model):
    name = cfg.SOLVER.OPTIMIZER
    optimizer = D2GO_OPTIM_MAPPER_REGISTRY.get(name.lower())(cfg, model)

    def _param_group_str(group):
        ret = {x: y if x != "params" else len(y) for x, y in group.items()}
        ret = sorted(ret.items())
        ret = [f"{x[0]}: {x[1]}" for x in ret]
        ret = "{" + ", ".join(ret) + "}"
        return ret

    def _param_groups_str(groups):
        ret = ""
        for idx, group in enumerate(groups):
            ret += f"Param group {idx}: {_param_group_str(group)}\n"
        return ret

    logger.info(
        f"optimizer parameter groups:\n{_param_groups_str(optimizer.param_groups)}"
    )

    return optimizer
