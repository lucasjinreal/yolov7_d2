from functools import partial

import torch.nn as nn

from detectron2.layers import (BatchNorm2d, NaiveSyncBatchNorm,
                               FrozenBatchNorm2d)
from detectron2.utils import env


norms = {
    "BN": BatchNorm2d,
    # Fixed in https://github.com/pytorch/pytorch/pull/36382
    "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (
        1, 5) else nn.SyncBatchNorm,
    "FrozenBN": FrozenBatchNorm2d,
    "GN": lambda channels: nn.GroupNorm(32, channels),
    # for debugging:
    "nnSyncBN": nn.SyncBatchNorm,
    "naiveSyncBN": NaiveSyncBatchNorm,
}


def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        kwargs: Additional parameters in normalization layers,
            such as, eps, momentum

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        assert norm in norms.keys(), 'normtype must be: {}'.format(norms.keys())
        norm = norms[norm]
    return norm(out_channels, **kwargs)


def get_activation(activation):
    """
    Only support `ReLU` and `LeakyReLU` now.

    Args:
        activation (str or callable):

    Returns:
        nn.Module: the activation layer
    """

    act = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
    }[activation]
    if activation == "LeakyReLU":
        act = partial(act, negative_slope=0.1)
    return act(inplace=True)
