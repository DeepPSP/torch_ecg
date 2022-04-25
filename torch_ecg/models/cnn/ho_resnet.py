"""
Higher Order ResNet

References
----------
[1] Luo, Z., Sun, Z., Zhou, W., & Kamata, S. I. (2021). Rethinking ResNets: Improved Stacking Strategies With High Order Schemes. arXiv preprint arXiv:2103.15244.

"""

from typing import NoReturn

import torch
from torch import nn

from ...cfg import DEFAULTS
from ...models._nets import (  # noqa: F401
    Activations,
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
    ZeroPadding,
)
from ...utils.utils_nn import SizeMixin

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "MidPointResNet",
    "RK4ResNet",
    "RK8ResNet",
]


class MidPointResNet(nn.Module, SizeMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        raise NotImplementedError


class RK4ResNet(nn.Module, SizeMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        raise NotImplementedError


class RK8ResNet(nn.Module, SizeMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        raise NotImplementedError
