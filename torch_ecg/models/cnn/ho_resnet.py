"""
Higher Order ResNet

References
----------
[1] Luo, Z., Sun, Z., Zhou, W., & Kamata, S. I. (2021). Rethinking ResNets: Improved Stacking Strategies With High Order Schemes. arXiv preprint arXiv:2103.15244.
"""
from copy import deepcopy
from itertools import repeat
from typing import Union, Optional, Sequence, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import Tensor

from ...cfg import CFG, DEFAULTS
from ...utils.utils_nn import compute_module_size, SizeMixin
from ...utils.misc import dict_to_str
from ...models._nets import (
    Activations,
    Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "MidPointResNet",
    "RK4ResNet",
    "RK8ResNet",
]


class MidPointResNet(SizeMixin, nn.Module):
    """
    """
    def __init__(self, in_channels:int, **config) -> NoReturn:
        """
        """
        raise NotImplementedError


class RK4ResNet(SizeMixin, nn.Module):
    """
    """
    def __init__(self, in_channels:int, **config) -> NoReturn:
        """
        """
        raise NotImplementedError


class RK8ResNet(SizeMixin, nn.Module):
    """
    """
    def __init__(self, in_channels:int, **config) -> NoReturn:
        """
        """
        raise NotImplementedError
