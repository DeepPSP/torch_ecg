"""
3rd place (entry 0436) of CPSC2019
and variations
"""

import sys
from copy import deepcopy
from collections import OrderedDict
from itertools import repeat
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from ...cfg import Cfg
from ...utils.utils_nn import compute_deconv_output_shape
from ...utils.misc import dict_to_str
from ..nets import (
    Conv_Bn_Activation,
    DownSample, ZeroPadding,
    GlobalContextBlock,
)

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_YOLO",
]


class ResNetGCBlock(nn.Module):
    """ NOT finished, NOT checked,

    ResNet block with global context

    References:
    -----------
    [1] entry 0436 of CPSC2019
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    __DEBUG__ = True
    __name__ = "ResNetGCBlock"

    def __init__(self,) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self,):
        """
        """
        raise NotImplementedError

    def compute_output_shape(self,):
        """
        """
        raise NotImplementedError


class StageModule(nn.Module):
    """ NOT finished, NOT checked,
    """
    __DEBUG__ = True
    __name__ = "StageModule"

    def __init__(self,) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self,):
        """
        """
        raise NotImplementedError

    def compute_output_shape(self,):
        """
        """
        raise NotImplementedError



class ECG_YOLO(nn.Module):
    """ NOT finished, NOT checked,

    """
    __DEBUG__ = True
    __name__ = "ECG_YOLO"

    def __init__(self,) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self,):
        """
        """
        raise NotImplementedError

    def compute_output_shape(self,):
        """
        """
        raise NotImplementedError
