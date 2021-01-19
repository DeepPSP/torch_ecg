"""
EfficientNet,

References:
-----------
[1] Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
"""

from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Sequence, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import Tensor
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg
from torch_ecg.utils.utils_nn import compute_module_size
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.models.nets import (
    Conv_Bn_Activation,
    DownSample,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)


if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "EfficientNet",
]


class EfficientNet(nn.Module):
    """
    """
    __DEBUG__ = True
    __name__ = "EfficientNet"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self,):
        """
        """
        raise NotImplementedError

    def compute_output_shape(self):
        """
        """
        raise NotImplementedError
