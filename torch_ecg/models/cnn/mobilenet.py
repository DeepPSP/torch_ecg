"""
MobileNets, from V1 to V3

References
----------
[1] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
[2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
[3] Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1314-1324).
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
    "MobileNetV1",
    "MobileNetV2",
    "MobileNetV3",
]


class InvertedResidual(nn.Module):
    """
    """
    __DEBUG__ = True
    __name__ = "InvertedResidual"

    def __init__(self,) -> NoReturn:
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


class MobileNetV1(nn.Module):
    """
    """
    __DEBUG__ = True
    __name__ = "MobileNetV1"

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


class MobileNetV2(nn.Module):
    """

    References:
    -----------
    [1] https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    """
    __DEBUG__ = True
    __name__ = "MobileNetV2"

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


class MobileNetV3(nn.Module):
    """

    References:
    -----------
    [1] https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py
    """
    __DEBUG__ = True
    __name__ = "MobileNetV3"

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
