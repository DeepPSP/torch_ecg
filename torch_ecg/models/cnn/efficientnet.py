"""
EfficientNet,

References
----------
[1] Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
[2] Tan, M., & Le, Q. V. (2021). Efficientnetv2: Smaller models and faster training. arXiv preprint arXiv:2104.00298.
[3] https://github.com/google/automl
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

from ...cfg import CFG, DEFAULTS
from ...utils.utils_nn import compute_module_size, SizeMixin
from ...utils.misc import dict_to_str
from ...models._nets import (
    Conv_Bn_Activation,
    DownSample,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)


if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "EfficientNet",
]


class EfficientNet(SizeMixin, nn.Module):
    """

    Reference
    ---------
    1. Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
    2. https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    3. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    3. https://github.com/google/automl
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


class EfficientNetV2(SizeMixin, nn.Module):
    """

    Reference
    ---------
    1. Tan, M., & Le, Q. V. (2021). Efficientnetv2: Smaller models and faster training. arXiv preprint arXiv:2104.00298.
    2. https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    3. https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    4. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    5. https://github.com/google/automl
    """
    __DEBUG__ = True
    __name__ = "EfficientNetV2"

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
