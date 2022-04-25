"""
EfficientNet,

References
----------
[1] Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
[2] Tan, M., & Le, Q. V. (2021). Efficientnetv2: Smaller models and faster training. arXiv preprint arXiv:2104.00298.
[3] https://github.com/google/automl

"""

from typing import NoReturn

import torch
from torch import nn

from ...cfg import DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
)
from ...utils.utils_nn import SizeMixin

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "EfficientNet",
]


class EfficientNet(nn.Module, SizeMixin):
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

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
    ):
        """ """
        raise NotImplementedError

    def compute_output_shape(self):
        """ """
        raise NotImplementedError


class EfficientNetV2(nn.Module, SizeMixin):
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

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
    ):
        """ """
        raise NotImplementedError

    def compute_output_shape(self):
        """ """
        raise NotImplementedError
