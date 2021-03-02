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

from ...cfg import Cfg
from ...utils.utils_nn import compute_module_size
from ...utils.misc import dict_to_str
from ...models.nets import (
    Conv_Bn_Activation, MultiConv,
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

    def __init__(self, in_channels:int, out_channels:int, expanded_channels:int, filter_length:int, stride:int, groups:int, activation:str, use_se:bool, width_mult:float) -> NoReturn:
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


class MobileNetV1(nn.Sequential):
    """

    Similar to Xception, but without skip connections

    normal conv
    --> entry flow (separable convs, down sample and double channels every other conv)
    --> middle flow (separable convs, no down sampling, stationary number of channels)
    --> exit flow (separable convs, down sample and double channels at each conv)
    """
    __DEBUG__ = True
    __name__ = "MobileNetV1"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, NOT checked,

        Parameters:
        -----------
        in_channels: int,
        config: dict,
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))

        self.add_module(
            "init_conv",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_length,
                stride=self.config.init_subsample_length,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
            )
        )

        _, entry_flow_in_channels, _ = self.init_conv.compute_output_shape()
        entry_flow = MultiConv(
            in_channels=entry_flow_in_channels,
            **(self.config.entry_flow)
        )
        self.add_module(
            "entry_flow",
            entry_flow
        )

        _, middle_flow_in_channels, _ = entry_flow.compute_output_shape()
        middle_flow = MultiConv(
            in_channels=middle_flow_in_channels,
            **(self.config.middle_flow)
        )
        self.add_module(
            "middle_flow",
            middle_flow
        )

        _, exit_flow_in_channels, _ = middle_flow.compute_output_shape()
        exit_flow = MultiConv(
            in_channels=exit_flow_in_channels,
            **(self.config.exit_flow)
        )
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, NOT checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, NOT checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        return compute_module_size(self)


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
