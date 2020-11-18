"""
3rd place (entry 0436) of CPSC2019
and variations
"""

import sys
from copy import deepcopy
from collections import OrderedDict
from itertools import repeat
from typing import Union, Optional, Sequence, List, NoReturn
from numbers import Real

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg
from torch_ecg.utils.utils_nn import compute_deconv_output_shape, compute_module_size
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.models.nets import (
    Conv_Bn_Activation, MultiConv,
    DownSample, ZeroPadding,
    GlobalContextBlock,
)

if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_YOLO",
]


class ResNetGCBlock(nn.Module):
    """ NOT finished, NOT checked,

    ResNet (basic, not bottleneck) block with global context

    References:
    -----------
    [1] entry 0436 of CPSC2019
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    __DEBUG__ = True
    __name__ = "ResNetGCBlock"

    def __init__(self, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, groups:int=1, dilation:int=1, dropouts:Union[float, Sequence[float]]=0, **config) -> NoReturn:
        """ finished, NOT checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        num_filters: int,
            number of filters for the convolutional layers
        filter_length: int,
            length (size) of the filter kernels
        subsample_lengths: int,
            subsample length,
            including pool size for short cut, and stride for the top convolutional layer
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        dilation: int, default 1,
            dilation of the convolutional layers
        dropouts: float, or sequence of float, default 0.0,
            dropout ratio after each convolution (and batch normalization, and activation, etc.)
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer,
            and short cut patterns, etc.
        """
        super().__init__()
        self.__num_convs = 2
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__kernel_size = filter_length
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.__groups = groups
        self.__dilation = dilation
        if isinstance(dropouts, float):
            self.__dropouts = list(repeat(dropouts, self.__num_convs))
        else:
            self.__dropouts = list(dropouts)
        assert len(self.__dropouts) == self.__num_convs
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.__increase_channels = (self.__out_channels > self.__in_channels)
        self.short_cut = self._make_short_cut_layer()

        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            conv_activation = (self.config.activation if i < self.__num_convs-1 else None)
            self.main_stream.add_module(
                f"cba_{i}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.__kernel_size,
                    stride=(self.__stride if i == 0 else 1),
                    dilation=self.__dilation,
                    groups=self.__groups,
                    batch_norm=True,
                    activation=conv_activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = self.__out_channels
            if i == 0 and self.__dropouts[i] > 0:
                self.main_stream.add_module(
                    f"dropout_{i}",
                    nn.Dropout(self.__dropouts[i])
                )
            if i == 1:
                self.main_stream.add_module(
                    f"gcb",
                    GlobalContextBlock(
                        in_channels=self.__out_channels,
                        ratio=self.config.gcb.ratio,
                        reduction=self.config.gcb.reduction,
                        pooling_type=self.config.gcb.pooling_type,
                        fusion_types=self.config.gcb.fusion_types,
                    )
                )

        if isinstance(self.config.activation, str):
            self.out_activation = \
                Activations[self.config.activation.lower()](**self.config.kw_activation)
        else:
            self.out_activation = \
                self.config.activation(**self.config.kw_activation)

        if self.__dropouts[1] > 0:
            self.out_dropout = nn.Dropout(self.__dropouts[1])
        else:
            self.out_dropout = None
    
    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """ finished, NOT checked,
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == 'conv':
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    groups=self.__groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            if self.config.increase_channels_method.lower() == 'zero_padding':
                batch_norm = False if self.config.subsample_mode.lower() != 'conv' else True
                short_cut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        batch_norm=batch_norm,
                        mode=self.config.subsample_mode,
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels),
                )
        else:
            short_cut = None
        return short_cut

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
        identity = input

        output = self.main_stream(input)

        if self.short_cut is not None:
            identity = self.short_cut(input)

        output += identity
        output = self.out_activation(output)

        if self.out_dropout:
            output = self.out_dropout(output)

        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self.main_stream:
            if type(module).__name__ == type(nn.Dropout).__name__:
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class StageModule(nn.Module):
    """ NOT finished, NOT checked,
    """
    __DEBUG__ = True
    __name__ = "StageModule"

    def __init__(self, stage:int, out_branches:int, in_channels:int, **config) -> NoReturn:
        """ NOT finished, NOT checked,
        """
        super().__init__()
        self.stage = stage
        self.out_branches = out_branches
        self.in_channels = in_channels
        self.config = ED(config)

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = in_channels * (2**i)
            branch = nn.Sequential(
                ResNetGCBlock(in_channels=w, num_filters=w, **(config.resnet_gc)),
                ResNetGCBlock(in_channels=w, num_filters=w, **(config.resnet_gc)),
                ResNetGCBlock(in_channels=w, num_filters=w, **(config.resnet_gc)),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.out_branches):
            fl = nn.ModuleList()
            for j in range(self.stage):
                if i == j :
                    fl.append(nn.Sequential())
                elif i < j :
                    if i == 0:
                        fl.append(nn.Sequential(
                            nn.Conv1d(in_channels * (2 ** j), in_channels * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm1d(in_channels * (2 ** i)),
                            nn.Upsample(size=625),
                        ))
                    elif i == 1:
                        fl.append(nn.Sequential(
                            nn.Conv1d(in_channels * (2 ** j), in_channels * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm1d(in_channels * (2 ** i)),
                            nn.Upsample(size=313)
                        ))
                    elif i == 2:
                        fl.append(nn.Sequential(
                            nn.Conv1d(in_channels * (2 ** j), in_channels * (2 ** i), kernel_size=1, stride=1),
                            nn.BatchNorm1d(in_channels * (2 ** i)),
                            nn.Upsample(size=157)
                        ))

                elif i > j:
                    opts = []
                    if i == j+1:
                        opts.append(Conv_Bn_Activation(
                            in_channels=in_channels * (2 ** j),
                            out_channels=in_channels * (2 ** i),
                            kernel_size=7,
                            stride=2,
                            batch_norm=True,
                            activation=None,
                        ))
                    elif i == j+2:
                        opts.append(MultiConv(
                            in_channels=in_channels * (2 ** j),
                            out_channels=[in_channels * (2 ** (j+1)), in_channels * (2 ** (j + 2))],
                            filter_lengths=7,
                            subsample_lengths=2,
                            out_activation=False,
                        ))
                    elif i == j+3:
                        opts.append(MultiConv(
                            in_channels=in_channels * (2 ** j),
                            out_channels=[in_channels * (2 ** (j+1)), in_channels * (2 ** (j + 2)), in_channels * (2 ** (j + 3))],
                            filter_lengths=7,
                            subsample_lengths=2,
                            out_activation=False,
                        ))
                    fl.append(nn.Sequential(*opts))
            self.fuse_layers.append(fl)
        self.fuse_activation = nn.ReLU(inplace=True)

    def forward(self, inputs:Sequence[Tensor]) -> List[Tensor]:
        """ NOT finished, NOT checked,
        """
        assert len(self.branches) == len(inputs)
        x = [branch(b) for branch, b in zip(self.branches, inputs)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


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
