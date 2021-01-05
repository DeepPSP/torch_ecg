"""
The most frequently used (can serve as baseline) CNN family of physiological signal processing,
whose performance however seems exceeded by newer networks
"""
from copy import deepcopy
from itertools import repeat
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
    Activations,
    Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)


if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ResNet",
    "ResNetBasicBlock",
    "ResNetBottleNeck",
]


class ResNetBasicBlock(nn.Module):
    """ finished, checked,

    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`
    """
    __DEBUG__ = False
    __name__ = "ResNetBasicBlock"
    expansion = 1  # not used

    def __init__(self,
                 in_channels:int,
                 num_filters:int,
                 filter_length:int,
                 subsample_length:int,
                 groups:int=1,
                 dilation:int=1,
                 **config) -> NoReturn:
        """ finished, checked,

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
            not used
        config: dict,
            other hyper-parameters, including
            increase channel method, subsample method,
            activation choices, weight initializer, and short cut patterns, etc.
        """
        super().__init__()
        if dilation > 1:
            raise NotImplementedError(f"Dilation > 1 not supported in {self.__name__}")
        self.__num_convs = 2
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__kernel_size = filter_length
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.__groups = groups
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        if self.config.increase_channels_method.lower() == "zero_padding" and self.__groups != 1:
            raise ValueError("zero padding for increasing channels can not be used with groups != 1")
        
        self.__increase_channels = (self.__out_channels > self.__in_channels)
        self.shortcut = self._make_shortcut_layer()

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

        if isinstance(self.config.activation, str):
            self.out_activation = \
                Activations[self.config.activation.lower()](**self.config.kw_activation)
        else:
            self.out_activation = \
                self.config.activation(**self.config.kw_activation)
    
    def _make_shortcut_layer(self) -> Union[nn.Module, type(None)]:
        """ finished, checked,
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                shortcut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    groups=self.__groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = False if self.config.subsample_mode.lower() != "conv" else True
                shortcut = nn.Sequential(
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
            shortcut = None
        return shortcut

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        out: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        identity = input

        out = self.main_stream(input)

        if self.shortcut is not None:
            identity = self.shortcut(input)

        out += identity
        out = self.out_activation(out)

        return out

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

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
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class ResNetBottleNeck(nn.Module):
    """ finished, NOT checked,

    bottle neck blocks for `ResNet`, as implemented in ref. [2] of `ResNet`,
    as for 1D ECG, should be of the "baby-giant-baby" pattern?
    """
    __DEBUG__ = False
    __name__ = "ResNetBottleNeck"
    expansion = 4
    __DEFAULT_BASE_WIDTH__ = 12 * 4

    def __init__(self,
                 in_channels:int,
                 num_filters:int,
                 filter_length:int,
                 subsample_length:int,
                 groups:int=1,
                 dilation:int=1,
                 base_width:int=12*4,
                 base_groups:int=1,
                 base_filter_length:int=1,
                 **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        num_filters: sequence of int,
            number of filters for the neck conv layer
        filter_length: int,
            lengths (sizes) of the filter kernels for the neck conv layer
        subsample_length: int,
            subsample length,
            including pool size for short cut,
            and stride for the (top or neck) conv layer
        groups: int, default 1,
            pattern of connections between inputs and outputs of the neck conv layer,
            for more details, ref. `nn.Conv1d`
        dilation: int, default 1,
            dilation of the conv layers
        base_width: int, default 12*4,
            number of filters per group for the neck conv layer
            usually number of filters of the initial conv layer of the whole ResNet
        base_groups: int, default 1,
            pattern of connections between inputs and outputs of conv layers at the two ends,
            should divide `groups`
        base_filter_length: int, default 1,
            lengths (sizes) of the filter kernels for conv layers at the two ends
        config: dict,
            other hyper-parameters, including
            increase channel method, subsample method,
            activation choices, weight initializer, and short cut patterns, etc.
        """
        super().__init__()
        self.__num_convs = 3
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        self.expansion = self.config.get("expansion", self.expansion)

        self.__in_channels = in_channels
        # update denominator of computing neck_num_filters by init_num_filters
        self.__DEFAULT_BASE_WIDTH__ = \
            self.config.get("init_num_filters", self.__DEFAULT_BASE_WIDTH__)
        neck_num_filters = \
            int(num_filters * (base_width / self.__DEFAULT_BASE_WIDTH__)) * groups
        self.__out_channels = [
            neck_num_filters,
            neck_num_filters,
            num_filters * self.expansion,
        ]
        if self.__DEBUG__:
            print(f"__DEFAULT_BASE_WIDTH__ = {self.__DEFAULT_BASE_WIDTH__}, in_channels = {in_channels}, num_filters = {num_filters}, base_width = {base_width}, neck_num_filters = {neck_num_filters}")
        self.__base_filter_length = base_filter_length
        self.__kernel_size = [
            base_filter_length,
            filter_length,
            base_filter_length,
        ]
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        if groups % base_groups != 0:
            raise ValueError("`groups` should be divisible by `base_groups`")
        self.__base_groups = base_groups
        self.__groups = [
            base_groups,
            groups,
            base_groups,
        ]

        if self.config.increase_channels_method.lower() == "zero_padding" and self.__groups != 1:
            raise ValueError("zero padding for increasing channels can not be used with groups != 1")

        self.__increase_channels = (self.__out_channels[-1] > self.__in_channels)
        self.shortcut = self._make_shortcut_layer()

        self.main_stream = nn.Sequential()
        conv_names = {0:"cba_head", 1:"cba_neck", 2:"cba_tail",}
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            conv_activation = (self.config.activation if i < self.__num_convs-1 else None)
            conv_out_channels = self.__out_channels[i]
            self.main_stream.add_module(
                conv_names[i],
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=self.__kernel_size[i],
                    stride=(self.__stride if i == self.config.subsample_at else 1),
                    groups=self.__groups[i],
                    batch_norm=True,
                    activation=conv_activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = conv_out_channels

        if isinstance(self.config.activation, str):
            self.out_activation = \
                Activations[self.config.activation.lower()](**self.config.kw_activation)
        else:
            self.out_activation = \
                self.config.activation(**self.config.kw_activation)
        
    def _make_shortcut_layer(self) -> Union[nn.Module, type(None)]:
        """ finished, checked,
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                shortcut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels[-1],
                    groups=self.__base_groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = False if self.config.subsample_mode.lower() != "conv" else True
                shortcut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        batch_norm=batch_norm,
                        mode=self.config.subsample_mode,
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels[-1]),
                )
        else:
            shortcut = None
        return shortcut

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        out: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        identity = input

        out = self.main_stream(input)

        if self.shortcut is not None:
            identity = self.shortcut(input)

        out += identity
        out = self.out_activation(out)

        return out

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

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
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class ResNetMacroBlock(nn.Sequential):
    """ NOT finished, NOT checked,
    """
    __DEBUG__ = True
    __name__ = "ResNetMacroBlock"

    def __init__(self) -> NoReturn:
        """
        """
        super().__init__()
        raise NotImplementedError

    def forward(self):
        """
        """
        raise NotImplementedError

    def compute_output_shape():
        """
        """
        raise NotImplementedError


class ResNet(nn.Sequential):
    """ finished, checked,

    References:
    -----------
    [1] https://github.com/awni/ecg
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    TODO:
    -----
    1. check performances of activations other than "nn.ReLU", especially mish and swish
    2. add functionality of "replace_stride_with_dilation"
    """
    __DEBUG__ = True
    __name__ = "ResNet"
    building_block = ResNetBasicBlock
    __DEFAULT_CONFIG__ = ED(
        activation="relu", kw_activation={"inplace": True},
        kernel_initializer="he_normal", kw_initializer={},
        init_subsample_mode="max",
    )

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set:
            init_num_filters: sequence of int,
                number of filters of the first convolutional layer
            init_filter_length: sequence of int,
                filter length (kernel size) of the first convolutional layer
            init_conv_stride: int,
                stride of the first convolutional layer
            init_pool_size: int,
                pooling kernel size of the first pooling layer
            init_pool_stride: int,
                pooling stride of the first pooling layer
            bias: bool,
                if True, each convolution will have a bias term
            num_blocks: sequence of int,
                number of building blocks in each macro block
            filter_lengths: int or sequence of int or sequence of sequences of int,
                filter length(s) (kernel size(s)) of the convolutions,
                with granularity to the whole network, to each macro block,
                or to each building block
            subsample_lengths: int or sequence of int or sequence of sequences of int,
                subsampling length(s) (ratio(s)) of all blocks,
                with granularity to the whole network, to each macro block,
                or to each building block,
                the former 2 subsample at the first building block
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for the building blocks
            for a full list of configurable parameters, ref. corr. config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        if self.config.get("building_block", "").lower() in ["bottleneck", "bottle_neck",]:
            self.building_block = ResNetBottleNeck
            # additional parameters for bottleneck
            self.additional_kw = ED({
                k: self.config[k] for k in ["base_width", "base_groups", "base_filter_length",] \
                    if k in self.config.keys()
            })
        else:
            self.additional_kw = ED()
        if self.__DEBUG__:
            print(f"additional_kw = {self.additional_kw}")
        
        self.add_module(
            "init_cba",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_length,
                stride=self.config.init_conv_stride,
                # bottleneck use "base_groups"
                groups=self.additional_kw.get("base_groups", self.config.groups),
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
                bias=self.config.bias,
            )
        )
        
        if self.config.init_pool_stride > 0:
            self.add_module(
                "init_pool",
                DownSample(
                    down_scale=self.config.init_pool_stride,
                    in_channels=self.config.init_num_filters,
                    kernel_size=self.config.init_pool_size,
                    padding=(self.config.init_pool_size-1)//2,
                    mode=self.config.init_subsample_mode.lower(),
                ),
            )

        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = \
                list(repeat(self.config.filter_lengths, len(self.config.num_blocks)))
        else:
            self.__filter_lengths = self.config.filter_lengths
            assert len(self.__filter_lengths) == len(self.config.num_blocks), \
                f"`config.filter_lengths` indicates {len(self.__filter_lengths)} macro blocks, while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = \
                list(repeat(self.config.subsample_lengths, len(self.config.num_blocks)))
        else:
            self.__subsample_lengths = self.config.subsample_lengths
            assert len(self.__subsample_lengths) == len(self.config.num_blocks), \
                f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)} macro blocks, while `config.num_blocks` indicates {len(self.config.num_blocks)}"

        # grouped resnet (basic) blocks,
        # number of channels are doubled at the first block of each macro-block
        macro_in_channels = self.config.init_num_filters
        for macro_idx, nb in enumerate(self.config.num_blocks):
            macro_num_filters = (2**macro_idx) * self.config.init_num_filters
            macro_filter_lengths = self.__filter_lengths[macro_idx]
            macro_subsample_lengths = self.__subsample_lengths[macro_idx]
            block_in_channels = macro_in_channels
            block_num_filters = macro_num_filters
            if isinstance(macro_filter_lengths, int):
                block_filter_lengths = list(repeat(macro_filter_lengths, nb))
            else:
                block_filter_lengths = macro_filter_lengths
            assert len(block_filter_lengths) == nb, \
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            if isinstance(macro_subsample_lengths, int):
                # subsample at the first building block
                block_subsample_lengths = list(repeat(1, nb))
                block_subsample_lengths[0] = macro_subsample_lengths
            else:
                block_subsample_lengths = macro_subsample_lengths
            assert len(block_subsample_lengths) == nb, \
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            for block_idx in range(nb):
                self.add_module(
                    f"{self.building_block.__name__}_{macro_idx}_{block_idx}",
                    self.building_block(
                        in_channels=block_in_channels,
                        num_filters=block_num_filters,
                        filter_length=block_filter_lengths[block_idx],
                        subsample_length=block_subsample_lengths[block_idx],
                        groups=self.config.groups,
                        dilation=1,
                        init_num_filters=self.config.init_num_filters,
                        **(self.additional_kw),
                        **(self.config.block),
                    )
                )
                block_in_channels = block_num_filters * self.building_block.expansion
            macro_in_channels = macro_num_filters * self.building_block.expansion

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

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
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)
