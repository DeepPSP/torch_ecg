"""
named CNNs, which are frequently used by more complicated models, including
1. vgg
2. resnet
3. variants of resnet (with se, gc, etc.)
4. multi_scopic
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.utils.utils_nn import compute_conv_output_shape, compute_module_size
from torch_ecg.utils.misc import dict_to_str
from .nets import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    AttentionWithContext,
    AttentivePooling,
    NonLocalBlock, SEBlock, GlobalContextBlock,
    SeqLin,
)


if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "VGGBlock", "VGG16",
    "ResNetBasicBlock", "ResNetBottleNeck", "ResNet",
    "MultiScopicBasicBlock", "MultiScopicBranch", "MultiScopicCNN",
]


class VGGBlock(nn.Sequential):
    """ finished, checked,

    building blocks of the CNN feature extractor `VGG16`
    """
    __DEBUG__ = False
    __name__ = "VGGBlock"

    def __init__(self, num_convs:int, in_channels:int, out_channels:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        num_convs: int,
            number of convolutional layers of this block
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other parameters, including
            filter length (kernel size), activation choices,
            weight initializer, batch normalization choices, etc. for the convolutional layers;
            and pool size for the pooling layer
        """
        super().__init__()
        self.__num_convs = num_convs
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__groups = groups
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=self.config.filter_length,
                stride=self.config.subsample_length,
                groups=self.__groups,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
                batch_norm=self.config.batch_norm,
            )
        )
        for idx in range(num_convs-1):
            self.add_module(
                f"cba_{idx+2}",
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=self.config.filter_length,
                    stride=self.config.subsample_length,
                    groups=self.__groups,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    batch_norm=self.config.batch_norm,
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.config.pool_size, self.config.pool_stride)
        )

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
        num_layers = 0
        for module in self:
            if num_layers < self.__num_convs:
                output_shape = module.compute_output_shape(seq_len, batch_size)
                _, _, seq_len = output_shape
            else:
                output_shape = compute_conv_output_shape(
                    input_shape=[batch_size, self.__out_channels, seq_len],
                    num_filters=self.__out_channels,
                    kernel_size=self.config.pool_size,
                    stride=self.config.pool_size,
                    channel_last=False,
                )
            num_layers += 1
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class VGG16(nn.Sequential):
    """ finished, checked,

    CNN feature extractor of the CRNN models proposed in refs of `ECG_CRNN`
    """
    __DEBUG__ = False
    __name__ = "VGG16"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, including
            number of convolutional layers, number of filters for each layer,
            and more for `VGGBlock`
        """
        super().__init__()
        self.__in_channels = in_channels
        # self.config = deepcopy(ECG_CRNN_CONFIG.cnn.vgg16)
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        module_in_channels = in_channels
        for idx, (nc, nf) in enumerate(zip(self.config.num_convs, self.config.num_filters)):
            module_name = f"vgg_block_{idx+1}"
            self.add_module(
                name=module_name,
                module=VGGBlock(
                    num_convs=nc,
                    in_channels=module_in_channels,
                    out_channels=nf,
                    groups=self.config.groups,
                    **(self.config.block),
                )
            )
            module_in_channels = nf

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
        for module in self:
            output_shape = module.compute_output_shape(seq_len, batch_size)
            _, _, seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class ResNetBasicBlock(nn.Module):
    """ finished, checked,

    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`
    """
    __DEBUG__ = False
    __name__ = "ResNetBasicBlock"
    expansion = 1  # not used

    def __init__(self, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, groups:int=1, dilation:int=1, **config) -> NoReturn:
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
            filter length (kernel size), activation choices, weight initializer,
            and short cut patterns, etc.
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
    
    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """ finished, checked,
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    groups=self.__groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = False if self.config.subsample_mode.lower() != "conv" else True
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

        if self.short_cut is not None:
            identity = self.short_cut(input)

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

    def __init__(self, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, groups:int=1, dilation:int=1, base_width:int=12*4, base_groups:int=1, base_filter_length:int=1, **config) -> NoReturn:
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
            filter length (kernel size), activation choices, weight initializer,
            and short cut patterns, etc.
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
        self.short_cut = self._make_short_cut_layer()

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
        
    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """ finished, checked,
        """
        if self.__DEBUG__:
            print(f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}")
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels[-1],
                    groups=self.__base_groups,
                    batch_norm=True,
                    mode=self.config.subsample_mode,
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = False if self.config.subsample_mode.lower() != "conv" else True
                short_cut = nn.Sequential(
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
            short_cut = None
        return short_cut

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

        if self.short_cut is not None:
            identity = self.short_cut(input)

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

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        if self.config.get("block_name", "").lower() in ["bottleneck", "bottle_neck",]:
            self.building_block = ResNetBottleNeck
            # additional parameters for bottleneck
            self.additional_kw = ED({
                k: self.config[k] for k in ["base_width", "base_groups", "base_filter_length"] \
                    if k in self.config.keys()
            })
        else:
            self.additional_kw = ED()
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
        
        if self.config.init_pool_size > 0:
            self.add_module(
                "init_pool",
                nn.MaxPool1d(
                    kernel_size=self.config.init_pool_size,
                    stride=self.config.init_pool_stride,
                    padding=(self.config.init_pool_size-1)//2,
                )
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
            if type(module).__name__ == "MaxPool1d":
                output_shape = compute_conv_output_shape(
                    input_shape=(batch_size, self.config.init_filter_length, _seq_len),
                    num_filters=self.config.init_filter_length,
                    kernel_size=self.config.init_pool_size,
                    stride=self.config.init_pool_stride,
                    padding=(self.config.init_pool_size-1)//2,
                )
            else:
                output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class MultiScopicBasicBlock(nn.Sequential):
    """ finished, checked,

    basic building block of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)

    (conv -> activation) * N --> bn --> down_sample
    """
    __DEBUG__ = False
    __name__ = "MultiScopicBasicBlock"

    def __init__(self, in_channels:int, scopes:Sequence[int], num_filters:Union[int,Sequence[int]], filter_lengths:Union[int,Sequence[int]], subsample_length:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        scopes: sequence of int,
            scopes of the convolutional layers, via `dilation`
        num_filters: int or sequence of int,
        filter_lengths: int or sequence of int,
        subsample_length: int,
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_convs = len(self.__scopes)
        if isinstance(num_filters, int):
            self.__out_channels = list(repeat(num_filters, self.__num_convs))
        else:
            self.__out_channels = num_filters
            assert len(self.__out_channels) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `num_filters` indicates {len(self.__out_channels)}"
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = filter_lengths
            assert len(self.__filter_lengths) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        self.__subsample_length = subsample_length
        self.__groups = groups
        self.config = ED(deepcopy(config))

        conv_in_channels = self.__in_channels
        for idx in range(self.__num_convs):
            self.add_module(
                f"ca_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels[idx],
                    kernel_size=self.__filter_lengths[idx],
                    stride=1,
                    dilation=self.__scopes[idx],
                    groups=self.__groups,
                    batch_norm=False,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = self.__out_channels[idx]
        self.add_module(
            "bn",
            nn.BatchNorm1d(self.__out_channels[-1])
        )
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels[-1],
                groups=self.__groups,
                # padding=
                batch_norm=False,
                mode=self.config.subsample_mode,
            )
        )
        if self.config.dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.config.dropout, inplace=False)
            )

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
        for idx, module in enumerate(self):
            if idx == self.__num_convs:  # bn layer
                continue
            elif self.config.dropout > 0 and idx == len(self)-1:  # dropout layer
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class MultiScopicBranch(nn.Sequential):
    """ finished, checked,
    
    branch path of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = False
    __name__ = "MultiScopicBranch"

    def __init__(self, in_channels:int, scopes:Sequence[Sequence[int]], num_filters:Union[Sequence[int],Sequence[Sequence[int]]], filter_lengths:Union[Sequence[int],Sequence[Sequence[int]]], subsample_lengths:Union[int,Sequence[int]], groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        scopes: sequence of sequence of int,
            scopes (in terms of `dilation`) for the convolutional layers,
            each sequence of int is for one branch
        num_filters: sequence of int, or sequence of sequence of int,
            number of filters for the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same number of filters
        filter_lengths: sequence of int, or sequence of sequence of int,
            filter length (kernel size) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same filter length
        subsample_lengths: int, or sequence of int,
            subsample length (stride) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same subsample length
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other hyper-parameters, including
            dropout, activation choices, weight initializer, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_blocks = len(self.__scopes)
        self.__num_filters = num_filters
        assert len(self.__num_filters) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `num_filters` indicates {len(self.__num_filters)}"
        self.__filter_lengths = filter_lengths
        assert len(self.__filter_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `filter_lengths` indicates {llen(self.__filter_lengths)}"
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(repeat(subsample_lengths, self.__num_blocks))
        else:
            self.__subsample_lengths = filter_lengths
            assert len(self.__subsample_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `subsample_lengths` indicates {llen(self.__subsample_lengths)}"
        self.__groups = groups
        self.config = ED(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx in range(self.__num_blocks):
            self.add_module(
                f"block_{idx}",
                MultiScopicBasicBlock(
                    in_channels=block_in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.__num_filters[idx],
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.__groups,
                    dropout=self.config.dropouts[idx],
                    **(self.config.block)
                )
            )
            block_in_channels = self.__num_filters[idx]

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
        for idx, module in enumerate(self):
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class MultiScopicCNN(nn.Module):
    """ finished, checked,

    CNN part of the SOTA model from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = False
    __name__ = "MultiScopicCNN"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        self.__scopes = self.config.scopes
        self.__num_branches = len(self.__scopes)

        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.branches = nn.ModuleDict()
        for idx in range(self.__num_branches):
            self.branches[f"branch_{idx}"] = \
                MultiScopicBranch(
                    in_channels=self.__in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.config.num_filters[idx],
                    filter_lengths=self.config.filter_lengths[idx],
                    subsample_lengths=self.config.subsample_lengths[idx],
                    dropouts=self.config.dropouts[idx],
                    block=self.config.block,  # a dict
                )

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
        branch_out = OrderedDict()
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            branch_out[key] = self.branches[key].forward(input)
        output = torch.cat(
            [branch_out[f"branch_{idx}"] for idx in range(self.__num_branches)],
            dim=1,  # along channels
        )
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
        out_channels = 0
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            _, _branch_oc, _seq_len = \
                self.branches[key].compute_output_shape(seq_len, batch_size)
            out_channels += _branch_oc
        return (batch_size, out_channels, _seq_len)

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)
