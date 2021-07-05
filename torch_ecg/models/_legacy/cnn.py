"""
named CNNs, which are frequently used by more complicated models, including
1. vgg
2. resnet
3. variants of resnet (with se, gc, etc.)
4. multi_scopic
"""
import math
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
from torch_ecg.utils.utils_nn import (
    compute_conv_output_shape,
    compute_maxpool_output_shape,
    compute_avgpool_output_shape,
    compute_module_size,
)
from torch_ecg.utils.misc import dict_to_str, list_sum
from torch_ecg.models._nets import (
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
    "DenseBasicBlock", "DenseBottleNeck", "DenseMacroBlock", "DenseTransition", "DenseNet",
    "CPSCBlock", "CPSCCNN",
]


class VGGBlock(nn.Sequential):
    """ finished, checked,

    building blocks of the CNN feature extractor `VGG16`
    """
    __DEBUG__ = False
    __name__ = "VGGBlock"

    def __init__(self, num_convs:int, in_channels:int, out_channels:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        num_layers = 0
        for module in self:
            if num_layers < self.__num_convs:
                output_shape = module.compute_output_shape(seq_len, batch_size)
                _, _, seq_len = output_shape
            else:
                output_shape = compute_maxpool_output_shape(
                    input_shape=[batch_size, self.__out_channels, seq_len],
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
        
        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, including
            number of convolutional layers, number of filters for each layer,
            and more for `VGGBlock`.
            key word arguments that have to be set:
            num_convs: sequence of int,
                number of convolutional layers for each `VGGBlock`
            num_filters: sequence of int,
                number of filters for each `VGGBlock`
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for `VGGBlock`
            for a full list of configurable parameters, ref. corr. config file
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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


class ResNetBasicBlock(nn.Module):
    """ finished, checked,

    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`
    """
    __DEBUG__ = False
    __name__ = "ResNetBasicBlock"
    expansion = 1  # not used

    def __init__(self, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, groups:int=1, dilation:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
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
    
    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
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

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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

        Parameters
        ----------
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
        
    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
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

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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

    References
    ----------
    [1] https://github.com/awni/ecg
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    TODO
    ----
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
        
        Parameters
        ----------
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        scopes: sequence of int,
            scopes of the convolutional layers, via `dilation`
        num_filters: int or sequence of int,
            number of filters of the convolutional layer(s)
        filter_lengths: int or sequence of int,
            filter length(s) (kernel size(s)) of the convolutional layer(s)
        subsample_length: int,
            subsample length (ratio) at the last layer of the block
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        scopes: sequence of sequences of int,
            scopes (in terms of `dilation`) for the convolutional layers,
            each sequence of int is for one branch
        num_filters: sequence of int, or sequence of sequences of int,
            number of filters for the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same number of filters
        filter_lengths: sequence of int, or sequence of sequences of int,
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

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set:
            scopes: sequence of sequences of sequences of int,
                scopes (in terms of dilation) of each convolution
            num_filters: sequence of sequences (of int or of sequences of int),
                number of filters of the convolutional layers,
                with granularity to each block of each branch,
                or to each convolution of each block of each branch
            filter_lengths: sequence of sequences (of int or of sequences of int),
                filter length(s) (kernel size(s)) of the convolutions,
                with granularity to each block of each branch,
                or to each convolution of each block of each branch
            subsample_lengths: sequence of int or sequence of sequences of int,
                subsampling length(s) (ratio(s)) of all blocks,
                with granularity to each branch or to each block of each branch,
                each subsamples after the last convolution of each block
            dropouts: sequence of int or sequence of sequences of int,
                dropout rates of all blocks,
                with granularity to each branch or to each block of each branch,
                each dropouts at the last of each block
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for the building blocks
            for a full list of configurable parameters, ref. corr. config file
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
        
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
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
    
    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        out_channels = 0
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            _, _branch_oc, _seq_len = \
                self.branches[key].compute_output_shape(seq_len, batch_size)
            out_channels += _branch_oc
        output_shape = (batch_size, out_channels, _seq_len)
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class DenseBasicBlock(nn.Module):
    """ finished, checked,

    the basic building block for DenseNet,
    consisting of normalization -> activation -> convolution (-> dropout (optional)),
    the output Tensor is the concatenation of old features (input) with new features
    """
    __DEBUG__ = True
    __name__ = "DenseBasicBlock"
    __DEFAULT_CONFIG__ = ED(
        activation="relu", kw_activation={"inplace": True}, memory_efficient=False,
    )

    def __init__(self, in_channels:int, growth_rate:int, filter_length:int, groups:int=1, bias:bool=False, dropout:float=0.0, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        growth_rate: int,
            number of features of (channels) output from the main stream,
            further concatenated to the shortcut,
            hence making the final number of output channels grow by this value
        filter_length: int,
            length (size) of the filter kernels
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            activation choices, memory_efficient choices, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__growth_rate = growth_rate
        self.__kernel_size = filter_length
        self.__groups = groups
        self.config = ED(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        assert all([in_channels % groups == 0, growth_rate % groups == 0])

        self.bac = Conv_Bn_Activation(
            in_channels=self.__in_channels,
            out_channels=self.__growth_rate,
            kernel_size=self.__kernel_size,
            stride=1,
            dilation=1,
            groups=self.__groups,
            batch_norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        
        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        new_features = self.bac(input)
        if self.dropout:
            new_features = self.dropout(new_features)
        if self.__groups == 1:
            output = torch.cat([input, new_features], dim=1)
        else:  # see TODO of `DenseNet`
            # input width per group
            iw_per_group = self.__in_channels // self.__groups
            # new features width per group
            nfw_per_group = self.__growth_rate // self.__groups
            output = torch.cat(
                list_sum(
                    [
                        [
                            input[..., iw_per_group*i: iw_per_group*(i+1), ...],
                            new_features[..., nfw_per_group*i: nfw_per_group*(i+1), ...]
                        ]  for i in range(self.__groups)
                    ]
                ), 1
            )
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        out_channels = self.__in_channels + self.__growth_rate
        output_shape = (batch_size, out_channels, seq_len)
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class DenseBottleNeck(nn.Module):
    """ finished, checked,

    bottleneck modification of `DenseBasicBlock`,
    with an additional prefixed sequence of
    (normalization -> activation -> convolution of kernel size 1)
    """
    __DEBUG__ = True
    __name__ = "DenseBottleNeck"
    __DEFAULT_CONFIG__ = ED(
        activation="relu", kw_activation={"inplace": True}, memory_efficient=False,
    )

    def __init__(self, in_channels:int, growth_rate:int, bn_size:int, filter_length:int, groups:int=1, bias:bool=False, dropout:float=0.0, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        growth_rate: int,
            number of features of (channels) output from the main stream,
            further concatenated to the shortcut,
            hence making the final number of output channels grow by this value
        bn_size: int,
            base width of intermediate layers (the bottleneck)
        filter_length: int,
            length (size) of the filter kernels of the second convolutional layer
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            activation choices, memory_efficient choices, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__growth_rate = growth_rate
        self.__bn_size = bn_size
        self.__kernel_size = filter_length
        self.__groups = groups
        self.config = ED(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        bottleneck_channels = self.__bn_size * self.__growth_rate

        self.neck_conv = Conv_Bn_Activation(
            in_channels=self.__in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=groups,
            batch_norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        self.main_conv = Conv_Bn_Activation(
            in_channels=bottleneck_channels,
            out_channels=self.__growth_rate,
            kernel_size=self.__kernel_size,
            stride=1,
            dilation=1,
            groups=self.__groups,
            batch_norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def bn_function(self, input:Tensor) -> Tensor:
        """ finished, checked,

        the `not memory_efficient` way

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        bottleneck_output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        bottleneck_output = self.neck_conv(input)
        return bottleneck_output

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        if self.config.memory_efficient:
            raise NotImplementedError
        else:
            new_features = self.bn_function(input)
        new_features = self.main_conv(new_features)
        if self.dropout:
            new_features = self.dropout(new_features)
        if self.__groups == 1:
            output = torch.cat([input, new_features], dim=1)
        else:  # see TODO of `DenseNet`
            # input width per group
            iw_per_group = self.__in_channels // self.__groups
            # new features width per group
            nfw_per_group = self.__growth_rate // self.__groups
            output = torch.cat(
                list_sum(
                    [
                        [
                            input[..., iw_per_group*i: iw_per_group*(i+1), ...],
                            new_features[..., nfw_per_group*i: nfw_per_group*(i+1), ...]
                        ]  for i in range(self.__groups)
                    ]
                ), 1
            )
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        out_channels = self.__in_channels + self.__growth_rate
        output_shape = (batch_size, out_channels, seq_len)
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class DenseMacroBlock(nn.Sequential):
    """ finished, checked,

    macro blocks for `DenseNet`,
    stacked sequence of builing blocks of similar pattern
    """
    __DEBUG__ = True
    __name__ = "DenseMacroBlock"
    building_block = DenseBottleNeck

    def __init__(self, in_channels:int, num_layers:int, growth_rates:Union[Sequence[int],int], bn_size:int, filter_lengths:Union[Sequence[int],int], groups:int=1, bias:bool=False, dropout:float=0.0, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        num_layers: int,
            number of building block layers
        growth_rates: sequence of int, or int,
            growth rate(s) for each building block layers,
            if is sequence of int, should have length equal to `num_layers`
        bn_size: int,
            base width of intermediate layers for `DenseBottleNeck`,
            not used for `DenseBasicBlock`
        filter_lengths: sequence of int, or int,
            filter lengths(s) (kernel size(s)) for each building block layers,
            if is sequence of int, should have length equal to `num_layers`
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            extra kw for `DenseBottleNeck`, and
            activation choices, memory_efficient choices, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_layers = num_layers
        if isinstance(growth_rates, int):
            self.__growth_rates = list(repeat(growth_rates, num_layers))
        else:
            self.__growth_rates = list(growth_rates)
        assert len(self.__growth_rates) == self.__num_layers
        self.__bn_size = bn_size
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, num_layers))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert len(self.__filter_lengths) == self.__num_layers
        self.__groups = groups
        self.config = deepcopy(config)
        if self.config.get("building_block", "").lower() in ["basic", "basic_block",]:
            self.building_block = DenseBasicBlock

        for idx in range(self.__num_layers):
            self.add_module(
                f"dense_building_block_{idx}",
                self.building_block(
                    in_channels=self.__in_channels + idx * self.__growth_rates[idx],
                    growth_rate=self.__growth_rates[idx],
                    bn_size=self.__bn_size,
                    filter_length=self.__filter_lengths[idx],
                    bias=bias,
                    dropout=dropout,
                    **(self.config),
                )
            )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        
        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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


class DenseTransition(nn.Sequential):
    """ finished, checked,

    transition blocks between `DenseMacroBlock`s,
    used to perform sub-sampling,
    and compression of channels if specified
    """
    __DEBUG__ = True
    __name__ = "DenseTransition"
    __DEFAULT_CONFIG__ = ED(
        activation="relu", kw_activation={"inplace": True}, subsample_mode="avg",
    )

    def __init__(self, in_channels:int, compression:float=1.0, subsample_length:int=2, groups:int=1, bias:bool=False, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        compression: float, default 1.0,
            compression factor,
            proportion of the number of output channels to the number of input channels
        subsample_length: int, default 2,
            subsampling length (size)
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        config: dict,
            other parameters, including
            activation choices, subsampling mode (method), etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__compression = compression
        self.__subsample_length = subsample_length
        self.__groups = groups
        assert 0 < self.__compression <= 1.0 and self.__in_channels % self.__groups == 0
        self.config = ED(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))

        # input width per group
        iw_per_group = self.__in_channels // self.__groups
        # new feature widths per group
        nfw_per_group = math.floor(iw_per_group * self.__compression)
        self.__out_channels = nfw_per_group * self.__groups
        
        self.add_module(
            "bac",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=self.__groups,
                batch_norm=True,
                activation=self.config.activation.lower(),
                kw_activation=self.config.kw_activation,
                bias=bias,
                ordering="bac",
            )
        )
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels,
                mode=self.config.subsample_mode.lower(),
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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


class DenseNet(nn.Sequential):
    """ finished, checked,

    The core part of the SOTA model (framework) of CPSC2020

    References
    ----------
    [1] G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.
    [2] G. Huang, Z. Liu, G. Pleiss, L. Van Der Maaten and K. Weinberger, "Convolutional Networks with Dense Connectivity," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2019.2918284.
    [3] https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    [4] https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
    [5] https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
    [6] https://github.com/liuzhuang13/DenseNet/tree/master/models

    NOTE
    ----
    the difference of forward output of [5] from others, however [5] doesnot support dropout

    TODO
    ----
    1. for `groups` > 1, the concatenated output should be re-organized in the channel dimension?
    2. memory-efficient mode, i.e. storing the `new_features` in a shared memory instead of stacking in newly created `Tensor`s after each mini-block
    """
    __DEBUG__ = True
    __name__ = "DenseNet"
    __DEFAULT_CONFIG__ = ED(
        bias=False,
        activation="relu", kw_activation={"inplace": True},
        kernel_initializer="he_normal", kw_initializer={},
        init_subsample_mode="avg",
    )

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set:
            num_layers: sequence of int,
                number of building block layers of each dense (macro) block
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
            growth_rates: int or sequence of int or sequence of sequences of int,
                growth rates of the building blocks,
                with granularity to the whole network, or to each dense (macro) block,
                or to each building block
            filter_lengths: int or sequence of int or sequence of sequences of int,
                filter length(s) (kernel size(s)) of the convolutions,
                with granularity to the whole network, or to each macro block,
                or to each building block
            subsample_lengths: int or sequence of int,
                subsampling length(s) (ratio(s)) of the transition blocks
            compression: float,
                compression factor of the transition blocks
            bn_size: int,
                bottleneck base width, used only when building block is `DenseBottleNeck`
            dropouts: int,
                dropout ratio of each building block
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
        self.__num_blocks = len(self.config.num_layers)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "init_cba",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_length,
                stride=1,
                dilation=1,
                groups=self.config.groups,
                batch_norm=True,
                activation=self.config.activation.lower(),
                kw_activation=self.config.kw_activation,
                bias=self.config.bias,
            )
        )
        self.add_module(
            "init_pool",
            DownSample(
                down_scale=self.config.init_pool_stride,
                in_channels=self.config.init_num_filters,
                kernel_size=self.config.init_pool_size,
                padding=(self.config.init_pool_size-1)//2,
                mode=self.config.init_subsample_mode.lower(),
            )
        )

        if isinstance(self.config.growth_rates, int):
            self.__growth_rates = list(repeat(self.config.growth_rates, self.__num_blocks))
        else:
            self.__growth_rates = list(self.config.growth_rates)
        assert len(self.__growth_rates) == self.__num_blocks, \
                f"`config.growth_rates` indicates {len(self.__growth_rates)} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"
        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = \
                list(repeat(self.config.filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(self.config.filter_lengths)
            assert len(self.__filter_lengths) == self.__num_blocks, \
                f"`config.filter_lengths` indicates {len(self.__filter_lengths)} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = \
                list(repeat(self.config.subsample_lengths, self.__num_blocks-1))
        else:
            self.__subsample_lengths = list(self.config.subsample_lengths)
            assert len(self.__subsample_lengths) == self.__num_blocks-1, \
                f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)+1} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"

        macro_in_channels = self.config.init_num_filters
        for idx, macro_num_layers in enumerate(self.config.num_layers):
            dmb = DenseMacroBlock(
                in_channels=macro_in_channels,
                num_layers=macro_num_layers,
                growth_rates=self.__growth_rates[idx],
                bn_size=self.config.bn_size,
                filter_lengths=self.__filter_lengths[idx],
                groups=self.config.groups,
                bias=self.config.bias,
                dropout=self.config.dropout,
                **(self.config.block),
            )
            _, transition_in_channels, _ = dmb.compute_output_shape()
            self.add_module(
                f"dense_macro_block_{idx}",
                dmb
            )
            if idx < self.__num_blocks-1:
                dt = DenseTransition(
                    in_channels=transition_in_channels,
                    compression=self.config.compression,
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.config.groups,
                    bias=self.config.bias,
                    **(self.config.transition),
                )
                _, macro_in_channels, _ = dt.compute_output_shape()
                self.add_module(
                    f"transition_{idx}",
                    dt
                )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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



# the (SOTA) model of CPSC 2018
# seldom used

class CPSCBlock(nn.Sequential):
    """ finished, checked,

    building block of the SOTA model of CPSC2018 challenge
    """
    __DEBUG__ = True
    __name__ = "CPSCBlock"

    def __init__(self, in_channels:int, num_filters:int, filter_lengths:Sequence[int], subsample_lengths:Sequence[int], dropout:Optional[float]=None, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        num_filters:int,
            number of filters for the convolutional layers
        filter_lengths: sequence of int,
            filter length (kernel size) of each convolutional layer
        subsample_lengths: sequence of int,
            subsample length (stride) of each convolutional layer
        dropout: float, optional,
            if positive, a `Dropout` layer will be introduced with this dropout probability
        config: dict,
            other hyper-parameters, including
            activation choices, weight initializer, etc.
        """
        super().__init__()
        self.__num_convs = len(filter_lengths)
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__dropout = dropout or 0.0
        self.config = deepcopy(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        for idx, (kernel_size, stride) in enumerate(zip(filter_lengths[:-1], subsample_lengths[:-1])):
            self.add_module(
                f"baby_{idx+1}",
                Conv_Bn_Activation(
                    self.__in_channels, self.__out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    batch_norm=self.config.batch_norm,
                )
            )
        self.add_module(
            "giant",
            Conv_Bn_Activation(
                self.__out_channels, self.__out_channels,
                kernel_size=filter_lengths[-1],
                stride=subsample_lengths[-1],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                batch_norm=self.config.batch_norm,
            )
        )
        if self.__dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.__dropout),
            )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        n_layers = 0
        _seq_len = seq_len
        for module in self:
            if n_layers >= self.__num_convs:
                break
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
            n_layers += 1
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


class CPSCCNN(nn.Sequential):
    """ finished, checked,

    CNN part of the SOTA model of the CPSC2018 challenge
    """
    __DEBUG__ = True
    __name__ = "CPSCCNN"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters
        ----------
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

        num_filters = self.config.num_filters
        filter_lengths = self.config.filter_lengths
        subsample_lengths = self.config.subsample_lengths
        dropouts = self.config.dropouts
        blk_in = self.__in_channels
        for blk_idx, (blk_nf, blk_fl, blk_sl, blk_dp) \
            in enumerate(zip(num_filters, filter_lengths, subsample_lengths, dropouts)):
            self.add_module(
                f"cpsc_block_{blk_idx+1}",
                CPSCBlock(
                    in_channels=blk_in,
                    num_filters=blk_nf,
                    filter_lengths=blk_fl,
                    subsample_lengths=blk_sl,
                    dropout=blk_dp,
                )
            )
            blk_in = blk_nf[-1]

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output
    
    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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
