"""
UNet structure models,
mainly for ECG wave delineation

References:
-----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
[2] https://github.com/milesial/Pytorch-UNet/
"""
import sys
from copy import deepcopy
from collections import OrderedDict
from itertools import repeat
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from ..cfg import Cfg
from .nets import (
    Conv_Bn_Activation,
    DownSample, ZeroPadding,
)
from ..utils.utils_nn import compute_deconv_output_shape
from ..utils.misc import dict_to_str

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_UNET",
]


class DoubleConv(nn.Sequential):
    """

    building blocks of UNet
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    __DEBUG__ = False
    __name__ = "DoubleConv"

    def __init__(self, in_channels:int, out_channels:int, filter_lengths:Union[Sequence[int],int], subsample_lengths:Union[Sequence[int],int]=1, groups:int=1, mid_channels:Optional[int]=None, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the last convolutional layer
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
        """
        super().__init__()
        self.__num_convs = 2
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        if isinstance(filter_lengths, int):
            kernel_sizes = list(repeat(filter_lengths, self.__num_convs))
        else:
            kernel_sizes = filter_lengths
        assert len(kernel_sizes) == self.__num_convs

        if isinstance(subsample_lengths, int):
            strides = list(repeat(subsample_lengths, self.__num_convs))
        else:
            strides = subsample_lengths
        assert len(strides) == self.__num_convs

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.__mid_channels,
                kernel_size=kernel_sizes[0],
                stride=strides[0],
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
            ),
        )
        self.add_module(
            "cba_2",
            Conv_Bn_Activation(
                in_channels=self.__mid_channels,
                out_channels=self.__out_channels,
                kernel_size=kernel_sizes[1],
                stride=strides[1],
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
            )
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
            the output shape of this `DoubleConv` layer, given `seq_len` and `batch_size`
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
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class DownDoubleConv(nn.Sequential):
    """
    Downscaling with maxpool then double conv
    down sample (maxpool) --> double conv (conv --> conv)

    channels are increased after down sampling
    
    References:
    -----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
    """
    __DEBUG__ = False
    __name__ = "DownDoubleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(self, down_scale:int, in_channels:int, out_channels:int, filter_lengths:Union[Sequence[int],int], groups:int=1, mid_channels:Optional[int]=None, mode:str='max', **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the last convolutional layer
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        mode: str, default 'max',
            mode for down sampling,
            can be one of `DownSample.__MODES__`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
        """
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
        self.__out_channels = out_channels
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "down_sample",
            DownSample(
                down_scale=self.__down_scale,
                in_channels=self.__in_channels,
                batch_norm=False,
                mode=mode,
            )
        )
        self.add_module(
            "double_conv",
            DoubleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_lengths=filter_lengths,
                subsample_lengths=1,
                groups=groups,
                mid_channels=self.__mid_channels,
                **(self.config)
            ),
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(seq_len=_seq_len)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class UpDoubleConv(nn.Module):
    """
    Upscaling then double conv, with input of corr. down layer concatenated
    up sampling --> conv (conv --> conv)
        ^
        |
    extra input

    channels are shrinked after up sampling
    """
    __DEBUG__ = False
    __name__ = "UpDoubleConv"
    __MODES__ = ['nearest', 'linear', 'area', 'deconv',]

    def __init__(self, up_scale:int, in_channels:int, out_channels:int, filter_lengths:Union[Sequence[int],int], deconv_filter_length:Optional[int]=None, groups:int=1, mode:str='deconv', mid_channels:Optional[int]=None, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        up_scale: int,
            scale of up sampling
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size) of the convolutional layers
        deconv_filter_length: int,
            only used when `mode` == 'conv'
            length(s) of the filters (kernel size) of the deconvolutional upsampling layer
        groups: int, default 1, not used currently,
            connection pattern (of channels) of the inputs and outputs
        mode: str, default 'deconv', case insensitive,
            mode of up sampling
        mid_channels: int, optional,
            number of channels produced by the first deconvolutional layer,
            defaults to `out_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the deconvolutional layers
        """
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else in_channels // 2
        self.__out_channels = out_channels
        self.__deconv_filter_length = deconv_filter_length
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        # the following has to be checked
        # if bilinear, use the normal convolutions to reduce the number of channels
        if self.__mode == 'deconv':
            self.__deconv_padding = max(0, (self.__deconv_filter_length - self.__up_scale)//2)
            self.up = nn.ConvTranspose1d(
                in_channels=self.__in_channels,
                out_channels=self.__in_channels,
                kernel_size=self.__deconv_filter_length,
                stride=self.__up_scale,
                padding=self.__deconv_padding,
            )
        else:
            self.up = nn.Upsample(
                scale_factor=self.__up_scale,
                mode=mode,
            )
        self.zero_pad = ZeroPadding(
            in_channels=self.__in_channels,
            out_channels=self.__in_channels+self.__in_channels//2,
            loc="head",
        )
        self.conv = DoubleConv(
            in_channels=2*self.__in_channels,
            out_channels=self.__out_channels,
            filter_lengths=filter_lengths,
            subsample_lengths=1,
            groups=groups,
            **(self.config),
        )

    def forward(self, input:Tensor, down_output:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        input: Tensor,
            input tensor from the previous layer
        down_output:Tensor: Tensor,
            input tensor of the last layer of corr. down block
        """
        output = self.up(input)
        output = self.zero_pad(output)

        diff_sig_len = down_output.shape[-1] - output.shape[-1]

        output = F.pad(output, [diff_sig_len // 2, diff_sig_len - diff_sig_len // 2])
        output = torch.cat([down_output, output], dim=1)  # concate along the channel axis
        output = self.conv(output)

        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        _sep_len = seq_len
        if self.__mode == 'deconv':
            output_shape = compute_deconv_output_shape(
                input_shape=[batch_size, self.__in_channels, _sep_len],
                num_filters=self.__in_channels,
                kernel_size=self.__deconv_filter_length,
                stride=self.__up_scale,
                padding=self.__deconv_padding,
            )
        else:
            output_shape = [batch_size, self.__in_channels, self.__up_scale*_sep_len]
        _, _, _seq_len = output_shape
        output_shape = self.zero_pad.compute_output_shape(_seq_len, batch_size)
        _, _, _seq_len = output_shape
        output_shape = self.conv.compute_output_shape(_seq_len, batch_size)
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ECG_UNET(nn.Module):
    """
    """
    __DEBUG__ = False
    __name__ = "ECG_UNET"
    
    def __init__(self, classes:Sequence[str], n_leads:int, config:dict) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: sequence of int,
            name of the classes
        n_leads: int,
            number of input leads
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)  # final out_channels
        self.__in_channels = n_leads
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.init_conv = DoubleConv(
            in_channels=self.__in_channels,
            out_channels=self.n_classes,
            filter_lengths=self.config.init_filter_length,
            subsample_lengths=1,
            groups=self.config.groups,
            batch_norm=self.config.batch_norm,
            activation=self.config.activation,
            kw_activation=self.config.kw_activation,
            kernel_initializer=self.config.kernel_initializer,
            kw_initializer=self.config.kw_initializer,
        )

        self.down_blocks = nn.ModuleDict()
        in_channels = self.n_classes
        for idx in range(self.config.down_up_block_num):
            self.down_blocks[f'down_{idx}'] = \
                DownDoubleConv(
                    down_scale=self.config.down_scales[idx],
                    in_channels=in_channels,
                    out_channels=self.config.down_num_filters[idx],
                    filter_lengths=self.config.down_filter_lengths[idx],
                    groups=self.config.groups,
                    mode=self.config.down_mode,
                    **(self.config.down_block)
                )
            in_channels = self.config.down_num_filters[idx]

        self.up_blocks = nn.ModuleDict()
        in_channels = self.config.down_num_filters[-1]
        for idx in range(self.config.down_up_block_num):
            self.up_blocks[f'up_{idx}'] = \
                UpDoubleConv(
                    up_scale=self.config.up_scales[idx],
                    in_channels=in_channels,
                    out_channels=self.config.up_num_filters[idx],
                    filter_lengths=self.config.up_conv_filter_lengths[idx],
                    deconv_filter_length=self.config.up_deconv_filter_lengths[idx],
                    groups=self.config.groups,
                    mode=self.config.up_mode,
                    **(self.config.up_block)
                )
            in_channels = self.config.up_num_filters[idx]

        self.out_conv = Conv_Bn_Activation(
            in_channels=self.config.up_num_filters[-1],
            out_channels=self.n_classes,
            kernel_size=self.config.out_filter_length,
            stride=1,
            groups=self.config.groups,
            batch_norm=self.config.batch_norm,
            activation=self.config.activation,
            kw_activation=self.config.kw_activation,
            kernel_initializer=self.config.kernel_initializer,
            kw_initializer=self.config.kw_initializer,
        )

        self.softmax = nn.Softmax(dim=-1)  # for making inference

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,
        """
        to_concat = [self.init_conv(input)]
        for idx in range(self.config.down_up_block_num):
            to_concat.append(self.down_blocks[f"down_{idx}"](to_concat[-1]))
        up_input = to_concat[-1]
        to_concat = to_concat[-2::-1]
        for idx in range(self.config.down_up_block_num):
            up_output = self.up_blocks[f"up_{idx}"](up_input, to_concat[idx])
            up_input = up_output
        output = self.out_conv(up_output)

        return output

    @torch.no_grad()
    def inference(self, input:Tensor, class_map:Optional[dict]=None) -> np.ndarray:
        """ finished, checked,
        """
        output_masks = self.forward(input)  # batch_size, channels(n_classes), seq_len
        output_masks = output_masks.permute(0,2,1)  # batch_size, seq_len, channels(n_classes)
        output_masks = self.softmax(output_masks)  # batch_size, seq_len, channels(n_classes)
        output_masks = output_masks.cpu().detach().numpy()
        output_masks = np.argmax(output_masks, axis=-1)
        return output_masks

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.n_classes, seq_len)
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params
