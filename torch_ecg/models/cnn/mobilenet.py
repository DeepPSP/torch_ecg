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
from typing import Union, Optional, Sequence, NoReturn, Any

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import Tensor

from ...cfg import CFG, DEFAULTS
from ...utils.utils_nn import compute_module_size, SizeMixin
from ...utils.misc import dict_to_str
from ...models._nets import (
    Conv_Bn_Activation, MultiConv,
    DownSample,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)


if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "MobileNetV1",
    "MobileNetV2",
    "MobileNetV3",
]


_DEFAULT_CONV_CONFIGS_MobileNetV1 = CFG(
    ordering="cba",
    conv_type="separable",
    batch_norm=True,
    activation="relu6",
    kw_activation={"inplace": True},
    # kernel_initializer="he_normal",
    # kw_initializer={},
)


class MobileNetSeparableConv(SizeMixin, nn.Sequential):
    """

    similar to `_nets.SeparableConv`,
    the difference is that there are normalization and activation between depthwise conv and pointwise conv
    """
    __DEBUG__ = True
    __name__ = "MobileNetSeparableConv"
    
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 padding:Optional[int]=None,
                 dilation:int=1,
                 groups:int=1,
                 batch_norm:Union[bool,str,nn.Module]=True,
                 activation:Optional[Union[str,nn.Module]]="relu6",
                 kernel_initializer:Optional[Union[str,callable]]=None,
                 bias:bool=True,
                 depth_multiplier:int=1,
                 width_multiplier:float=1.0,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size (length) of the convolution kernel
        stride: int,
            stride (subsample length) of the convolution
        padding: int, optional,
            zero-padding added to both sides of the input
        dilation: int, default 1,
            spacing between the kernel points
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, default "relu6",
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear", "hardswish", "relu6"
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        depth_multiplier: int, default 1,
            multiplier of the number of output channels of the depthwise convolution
        width_multiplier: float, default 1.0,
            multiplier of the number of output channels of the pointwise convolution
        kwargs: dict, optional,
            extra parameters, including `ordering`, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation
        if padding is None:
            # "same" padding
            self.__padding = (self.__dilation * (self.__kernel_size - 1)) // 2
        elif isinstance(padding, int):
            self.__padding = padding
        self.__groups = groups
        self.__bias = bias
        ordering = kwargs.get("ordering", _DEFAULT_CONV_CONFIGS_MobileNetV1["ordering"])
        kw_initializer = kwargs.get("kw_initializer", {})
        self.__depth_multiplier = depth_multiplier
        dc_out_channels = int(self.__in_channels * self.__depth_multiplier)
        assert dc_out_channels % self.__in_channels == 0, \
            f"depth_multiplier (input is {self.__depth_multiplier}) should be positive integers"
        self.__width_multiplier = width_multiplier
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert self.__out_channels % self.__groups == 0, \
            f"width_multiplier (input is {self.__width_multiplier}) makes `out_channels` not divisible by `groups` (= {self.__groups})"

        self.add_module(
            "depthwise_conv",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=dc_out_channels,
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__in_channels,
                bias=self.__bias,
                batch_norm=batch_norm,
                activation=activation,
                ordering=ordering,
            )
        )
        self.add_module(
            "pointwise_conv",
            Conv_Bn_Activation(
                in_channels=dc_out_channels,
                out_channels=self.__out_channels,
                groups=self.__groups,
                bias=self.__bias,
                kernel_size=1, stride=1, padding=0, dilation=1,
                batch_norm=batch_norm,
                activation=activation,
                ordering=ordering,
            )
        )

        if kernel_initializer:
            if callable(kernel_initializer):
                for module in self:
                    kernel_initializer(module.weight)
            elif isinstance(kernel_initializer, str) and kernel_initializer.lower() in Initializers.keys():
                for module in self:
                    Initializers[kernel_initializer.lower()](module.weight, **kw_initializer)
            else:  # TODO: add more initializers
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
    
    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channles, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channles, seq_len)
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
            the output shape, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class MobileNetV1(SizeMixin, nn.Sequential):
    """

    Similar to Xception, but without skip connections,
    separable convolutions are slightly different too

    normal conv
    --> entry flow (separable convs, down sample and double channels every other conv)
    --> middle flow (separable convs, no down sampling, stationary number of channels)
    --> exit flow (separable convs, down sample and double channels at each conv)

    References
    ----------
    1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
    2. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/mobilenet.py
    """
    __DEBUG__ = True
    __name__ = "MobileNetV1"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set in 3 sub-dict,
            namely in "entry_flow", "middle_flow", and "exit_flow", including
            out_channels: int,
                number of channels of the output
            kernel_size: int,
                kernel size of down sampling,
                if not specified, defaults to `down_scale`,
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            padding: int,
                zero-padding added to both sides of the input
            batch_norm: bool or Module,
                batch normalization,
                the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))

        if isinstance(self.config.init_num_filters, int):
            init_convs = Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_lengths,
                stride=self.config.init_subsample_lengths,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        else:
            init_convs = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                filter_lengths=self.config.init_filter_lengths,
                subsample_lengths=self.config.init_subsample_lengths,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        self.add_module(
            "init_convs",
            init_convs,
        )

        _, entry_flow_in_channels, _ = self.init_convs.compute_output_shape()
        entry_flow = self._generate_flow(
            in_channels=entry_flow_in_channels,
            depth_multiplier=self.config.depth_multiplier,
            width_multiplier=self.config.width_multiplier,
            **self.config.entry_flow
        )
        self.add_module(
            "entry_flow",
            entry_flow
        )

        _, middle_flow_in_channels, _ = entry_flow[-1].compute_output_shape()
        middle_flow = self._generate_flow(
            in_channels=middle_flow_in_channels,
            depth_multiplier=self.config.depth_multiplier,
            width_multiplier=self.config.width_multiplier,
            **self.config.middle_flow
        )
        self.add_module(
            "middle_flow",
            middle_flow
        )

        _, exit_flow_in_channels, _ = middle_flow[-1].compute_output_shape()
        exit_flow = self._generate_flow(
            in_channels=exit_flow_in_channels,
            depth_multiplier=self.config.depth_multiplier,
            width_multiplier=self.config.width_multiplier,
            **self.config.exit_flow
        )
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    def _generate_flow(self,
                       in_channels:int, 
                       out_channels:Sequence[int],
                       filter_lengths:Union[Sequence[int], int],
                       subsample_lengths:Union[Sequence[int],int]=1,
                       dilations:Union[Sequence[int],int]=1,
                       groups:int=1,
                       batch_norm:Union[bool,str,nn.Module]=True,
                       activation:Optional[Union[str,nn.Module]]="relu6",
                       depth_multiplier:int=1,
                       width_multiplier:float=1.0,
                       ordering:str="cba") -> nn.Sequential:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        dilations: int or sequence of int, default 1,
            spacing between the kernel points of (each) convolutional layer
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, default "relu6",
            name or Module of the activation,
        depth_multiplier: int, default 1,
            multiplier of the number of output channels of the depthwise convolution
        width_multiplier: float, default 1.0,
            multiplier of the number of output channels of the pointwise convolution

        Returns
        -------
        flow: nn.Sequential,
            the sequential flow of consecutive separable convolutions, each followed by bn and relu6
        """
        n_convs = len(out_channels)
        _filter_lengths = list(repeat(filter_lengths, n_convs)) if isinstance(filter_lengths, int) else filter_lengths
        _subsample_lengths = list(repeat(subsample_lengths, n_convs)) if isinstance(subsample_lengths, int) else subsample_lengths
        _dilations = list(repeat(dilations, n_convs)) if isinstance(dilations, int) else dilations
        assert n_convs == len(_filter_lengths) == len(_subsample_lengths) == len(_dilations)
        ic = in_channels
        flow = nn.Sequential()
        for idx, (oc, fl, sl, dl) in enumerate(zip(out_channels, _filter_lengths, _subsample_lengths, _dilations)):
            sc_layer = MobileNetSeparableConv(
                in_channels=ic,
                out_channels=oc,
                kernel_size=fl,
                stride=sl,
                dilation=dl,
                groups=groups,
                batch_norm=batch_norm,
                activation=activation,
                depth_multiplier=depth_multiplier,
                width_multiplier=width_multiplier,
                ordering=ordering,
            )
            flow.add_module(
                f"separable_conv_{idx}",
                sc_layer,
            )
            _, ic, _ = sc_layer.compute_output_shape()
        return flow

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
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        _, _, _seq_len = self.init_convs.compute_output_shape(_seq_len, batch_size)
        for module in self.entry_flow:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        for module in self.middle_flow:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        for module in self.exit_flow:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class InvertedResidual(SizeMixin, nn.Module):
    """

    inverted residual block

    expansion (via pointwise conv) --> depthwise conv --> pointwise conv (without activation) ---> output
        |                                                                                      |
        |----------------------- shortcut (only under certain condition) ----------------------|
    """
    __DEBUG__ = True
    __name__ = "InvertedResidual"

    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 expansion:float,
                 filter_length:int,
                 stride:int,
                 groups:int,
                 batch_norm:Union[bool,str,nn.Module]=True,
                 activation:Optional[Union[str,nn.Module]]="relu6",
                 width_multiplier:float=1.0,
                 use_se:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        expansion: float,
            expansion of the first pointwise convolution
        filter_length: int,
            size (length) of the middle depthwise convolution kernel
        stride: int,
            stride (subsample length) of the middle depthwise convolution
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, default "relu6",
            name or Module of the activation, except for the last pointwise convolution
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__expansion = expansion
        self.__mid_channels = int(round(self.__expansion * self.__in_channels))
        self.__filter_length = filter_length
        self.__stride = stride
        self.__groups = groups
        assert self.__mid_channels % self.__groups == 0, \
            f"expansion ratio (input is {self.__expansion}) makes mid-channels (= {self.__mid_channels}) not divisible by `groups` (={self.__groups})"
        self.__width_multiplier = width_multiplier

        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        if self.__expansion != 1:
            # expand, pointwise
            expansion = Conv_Bn_Activation(
                conv_in_channels,
                self.__mid_channels,
                kernel_size=1,
                stride=1,
                groups=self.__groups,
                batch_norm=batch_norm,
                activation=activation,
                # width_multiplier=width_multiplier,
            )
            self.main_stream.add_module(
                "expansion",
                expansion,
            )
            _, conv_in_channels, _ = expansion.compute_output_shape()
        # depthwise convolution
        dw = Conv_Bn_Activation(
            conv_in_channels,
            conv_in_channels,
            kernel_size=self.__filter_length,
            stride=self.__stride,
            groups=conv_in_channels,
            batch_norm=batch_norm,
            activation=activation,
            # width_multiplier=width_multiplier,
        )
        self.main_stream.add_module(
            "depthwise_conv",
            dw,
        )
        _, conv_in_channels, _ = dw.compute_output_shape()
        # pointwise conv without non-linearity (activation)
        pw_linear = Conv_Bn_Activation(
            conv_in_channels,
            self.__out_channels,
            kernel_size=1,
            stride=1,
            groups=self.__groups,
            bias=False,
            batch_norm=batch_norm,
            activation=None,
            width_multiplier=width_multiplier,
        )
        self.main_stream.add_module(
            "pointwise_conv",
            pw_linear,
        )
        _, self.__out_channels, _ = pw_linear.compute_output_shape()

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
        out = self.main_stream(input)

        if self.__stride == 1 and self.__in_channels == self.__out_channels:
            # NOTE the condition that skip connections are done
            # which is different from ResNet
            out = out + input

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


class MobileNetV2(SizeMixin, nn.Sequential):
    """

    References
    ----------
    1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    3. https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py
    """
    __DEBUG__ = True
    __name__ = "MobileNetV2"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        # init conv(s)
        if isinstance(self.config.init_num_filters, int):
            init_convs = Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_lengths,
                stride=self.config.init_subsample_lengths,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        else:
            init_convs = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                filter_lengths=self.config.init_filter_lengths,
                subsample_lengths=self.config.init_subsample_length,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        self.add_module(
            "init_convs",
            init_convs,
        )

        # inverted residual blocks
        inv_res_cfg = zip(
            self.config.inv_res.expansions,
            self.config.inv_res.out_channels,
            self.config.inv_res.n_blocks,
            self.config.inv_res.strides,
            self.config.inv_res.filter_lengths,
        )
        _, inv_res_in_channels, _ = init_convs.compute_output_shape()
        idx = 0
        for t, c, n, s, k in inv_res_cfg:
            for i in range(n):
                inv_res_blk = InvertedResidual(
                    inv_res_in_channels,
                    out_channels=c,
                    expansion=t,
                    filter_length=k,
                    stride=s if i == 0 else 1,
                    groups=self.config.groups,
                    batch_norm=self.config.batch_norm,
                    activation=self.config.activation,
                    width_multiplier=self.config.width_multiplier,
                )
                self.add_module(
                    f"inv_res_{idx}",
                    inv_res_blk,
                )
                _, inv_res_in_channels, _ = inv_res_blk.compute_output_shape()
                idx += 1

        # final conv(s)
        # no alpha applied to last conv as stated in the paper
        if isinstance(self.config.final_num_filters, int):
            init_convs = Conv_Bn_Activation(
                in_channels=inv_res_in_channels,
                out_channels=self.config.final_num_filters,
                kernel_size=self.config.final_filter_lengths,
                stride=self.config.final_subsample_lengths,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        else:
            init_convs = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.final_num_filters,
                filter_lengths=self.config.final_filter_lengths,
                subsample_lengths=self.config.final_subsample_length,
                groups=self.config.groups,
                batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        self.add_module(
            "final_convs",
            final_convs,
        )

    def forward(self, input:Tensor) -> Tensor:
        """ NOT finished, NOT checked,

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
        """ NOT finished, NOT checked,

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


class MobileNetV3(SizeMixin, nn.Module):
    """

    References
    ----------
    1. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1314-1324).
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py
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
