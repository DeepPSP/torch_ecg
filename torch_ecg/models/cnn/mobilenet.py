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
from typing import Any, NoReturn, Optional, Sequence, Union
from numbers import Real

import torch
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
    Initializers,
    MultiConv,
    make_attention_layer,
)
from ...utils.misc import dict_to_str, deprecate_kwargs, add_docstring
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
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


class MobileNetSeparableConv(nn.Sequential, SizeMixin):
    """

    similar to `_nets.SeparableConv`,
    the difference is that there are normalization and activation between depthwise conv and pointwise conv

    """

    __DEBUG__ = True
    __name__ = "MobileNetSeparableConv"

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[Union[str, nn.Module]] = "relu6",
        kernel_initializer: Optional[Union[str, callable]] = None,
        bias: bool = True,
        depth_multiplier: int = 1,
        width_multiplier: float = 1.0,
        **kwargs: Any,
    ) -> NoReturn:
        """

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
        assert (
            dc_out_channels % self.__in_channels == 0
        ), f"depth_multiplier (input is {self.__depth_multiplier}) should be positive integers"
        self.__width_multiplier = width_multiplier
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert (
            self.__out_channels % self.__groups == 0
        ), f"width_multiplier (input is {self.__width_multiplier}) makes `out_channels` not divisible by `groups` (= {self.__groups})"

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
                norm=batch_norm,
                activation=activation,
                ordering=ordering,
            ),
        )
        self.add_module(
            "pointwise_conv",
            Conv_Bn_Activation(
                in_channels=dc_out_channels,
                out_channels=self.__out_channels,
                groups=self.__groups,
                bias=self.__bias,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                norm=batch_norm,
                activation=activation,
                ordering=ordering,
            ),
        )

        if kernel_initializer:
            if callable(kernel_initializer):
                for module in self:
                    kernel_initializer(module.weight)
            elif (
                isinstance(kernel_initializer, str)
                and kernel_initializer.lower() in Initializers.keys()
            ):
                for module in self:
                    Initializers[kernel_initializer.lower()](
                        module.weight, **kw_initializer
                    )
            else:  # TODO: add more initializers
                raise ValueError(f"initializer `{kernel_initializer}` not supported")

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class MobileNetV1(nn.Sequential, SizeMixin):
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

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """

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
                norm=self.config.get("norm", self.config.get("batch_norm")),
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
                norm=self.config.batch_norm,
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
            **self.config.entry_flow,
        )
        self.add_module("entry_flow", entry_flow)

        _, middle_flow_in_channels, _ = entry_flow[-1].compute_output_shape()
        middle_flow = self._generate_flow(
            in_channels=middle_flow_in_channels,
            depth_multiplier=self.config.depth_multiplier,
            width_multiplier=self.config.width_multiplier,
            **self.config.middle_flow,
        )
        self.add_module("middle_flow", middle_flow)

        _, exit_flow_in_channels, _ = middle_flow[-1].compute_output_shape()
        exit_flow = self._generate_flow(
            in_channels=exit_flow_in_channels,
            depth_multiplier=self.config.depth_multiplier,
            width_multiplier=self.config.width_multiplier,
            **self.config.exit_flow,
        )
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    def _generate_flow(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
        dilations: Union[Sequence[int], int] = 1,
        groups: int = 1,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[Union[str, nn.Module]] = "relu6",
        depth_multiplier: int = 1,
        width_multiplier: float = 1.0,
        ordering: str = "cba",
    ) -> nn.Sequential:
        """

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
        _filter_lengths = (
            list(repeat(filter_lengths, n_convs))
            if isinstance(filter_lengths, int)
            else filter_lengths
        )
        _subsample_lengths = (
            list(repeat(subsample_lengths, n_convs))
            if isinstance(subsample_lengths, int)
            else subsample_lengths
        )
        _dilations = (
            list(repeat(dilations, n_convs))
            if isinstance(dilations, int)
            else dilations
        )
        assert (
            n_convs
            == len(_filter_lengths)
            == len(_subsample_lengths)
            == len(_dilations)
        )
        ic = in_channels
        flow = nn.Sequential()
        for idx, (oc, fl, sl, dl) in enumerate(
            zip(out_channels, _filter_lengths, _subsample_lengths, _dilations)
        ):
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

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

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

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class InvertedResidual(nn.Module, SizeMixin):
    """

    inverted residual block

    expansion (via pointwise conv) --> depthwise conv --> pointwise conv (without activation) ---> output
        |                                                                                      |
        |----------------------- shortcut (only under certain condition) ----------------------|

    """

    __DEBUG__ = True
    __name__ = "InvertedResidual"

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float,
        filter_length: int,
        stride: int,
        groups: int = 1,
        dilation: int = 1,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[Union[str, nn.Module]] = "relu6",
        width_multiplier: float = 1.0,
        attn: Optional[CFG] = None,
    ) -> NoReturn:
        """

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
        dilation: int, default 1,
            spacing between the kernel points of (each) convolutional layer
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, default "relu6",
            name or Module of the activation, except for the last pointwise convolution
        width_multiplier: float, default 1.0,
            multiplier of the number of output channels of the pointwise convolution
        attn: dict, optional,
            attention mechanism for the neck conv layer,
            if None, no attention mechanism is used,
            keys:
                "name": str, can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
                "pos": int, position of the attention mechanism,
                other keys are specific to the attention mechanism

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__expansion = expansion
        self.__mid_channels = int(round(self.__expansion * self.__in_channels))
        self.__filter_length = filter_length
        self.__stride = stride
        self.__groups = groups
        assert self.__mid_channels % self.__groups == 0, (
            f"expansion ratio (input is {self.__expansion}) makes mid-channels "
            f"(= {self.__mid_channels}) not divisible by `groups` (={self.__groups})"
        )
        self.__dilation = dilation
        if self.__dilation > 1:
            self.__stride = 1
        self.__width_multiplier = width_multiplier
        self.__attn = attn
        if self.__attn:
            self.__attn = CFG(self.__attn)

        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        current_pos = 0
        current_pos = self._add_attn_layer_if_needed(conv_in_channels, current_pos)
        if self.__expansion != 1:
            # expand, pointwise
            expansion = Conv_Bn_Activation(
                conv_in_channels,
                self.__mid_channels,
                kernel_size=1,
                stride=1,
                groups=self.__groups,
                norm=batch_norm,
                activation=activation,
                # width_multiplier=width_multiplier,
            )
            self.main_stream.add_module(
                "expansion",
                expansion,
            )
            current_pos += 1
            current_pos = self._add_attn_layer_if_needed(conv_in_channels, current_pos)
            _, conv_in_channels, _ = expansion.compute_output_shape()
        # depthwise convolution
        dw = Conv_Bn_Activation(
            conv_in_channels,
            conv_in_channels,
            kernel_size=self.__filter_length,
            stride=self.__stride,
            groups=conv_in_channels,
            dilation=self.__dilation,
            norm=batch_norm,
            activation=activation,
            # width_multiplier=width_multiplier,
        )
        self.main_stream.add_module(
            "depthwise_conv",
            dw,
        )
        current_pos += 1
        current_pos = self._add_attn_layer_if_needed(conv_in_channels, current_pos)
        _, conv_in_channels, _ = dw.compute_output_shape()
        # pointwise conv without non-linearity (activation)
        pw_linear = Conv_Bn_Activation(
            conv_in_channels,
            self.__out_channels,
            kernel_size=1,
            stride=1,
            groups=self.__groups,
            bias=False,
            norm=batch_norm,
            activation=None,
            width_multiplier=self.__width_multiplier,
        )
        self.main_stream.add_module(
            "pointwise_conv",
            pw_linear,
        )
        current_pos += 1
        current_pos = self._add_attn_layer_if_needed(conv_in_channels, current_pos)
        _, self.__out_channels, _ = pw_linear.compute_output_shape()

    def _add_attn_layer_if_needed(self, in_channels: int, current_pos: int) -> NoReturn:
        """

        add attention layer at the position specified by `self.__attn.pos`

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        current_pos: int,
            position of the current layer

        """
        if self.__attn and self.__attn["pos"] == current_pos:
            self.main_stream.add_module(
                self.__attn["name"],
                make_attention_layer(in_channels, **self.__attn),
            )
            return current_pos + 1
        return current_pos

    def forward(self, input: Tensor) -> Tensor:
        """

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

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

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
            if isinstance(module, nn.Dropout):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class MobileNetV2(nn.Sequential, SizeMixin):
    """

    References
    ----------
    1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    3. https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

    """

    __DEBUG__ = True
    __name__ = "MobileNetV2"

    def __init__(self, in_channels: int, **config: CFG) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            keyword arguments that have to be set:
            groups: int,
                number of groups in the pointwise convolutional layer(s)
            norm: bool or str or Module,
                normalization layer
            activation: str or Module,
                activation layer
            bias: bool,
                whether to use bias in the convolutional layer(s)
            width_multiplier: float,
                multiplier of the number of output channels of the pointwise convolution
            stem: CFG,
                config of the stem block, with the following keys:
                num_filters: int or sequence of int,
                    number of filters in the first convolutional layer(s)
                filter_lengths: int or sequence of int,
                    filter lengths (kernel sizes) in the first convolutional layer(s)
                subsample_lengths: int or sequence of int,
                    subsample lengths (strides) in the first convolutional layer(s)
            inv_res: CFG,
                config of the inverted residual blocks, with the following keys:
                expansions: sequence of int,
                    expansion ratios of the inverted residual blocks
                out_channels: sequence of int,
                    number of output channels in each block
                n_blocks: sequence of int,
                    number of inverted residual blocks
                strides: sequence of int,
                    strides of the inverted residual blocks
                filter_lengths: sequence of int,
                    filter lengths (kernel sizes) in each block
            exit_flow: CFG,
                config of the exit flow blocks, with the following keys:
                num_filters: int or sequence of int,
                    number of filters in the final convolutional layer(s)
                filter_lengths: int or sequence of int,
                    filter lengths (kernel sizes) in the final convolutional layer(s)
                subsample_lengths: int or sequence of int,
                    subsample lengths (strides) in the final convolutional layer(s)

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        # stem
        if isinstance(self.config.stem.num_filters, int):
            stem = Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.stem.num_filters,
                kernel_size=self.config.stem.filter_lengths,
                stride=self.config.stem.subsample_lengths,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        else:
            stem = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.stem.num_filters,
                filter_lengths=self.config.stem.filter_lengths,
                subsample_lengths=self.config.stem.subsample_lengths,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
            )
        self.add_module(
            "stem",
            stem,
        )

        # inverted residual blocks
        inv_res_cfg = zip(
            self.config.inv_res.expansions,
            self.config.inv_res.out_channels,
            self.config.inv_res.n_blocks,
            self.config.inv_res.strides,
            self.config.inv_res.filter_lengths,
        )
        _, inv_res_in_channels, _ = stem.compute_output_shape()
        idx = 0
        for t, c, n, s, k in inv_res_cfg:
            # t: expansion
            # c: output channels
            # n: number of blocks
            # s: stride
            # k: kernel size
            for i in range(n):
                inv_res_blk = InvertedResidual(
                    inv_res_in_channels,
                    out_channels=c,
                    expansion=t,
                    filter_length=k,
                    stride=s if i == 0 else 1,
                    groups=self.config.groups,
                    norm=self.config.get("norm", self.config.get("batch_norm")),
                    activation=self.config.activation,
                    width_multiplier=self.config.width_multiplier,
                )
                self.add_module(
                    f"inv_res_{idx}",
                    inv_res_blk,
                )
                _, inv_res_in_channels, _ = inv_res_blk.compute_output_shape()
                idx += 1

        # exit_flow
        # no alpha applied to last conv as stated in the paper
        if isinstance(self.config.exit_flow.num_filters, int):
            exit_flow = Conv_Bn_Activation(
                in_channels=inv_res_in_channels,
                out_channels=self.config.exit_flow.num_filters,
                kernel_size=self.config.exit_flow.filter_lengths,
                stride=self.config.exit_flow.subsample_lengths,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        else:
            exit_flow = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.exit_flow.num_filters,
                filter_lengths=self.config.exit_flow.filter_lengths,
                subsample_lengths=self.config.exit_flow.subsample_length,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class InvertedResidualBlock(nn.Sequential, SizeMixin):
    """ """

    __DEBUG__ = True
    __name__ = "InvertedResidualBlock"

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        in_channels: int,
        n_blocks: int,
        expansion: Union[float, Sequence[float]],
        filter_length: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = None,
        groups: int = 1,
        dilation: Union[int, Sequence[int]] = 1,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[
            Union[str, nn.Module, Sequence[Union[str, nn.Module]]]
        ] = "relu",
        width_multiplier: Union[float, Sequence[float]] = 1.0,
        out_channels: Union[int, Sequence[int]] = None,
        attn: Optional[Union[CFG, Sequence[CFG]]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of input channels
        n_blocks: int,
            number of inverted residual blocks
        expansion: float or sequence of floats,
            expansion ratios of the inverted residual blocks
        filter_length: int or sequence of ints,
            filter length of the depthwise convolution in the inverted residual blocks
        stride: int or sequence of ints, optional,
            stride of the depthwise convolution in the inverted residual blocks,
            defaults to `[2] + [1] * (n_blocks - 1)`
        groups: int, default 1,
            number of groups in the expansion and pointwise convolution in the inverted residual blocks,
        dilation: int or sequence of ints, default 1,
            dilation of the depthwise convolution in the inverted residual blocks
        batch_norm: bool or str or nn.Module, default True,
            normalization layer to use, defaults to batch normalization
        activation: str or nn.Module or sequence of str or nn.Module, default "relu",
            activation function to use
        width_multiplier: float or sequence of floats, default 1.0,
            width multiplier of the inverted residual blocks
        out_channels: int or sequence of ints, optional,
            number of output channels of the inverted residual blocks,
            defaults to `2 * in_channels`
        attn: CFG or sequence of CFG, optional,
            config of attention layer to use, defaults to None

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__n_blocks = n_blocks
        self.__expansion = expansion
        if isinstance(expansion, Real):
            self.__expansion = list(repeat(expansion, self.n_blocks))
        else:
            self.__expansion = expansion
        assert (
            len(self.__expansion) == self.n_blocks
        ), f"expansion must be an integer or a sequence of length {self.n_blocks}"
        if isinstance(filter_length, int):
            self.__filter_length = list(repeat(filter_length, self.n_blocks))
        else:
            self.__filter_length = filter_length
        assert (
            len(self.__filter_length) == self.n_blocks
        ), f"filter_length must be an integer or a sequence of length {self.n_blocks}"
        if stride is None:
            self.__stride = [2] + list(repeat(1, self.n_blocks - 1))
        elif isinstance(stride, int):
            self.__stride = list(repeat(stride, self.n_blocks))
        else:
            self.__stride = stride
        assert (
            len(self.__stride) == self.n_blocks
        ), f"stride must be an integer or a sequence of length {self.n_blocks}"
        self.__groups = groups
        if isinstance(dilation, int):
            self.__dilation = list(repeat(dilation, self.n_blocks))
        else:
            self.__dilation = dilation
        assert (
            len(self.__dilation) == self.n_blocks
        ), f"dilation must be an integer or a sequence of length {self.n_blocks}"
        self.__batch_norm = batch_norm
        if isinstance(activation, (str, nn.Module)):
            self.__activation = [deepcopy(activation) for _ in range(self.n_blocks)]
        else:
            self.__activation = activation
        assert (
            len(self.__activation) == self.n_blocks
        ), f"activation must be a string or Module or a sequence of length {self.n_blocks}"
        if isinstance(width_multiplier, float):
            self.__width_multiplier = list(repeat(width_multiplier, self.n_blocks))
        else:
            self.__width_multiplier = width_multiplier
        assert (
            len(self.__width_multiplier) == self.n_blocks
        ), f"width_multiplier must be a float or a sequence of length {self.n_blocks}"
        if out_channels is None:
            self.__out_channels = list(repeat(2 * in_channels, self.n_blocks))
        elif isinstance(out_channels, int):
            self.__out_channels = list(repeat(out_channels, self.n_blocks))
        else:
            self.__out_channels = out_channels
        assert (
            len(self.__out_channels) == self.n_blocks
        ), f"out_channels must be an integer or a sequence of length {self.n_blocks}"
        if attn is None or isinstance(attn, CFG):
            self.__attn = [deepcopy(attn) for _ in range(self.n_blocks)]
        else:
            self.__attn = attn
        assert (
            len(self.__attn) == self.n_blocks
        ), f"attn must be a CFG or a sequence of length {self.n_blocks}"

        ivt_res_in_channels = self.__in_channels
        for idx, exp in enumerate(self.__expansion):
            self.add_module(
                f"inv_res_{idx}",
                InvertedResidual(
                    in_channels=ivt_res_in_channels,
                    out_channels=self.__out_channels[idx],
                    expansion=exp,
                    filter_length=self.__filter_length[idx],
                    stride=self.__stride[idx] if idx == 0 else 1,
                    dilation=self.__dilation[idx],
                    groups=self.__groups,
                    norm=self.__batch_norm,
                    activation=self.__activation[idx],
                    width_multiplier=self.__width_multiplier[idx],
                    attn=self.__attn[idx],
                ),
            )
            ivt_res_in_channels = self.__out_channels[idx]

    @property
    def n_blocks(self) -> int:
        return self.__n_blocks

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class MobileNetV3_STEM(nn.Sequential, SizeMixin):
    """ """

    __DEBUG__ = True
    __name__ = "MobileNetV3_STEM"

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        in_channels: int,
        groups: int = 1,
        bias: bool = True,
        batch_norm: Union[bool, str, nn.Module] = True,
        activation: Optional[
            Union[str, nn.Module, Sequence[Union[str, nn.Module]]]
        ] = "relu",
        width_multiplier: Union[float, Sequence[float]] = 1.0,
        **config: CFG,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        groups: int, default 1,
            number of groups in the expansion and pointwise convolution
        bias: bool,
            whether to use bias in the convolutional layer(s)
        batch_norm: bool or str or nn.Module, default True,
            normalization layer to use, defaults to batch normalization
        activation: str or nn.Module or sequence of str or nn.Module, default "relu",
            activation function to use
        width_multiplier: float or sequence of floats, default 1.0,
            width multiplier of the inverted residual blocks
        config: CFG,
            config of the stem block, with the following keys:
            num_filters: int or sequence of int,
                number of filters in the first convolutional layer(s)
            filter_lengths: int or sequence of int,
                filter lengths (kernel sizes) in the first convolutional layer(s)
            subsample_lengths: int or sequence of int,
                subsample lengths (strides) in the first convolutional layer(s)

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        if isinstance(self.config.num_filters, int):
            out_channels = self.config.num_filters
        else:
            out_channels = self.config.num_filters[0]
        self.add_module(
            "init_conv",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=out_channels,
                kernel_size=self.config.filter_lengths,
                stride=self.config.subsample_lengths,
                groups=groups,
                norm=batch_norm,
                activation=activation,
                bias=bias,
                width_multiplier=width_multiplier,
            ),
        )
        in_channels = out_channels
        if not isinstance(self.config.num_filters, int):
            for idx, out_channels in enumerate(self.config.num_filters[1:]):
                inv_res = InvertedResidual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=1.0,
                    filter_length=self.config.filter_lengths,
                    stride=1,
                    groups=groups,
                    dilation=1,
                    norm=batch_norm,
                    activation=activation,
                )
                self.add_module(f"inv_res_{idx}", inv_res)
                _, in_channels, _ = inv_res.compute_output_shape()

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class MobileNetV3(nn.Sequential, SizeMixin):
    """

    References
    ----------
    1. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1314-1324).
    2. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

    """

    __DEBUG__ = True
    __name__ = "MobileNetV3"

    def __init__(self, in_channels: int, **config: CFG) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            keyword arguments that have to be set:
            groups: int,
                number of groups in the convolutional layer(s) other than depthwise convolutions
            norm: bool or str or Module,
                normalization layer
            bias: bool,
                whether to use bias in the convolutional layer(s)
            width_multiplier: float,
                multiplier of the number of output channels of the pointwise convolution
            stem: CFG,
                config of the stem block, with the following keys:
                num_filters: int or sequence of int,
                    number of filters in the first convolutional layer(s)
                filter_lengths: int or sequence of int,
                    filter lengths (kernel sizes) in the first convolutional layer(s)
                subsample_lengths: int or sequence of int,
                    subsample lengths (strides) in the first convolutional layer(s)
            inv_res: CFG,
                config of the inverted residual blocks, with the following keys:
                in_channels: sequence of int,
                    number of input channels
                n_blocks: sequence of int,
                    number of inverted residual blocks
                expansions: sequence of floats or sequence of sequence of floats,
                    expansion ratios of the inverted residual blocks
                filter_lengths: sequence of ints or sequence of sequence of ints,
                    filter length of the depthwise convolution in the inverted residual blocks
                stride: sequence of ints or sequence of sequence of ints, optional,
                    stride of the depthwise convolution in the inverted residual blocks,
                    defaults to `[2] + [1] * (n_blocks - 1)`
                groups: int, default 1,
                    number of groups in the expansion and pointwise convolution in the inverted residual blocks,
                dilation: sequence of ints or sequence of sequence of ints, optional,
                    dilation of the depthwise convolution in the inverted residual blocks
                batch_norm: bool or str or nn.Module, default True,
                    normalization layer to use, defaults to batch normalization
                activation: str or nn.Module or sequence of str or nn.Module
                    activation function to use
                width_multiplier: float or sequence of floats, default 1.0,
                    width multiplier of the inverted residual blocks
                out_channels: sequence of ints or sequence of sequence of int, optional,
                    number of output channels of the inverted residual blocks,
                    defaults to `2 * in_channels`
                attn: sequence of CFG or sequence of sequence of CFG, optional,
                    config of attention layer to use, defaults to None
            exit_flow: CFG,
                config of the exit flow blocks, with the following keys:
                num_filters: int or sequence of int,
                    number of filters in the final convolutional layer(s)
                filter_lengths: int or sequence of int,
                    filter lengths (kernel sizes) in the final convolutional layer(s)
                subsample_lengths: int or sequence of int,
                    subsample lengths (strides) in the final convolutional layer(s)

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        # stem
        self.add_module(
            "stem",
            MobileNetV3_STEM(
                in_channels=in_channels,
                groups=self.config.groups,
                bias=self.config.bias,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                width_multiplier=self.config.width_multiplier,
                **self.config.stem,
            ),
        )

        # inverted residual blocks
        _, inv_res_in_channels, _ = self.stem.compute_output_shape()
        strides = self.config.inv_res.get(
            "strides", list(repeat(None, len(self.config.inv_res.n_blocks)))
        )
        out_channels = self.config.inv_res.get(
            "out_channels", list(repeat(None, len(self.config.inv_res.n_blocks)))
        )
        for idx, n_blocks in enumerate(self.config.inv_res.n_blocks):
            block = InvertedResidualBlock(
                in_channels=inv_res_in_channels,
                n_blocks=n_blocks,
                expansion=self.config.inv_res.expansions[idx],
                filter_length=self.config.inv_res.filter_lengths[idx],
                stride=strides[idx],
                groups=self.config.inv_res.groups,
                dilation=self.config.inv_res.dilations[idx],
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.inv_res.activations[idx],
                bias=self.config.bias,
                width_multiplier=self.config.width_multiplier,
                out_channels=out_channels[idx],
                attn=self.config.inv_res.attns[idx],
            )
            self.add_module(
                f"block_{idx}",
                block,
            )
            _, inv_res_in_channels, _ = block.compute_output_shape()

        # exit_flow
        # no alpha applied to last conv as stated in the paper
        if isinstance(self.config.exit_flow.num_filters, int):
            exit_flow = Conv_Bn_Activation(
                in_channels=inv_res_in_channels,
                out_channels=self.config.exit_flow.num_filters,
                kernel_size=self.config.exit_flow.filter_lengths,
                stride=self.config.exit_flow.subsample_lengths,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        else:
            exit_flow = MultiConv(
                in_channels=self.__in_channels,
                out_channels=self.config.exit_flow.num_filters,
                filter_lengths=self.config.exit_flow.filter_lengths,
                subsample_lengths=self.config.exit_flow.subsample_length,
                groups=self.config.groups,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                activation=self.config.activation,
                bias=self.config.bias,
                width_multiplier=max(1.0, self.config.width_multiplier),
            )
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)
