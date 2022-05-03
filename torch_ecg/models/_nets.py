"""
basic building blocks, for 1d signal (time series)
"""

from copy import deepcopy
from itertools import repeat
from math import sqrt
from numbers import Real
from typing import Any, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from deprecated import deprecated
from packaging import version
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence

from ..cfg import CFG
from ..utils.misc import dict_to_str, isclass, list_sum, deprecate_kwargs
from ..utils.utils_nn import SizeMixin  # compute_output_shape,
from ..utils.utils_nn import (
    compute_avgpool_output_shape,
    compute_conv_output_shape,
    compute_maxpool_output_shape,
)

__all__ = [
    "Mish",
    "Swish",
    "Hardswish",
    "Initializers",
    "Activations",
    "Bn_Activation",
    "Conv_Bn_Activation",
    "CBA",
    "MultiConv",
    "BranchedConv",
    "SeparableConv",
    "DeformConv",
    "AntiAliasConv",
    "DownSample",
    "BlurPool",
    "BidirectionalLSTM",
    "StackedLSTM",
    # "AML_Attention", "AML_GatedAttention",
    "AttentionWithContext",
    "MultiHeadAttention",
    "SelfAttention",
    "AttentivePooling",
    "ZeroPadding",
    "ZeroPad1d",
    "SeqLin",
    "MLP",
    "NonLocalBlock",
    "SEBlock",
    "GlobalContextBlock",
    "CBAMBlock",
    "BAMBlock",
    "CoordAttention",
    "CRF",
    "ExtendedCRF",
    "SpaceToDepth",
    "MLDecoder",
    "DropPath",
    "make_attention_layer",
]


if version.parse(torch.__version__) >= version.parse("1.5.0"):

    def _true_divide(dividend, divisor):
        return torch.true_divide(dividend, divisor)

else:

    def _true_divide(dividend, divisor):
        return dividend / divisor


# ---------------------------------------------
# activations
try:
    Mish = nn.Mish  # pytorch added in version 1.9
    Mish.__name__ = "Mish"
except Exception:

    class Mish(nn.Module):
        """The Mish activation"""

        __name__ = "Mish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return input * (torch.tanh(F.softplus(input)))


try:
    Swish = nn.SiLU  # pytorch added in version 1.7
    Swish.__name__ = "Swish"
    SiLU = nn.SiLU
except Exception:

    class Swish(nn.Module):
        """The Swish activation"""

        __name__ = "Swish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return input * F.sigmoid(input)

    SiLU = Swish
    SiLU.__name__ = "SiLU"


try:
    Hardswish = nn.Hardswish  # pytorch added in version 1.6
    Hardswish.__name__ = "Hardswish"
except Exception:

    class Hardswish(nn.Module):
        r"""Applies the hardswish function, element-wise, as described in the paper:
        `Searching for MobileNetV3`_.
        .. math::
            \text{Hardswish}(x) = \begin{cases}
                0 & \text{if~} x \le -3, \\
                x & \text{if~} x \ge +3, \\
                x \cdot (x + 3) /6 & \text{otherwise}
            \end{cases}
        .. _`Searching for MobileNetV3`:
            https://arxiv.org/abs/1905.02244
        """
        __name__ = "Hardswish"

        def __init__(self):
            """ """
            super().__init__()

        def forward(self, input: Tensor) -> Tensor:
            """ """
            return torch.clamp(input * (3 + input) / 6, min=0, max=input)


# ---------------------------------------------
# initializers
Initializers = CFG()
Initializers.he_normal = nn.init.kaiming_normal_
Initializers.kaiming_normal = nn.init.kaiming_normal_
Initializers.he_uniform = nn.init.kaiming_uniform_
Initializers.kaiming_uniform = nn.init.kaiming_uniform_
Initializers.xavier_normal = nn.init.xavier_normal_
Initializers.glorot_normal = nn.init.xavier_normal_
Initializers.xavier_uniform = nn.init.xavier_uniform_
Initializers.glorot_uniform = nn.init.xavier_uniform_
Initializers.normal = nn.init.normal_
Initializers.uniform = nn.init.uniform_
Initializers.orthogonal = nn.init.orthogonal_
Initializers.zeros = nn.init.zeros_
Initializers.ones = nn.init.ones_
Initializers.constant = nn.init.constant_


# ---------------------------------------------
# activations
Activations = CFG()
Activations.mish = Mish
Activations.swish = Swish
Activations.hardswish = Hardswish
Activations.hard_swish = Hardswish
Activations.relu = nn.ReLU
Activations.relu6 = nn.ReLU6
Activations.rrelu = nn.RReLU
Activations.leaky = nn.LeakyReLU
Activations.leaky_relu = Activations.leaky
Activations.gelu = nn.GELU
Activations.silu = SiLU
Activations.elu = nn.ELU
Activations.celu = nn.CELU
Activations.selu = nn.SELU
Activations.glu = nn.GLU
Activations.prelu = nn.PReLU
Activations.tanh = nn.Tanh
Activations.hardtanh = nn.Hardtanh
Activations.sigmoid = nn.Sigmoid
Activations.hardsigmoid = nn.Hardsigmoid
Activations.softmax = nn.Softmax
# Activations.linear = None


def get_activation(
    act: Union[str, nn.Module, type(None)], kw_act: Optional[dict] = None
) -> Optional[nn.Module]:
    """

    Parameters
    ----------
    act: str or nn.Module or None,
        name or the class or an instance of the activation, or None
    kw_act: dict, optional,
        keyword arguments for the activation

    Returns
    -------
    nn.Module or None,
        the class of the activation or an instance of the activation, or None
    """
    if act is None:
        return act
    if isclass(act):
        _act = act
    elif isinstance(act, str):
        if act.lower() not in Activations:
            raise ValueError(f"activation `{act}` not supported")
        _act = Activations[act.lower()]
    elif isinstance(act, nn.Module):
        return act
    else:
        raise ValueError(f"activation `{act}` not supported")
    if kw_act is None:
        return _act
    return _act(**kw_act)


# ---------------------------------------------
# normalizations
Normalizations = CFG()
Normalizations.batch_norm = nn.BatchNorm1d
Normalizations.batch_normalization = Normalizations.batch_norm
Normalizations.group_norm = nn.GroupNorm
Normalizations.group_normalization = Normalizations.group_norm
Normalizations.layer_norm = nn.LayerNorm
Normalizations.layer_normalization = Normalizations.layer_norm
Normalizations.instance_norm = nn.InstanceNorm1d
Normalizations.instance_normalization = Normalizations.instance_norm
Normalizations.local_response_norm = nn.LocalResponseNorm
Normalizations.local_response_normalization = Normalizations.local_response_norm
# other normalizations:
# weight normalization
# batch re-normalization
# batch-instance normalization
# switchable normalization
# spectral normalization
# ScaleNorm
# batch group normalization
# ref. https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/
# and ref. Zhou, Xiao-Yun, et al. "Batch Group Normalization." arXiv preprint arXiv:2012.02782 (2020).
# problem: parameters of different normalizations are different


def get_normalization(
    norm: Union[str, nn.Module, type(None)], kw_norm: Optional[dict] = None
) -> Optional[nn.Module]:
    """

    Parameters
    ----------
    norm: str or nn.Module or None,
        name or the class or an instance of the normalization, or None
    kw_norm: dict, optional,
        keyword arguments for the normalization

    Returns
    -------
    nn.Module or None,
        the class of the normalization or an instance of the normalization, or None
    """
    if norm is None:
        return norm
    if isclass(norm):
        _norm = norm
    elif isinstance(norm, str):
        if norm.lower() not in Normalizations:
            raise ValueError(f"normalization `{norm}` not supported")
        _norm = Normalizations.get(norm.lower())
    elif isinstance(norm, nn.Module):
        return norm
    else:
        raise ValueError(f"normalization `{norm}` not supported")
    if kw_norm is None:
        return _norm
    return _norm(**kw_norm)


# ---------------------------------------------

_DEFAULT_CONV_CONFIGS = CFG(
    norm=True,
    activation="relu",
    kw_activation={"inplace": True},
    kernel_initializer="he_normal",
    kw_initializer={},
    ordering="cba",
    conv_type=None,
    width_multiplier=1.0,
)


# ---------------------------------------------
# basic building blocks of CNN
class Bn_Activation(nn.Sequential, SizeMixin):
    """

    batch normalization --> activation

    """

    __name__ = "Bn_Activation"

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        num_features: int,
        batch_norm: Union[bool, str, nn.Module],
        activation: Union[str, nn.Module],
        kw_activation: Optional[dict] = None,
        dropout: float = 0.0,
    ) -> NoReturn:
        """

        Parameters
        ----------
        num_features: int,
            number of features (channels) of the input (and output)
        batch_norm: bool or str or Module, default True,
            (batch) normalization, or other normalizations, e.g. group normalization
            (the name of) the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module,
            name of the activation or an activation `Module`
        kw_activation: dict, optional,
            key word arguments for `activation`
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer at the end of the block

        """
        super().__init__()
        self.__num_features = num_features
        self.__kw_activation = kw_activation or {}
        self.__dropout = dropout
        act_layer = get_activation(activation, kw_activation or {})
        act_name = f"activation_{type(act_layer).__name__}"

        self.add_module(  # TODO: add other normalizations
            "batch_norm",
            nn.BatchNorm1d(num_features),
        )
        self.add_module(
            act_name,
            act_layer,
        )
        if self.__dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.__dropout),
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        output = super().forward(input)
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (batch_size, self.__num_features, seq_len)
        return output_shape


class Conv_Bn_Activation(nn.Sequential, SizeMixin):
    """

    1d convolution --> batch normalization (optional) -- > activation (optional),
    orderings can be adjusted,
    with "same" padding as default padding

    """

    __name__ = "Conv_Bn_Activation"

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
        activation: Optional[Union[str, nn.Module]] = None,
        kernel_initializer: Optional[Union[str, callable]] = None,
        bias: bool = True,
        ordering: str = "cba",
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
        activation: str or Module, optional,
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear", "hardswish", "relu6"
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        ordering: str, default "cba",
            ordering of the layers, case insensitive
        kwargs: dict, optional,
            other key word arguments, including
            `conv_type`, `kw_activation`, `kw_initializer`, `kw_bn`,
            `alpha` (alias `width_multiplier`), etc.

        NOTE that if `padding` is not specified (default None),
        then the actual padding used for the convolutional layer is automatically computed
        to fit the "same" padding (not actually "same" for even kernel sizes)

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
        self.__ordering = ordering.lower()
        assert "c" in self.__ordering

        kw_activation = kwargs.get("kw_activation", {})
        kw_initializer = kwargs.get("kw_initializer", {})
        kw_bn = kwargs.get("kw_bn", {})
        self.__conv_type = kwargs.get("conv_type", None)
        if isinstance(self.__conv_type, str):
            self.__conv_type = self.__conv_type.lower()
        self.__width_multiplier = (
            kwargs.get("width_multiplier", None) or kwargs.get("alpha", None) or 1.0
        )
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert self.__out_channels % self.__groups == 0, (
            f"width_multiplier (input is {self.__width_multiplier}) makes "
            f"`out_channels` (= {self.__out_channels}) "
            f"not divisible by `groups` (= {self.__groups})"
        )

        if self.__conv_type is None:
            conv_layer = nn.Conv1d(
                self.__in_channels,
                self.__out_channels,
                self.__kernel_size,
                self.__stride,
                self.__padding,
                self.__dilation,
                self.__groups,
                bias=self.__bias,
            )
            if kernel_initializer:
                if callable(kernel_initializer):
                    kernel_initializer(conv_layer.weight)
                elif (
                    isinstance(kernel_initializer, str)
                    and kernel_initializer.lower() in Initializers.keys()
                ):
                    Initializers[kernel_initializer.lower()](
                        conv_layer.weight, **kw_initializer
                    )
                else:  # TODO: add more initializers
                    raise ValueError(
                        f"initializer `{kernel_initializer}` not supported"
                    )
        elif self.__conv_type == "separable":
            conv_layer = SeparableConv(
                in_channels=self.__in_channels,
                # out_channels=self.__out_channels,
                out_channels=out_channels,  # note the existence of `width_multiplier` in `kwargs`
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__groups,
                kernel_initializer=kernel_initializer,
                bias=self.__bias,
                **kwargs,
            )
        elif self.__conv_type in [
            "anti_alias",
            "aa",
        ]:
            conv_layer = AntiAliasConv(
                self.__in_channels,
                self.__out_channels,
                self.__kernel_size,
                self.__stride,
                self.__padding,
                self.__dilation,
                self.__groups,
                bias=self.__bias,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"convolution of type {self.__conv_type} not implemented yet!"
            )

        if "b" in self.__ordering and self.__ordering.index(
            "c"
        ) < self.__ordering.index("b"):
            bn_in_channels = self.__out_channels
        else:
            bn_in_channels = self.__in_channels
        if batch_norm:
            if isinstance(batch_norm, bool):
                bn_layer = nn.BatchNorm1d(bn_in_channels, **kw_bn)
            elif isinstance(batch_norm, str):
                if batch_norm.lower() in [
                    "batch_norm",
                    "batch_normalization",
                ]:
                    bn_layer = nn.BatchNorm1d(bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "instance_norm",
                    "instance_normalization",
                ]:
                    bn_layer = nn.InstanceNorm1d(bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "group_norm",
                    "group_normalization",
                ]:
                    bn_layer = nn.GroupNorm(self.__groups, bn_in_channels, **kw_bn)
                elif batch_norm.lower() in [
                    "layer_norm",
                    "layer_normalization",
                ]:
                    bn_layer = nn.LayerNorm(**kw_bn)
                else:
                    raise ValueError(
                        f"normalization method {batch_norm} not supported yet!"
                    )
            else:
                bn_layer = batch_norm
        else:
            bn_layer = None

        act_layer = get_activation(activation, kw_activation)
        if act_layer is not None:
            act_name = f"activation_{type(act_layer).__name__}"

        if self.__ordering in ["cba", "cb", "ca"]:
            self.add_module("conv1d", conv_layer)
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
        elif self.__ordering in ["cab"]:
            self.add_module("conv1d", conv_layer)
            self.add_module(act_name, act_layer)
            self.add_module("batch_norm", bn_layer)
        elif self.__ordering in ["bac", "bc"]:
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
            self.add_module("conv1d", conv_layer)
        elif self.__ordering in ["acb", "ac"]:
            if act_layer:
                self.add_module(act_name, act_layer)
            self.add_module("conv1d", conv_layer)
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
        elif self.__ordering in ["bca"]:
            if bn_layer:
                self.add_module("batch_norm", bn_layer)
            self.add_module("conv1d", conv_layer)
            if act_layer:
                self.add_module(act_name, act_layer)
        else:
            raise ValueError(f"ordering \042{self.__ordering}\042 not supported!")

    def _assign_weights_lead_wise(
        self, other: "Conv_Bn_Activation", indices: Sequence[int]
    ) -> NoReturn:
        """

        assign weights of `self` to `other` according to `indices` in the `lead-wise` manner

        Parameters
        ----------
        other: `Conv_Bn_Activation`,
            the target instance of `Conv_Bn_Activation`
        indices: sequence of int,
            the indices of weights (weight and bias (if not None))
            to be assigned to `other`

        Examples
        --------
        import torch
        from torch_ecg.models._nets import Conv_Bn_Activation
        from torch_ecg.utils.misc import list_sum
        >>> units = 4
        >>> indices = [0, 1, 2, 3, 4, 10]
        >>> out_indices = list_sum([[i * units + j for j in range(units)] for i in indices])
        >>> cba12 = Conv_Bn_Activation(12, 12 * units, 3, 1, groups=12, batch_norm="group_norm")
        >>> cba6 = Conv_Bn_Activation(6, 6*units, 3, 1, groups=6, batch_norm="group_norm")
        >>> (cba12[0].weight.data[out_indices] == cba6[0].weight.data).all()
        tensor(False)
        >>> (cba12[0].bias.data[out_indices] == cba6[0].bias.data).all()
        tensor(False)
        >>> cba12._assign_weights_lead_wise(cba6, indices)
        >>> (cba12[0].weight.data[out_indices] == cba6[0].weight.data).all()
        tensor(True)
        >>> (cba12[0].bias.data[out_indices] == cba6[0].bias.data).all()
        tensor(True)
        >>> tensor12 = torch.zeros(1, 12, 200)
        >>> tensor6 = torch.rand(1, 6, 200)
        >>> tensor12[:, indices, :] = tensor6
        >>> (cba12(tensor12)[:, out_indices, :] == cba6(tensor6)).all()
        tensor(True)

        """
        assert (
            self.conv_type is None and other.conv_type is None
        ), "only normal convolution supported!"
        assert (
            self.in_channels * other.groups == other.in_channels * self.groups
        ), "in_channels should be in proportion to groups"
        assert (
            self.out_channels * other.groups == other.out_channels * self.groups
        ), "out_channels should be in proportion to groups"
        assert (
            len(indices) == other.groups
        ), "`indices` should have length equal to `groups` of `other`"
        assert len(set(indices)) == len(
            indices
        ), "`indices` should not contain duplicates"
        assert not any([isinstance(m, nn.LayerNorm) for m in self]) and not any(
            [isinstance(m, nn.LayerNorm) for m in other]
        ), "Lead-wise assignment of weights is not supported for the existence of `LayerNorm` layers"
        for field in [
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "bias",
            "ordering",
        ]:
            if getattr(self, field) != getattr(other, field):
                raise ValueError(
                    f"{field} of self and other should be the same, "
                    f"but got {getattr(self, field)} and {getattr(other, field)}"
                )
        units = self.out_channels // self.groups
        out_indices = list_sum([[i * units + j for j in range(units)] for i in indices])
        for m, om in zip(self, other):
            if isinstance(
                m, (nn.Conv1d, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)
            ):
                om.weight.data = m.weight.data[out_indices].clone()
                if m.bias is not None:
                    om.bias.data = m.bias.data[out_indices].clone()

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
            the output shape, given `seq_len` and `batch_size`

        """
        if self.__conv_type is None:
            input_shape = [batch_size, self.__in_channels, seq_len]
            output_shape = compute_conv_output_shape(
                input_shape=input_shape,
                num_filters=self.__out_channels,
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                dilation=self.__dilation,
                padding=self.__padding,
                channel_last=False,
            )
        elif self.__conv_type in [
            "separable",
            "anti_alias",
            "aa",
        ]:
            output_shape = self.conv1d.compute_output_shape(seq_len, batch_size)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def out_channels(self) -> int:
        return self.__out_channels

    @property
    def kernel_size(self) -> int:
        return self.__kernel_size

    @property
    def stride(self) -> int:
        return self.__stride

    @property
    def padding(self) -> int:
        return self.__padding

    @property
    def dilation(self) -> int:
        return self.__dilation

    @property
    def groups(self) -> int:
        return self.__groups

    @property
    def bias(self) -> bool:
        return self.__bias

    @property
    def ordering(self) -> str:
        return self.__ordering

    @property
    def conv_type(self) -> Optional[str]:
        return self.__conv_type


# alias
CBA = Conv_Bn_Activation


class MultiConv(nn.Sequential, SizeMixin):
    """

    a sequence (stack) of `Conv_Bn_Activation` blocks,
    perhaps with `Dropout` between the blocks

    """

    __DEBUG__ = False
    __name__ = "MultiConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
        dilations: Union[Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        out_activation: bool = True,
        **config,
    ) -> NoReturn:
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
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        out_activation: bool, default True,
            if True, the last mini-block of `Conv_Bn_Activation`
            will have activation as in `config`, otherwise None;
            if activation is before convolution,
            then `out_activation` refers to the first activation
        config: dict,
            other parameters, including
            type (separable or normal, etc.), width multipliers,
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
            and ordering of convolutions and batch normalizations, activations if applicable

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = list(out_channels)
        self.__num_convs = len(self.__out_channels)
        self.config = deepcopy(_DEFAULT_CONV_CONFIGS)
        self.config.update(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        if isinstance(filter_lengths, int):
            kernel_sizes = list(repeat(filter_lengths, self.__num_convs))
        else:
            kernel_sizes = list(filter_lengths)
        assert len(kernel_sizes) == self.__num_convs

        if isinstance(subsample_lengths, int):
            strides = list(repeat(subsample_lengths, self.__num_convs))
        else:
            strides = list(subsample_lengths)
        assert len(strides) == self.__num_convs

        if isinstance(dropouts, Real):
            _dropouts = list(repeat(dropouts, self.__num_convs))
        else:
            _dropouts = list(dropouts)
        assert len(_dropouts) == self.__num_convs

        if isinstance(dilations, int):
            _dilations = list(repeat(dilations, self.__num_convs))
        else:
            _dilations = list(dilations)
        assert len(_dilations) == self.__num_convs

        __ordering = self.config.ordering.lower()
        if "a" in __ordering and __ordering.index("a") < __ordering.index("c"):
            in_activation = out_activation
            out_activation = True
        else:
            in_activation = True

        conv_in_channels = self.__in_channels
        for idx, (oc, ks, sd, dl, dp) in enumerate(
            zip(self.__out_channels, kernel_sizes, strides, _dilations, _dropouts)
        ):
            activation = self.config.activation
            if idx == 0 and not in_activation:
                activation = None
            if idx == self.__num_convs - 1 and not out_activation:
                activation = None
            self.add_module(
                f"{__ordering}_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=oc,
                    kernel_size=ks,
                    stride=sd,
                    dilation=dl,
                    groups=groups,
                    norm=self.config.get("norm", self.config.get("batch_norm")),
                    activation=activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    ordering=self.config.ordering,
                    conv_type=self.config.conv_type,
                    width_multiplier=self.config.width_multiplier,
                ),
            )
            conv_in_channels = int(oc * self.config.width_multiplier)
            if dp > 0:
                self.add_module(
                    f"dropout_{idx}",
                    nn.Dropout(dp),
                )

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
            the output shape, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if (
                hasattr(module, "__name__")
                and module.__name__ == Conv_Bn_Activation.__name__
            ):
                output_shape = module.compute_output_shape(_seq_len, batch_size)
                _, _, _seq_len = output_shape
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class BranchedConv(nn.Module, SizeMixin):
    """

    branched `MultiConv` blocks

    """

    __DEBUG__ = False
    __name__ = "BranchedConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[Sequence[int]],
        filter_lengths: Union[Sequence[Sequence[int]], Sequence[int], int],
        subsample_lengths: Union[Sequence[Sequence[int]], Sequence[int], int] = 1,
        dilations: Union[Sequence[Sequence[int]], Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[Sequence[float]], Sequence[float], float] = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of sequence of int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        dilations: int or sequence of int, default 1,
            spacing between the kernel points of (each) convolutional layer
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = list(out_channels)
        assert all(
            [isinstance(item, (Sequence, np.ndarray)) for item in self.__out_channels]
        )
        self.__num_branches = len(self.__out_channels)
        self.config = deepcopy(_DEFAULT_CONV_CONFIGS)
        self.config.update(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        if isinstance(filter_lengths, int):
            kernel_sizes = list(repeat(filter_lengths, self.__num_branches))
        else:
            kernel_sizes = list(filter_lengths)
        assert len(kernel_sizes) == self.__num_branches

        if isinstance(subsample_lengths, int):
            strides = list(repeat(subsample_lengths, self.__num_branches))
        else:
            strides = list(subsample_lengths)
        assert len(strides) == self.__num_branches

        if isinstance(dropouts, Real):
            _dropouts = list(repeat(dropouts, self.__num_branches))
        else:
            _dropouts = list(dropouts)
        assert len(_dropouts) == self.__num_branches

        if isinstance(dilations, int):
            _dilations = list(repeat(dilations, self.__num_branches))
        else:
            _dilations = list(dilations)
        assert len(_dilations) == self.__num_branches

        self.branches = nn.ModuleDict()
        for idx, (oc, ks, sd, dl, dp) in enumerate(
            zip(self.__out_channels, kernel_sizes, strides, _dilations, _dropouts)
        ):
            self.branches[f"multi_conv_{idx}"] = MultiConv(
                in_channels=self.__in_channels,
                out_channels=oc,
                filter_lengths=ks,
                subsample_lengths=sd,
                dilations=dl,
                groups=groups,
                dropouts=dp,
                **(self.config),
            )

    def forward(self, input: Tensor) -> List[Tensor]:
        """
        input: of shape (batch_size, n_channels, seq_len)
        """
        out = []
        for idx in range(self.__num_branches):
            out.append(self.branches[f"multi_conv_{idx}"](input))
        return out

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> List[Sequence[Union[int, None]]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shapes: list of sequence,
            list of output shapes of each branch of this `BranchedConv` layer,
            given `seq_len` and `batch_size`

        """
        output_shapes = []
        for idx in range(self.__num_branches):
            branch_output_shape = self.branches[
                f"multi_conv_{idx}"
            ].compute_output_shape(seq_len, batch_size)
            output_shapes.append(branch_output_shape)
        return output_shapes

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class SeparableConv(nn.Sequential, SizeMixin):
    """

    (Super-)Separable Convolution

    References
    ----------
    [1] Kaiser, Lukasz, Aidan N. Gomez, and Francois Chollet. "Depthwise separable convolutions for neural machine translation." arXiv preprint arXiv:1706.03059 (2017).
    [2] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

    """

    __DEBUG__ = False
    __name__ = "SeparableConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        kernel_initializer: Optional[Union[str, callable]] = None,
        bias: bool = True,
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
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output
        kwargs: dict, optional,
            extra parameters, including `depth_multiplier`, `width_multiplier` (alias `alpha`), etc.

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
        kw_initializer = kwargs.get("kw_initializer", {})
        self.__depth_multiplier = kwargs.get("depth_multiplier", 1)
        dc_out_channels = int(self.__in_channels * self.__depth_multiplier)
        assert (
            dc_out_channels % self.__in_channels == 0
        ), f"depth_multiplier (input is {self.__depth_multiplier}) should be positive integers"
        self.__width_multiplier = (
            kwargs.get("width_multiplier", None) or kwargs.get("alpha", None) or 1
        )
        self.__out_channels = int(self.__width_multiplier * self.__out_channels)
        assert self.__out_channels % self.__groups == 0, (
            f"width_multiplier (input is {self.__width_multiplier}) "
            f"makes `out_channels` not divisible by `groups` (= {self.__groups})"
        )

        self.add_module(
            "depthwise_conv",
            nn.Conv1d(
                in_channels=self.__in_channels,
                out_channels=dc_out_channels,
                kernel_size=self.__kernel_size,
                stride=self.__stride,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__in_channels,
                bias=self.__bias,
            ),
        )
        self.add_module(
            "pointwise_conv",
            nn.Conv1d(
                in_channels=dc_out_channels,
                out_channels=self.__out_channels,
                groups=self.__groups,
                bias=self.__bias,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
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

    def forward(self, input: Tensor) -> Tensor:
        """

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
            the output shape, given `seq_len` and `batch_size`

        """
        # depthwise_conv
        output_shape = compute_conv_output_shape(
            input_shape=(batch_size, self.__in_channels, seq_len),
            num_filters=self.__in_channels,
            kernel_size=self.__kernel_size,
            stride=self.__stride,
            padding=self.__padding,
            dilation=self.__dilation,
        )
        # pointwise_conv
        output_shape = compute_conv_output_shape(
            input_shape=output_shape,
            num_filters=self.__out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
        )
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class DeformConv(nn.Module, SizeMixin):
    """

    Deformable Convolution

    References
    ----------
    [1] Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., & Wei, Y. (2017). Deformable convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 764-773).
    [2] Zhu, X., Hu, H., Lin, S., & Dai, J. (2019). Deformable convnets v2: More deformable, better results. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9308-9316).
    [3] https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/deform_conv.py

    """

    __name__ = "DeformConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: bool = False,
    ) -> NoReturn:
        """
        docstring, to write
        """
        raise NotImplementedError

    def forward(self, input: Tensor, offset: Tensor) -> Tensor:
        """
        docstring, to write
        """
        raise NotImplementedError

    def compute_output_shape(
        self,
    ) -> Sequence[Union[int, None]]:
        """
        docstring, to write
        """
        raise NotImplementedError


class DownSample(nn.Sequential, SizeMixin):
    """

    NOTE: this down sampling module allows changement of number of channels,
    via additional convolution, with some abuse of terminology

    the "conv" mode is not simply down "sampling" if `group` != `in_channels`

    """

    __name__ = "DownSample"
    __MODES__ = [
        "max",
        "avg",
        "lp",
        "lse",
        "conv",
        "nearest",
        "area",
        "linear",
        "blur",
    ]

    @deprecate_kwargs([["norm", "batch_norm"]])
    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: Optional[int] = None,
        groups: Optional[int] = None,
        padding: int = 0,
        batch_norm: Union[bool, nn.Module] = False,
        mode: str = "max",
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        down_scale: int,
            scale (in terms of stride) of down sampling
        in_channels: int,
            number of channels of the input
        out_channels: int, optional,
            number of channels of the output
        kernel_size: int, optional,
            kernel size of down sampling,
            if not specified, defaults to `down_scale`,
        groups: int, optional,
            connection pattern (of channels) of the inputs and outputs
        padding: int, default 0,
            zero-padding added to both sides of the input
        batch_norm: bool or Module, default False,
            (batch) normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        mode: str, default "max",
            can be one of `self.__MODES__`

        """
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__kernel_size = kernel_size or down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels or in_channels
        self.__groups = groups or self.__in_channels
        self.__padding = padding

        if self.__mode == "max":
            if self.__in_channels == self.__out_channels:
                down_layer = nn.MaxPool1d(
                    kernel_size=self.__kernel_size,
                    stride=self.__down_scale,
                    padding=self.__padding,
                )
            else:
                down_layer = nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=self.__kernel_size,
                        stride=self.__down_scale,
                        padding=self.__padding,
                    ),
                    nn.Conv1d(
                        self.__in_channels,
                        self.__out_channels,
                        kernel_size=1,
                        groups=self.__groups,
                        bias=False,
                    ),
                )
        elif self.__mode == "avg":
            if self.__in_channels == self.__out_channels:
                down_layer = nn.AvgPool1d(
                    kernel_size=self.__kernel_size,
                    stride=self.__down_scale,
                    padding=self.__padding,
                )
            else:
                down_layer = nn.Sequential(
                    (
                        nn.AvgPool1d(
                            kernel_size=self.__kernel_size,
                            stride=self.__down_scale,
                            padding=self.__padding,
                        ),
                        nn.Conv1d(
                            self.__in_channels,
                            self.__out_channels,
                            kernel_size=1,
                            groups=self.__groups,
                            bias=False,
                        ),
                    )
                )
        elif self.__mode == "conv":
            down_layer = nn.Conv1d(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                kernel_size=1,
                groups=self.__groups,
                bias=False,
                stride=self.__down_scale,
            )
        elif self.__mode == "nearest":
            raise NotImplementedError
        elif self.__mode == "area":
            raise NotImplementedError
        elif self.__mode == "linear":
            raise NotImplementedError
        elif self.__mode == "blur":
            if self.__in_channels == self.__out_channels:
                down_layer = BlurPool(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    **kwargs,
                )
            else:
                down_layer = nn.Sequential(
                    BlurPool(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        **kwargs,
                    ),
                    nn.Conv1d(
                        self.__in_channels,
                        self.__out_channels,
                        kernel_size=1,
                        groups=self.__groups,
                        bias=False,
                    ),
                )
        else:
            down_layer = None
        if down_layer:
            self.add_module(
                "down_sample",
                down_layer,
            )

        if batch_norm:
            bn_layer = (
                nn.BatchNorm1d(self.__out_channels)
                if isinstance(batch_norm, bool)
                else batch_norm(self.__out_channels)
            )
            self.add_module(
                "batch_normalization",
                bn_layer,
            )

    def forward(self, input: Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        if self.__mode in [
            "max",
            "avg",
            "conv",
            "blur",
        ]:
            output = super().forward(input)
        else:
            # align_corners = False if mode in ["nearest", "area"] else True
            output = F.interpolate(
                input=input,
                scale_factor=1 / self.__down_scale,
                mode=self.__mode,
                # align_corners=align_corners,
            )
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        if self.__mode == "conv":
            out_seq_len = compute_conv_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == "max":
            out_seq_len = compute_maxpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__kernel_size,
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == "blur":
            if self.__in_channels == self.__out_channels:
                out_seq_len = self.down_sample.compute_output_shape(
                    seq_len, batch_size
                )[-1]
            else:
                out_seq_len = self.down_sample[0].compute_output_shape(
                    seq_len, batch_size
                )[-1]
        elif self.__mode in [
            "avg",
            "nearest",
            "area",
            "linear",
        ]:
            out_seq_len = compute_avgpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__kernel_size,
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        output_shape = (batch_size, self.__out_channels, out_seq_len)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class ZeroPad1d(nn.ConstantPad1d, SizeMixin):
    """Pads the input tensor boundaries with zero.
    do NOT be confused with `ZeroPadding`, which pads along the channel dimension
    """

    __name__ = "ZeroPad1d"

    def __init__(self, padding: Sequence[int]) -> NoReturn:
        """

        Parameters
        ----------
        padding: 2-sequence of int,
            the padding to be applied to the input tensor

        """
        assert len(padding) == 2 and all(
            [isinstance(i, int) for i in padding]
        ), "padding must be a 2-sequence of int"
        super().__init__(padding, 0.0)


class BlurPool(nn.Module, SizeMixin):
    """

    Blur Pooling, also named as AntiAliasDownsample

    References
    ----------
    1. Zhang, Richard. "Making convolutional networks shift-invariant again." International conference on machine learning. PMLR, 2019.
    2. https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
    3. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/blur_pool.py
    4. https://github.com/kornia/kornia/blob/master/kornia/filters/blur_pool.py

    """

    __DEBUG__ = False
    __name__ = "BlurPool"

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        filt_size: int = 3,
        pad_type: str = "reflect",
        pad_off: int = 0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        down_scale: int,
            scale (in terms of stride) of down sampling
        in_channels: int,
            number of input channels
        filt_size: int, default 3,
            size (length) of the filter
        pad_type: str, default "reflect",
            type of padding, can be "reflect" or "replicate" or "zero"
        pad_off: int, default 0,
            padding offset
        kwargs: keyword arguments,

        """
        super().__init__()
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__filt_size = filt_size
        self.__pad_type = pad_type.lower()
        self.__pad_off = pad_off
        self.__pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.__pad_sizes = [pad_size + pad_off for pad_size in self.__pad_sizes]
        self.__off = int((self.__down_scale - 1) / 2.0)
        if self.__filt_size == 1:
            a = np.array([1.0])
        elif self.__filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.__filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.__filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.__filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.__filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.__filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        # saved and restored in the state_dict, but not trained by the optimizer
        filt = Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer(
            "filt", filt.unsqueeze(0).unsqueeze(0).repeat((self.__in_channels, 1, 1))
        )

        self.pad = self._get_pad_layer()

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        Tensor,
            the blur-pooled output of the input tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        if self.__filt_size == 1:
            if self.__pad_off == 0:
                return input[..., :: self.__down_scale]
            else:
                return self.pad(input)[..., :: self.__down_scale]
        else:
            return F.conv1d(
                self.pad(input),
                self.filt,
                stride=self.__down_scale,
                groups=self.__in_channels,
            )

    def _get_pad_layer(self) -> nn.Module:
        """
        get the padding layer by `self.__pad_type` and `self.__pad_sizes`
        """
        if self.__pad_type in [
            "refl",
            "reflect",
        ]:
            PadLayer = nn.ReflectionPad1d
        elif self.__pad_type in [
            "repl",
            "replicate",
        ]:
            PadLayer = nn.ReplicationPad1d
        elif self.__pad_type == "zero":
            PadLayer = ZeroPad1d
        else:
            print(f"Pad type [{self.__pad_type}] not recognized")
        return PadLayer(self.__pad_sizes)

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
            the output shape, given `seq_len` and `batch_size`

        """
        if self.__filt_size == 1:
            if seq_len is None:
                output_shape = (batch_size, self.__in_channels, None)
            else:
                output_shape = (
                    batch_size,
                    self.__in_channels,
                    (np.sum(self.__pad_sizes) + seq_len - 1) // self.__down_scale + 1,
                )
            return output_shape
        if seq_len is None:
            padded_len = None
        else:
            padded_len = np.sum(self.__pad_sizes) + seq_len
        kernel_size = self.filt.shape[-1]
        output_shape = compute_conv_output_shape(
            input_shape=(batch_size, self.__in_channels, padded_len),
            num_filters=self.__in_channels,
            kernel_size=kernel_size,
            stride=self.__down_scale,
        )
        return output_shape

    def extra_repr(self):
        return "down_scale={}, in_channels={}, filt_size={}, pad_type={}, pad_off={},".format(
            self.__down_scale,
            self.__in_channels,
            self.__filt_size,
            self.__pad_type,
            self.__pad_off,
        )

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class AntiAliasConv(nn.Sequential, SizeMixin):
    """ """

    __DEBUG__ = False
    __name__ = "AntiAliasConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
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
        bias: bool, default True,
            if True, adds a learnable bias to the output
        kwargs: keyword arguments

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = (
            dilation * (kernel_size - 1) // 2 if padding is None else padding
        )
        self.__dilation = dilation
        self.__groups = groups
        self.add_module(
            "conv",
            nn.Conv1d(
                self.__in_channels,
                self.__out_channels,
                self.__kernel_size,
                stride=1,
                padding=self.__padding,
                dilation=self.__dilation,
                groups=self.__groups,
                bias=bias,
            ),
        )
        if self.__stride > 1:
            self.add_module(
                "aa",
                BlurPool(
                    self.__stride,
                    self.__out_channels,
                    **kwargs,
                ),
            )

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = compute_conv_output_shape(
            input_shape=(batch_size, self.__in_channels, seq_len),
            num_filters=self.__out_channels,
            kernel_size=self.__kernel_size,
            stride=1,
            dilation=self.__dilation,
            padding=self.__padding,
            channel_last=False,
        )
        if self.__stride > 1:
            output_shape = self.aa.compute_output_shape(output_shape[-1], batch_size)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class BidirectionalLSTM(nn.Module, SizeMixin):
    """
    from crnn_torch of references.ati_cnn
    """

    __name__ = "BidirectionalLSTM"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        return_sequences: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        input_size: int,
            the number of features in the input
        hidden_size: int,
            the number of features in the hidden state
        num_layers: int, default 1,
            number of lstm layers
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
        return_sequences: bool, default True,
            if True, returns the the full output sequence,
            otherwise the last output in the output sequence
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        self.__output_size = 2 * hidden_size
        self.return_sequence = return_sequences

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bias=bias,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)

        """
        output, _ = self.lstm(input)  # seq_len, batch_size, 2 * hidden_size
        if not self.return_sequence:
            output = output[-1, ...]
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (seq_len, batch_size, self.__output_size)
        return output_shape


class StackedLSTM(nn.Sequential, SizeMixin):
    """finished, checked (no bugs, but correctness is left further to check),

    stacked LSTM, which allows different hidden sizes for each LSTM layer

    NOTE
    ----
    1. `batch_first` is fixed `False`
    2. currently, how to correctly pass the argument `hx` between LSTM layers is not known to me, hence should be careful (and not recommended, use `nn.LSTM` and set `num_layers` instead) to use

    """

    __DEBUG__ = False
    __name__ = "StackedLSTM"

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        bias: Union[Sequence[bool], bool] = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        bidirectional: bool = True,
        return_sequences: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        input_size: int,
            the number of features in the input
        hidden_sizes: sequence of int,
            the number of features in the hidden state of each LSTM layer
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropouts: float or sequence of float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
            or corresponding value in the sequence (except for the last LSTM layer)
        bidirectional: bool, default True,
            if True, each LSTM layer becomes bidirectional
        return_sequences: bool, default True,
            if True, returns the the full output sequence,
            otherwise the last output in the output sequence
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        self.__hidden_sizes = hidden_sizes
        self.num_lstm_layers = len(hidden_sizes)
        l_bias = (
            bias
            if isinstance(bias, Sequence)
            else list(repeat(bias, self.num_lstm_layers))
        )
        self.__dropouts = (
            dropouts
            if isinstance(dropouts, Sequence)
            else list(repeat(dropouts, self.num_lstm_layers))
        )
        self.bidirectional = bidirectional
        self.batch_first = False
        self.return_sequences = return_sequences

        module_name_prefix = "bidirectional_lstm" if bidirectional else "lstm"
        self.__module_names = []
        for idx, (hs, b) in enumerate(zip(hidden_sizes, l_bias)):
            if idx == 0:
                _input_size = input_size
            else:
                _input_size = hidden_sizes[idx - 1]
                if self.bidirectional:
                    _input_size = 2 * _input_size
            self.add_module(
                name=f"{module_name_prefix}_{idx+1}",
                module=nn.LSTM(
                    input_size=_input_size,
                    hidden_size=hs,
                    num_layers=1,
                    bias=b,
                    batch_first=self.batch_first,
                    bidirectional=self.bidirectional,
                ),
            )
            self.__module_names.append("lstm")
            if self.__dropouts[idx] > 0 and idx < self.num_lstm_layers - 1:
                self.add_module(
                    name=f"dropout_{idx+1}",
                    module=nn.Dropout(self.__dropouts[idx]),
                )
                self.__module_names.append("dp")

    def forward(
        self,
        input: Union[Tensor, PackedSequence],
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        keep up with `nn.LSTM.forward`, parameters ref. `nn.LSTM.forward`

        Parameters
        ----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        hx: 2-tuple of Tensor, optional,

        Returns
        -------
        final_output: Tensor,
            of shape (seq_len, batch_size, n_channels) if `return_sequences` is True,
            otherwise of shape (batch_size, n_channels)

        """
        output, _hx = input, hx
        for idx, (name, module) in enumerate(zip(self.__module_names, self)):
            if name == "dp":
                output = module(output)
            elif name == "lstm":
                if idx > 0:
                    _hx = None
                module.flatten_parameters()
                output, _hx = module(output, _hx)
        if self.return_sequences:
            final_output = output  # seq_len, batch_size, n_direction*hidden_size
        else:
            final_output = output[-1, ...]  # batch_size, n_direction*hidden_size
        return final_output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_size = self.__hidden_sizes[-1]
        if self.bidirectional:
            output_size *= 2
        if self.return_sequences:
            output_shape = (seq_len, batch_size, output_size)
        else:
            output_shape = (batch_size, output_size)
        return output_shape


# ---------------------------------------------
# attention mechanisms, from various sources
@deprecated(reason="not checked yet")
class AML_Attention(nn.Module, SizeMixin):
    """NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    References
    ----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L6

    """

    __name__ = "AML_Attention"

    def __init__(self, L: int, D: int, K: int):
        """NOT checked,"""
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

    def forward(self, input):
        """ """
        A = self.attention(input)  # NxK
        return A


@deprecated(reason="not checked yet")
class AML_GatedAttention(nn.Module, SizeMixin):
    """NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    TODO: compare with `nn.MultiheadAttention`

    References
    ----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L72

    """

    __name__ = "AML_GatedAttention"

    def __init__(self, L: int, D: int, K: int):
        """NOT checked,"""
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, input):
        """ """
        A_V = self.attention_V(input)  # NxD
        A_U = self.attention_U(input)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        return A


class AttentionWithContext(nn.Module, SizeMixin):
    """finished, checked (might have bugs),

    from 0236 of CPSC2018 challenge

    """

    __DEBUG__ = False
    __name__ = "AttentionWithContext"

    def __init__(
        self, in_channels: int, bias: bool = True, initializer: str = "glorot_uniform"
    ):
        """finished, checked (might have bugs),

        Parameters
        ----------
        in_channels: int,
            number of channels in the input signal
        bias: bool, default True,
            if True, adds a learnable bias to the output
        initializer: str, default "glorot_uniform",
            weight initializer

        """
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias

        self.W = Parameter(torch.Tensor(in_channels, in_channels))
        if self.__DEBUG__:
            print(f"AttentionWithContext W.shape = {self.W.shape}")
        self.init(self.W)

        if self.bias:
            self.b = Parameter(torch.Tensor(in_channels))
            Initializers.zeros(self.b)
            if self.__DEBUG__:
                print(f"AttentionWithContext b.shape = {self.b.shape}")
            # Initializers["zeros"](self.b)
            self.u = Parameter(torch.Tensor(in_channels))
            Initializers.constant(self.u, 1 / in_channels)
            if self.__DEBUG__:
                print(f"AttentionWithContext u.shape = {self.u.shape}")
            # self.init(self.u)
        else:
            self.register_parameter("b", None)
            self.register_parameter("u", None)

    def compute_mask(self, input: Tensor, input_mask: Optional[Tensor] = None):
        """

        Parameters
        ----------
        to write

        """
        return None

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, seq_len, n_channels)
        mask: Tensor, optional,

        """
        if self.__DEBUG__:
            print(
                f"AttentionWithContext forward: input.shape = {input.shape}, W.shape = {self.W.shape}"
            )

        # linear + activation
        # (batch_size, seq_len, n_channels) x (n_channels, n_channels)
        # -> (batch_size, seq_len, n_channels)
        uit = torch.tensordot(input, self.W, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: uit.shape = {uit.shape}")
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)

        # scores (weights)
        # (batch_size, seq_len, n_channels) x (n_channels,)
        # -> (batch_size, seq_len)
        ait = torch.tensordot(uit, self.u, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: ait.shape = {ait.shape}")

        # softmax along seq_len
        # (batch_size, seq_len)
        a = torch.exp(ait)
        if mask is not None:
            a_masked = a * mask
        else:
            a_masked = a
        a_masked = _true_divide(
            a_masked,
            torch.sum(a_masked, dim=-1, keepdim=True) + torch.finfo(torch.float32).eps,
        )
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: a_masked.shape = {a_masked.shape}")

        # weighted -> sum
        # (batch_size, seq_len, n_channels) x (batch_size, seq_len, 1)
        # -> (batch_size, seq_len, n_channels)
        weighted_input = input * a[..., np.newaxis]
        if self.__DEBUG__:
            print(
                f"AttentionWithContext forward: weighted_input.shape = {weighted_input.shape}"
            )
        output = torch.sum(weighted_input, dim=-1)
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: output.shape = {output.shape}")
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape


class _ScaledDotProductAttention(nn.Module, SizeMixin):
    """
    References
    ----------
    [1] https://github.com/CyberZHG/torch-multi-head-attention

    """

    __DEBUG__ = False
    __name__ = "_ScaledDotProductAttention"

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        all tensors of shape (batch_size, seq_len, features)
        """
        # if self.__DEBUG__:
        #     print(f"query.shape = {query.shape}, key.shape = {key.shape}, value.shape = {value.shape}")
        dk = query.shape[-1]
        scores = query.matmul(key.transpose(-2, -1)) / sqrt(
            dk
        )  # -> (batch_size, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = attention.matmul(value)
        # if self.__DEBUG__:
        #     print(f"scores.shape = {scores.shape}, attention.shape = {attention.shape}, output.shape = {output.shape}")
        return output


class MultiHeadAttention(nn.Module, SizeMixin):
    """

    Multi-head attention.

    References
    ----------
    [1] https://github.com/CyberZHG/torch-multi-head-attention

    """

    __DEBUG__ = False
    __name__ = "MultiHeadAttention"

    def __init__(
        self,
        in_features: int,
        head_num: int,
        bias: bool = True,
        activation: Optional[Union[str, nn.Module]] = "relu",
        **kwargs: Any,
    ):
        """

        Parameters
        ----------
        in_features: int,
            Size of each input sample
        head_num: int,
            Number of heads.
        bias: bool, default True,
            Whether to use the bias term.
        activation: str or Module, default "relu",
            The activation after each linear transformation.
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(
                f"`in_features`({in_features}) should be divisible by `head_num`({head_num})"
            )
        self.in_features = in_features
        self.head_num = head_num
        self.activation = get_activation(activation, kwargs.get("kw_activation", {}))
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        q, k, v are of shape (seq_len, batch_size, features)
        in order to keep accordance with `nn.MultiheadAttention`
        """
        # all (seq_len, batch_size, features) -> (batch_size, seq_len, features)
        q = self.linear_q(q.permute(1, 0, 2))
        k = self.linear_k(k.permute(1, 0, 2))
        v = self.linear_v(v.permute(1, 0, 2))
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = _ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        # shape back from (batch_size, seq_len, features) -> (seq_len, batch_size, features)
        y = y.permute(1, 0, 2)
        return y

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        """ """
        batch_size, seq_len, in_features = x.shape
        sub_dim = in_features // self.head_num
        # if self.__DEBUG__:
        #     print(f"batch_size = {batch_size}, seq_len = {seq_len}, in_features = {in_features}, sub_dim = {sub_dim}")
        reshaped = (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )
        return reshaped

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        """ """
        batch_size, seq_len, in_features = x.shape
        batch_size //= self.head_num
        out_dim = in_features * self.head_num
        # if self.__DEBUG__:
        #     print(f"batch_size = {batch_size}, seq_len = {seq_len}, in_features = {in_features}, out_dim = {out_dim}")
        reshaped = (
            x.reshape(batch_size, self.head_num, seq_len, in_features)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )
        return reshaped

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (seq_len, batch_size, self.in_features * self.head_num)
        return output_shape

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features,
            self.head_num,
            self.bias,
            type(self.activation).__name__,
        )


class SelfAttention(nn.Module, SizeMixin):
    """ """

    __DEBUG__ = False
    __name__ = "SelfAttention"

    def __init__(
        self,
        in_features: int,
        head_num: int,
        dropout: float = 0.0,
        bias: bool = True,
        activation: Optional[Union[str, nn.Module]] = "relu",
        **kwargs: Any,
    ):
        """

        Parameters
        ----------
        in_features: int,
            size of each input sample
        head_num: int,
            number of heads.
        dropout: float, default 0,
            dropout factor for out projection weight of MHA
        bias: bool, default True,
            whether to use the bias term
        activation: str or Module, default "relu",
            The activation after each linear transformation.
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(
                f"`in_features`({in_features}) should be divisible by `head_num`({head_num})"
            )
        self.in_features = in_features
        self.head_num = head_num
        self.dropout = dropout
        self.bias = bias
        self.mha = nn.MultiheadAttention(
            in_features,
            head_num,
            dropout=dropout,
            bias=bias,
        )
        # self.mha = MultiHeadAttention(
        #     in_features, head_num, bias=bias,
        # )

    def forward(self, input: Tensor) -> Tensor:
        """
        input of shape (seq_len, batch_size, features)
        output of shape (seq_len, batch_size, features)
        """
        output, _ = self.mha(input, input, input)
        # output = self.mha(input, input, input)
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (seq_len, batch_size, self.in_features)
        return output_shape


class AttentivePooling(nn.Module, SizeMixin):
    """ """

    __DEBUG__ = False
    __name__ = "AttentivePooling"

    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        activation: Optional[Union[str, nn.Module]] = "tanh",
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels of input Tensor
        mid_channels: int, optional,
            output channels of a intermediate linear layer
        activation: str or Module,
            name of the activation or an activation `Module`
        dropout: float, default 0.2,
            dropout ratio before computing attention scores
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = (mid_channels or self.__in_channels // 2) or 1
        self.__dropout = dropout
        self.activation = get_activation(activation, kwargs.get("kw_activation", {}))

        self.dropout = nn.Dropout(self.__dropout, inplace=False)
        self.mid_linear = nn.Linear(self.__in_channels, self.__mid_channels)
        self.contraction = nn.Linear(self.__mid_channels, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, seq_len, n_channels)

        """
        scores = self.dropout(input)
        scores = self.mid_linear(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.activation(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.contraction(scores)  # -> (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # -> (batch_size, seq_len)
        scores = self.softmax(scores)  # -> (batch_size, seq_len)
        weighted_input = input * (
            scores[..., np.newaxis]
        )  # -> (batch_size, seq_len, n_channels)
        output = weighted_input.sum(1)  # -> (batch_size, n_channels)
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (batch_size, self.__in_channels)
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class ZeroPadding(nn.Module, SizeMixin):
    """
    zero padding for increasing channels,
    degenerates to `identity` if in and out channels are equal

    """

    __name__ = "ZeroPadding"
    __LOC__ = [
        "head",
        "tail",
    ]

    def __init__(
        self, in_channels: int, out_channels: int, loc: str = "head"
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels for the output
        loc: str, default "top", case insensitive,
            padding to the head or the tail channel

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__increase_channels = out_channels - in_channels
        assert self.__increase_channels >= 0
        self.__loc = loc.lower()
        assert self.__loc in self.__LOC__
        # self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        batch_size, _, seq_len = input.shape
        _device = input.device
        if self.__increase_channels > 0:
            output = torch.zeros((batch_size, self.__increase_channels, seq_len))
            output = output.to(device=_device)
            if self.__loc == "head":
                output = torch.cat((output, input), dim=1)
            elif self.__loc == "tail":
                output = torch.cat((input, output), dim=1)
        else:
            output = input
        return output

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
            the output shape, given `seq_len` and `batch_size`

        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape


class SeqLin(nn.Sequential, SizeMixin):
    """
    Sequential linear,
    might be useful in learning non-linear classifying hyper-surfaces

    """

    __DEBUG__ = False
    __name__ = "SeqLin"

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        activation: Union[str, nn.Module] = "relu",
        kernel_initializer: Optional[str] = None,
        bias: bool = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of ouput channels for each linear layer
        activation: str or nn.Module, default "relu",
            name of activation after each linear layer
        kernel_initializer: str, optional,
            name of kernel initializer for `weight` of each linear layer
        bias: bool, default True,
            if True, each linear layer will have a learnable bias vector
        dropouts: float or sequence of float, default 0,
            dropout ratio(s) (if > 0) after each (activation after each) linear layer
        kwargs: dict, optional,
            extra parameters

        TODO
        ----
        Can one have grouped linear layers?

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__num_layers = len(self.__out_channels)
        kw_activation = kwargs.get("kw_activation", {})
        kw_initializer = kwargs.get("kw_initializer", {})
        act_layer = get_activation(activation)
        if not isclass(act_layer):
            raise TypeError("`activation` must be a class or str, not an instance")
        self.__activation = act_layer.__name__
        if kernel_initializer:
            if kernel_initializer.lower() in Initializers.keys():
                self.__kernel_initializer = Initializers[kernel_initializer.lower()]
            else:
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        else:
            self.__kernel_initializer = None
        self.__bias = bias
        if isinstance(dropouts, Real):
            if self.__num_layers > 1:
                self.__dropouts = list(repeat(dropouts, self.__num_layers - 1)) + [0.0]
            else:
                self.__dropouts = [dropouts]
        else:
            self.__dropouts = dropouts
            assert (
                len(self.__dropouts) == self.__num_layers
            ), f"`out_channels` indicates {self.__num_layers} linear layers, while `dropouts` indicates {len(self.__dropouts)}"
        self.__skip_last_activation = kwargs.get("skip_last_activation", False)

        lin_in_channels = self.__in_channels
        for idx in range(self.__num_layers):
            lin_layer = nn.Linear(
                in_features=lin_in_channels,
                out_features=self.__out_channels[idx],
                bias=self.__bias,
            )
            if self.__kernel_initializer:
                self.__kernel_initializer(lin_layer.weight, **kw_initializer)
            self.add_module(
                f"lin_{idx}",
                lin_layer,
            )
            if idx < self.__num_layers - 1 or not self.__skip_last_activation:
                self.add_module(
                    f"act_{idx}",
                    act_layer(**kw_activation),
                )
            if self.__dropouts[idx] > 0:
                self.add_module(
                    f"dropout_{idx}",
                    nn.Dropout(self.__dropouts[idx]),
                )
            lin_in_channels = self.__out_channels[idx]

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels),
            ndim in accordance with `input`

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self,
        seq_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        input_seq: bool = True,
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
        batch_size: int, optional,
            the batch size, can be None
        input_seq: bool, default True,
            if True, the input is a sequence (Tensor of dim 3) of vectors of features,
            otherwise a vector of features (Tensor of dim 2)

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        if input_seq:
            output_shape = (batch_size, seq_len, self.__out_channels[-1])
        else:
            output_shape = (batch_size, self.__out_channels[-1])
        return output_shape

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class MLP(SeqLin):
    """
    multi-layer perceptron,
    alias for sequential linear block

    """

    __DEBUG__ = False
    __name__ = "MLP"

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        activation: str = "relu",
        kernel_initializer: Optional[str] = None,
        bias: bool = True,
        dropouts: Union[float, Sequence[float]] = 0.0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of ouput channels for each linear layer
        activation: str, default "relu",
            name of activation after each linear layer
        kernel_initializer: str, optional,
            name of kernel initializer for `weight` of each linear layer
        bias: bool, default True,
            if True, each linear layer will have a learnable bias vector
        dropouts: float or sequence of float, default 0,
            dropout ratio(s) (if > 0) after each (activation after each) linear layer
        kwargs: dict, optional,
            extra parameters

        """
        super().__init__(
            in_channels,
            out_channels,
            activation,
            kernel_initializer,
            bias,
            dropouts,
            **kwargs,
        )


class NonLocalBlock(nn.Module, SizeMixin):
    """

    Non-local Neural Networks

    References
    ----------
    [1] Wang, Xiaolong, et al. "Non-local neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    [2] https://github.com/AlexHex7/Non-local_pytorch

    """

    __DEBUG__ = False
    __name__ = "NonLocalBlock"
    __MID_LAYERS__ = ["g", "theta", "phi", "W"]

    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        filter_lengths: Union[CFG, int] = 1,
        subsample_length: int = 2,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        mid_channels: int, optional,
            number of output channels for the mid layers ("g", "phi", "theta")
        filter_lengths: dict or int, default 1,
            filter lengths (kernel sizes) for each convolutional layers ("g", "phi", "theta", "W")
        subsample_length: int, default 2,
            subsample length (max pool size) of the "g" and "phi" layers
        config: dict,
            other parameters, including
            batch normalization choices, etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = (mid_channels or self.__in_channels // 2) or 1
        self.__out_channels = self.__in_channels
        if isinstance(filter_lengths, dict):
            assert set(filter_lengths.keys()) == set(self.__MID_LAYERS__)
            self.__kernel_sizes = CFG({k: v for k, v in filter_lengths.items()})
        else:
            self.__kernel_sizes = CFG({k: filter_lengths for k in self.__MID_LAYERS__})
        self.__subsample_length = subsample_length
        self.config = CFG(deepcopy(config))

        self.mid_layers = nn.ModuleDict()
        for k in ["g", "theta", "phi"]:
            self.mid_layers[k] = nn.Sequential()
            self.mid_layers[k].add_module(
                "conv",
                Conv_Bn_Activation(
                    in_channels=self.__in_channels,
                    out_channels=self.__mid_channels,
                    kernel_size=self.__kernel_sizes[k],
                    stride=1,
                    norm=False,
                    activation=None,
                ),
            )
            if self.__subsample_length > 1 and k != "theta":
                self.mid_layers[k].add_module(
                    "max_pool", nn.MaxPool1d(kernel_size=self.__subsample_length)
                )

        self.W = Conv_Bn_Activation(
            in_channels=self.__mid_channels,
            out_channels=self.__out_channels,
            kernel_size=self.__kernel_sizes["W"],
            stride=1,
            norm=self.config.get("norm", self.config.get("batch_norm")),
            activation=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        y: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        g_x = self.mid_layers["g"](x)  # --> batch_size, n_channels, seq_len'
        g_x = g_x.permute(0, 2, 1)  # --> batch_size, seq_len', n_channels

        theta_x = self.mid_layers["theta"](x)  # --> batch_size, n_channels, seq_len
        theta_x = theta_x.permute(0, 2, 1)  # --> batch_size, seq_len, n_channels
        phi_x = self.mid_layers["phi"](x)  # --> batch_size, n_channels, seq_len'
        # (batch_size, seq_len, n_channels) x (batch_size, n_channels, seq_len')
        f = torch.matmul(theta_x, phi_x)  # --> batch_size, seq_len, seq_len'
        f = F.softmax(f, dim=-1)  # --> batch_size, seq_len, seq_len'

        # (batch_size, seq_len, seq_len') x (batch_size, seq_len', n_channels)
        y = torch.matmul(f, g_x)  # --> (batch_size, seq_len, n_channels)
        y = y.permute(0, 2, 1).contiguous()  # --> (batch_size, n_channels, seq_len)
        y = self.W(y)
        y += x
        return y

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class SEBlock(nn.Module, SizeMixin):
    """

    Squeeze-and-Excitation Block

    References
    ----------
    [1] J. Hu, L. Shen, S. Albanie, G. Sun and E. Wu, "Squeeze-and-Excitation Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 8, pp. 2011-2023, 1 Aug. 2020, doi: 10.1109/TPAMI.2019.2913372.
    [2] J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.
    [3] https://github.com/hujie-frank/SENet
    [4] https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

    """

    __DEBUG__ = False
    __name__ = "SEBlock"
    __DEFAULT_CONFIG__ = CFG(
        bias=False, activation="relu", kw_activation={"inplace": True}, dropouts=0.0
    )

    def __init__(self, in_channels: int, reduction: int = 16, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        reduction: int, default 16,
            reduction ratio of mid-channels to `in_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, dropouts, etc.
            for the linear layers
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = in_channels // reduction
        self.__out_channels = in_channels
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            SeqLin(
                in_channels=self.__in_channels,
                out_channels=[self.__mid_channels, self.__out_channels],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                bias=self.config.bias,
                dropouts=self.config.dropouts,
                skip_last_activation=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        batch_size, n_channels, seq_len = input.shape
        y = self.avg_pool(input).squeeze(-1)  # --> batch_size, n_channels
        y = self.fc(y).unsqueeze(-1)  # --> batch_size, n_channels, 1
        # output = input * y.expand_as(input)  # equiv. to the following
        # (batch_size, n_channels, seq_len) x (batch_size, n_channels, 1)
        output = input * y  # --> (batch_size, n_channels, seq_len)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class GEBlock(nn.Module, SizeMixin):
    """

    Gather-excite Network

    References
    ----------
    [1] Hu, J., Shen, L., Albanie, S., Sun, G., & Vedaldi, A. (2018). Gather-excite: Exploiting feature context in convolutional neural networks. Advances in neural information processing systems, 31, 9401-9411.
    [2] https://github.com/hujie-frank/GENet
    [3] https://github.com/BayesWatch/pytorch-GENet
    [4] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/gather_excite.py
    """

    __DEBUG__ = True
    __name__ = "GEBlock"

    def __init__(self) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError


class SKBlock(nn.Module, SizeMixin):
    """

    Selective Kernel Networks

    References
    ----------
    [1] Li, X., Wang, W., Hu, X., & Yang, J. (2019). Selective kernel networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 510-519).
    [2] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/sknet.py

    """

    __DEBUG__ = True
    __name__ = "SKBlock"

    def __init__(self) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError


class GlobalContextBlock(nn.Module, SizeMixin):
    """

    Global Context Block

    References
    ----------
    [1] Cao, Yue, et al. "Gcnet: Non-local networks meet squeeze-excitation networks and beyond." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019.
    [2] https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    [3] entry 0436 of CPSC2019

    """

    __DEBUG__ = False
    __name__ = "GlobalContextBlock"
    __POOLING_TYPES__ = [
        "attn",
        "avg",
    ]
    __FUSION_TYPES__ = [
        "add",
        "mul",
    ]

    def __init__(
        self,
        in_channels: int,
        ratio: int,
        reduction: bool = True,
        pooling_type: str = "attn",
        fusion_types: Sequence[str] = [
            "add",
        ],
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        ratio: int,
            raise or reduction ratio of the mid-channels to `in_channels`
            in the "channel attention" sub-block
        reduction: bool, default True,
            if True, mid-channels would be `in_channels // ratio` (as in `SEBlock`),
            otherwise, mid-channels would be `in_channels * ratio` (might should not be used),
        pooling_type: str, default "attn",
            mode (or type) of subsampling (or pooling) of "spatial attention"
        fusion_types: sequence of str, default ["add",],
            types of fusion of context with the input

        """
        super().__init__()
        assert pooling_type in self.__POOLING_TYPES__
        assert all([f in self.__FUSION_TYPES__ for f in fusion_types])
        assert len(fusion_types) > 0, "at least one fusion should be used"
        self.__in_channels = in_channels
        self.__ratio = ratio
        if reduction:
            self.__mid_channels = in_channels // ratio
        else:
            self.__mid_channels = in_channels * ratio
        self.__pooling_type = pooling_type.lower()
        self.__fusion_types = [item.lower() for item in fusion_types]

        if self.__pooling_type == "attn":
            self.conv_mask = nn.Conv1d(self.__in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if "add" in self.__fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(self.__in_channels, self.__mid_channels, kernel_size=1),
                nn.LayerNorm([self.__mid_channels, 1]),
                Activations["relu"](inplace=True),
                nn.Conv1d(self.__mid_channels, self.__in_channels, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "mul" in self.__fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv1d(self.__in_channels, self.__mid_channels, kernel_size=1),
                nn.LayerNorm([self.__mid_channels, 1]),
                Activations["relu"](inplace=True),
                nn.Conv1d(self.__mid_channels, self.__in_channels, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        context: Tensor,
            of shape (batch_size, n_channels, 1)

        """
        if self.__pooling_type == "attn":
            input_x = x.unsqueeze(1)  # --> (batch_size, 1, n_channels, seq_len)
            context = self.conv_mask(x)  # --> (batch_size, 1, seq_len)
            context = self.softmax(context)  # --> (batch_size, 1, seq_len)
            context = context.unsqueeze(3)  # --> (batch_size, 1, seq_len, 1)
            # matmul: (batch_size, 1, n_channels, seq_len) x (batch_size, 1, seq_len, 1)
            context = torch.matmul(
                input_x, context
            )  # --> (batch_size, 1, n_channels, 1)
            context = context.squeeze(1)  # --> (batch_size, n_channels, 1)
        elif self.__pooling_type == "avg":
            context = self.avg_pool(x)  # --> (batch_size, n_channels, 1)
        return context

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        context = self.spatial_pool(input)  # --> (batch_size, n_channels, 1)
        output = input
        if self.channel_mul_conv is not None:
            channel_mul_term = self.channel_mul_conv(
                context
            )  # --> (batch_size, n_channels, 1)
            channel_mul_term = torch.sigmoid(
                channel_mul_term
            )  # --> (batch_size, n_channels, 1)
            # (batch_size, n_channels, seq_len) x (batch_size, n_channels, 1)
            output = output * channel_mul_term  # --> (batch_size, n_channels, seq_len)
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(
                context
            )  # --> (batch_size, n_channels, 1)
            output = output + channel_add_term  # --> (batch_size, n_channels, seq_len)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class BAMBlock(nn.Module, SizeMixin):
    """

    Bottleneck Attention Module (BMVC2018)

    References
    ----------
    1. Park, Jongchan, et al. "Bam: Bottleneck attention module." arXiv preprint arXiv:1807.06514 (2018).
    2. https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py

    """

    __DEBUG__ = True
    __name__ = "BAMBlock"

    def __init__(
        self,
    ):
        """ """
        raise NotImplementedError


class CBAMBlock(nn.Module, SizeMixin):
    """

    Convolutional Block Attention Module (ECCV2018)

    References
    ----------
    1. Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018.
    2. https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py

    """

    __DEBUG__ = False
    __name__ = "CBAMBlock"
    __POOL_TYPES__ = [
        "avg",
        "max",
        "lp",
        "lse",
    ]

    def __init__(
        self,
        gate_channels: int,
        reduction: int = 16,
        groups: int = 1,
        activation: Union[str, nn.Module] = "relu",
        gate: Union[str, nn.Module] = "sigmoid",
        pool_types: Sequence[str] = [
            "avg",
            "max",
        ],
        no_spatial: bool = False,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        gate_channels: int,
            number of input channels of the gates
        reduction: int, default 16,
            reduction ratio of the channel gate
        groups: int, default 1,
            not used currently, might be used later, for some possible ``grouped channel gate``
        activation: str or Module, default "relu",
            activation after the convolutions in the channel gate
        gate: str or Module, default "sigmoid",
            activation gate of the channel gate
        pool_types: sequence of str, default ["avg", "max",],
            pooling types of the channel gate
        no_spatial: bool, default False,
            if True, spatial gate would be skipped
        kwargs: dict, optional,
            extra parameters, including
            lp_norm_type: float, default 2.0,
                norm type for the possible lp norm pooling in the channel gate
            spatial_conv_kernel_size: int, default 7,
                kernel size of the convolution in the spatial gate
            spatial_conv_bn: bool or str or Module, default "batch_norm",
                normalization of the convolution in the spatial gate

        """
        super().__init__()
        self.__gate_channels = gate_channels
        self.__reduction = reduction
        self.__groups = groups
        self.__pool_types = pool_types
        self.__pool_funcs = {
            "avg": nn.AdaptiveAvgPool1d(1),
            "max": nn.AdaptiveMaxPool1d(1),
            "lp": self._lp_pool,
            "lse": self._lse_pool,
        }
        self.__lp_norm_type = kwargs.get("lp_norm_type", 2.0)
        self.__spatial_conv_kernel_size = kwargs.get("spatial_conv_kernel_size", 7)
        self.__spatial_conv_bn = kwargs.get("spatial_conv_bn", "batch_norm")

        # channel gate
        self.channel_gate_mlp = MLP(
            in_channels=self.__gate_channels,
            out_channels=[
                self.__gate_channels // reduction,
                self.__gate_channels,
            ],
            activation=activation,
            skip_last_activation=True,
        )
        # self.channel_gate_act = nn.Sigmoid()
        if isinstance(gate, str):
            self.channel_gate_act = Activations[gate.lower()]()
        elif isinstance(gate, nn.Module):
            self.channel_gate_act = gate
        else:
            raise ValueError(f"Unsupported gate activation {gate}!")

        # spatial gate
        if no_spatial:
            self.spatial_gate_conv = None
        else:
            self.spatial_gate_conv = Conv_Bn_Activation(
                in_channels=2,
                out_channels=1,
                kernel_size=self.__spatial_conv_kernel_size,
                stride=1,
                # groups=self.__groups,
                norm=self.__spatial_conv_bn,
                activation="sigmoid",
            )

    def _fwd_channel_gate(self, input: Tensor) -> Tensor:
        """

        forward function of the channel gate

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        channel_att_sum = None
        for pool_type in self.__pool_types:
            pool_func = self.__pool_funcs[pool_type]
            channel_att_raw = self.channel_gate_mlp(pool_func(input).flatten(1, -1))
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # scale = torch.sigmoid(channel_att_sum)
        scale = self.channel_gate_act(channel_att_sum)
        output = scale.unsqueeze(-1) * input
        return output

    def _fwd_spatial_gate(self, input: Tensor) -> Tensor:
        """

        forward function of the spatial gate

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        if self.spatial_gate_conv is None:
            return input
        # channel pool, `scale` has n_channels = 2
        scale = torch.cat(
            (input.max(dim=1, keepdim=True)[0], input.mean(dim=1, keepdim=True)), dim=1
        )
        scale = self.spatial_gate_conv(scale)
        output = scale * input
        return output

    def _lp_pool(self, input: Tensor) -> Tensor:
        """

        global power-average pooling over `input`

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        Tensor,
            of shape (batch_size, n_channels, 1)

        """
        return F.lp_pool1d(
            input, norm_type=self.__lp_norm_type, kernel_size=input.shape[-1]
        )

    def _lse_pool(self, input: Tensor) -> Tensor:
        """

        global logsumexp pooling over `input`

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        Tensor,
            of shape (batch_size, n_channels, 1)

        """
        return torch.logsumexp(input, dim=-1)

    def forward(self, input: Tensor) -> Tensor:
        """

        forward function of the `CBAMBlock`,
        first channel gate, then (optional) spatial gate

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        output = self._fwd_spatial_gate(self._fwd_channel_gate(input))
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.__gate_channels, seq_len)


class CoordAttention(nn.Module, SizeMixin):
    """

    Coordinate attention

    References
    ----------
    1. Hou, Qibin, Daquan Zhou, and Jiashi Feng. "Coordinate attention for efficient mobile network design." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
    2. https://github.com/Andrew-Qibin/CoordAttention

    """

    __DEBUG__ = True
    __name__ = "CoordAttention"

    def __init__(
        self,
    ):
        """ """
        raise NotImplementedError


class CRF(nn.Module, SizeMixin):
    """Conditional random field, modified from [1]

    This module implements a conditional random field [2]. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`.

    Args:
        num_tags:
            number of tags.
        batch_first:
            if True, input and output tensors are provided as (batch, seq_len, n_channels),
            otherwise as (seq_len, batch, n_channels)
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.

    References
    ----------
    [1] https://github.com/kmkurn/pytorch-crf
    [2] Lafferty, John, Andrew McCallum, and Fernando CN Pereira. "Conditional random fields: Probabilistic models for segmenting and labeling sequence data." (2001).
    [3] https://repository.upenn.edu/cis_papers/159/
    [4] https://en.wikipedia.org/wiki/Viterbi_algorithm
    [5] https://github.com/s14t284/TorchCRF
    [6] https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
    [7] https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

    """

    __DEBUG__ = True
    __name__ = "CRF"

    def __init__(self, num_tags: int, batch_first: bool = False) -> NoReturn:
        """

        Parameters
        ----------
        num_tags: int,
            number of tags.
        batch_first: bool, default False,
            if True, input and output tensors are provided as (batch, seq_len, num_tags),
            otherwise as (seq_len, batch, num_tags)

        """
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()
        # self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset_parameters(self) -> NoReturn:
        """
        Initialize the transition parameters.

        The parameters will be initialized randomly from
        a uniform distribution between -0.1 and 0.1.

        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__name__}(num_tags={self.num_tags})"

    def neg_log_likelihood(
        self,
        emissions: Tensor,
        tags: torch.LongTensor,
        mask: Optional[torch.ByteTensor] = None,
        reduction: str = "sum",
    ) -> Tensor:
        """
        Compute the negative conditional log likelihood (the loss function)
        of a sequence of tags given emission scores.

        Parameters
        ----------
        emissions: Tensor,
            emission score tensor of shape (seq_len, batch_size, num_tags)
        tags: torch.LongTensor,
            sequence of tags tensor of shape (seq_len, batch_size)
        mask: torch.ByteTensor,
            mask tensor of shape (seq_len, batch_size)
        reduction: str, default "sum",
            specifies the reduction to apply to the output:
            can be one of ``none|sum|mean|token_mean``, case insensitive
            ``none``: no reduction will be applied.
            ``sum``: the output will be summed over batches.
            ``mean``: the output will be averaged over batches.
            ``token_mean``: the output will be averaged over tokens.

        Returns
        -------
        nll: Tensor,
            The negative log likelihood.
            This will have size ``(batch_size,)`` if reduction is ``none``,
            otherwise ``()``.

        """
        self._validate(emissions, tags=tags, mask=mask)
        _device = next(self.parameters()).device
        _reduction = reduction.lower()
        if _reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {_reduction}")
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=_device)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator  # log likelihood
        nll = -llh  # negative log likelihood

        if _reduction == "none":
            pass
        elif _reduction == "sum":
            nll = nll.sum()
        elif _reduction == "mean":
            nll = nll.mean()
        # elif _reduction == "token_mean":
        else:
            nll = nll.sum() / mask.float().sum()
        return nll

    def forward(
        self, emissions: Tensor, mask: Optional[torch.ByteTensor] = None
    ) -> Tensor:
        """
        Find the most likely tag sequence using Viterbi algorithm.

        Parameters
        ----------
        emissions: Tensor,
            emission score tensor of shape (seq_len, batch_size, num_tags)
        mask: torch.ByteTensor
            mask tensor of shape (seq_len, batch_size)

        Returns
        -------
        output: Tensor,
            one hot encoding Tensor of the most likely tag sequence

        """
        self._validate(emissions, mask=mask)
        _device = next(self.parameters()).device
        if mask is None:
            mask = emissions.new_ones(
                emissions.shape[:2], dtype=torch.uint8, device=_device
            )
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        best_tags = Tensor(self._viterbi_decode(emissions, mask)).to(torch.int64)
        output = F.one_hot(best_tags.to(_device), num_classes=self.num_tags).permute(
            1, 0, 2
        )
        return output

    def _validate(
        self,
        emissions: Tensor,
        tags: Optional[torch.LongTensor] = None,
        mask: Optional[torch.ByteTensor] = None,
    ) -> NoReturn:
        """
        check validity of input `Tensor`s
        """
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.shape[2] != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.shape[2]}"
            )

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
                )

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(
        self, emissions: Tensor, tags: torch.LongTensor, mask: torch.ByteTensor
    ) -> Tensor:
        """
        # emissions: (seq_len, batch_size, num_tags)
        # tags: (seq_len, batch_size)
        # mask: (seq_len, batch_size)
        """
        assert emissions.dim() == 3 and tags.dim() == 2
        seq_len, batch_size, num_tags = emissions.shape
        assert emissions.shape[:2] == tags.shape
        assert emissions.shape[2] == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_len, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_len):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.ByteTensor
    ) -> Tensor:
        """
        # emissions: (seq_len, batch_size, num_tags)
        # mask: (seq_len, batch_size)
        """
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].all()

        seq_len, batch_size, num_tags = emissions.shape

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_len):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self, emissions: torch.FloatTensor, mask: torch.ByteTensor
    ) -> List[List[int]]:
        """
        # emissions: (seq_len, batch_size, num_tags)
        # mask: (seq_len, batch_size)
        """
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].all()

        seq_len, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_len):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[: seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        if self.batch_first:
            output_shape = (batch_size, seq_len, self.num_tags)
        else:
            output_shape = (seq_len, batch_size, self.num_tags)
        return output_shape


class ExtendedCRF(nn.Sequential, SizeMixin):
    """
    (possibly) combination of a linear (projection) layer and a `CRF` layer,
    which allows the input size to be unequal to (usually greater than) num_tags,
    as in ref.

    References
    ----------
    [1] https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/crf.py
    [2] https://github.com/tensorflow/addons/blob/master/tensorflow_addons/text/crf.py
    [3] https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/crf.py

    """

    __DEBUG__ = False
    __name__ = "ExtendedCRF"

    def __init__(self, in_channels: int, num_tags: int, bias: bool = True) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        num_tags: int,
            number of tags
        bias: bool, default True,
            if True, adds a learnable bias to the linear (projection) layer

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_tags = num_tags
        self.__bias = bias
        if self.__in_channels != self.__num_tags:
            self.add_module(
                name="proj",
                module=nn.Linear(
                    in_features=self.__in_channels,
                    out_features=self.__num_tags,
                    bias=self.__bias,
                ),
            )
        self.add_module(
            name="crf",
            module=CRF(
                num_tags=self.__num_tags,
                batch_first=True,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, seq_len, n_channels)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, seq_len, n_channels)

        """
        if self.__in_channels != self.__num_tags:
            output = self.proj(input)
        else:
            output = input
        output = output.permute(
            1, 0, 2
        )  # (batch_size, seq_len, n_channels) --> (seq_len, batch_size, n_channels)
        output = self.crf(output)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, seq_len, self.__num_tags)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class SpaceToDepth(nn.Module, SizeMixin):
    """

    References
    ----------
    1. https://github.com/Alibaba-MIIL/TResNet/blob/master/src/models/tresnet_v2/layers/space_to_depth.py

    """

    __DEBUG__ = False
    __name__ = "SpaceToDepth"

    def __init__(
        self, in_channels: int, out_channels: int, block_size: int = 4
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels in the output
        block_size: int, default 4,
            block size of converting from the space dim to depth dim

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.bs = block_size
        if self.__in_channels * self.bs != self.__out_channels:
            self.out_conv = Conv_Bn_Activation(
                in_channels=self.__in_channels * self.bs,
                out_channels=self.__out_channels,
                kernel_size=1,
                stride=1,
            )
        else:
            self.out_conv = None

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x: Tensor,
            of shape (batch, channel, seqlen)

        Returns
        -------
        Tensor,
            of shape (batch, channel', seqlen//bs)

        """
        batch, channel, seqlen = x.shape
        x = x[..., : seqlen // self.bs * self.bs]
        batch, channel, seqlen = x.shape
        x = x.view(batch, channel, seqlen // self.bs, self.bs)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, channel * self.bs, seqlen // self.bs)
        if self.out_conv is not None:
            x = self.out_conv(x)
        return x

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        tuple,
            the output shape, given `seq_len` and `batch_size`

        """
        if seq_len is not None:
            return (batch_size, self.__out_channels, seq_len // self.bs)
        else:
            return (batch_size, self.__out_channels, None)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


@torch.jit.script
class _GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(
        self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor
    ):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module, SizeMixin):
    """

    References
    ----------
    1. https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/src_files/ml_decoder/ml_decoder.py

    """

    __DEBUG__ = False
    __name__ = "MLDecoder"

    def __init__(
        self,
        in_channels: int,
        out_channels: Sequence[int],
        num_of_groups: int = -1,
        decoder_embedding: int = 768,
        zsl: bool = False,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels in the output
        num_of_groups: int, default -1,
            number of groups, if -1, then it defaults to min(100, out_channels)
        decoder_embedding: int, default 768,
            embedding size of the decoder,
            this value determines the size (in terms of n_params) of the whole module
        zsl: bool, default False,
            indicator of zero shot learning

        """
        super().__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > out_channels:
            embed_len_decoder = out_channels

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(in_channels, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            raise NotImplementedError
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        # layer_decode = TransformerDecoderLayerOptimal(
        layer_decode = nn.TransformerDecoderLayer(
            d_model=decoder_embedding,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=decoder_dropout,
        )
        self.decoder = nn.TransformerDecoder(
            layer_decode, num_layers=num_layers_decoder
        )
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = Parameter(Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = Parameter(Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.out_channels = out_channels
            self.decoder.duplicate_factor = int(
                out_channels / embed_len_decoder + 0.999
            )
            self.decoder.duplicate_pooling = Parameter(
                Tensor(
                    embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor
                )
            )
            self.decoder.duplicate_pooling_bias = Parameter(Tensor(out_channels))
        nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = _GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x: Tensor,
            of shape (batch, channel, seqlen)

        Returns
        -------
        Tensor,
            of shape (batch, out_channels)

        """
        embedding_spatial = x.permute(
            0, 2, 1
        )  # (batch, channel, seqlen) -> (batch, seqlen, channel)
        embedding_spatial = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial = F.relu(embedding_spatial, inplace=True)

        batch_size = embedding_spatial.shape[0]
        if self.zsl:
            query_embed = F.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = query_embed.unsqueeze(1).expand(
            -1, batch_size, -1
        )  # no allocation of memory with expand
        h = self.decoder(
            tgt, embedding_spatial.transpose(0, 1)
        )  # (embed_len_decoder, batch, decoder_embedding)
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(
            h.shape[0],
            h.shape[1],
            self.decoder.duplicate_factor,
            device=h.device,
            dtype=h.dtype,
        )
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, : self.decoder.out_channels]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        return logits

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        tuple,
            the output shape, given `seq_len` and `batch_size`

        """
        return (batch_size, self.decoder.out_channels)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class DropPath(nn.Module, SizeMixin):
    """

    References
    ----------
    1. Huang, Gao, et al. "Deep networks with stochastic depth." European conference on computer vision. Springer, Cham, 2016.
    2. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py

    """

    __name__ = "DropPath"

    def __init__(self, p: float = 0.2, inplace: bool = False) -> NoReturn:
        """

        Parameters
        ----------
        p: float, default 0.2,
            drop path probability
        inplace: bool, default False,
            whether to do inplace operation

        """
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        """ """
        return drop_path(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


def drop_path(
    x: Tensor, p: float = 0.2, training: bool = False, inplace: bool = False
) -> Tensor:
    """
    modified from timm.models.layers.drop_path

    Parameters
    ----------
    x: Tensor,
        of shape (batch, *)
    p: float, default 0.2,
        drop path probability
    training: bool, default False,
        whether in training mode
    inplace: bool, default False,
        whether to do inplace operation

    Returns
    -------
    Tensor,
        of shape (batch, *)

    """
    if p == 0.0 or not training:
        return x
    if not inplace:
        x = x.clone()
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    x.div_(keep_prob).mul_(random_tensor)
    return x


def make_attention_layer(in_channels: int, **config: dict) -> nn.Module:
    """

    make attention layer by config

    Parameters
    ----------
    in_channels: int,
        number of channels in the input
    config: dict,
        config of the attention layer

    Returns
    -------
    nn.Module,
        the attention layer

    Examples
    --------
    ```python
    from torch_ecg.model_configs.attn import squeeze_excitation
    from torch_ecg.models._nets import make_attention_layer
    layer = make_attention_layer(in_channels=128, name="se", **squeeze_excitation)
    ```

    """
    key = "name" if "name" in config else "type"
    name = config[key].lower()
    if name in ["se"]:
        return SEBlock(in_channels, **config)
    elif name in ["gc"]:
        return GlobalContextBlock(in_channels, **config)
    elif name in ["nl", "non-local", "nonlocal", "non_local"]:
        return NonLocalBlock(in_channels, **config)
    elif name in [
        "ca",
    ]:
        return CoordAttention(in_channels, **config)
    elif name in [
        "sk",
    ]:
        return SKBlock(in_channels, **config)
    elif name in [
        "ge",
    ]:
        return GEBlock(in_channels, **config)
    elif name in [
        "cbam",
    ]:
        return CBAMBlock(in_channels, **config)
    elif name in [
        "bam",
    ]:
        return BAMBlock(in_channels, **config)
    else:
        try:
            return eval(f"""{config[key]}(in_channels, **config)""")
        except Exception:
            raise ValueError(f"Unknown attention type: {config[key]}")
