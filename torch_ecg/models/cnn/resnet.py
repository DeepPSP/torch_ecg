"""
The most frequently used (can serve as baseline) CNN family of physiological signal processing,
whose performance however seems exceeded by newer networks
"""

from copy import deepcopy
from itertools import repeat
from numbers import Real
from typing import NoReturn, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    Activations,
    Conv_Bn_Activation,
    DownSample,
    MultiConv,
    SpaceToDepth,
    ZeroPadding,
    make_attention_layer,
)
from ...utils.misc import dict_to_str, add_docstring
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ResNet",
    "ResNetBasicBlock",
    "ResNetBottleNeck",
]


class ResNetBasicBlock(nn.Module, SizeMixin):
    """
    building blocks for `ResNet`, as implemented in ref. [2] of `ResNet`

    """

    __DEBUG__ = False
    __name__ = "ResNetBasicBlock"
    expansion = 1  # not used

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int,
        subsample_length: int,
        groups: int = 1,
        dilation: int = 1,
        attn: Optional[dict] = None,
        **config,
    ) -> NoReturn:
        """

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
        attn: dict, optional,
            attention mechanism for the neck conv layer,
            if None, no attention mechanism is used,
            keys:
                "name": str, can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
                "pos": int, position of the attention mechanism,
                other keys are specific to the attention mechanism
        config: dict,
            other hyper-parameters, including
            increase channel method, subsample method, dropouts,
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
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        if (
            self.config.increase_channels_method.lower() == "zero_padding"
            and self.__groups != 1
        ):
            raise ValueError(
                "zero padding for increasing channels can not be used with groups != 1"
            )

        self.__attn = attn
        if self.__attn:
            self.__attn = CFG(self.__attn)

        self.__increase_channels = self.__out_channels > self.__in_channels
        self.shortcut = self._make_shortcut_layer()

        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            conv_activation = (
                self.config.activation if i < self.__num_convs - 1 else None
            )
            if self.__attn and self.__attn["pos"] == i:
                self.main_stream.add_module(
                    self.__attn["name"],
                    make_attention_layer(conv_in_channels, **self.__attn),
                )
            self.main_stream.add_module(
                f"cba_{i}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.__kernel_size,
                    stride=(self.__stride if i == 0 else 1),
                    groups=self.__groups,
                    norm=True,
                    activation=conv_activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                    conv_type=self.config.get("conv_type", None),
                ),
            )
            if self.config.get("dropout", 0) > 0 and i < self.__num_convs - 1:
                self.main_stream.add_module(
                    f"dropout_{i}", nn.Dropout(self.config.dropout)
                )
            conv_in_channels = self.__out_channels
        if self.__attn and self.__attn["pos"] == -1:
            self.main_stream.add_module(
                self.__attn["name"],
                make_attention_layer(conv_in_channels, **self.__attn),
            )

        if isinstance(self.config.activation, str):
            self.out_activation = Activations[self.config.activation.lower()](
                **self.config.kw_activation
            )
        else:
            self.out_activation = self.config.activation(**self.config.kw_activation)

        if self.config.get("dropout", 0) > 0:
            self.out_dropout = nn.Dropout(self.config.dropout)
        else:
            self.out_dropout = None

    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
        """ """
        if self.__DEBUG__:
            print(
                f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}"
            )
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                shortcut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    groups=self.__groups,
                    norm=True,
                    mode=self.config.subsample_mode,
                    filt_size=self.config.get("filt_size", 3),  # for blur pool
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = (
                    False if self.config.subsample_mode.lower() != "conv" else True
                )
                shortcut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        norm=batch_norm,
                        mode=self.config.subsample_mode,
                        filt_size=self.config.get("filt_size", 3),  # for blur pool
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels),
                )
        else:
            shortcut = None
        return shortcut

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
        identity = input

        out = self.main_stream(input)

        if self.shortcut is not None:
            identity = self.shortcut(input)

        if identity.shape[-1] < out.shape[-1]:
            # downsampling using "avg" or "max" adopts no padding
            # hence shape[-1] of identity might be smaller by 1 than shape[-1] of out
            diff_sig_len = out.shape[-1] - identity.shape[-1]
            identity = F.pad(
                identity, [diff_sig_len // 2, diff_sig_len - diff_sig_len // 2]
            )

        out += identity
        out = self.out_activation(out)

        if self.out_dropout is not None:
            out = self.out_dropout(out)

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


class ResNetBottleNeck(nn.Module, SizeMixin):
    """

    bottle neck blocks for `ResNet`, as implemented in ref. [2] of `ResNet`,
    as for 1D ECG, should be of the "baby-giant-baby" pattern?

    """

    __DEBUG__ = False
    __name__ = "ResNetBottleNeck"
    expansion = 4
    __DEFAULT_BASE_WIDTH__ = 12 * 4

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int,
        subsample_length: int,
        groups: int = 1,
        dilation: int = 1,
        base_width: int = 12 * 4,
        base_groups: int = 1,
        base_filter_length: int = 1,
        attn: Optional[dict] = None,
        **config,
    ) -> NoReturn:
        """

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
        attn: dict, optional,
            attention mechanism for the neck conv layer,
            if None, no attention mechanism is used,
            keys:
                "name": str, can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
                "pos": int, position of the attention mechanism,
                other keys are specific to the attention mechanism
        config: dict,
            other hyper-parameters, including
            increase channel method, subsample method, dropout,
            activation choices, weight initializer, and short cut patterns, etc.

        """
        super().__init__()
        self.__num_convs = 3
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
        self.expansion = self.config.get("expansion", self.expansion)

        self.__in_channels = in_channels
        # update denominator of computing neck_num_filters by init_num_filters
        self.__DEFAULT_BASE_WIDTH__ = self.config.get(
            "init_num_filters", self.__DEFAULT_BASE_WIDTH__
        )
        neck_num_filters = (
            int(num_filters * (base_width / self.__DEFAULT_BASE_WIDTH__)) * groups
        )
        self.__out_channels = [
            neck_num_filters,
            neck_num_filters,
            num_filters * self.expansion,
        ]
        if self.__DEBUG__:
            print(
                f"__DEFAULT_BASE_WIDTH__ = {self.__DEFAULT_BASE_WIDTH__}, "
                f"in_channels = {in_channels}, num_filters = {num_filters}, "
                f"base_width = {base_width}, neck_num_filters = {neck_num_filters}."
            )
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
        self.__attn = attn
        if self.__attn:
            self.__attn = CFG(self.__attn)

        if (
            self.config.increase_channels_method.lower() == "zero_padding"
            and self.__groups != 1
        ):
            raise ValueError(
                "zero padding for increasing channels can not be used with groups != 1"
            )

        self.__increase_channels = self.__out_channels[-1] > self.__in_channels
        self.shortcut = self._make_shortcut_layer()

        self.main_stream = nn.Sequential()
        conv_names = {
            0: "cba_head",
            1: "cba_neck",
            2: "cba_tail",
        }
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            if self.__attn and self.__attn["pos"] == i:
                self.main_stream.add_module(
                    self.__attn["name"],
                    make_attention_layer(conv_in_channels, **self.__attn),
                )
            conv_activation = (
                self.config.activation if i < self.__num_convs - 1 else None
            )
            conv_out_channels = self.__out_channels[i]
            conv_type = self.config.get("conv_type", None)
            if self.__kernel_size[i] == 1 and conv_type == "separable":
                conv_type = None
            self.main_stream.add_module(
                conv_names[i],
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=self.__kernel_size[i],
                    stride=(self.__stride if i == self.config.subsample_at else 1),
                    groups=self.__groups[i],
                    norm=True,
                    activation=conv_activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                    conv_type=conv_type,
                ),
            )
            if self.config.get("dropout", 0) > 0 and i < self.__num_convs - 1:
                self.main_stream.add_module(
                    f"dropout_{i}", nn.Dropout(self.config.dropout)
                )
            conv_in_channels = conv_out_channels
        if self.__attn and self.__attn["pos"] == -1:
            self.main_stream.add_module(
                self.__attn["name"],
                make_attention_layer(conv_in_channels, **self.__attn),
            )

        if isinstance(self.config.activation, str):
            self.out_activation = Activations[self.config.activation.lower()](
                **self.config.kw_activation
            )
        else:
            self.out_activation = self.config.activation(**self.config.kw_activation)

        if self.config.get("dropout", 0) > 0:
            self.out_dropout = nn.Dropout(self.config.dropout)
        else:
            self.out_dropout = None

    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
        """ """
        if self.__DEBUG__:
            print(
                f"down_scale = {self.__down_scale}, increase_channels = {self.__increase_channels}"
            )
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == "conv":
                shortcut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels[-1],
                    groups=self.__base_groups,
                    norm=True,
                    mode=self.config.subsample_mode,
                    filt_size=self.config.get("filt_size", 3),  # for blur pool
                )
            elif self.config.increase_channels_method.lower() == "zero_padding":
                batch_norm = (
                    False if self.config.subsample_mode.lower() != "conv" else True
                )
                shortcut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        norm=batch_norm,
                        mode=self.config.subsample_mode,
                        filt_size=self.config.get("filt_size", 3),  # for blur pool
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels[-1]),
                )
        else:
            shortcut = None
        return shortcut

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
        identity = input

        out = self.main_stream(input)

        if self.shortcut is not None:
            identity = self.shortcut(input)

        if identity.shape[-1] < out.shape[-1]:
            # downsampling using "avg" or "max" adopts no padding
            # hence shape[-1] of identity might be smaller by 1 than shape[-1] of out
            diff_sig_len = out.shape[-1] - identity.shape[-1]
            identity = F.pad(
                identity, [diff_sig_len // 2, diff_sig_len - diff_sig_len // 2]
            )

        out += identity
        out = self.out_activation(out)

        if self.out_dropout is not None:
            out = self.out_dropout(out)

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


class ResNetMacroBlock(nn.Sequential, SizeMixin):
    """NOT finished, NOT checked,"""

    __DEBUG__ = True
    __name__ = "ResNetMacroBlock"

    def __init__(self) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(self):
        """ """
        raise NotImplementedError

    def compute_output_shape(self):
        """ """
        raise NotImplementedError


class ResNetStem(nn.Sequential, SizeMixin):
    """
    the input stem of ResNet
    """

    __name__ = "ResNetStem"

    def __init__(
        self,
        in_channels: int,
        out_channels: Union[int, Sequence[int]],
        filter_lengths: Union[int, Sequence[int]],
        conv_stride: int,
        pool_size: int,
        pool_stride: int,
        subsample_mode: str = "max",
        groups: int = 1,
        **config,
    ) -> NoReturn:
        f"""

        Parameters
        ----------
        in_channels: int,
            the number of input channels
        out_channels: int or sequence of int,
            the number of output channels
        filter_lengths: int or sequence of int,
            the length of the filter, or equivalently,
            the kernel size(s) of the convolutions
        conv_stride: int,
            the stride of the convolution
        pool_size: int,
            the size of the pooling window
        pool_stride: int,
            the stride of the pooling window
        subsample_mode: str,
            the mode of subsampling, can be one of
            {DownSample.__MODES__},
            or "s2d" (with aliases "space_to_depth", "SpaceToDepth")
        groups: int,
            the number of groups for the convolution
        config: dict,
            the other configs for convolution and pooling

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__filter_lengths = filter_lengths
        if subsample_mode.lower() in ["s2d", "space_to_depth", "SpaceToDepth"]:
            self.add_module(
                "s2d",
                SpaceToDepth(
                    self.__in_channels, self.__out_channels, config.get("block_size", 4)
                ),
            )
            return
        if isinstance(self.__filter_lengths, int):
            self.__filter_lengths = [self.__filter_lengths]
        if isinstance(self.__out_channels, int):
            self.__out_channels = [self.__out_channels]
        assert len(self.__filter_lengths) == len(self.__out_channels)

        conv_in_channels = self.__in_channels
        for idx, fl in enumerate(self.__filter_lengths):
            self.add_module(
                f"conv_{idx}",
                Conv_Bn_Activation(
                    conv_in_channels,
                    self.__out_channels[idx],
                    self.__filter_lengths[idx],
                    stride=conv_stride if idx == 0 else 1,
                    groups=groups,
                    **config,
                ),
            )
            conv_in_channels = self.__out_channels[idx]
        if pool_stride > 1:
            self.add_module(
                "pool",
                DownSample(
                    pool_stride,
                    conv_in_channels,
                    kernel_size=pool_size,
                    groups=groups,
                    padding=(pool_stride - 1) // 2,
                    mode=subsample_mode.lower(),
                    **config,
                ),
            )

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class ResNet(nn.Sequential, SizeMixin):
    """

    References
    ----------
    [1] https://github.com/awni/ecg
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    TODO
    ----
    1. check performances of activations other than "nn.ReLU", especially mish and swish
    2. add functionality of "replace_stride_with_dilation"

    """

    __DEBUG__ = False
    __name__ = "ResNet"
    building_block = ResNetBasicBlock
    __DEFAULT_CONFIG__ = CFG(
        activation="relu",
        kw_activation={"inplace": True},
        kernel_initializer="he_normal",
        kw_initializer={},
        dropouts=0,
    )

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            keyword arguments that have to be set:
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
            stem: dict,
                other parameters that can be set for the input stem
            block: dict,
                other parameters that can be set for the building blocks
            for a full list of configurable parameters, ref. corr. config file

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
        if not isinstance(self.config.get("building_block", None), str):
            # for those architectures that have multiple type of building blocks
            self.building_block = []
            self.additional_kw = []
            for b in self.config.building_block:
                if b.lower() in [
                    "bottleneck",
                    "bottle_neck",
                ]:
                    self.building_block.append(ResNetBottleNeck)
                    self.additional_kw.append(
                        CFG(
                            {
                                k: self.config[k]
                                for k in [
                                    "base_width",
                                    "base_groups",
                                    "base_filter_length",
                                ]
                                if k in self.config.keys()
                            }
                        )
                    )
                else:
                    self.building_block.append(ResNetBasicBlock)
                    self.additional_kw.append(CFG())
            assert isinstance(self.config.block, Sequence) and len(
                self.config.block
            ) == len(self.config.building_block) == len(self.config.num_blocks)
        elif self.config.get("building_block", "").lower() in [
            "bottleneck",
            "bottle_neck",
        ]:
            self.building_block = ResNetBottleNeck
            # additional parameters for bottleneck
            self.additional_kw = CFG(
                {
                    k: self.config[k]
                    for k in [
                        "base_width",
                        "base_groups",
                        "base_filter_length",
                    ]
                    if k in self.config.keys()
                }
            )
        else:
            self.additional_kw = CFG()
        if self.__DEBUG__:
            print(f"additional_kw = {self.additional_kw}")

        self.add_module(
            "input_stem",
            ResNetStem(
                in_channels=self.__in_channels,
                out_channels=self.config.stem.num_filters,
                # bottleneck use "base_groups"
                groups=self.config.get("base_groups", self.config.groups),
                activation=self.config.activation,
                **self.config.stem,
            ),
        )

        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = list(
                repeat(self.config.filter_lengths, len(self.config.num_blocks))
            )
        else:
            self.__filter_lengths = self.config.filter_lengths
        assert len(self.__filter_lengths) == len(self.config.num_blocks), (
            f"`config.filter_lengths` indicates {len(self.__filter_lengths)} macro blocks, "
            f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        )
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(self.config.subsample_lengths, len(self.config.num_blocks))
            )
        else:
            self.__subsample_lengths = self.config.subsample_lengths
        assert len(self.__subsample_lengths) == len(self.config.num_blocks), (
            f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)} macro blocks, "
            f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        )
        self.__num_filters = self.config.get(
            "num_filters",
            [
                (2**i) * self.input_stem.compute_output_shape()[1]
                for i in range(len(self.config.num_blocks))
            ],
        )
        assert len(self.__num_filters) == len(self.config.num_blocks), (
            f"`config.num_filters` indicates {len(self.__num_filters)} macro blocks, "
            f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        )
        if isinstance(self.config.dropouts, Real):
            self.__dropouts = list(
                repeat(self.config.dropouts, len(self.config.num_blocks))
            )
        else:
            self.__dropouts = self.config.dropouts
        assert len(self.__dropouts) == len(self.config.num_blocks), (
            f"`config.dropouts` indicates {len(self.__dropouts)} macro blocks, "
            f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
        )

        # grouped resnet (basic) blocks,
        # number of channels are doubled at the first block of each macro-block
        macro_in_channels = self.input_stem.compute_output_shape()[1]
        for macro_idx, nb in enumerate(self.config.num_blocks):
            macro_num_filters = self.__num_filters[macro_idx]
            macro_filter_lengths = self.__filter_lengths[macro_idx]
            macro_subsample_lengths = self.__subsample_lengths[macro_idx]
            block_in_channels = macro_in_channels
            block_num_filters = macro_num_filters
            if isinstance(macro_filter_lengths, int):
                block_filter_lengths = list(repeat(macro_filter_lengths, nb))
            else:
                block_filter_lengths = macro_filter_lengths
            assert len(block_filter_lengths) == nb, (
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} "
                f"building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            )
            if isinstance(macro_subsample_lengths, int):
                # subsample at the first building block
                block_subsample_lengths = list(repeat(1, nb))
                block_subsample_lengths[0] = macro_subsample_lengths
            else:
                block_subsample_lengths = macro_subsample_lengths
            assert len(block_subsample_lengths) == nb, (
                f"at the {macro_idx}-th macro block, `macro_subsample_lengths` indicates {len(macro_subsample_lengths)} "
                f"building blocks, while `config.num_blocks[{macro_idx}]` indicates {nb}"
            )
            if isinstance(self.building_block, Sequence):
                bb = self.building_block[macro_idx]
                bb_config = self.config.block[macro_idx]
                bb_kw = self.additional_kw[macro_idx]
            else:
                bb = self.building_block
                bb_config = self.config.block
                bb_kw = self.additional_kw
            for block_idx in range(nb):
                self.add_module(
                    f"{bb.__name__}_{macro_idx}_{block_idx}",
                    bb(
                        in_channels=block_in_channels,
                        num_filters=block_num_filters,
                        filter_length=block_filter_lengths[block_idx],
                        subsample_length=block_subsample_lengths[block_idx],
                        groups=self.config.groups,
                        dilation=1,
                        dropout=self.__dropouts[macro_idx],
                        **(bb_kw),
                        **(bb_config),
                    ),
                )
                block_in_channels = block_num_filters * bb.expansion
            macro_in_channels = macro_num_filters * bb.expansion

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels
