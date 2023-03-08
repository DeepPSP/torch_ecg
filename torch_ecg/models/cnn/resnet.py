"""
The most frequently used (can serve as baseline) CNN family of physiological signal processing,
whose performance however seems exceeded by newer networks
"""

import textwrap
from copy import deepcopy
from itertools import repeat
from numbers import Real
from typing import Optional, Sequence, Union, List

import torch.nn.functional as F
from torch import Tensor, nn

from ...cfg import CFG
from ...models._nets import (
    Activations,
    Conv_Bn_Activation,
    DownSample,
    SpaceToDepth,
    ZeroPadding,
    make_attention_layer,
)
from ...utils.misc import add_docstring, CitationMixin
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)


__all__ = [
    "ResNet",
    "ResNetBasicBlock",
    "ResNetBottleNeck",
]


if not hasattr(nn, "Dropout1d"):
    nn.Dropout1d = nn.Dropout  # added in pytorch 1.12


_DEFAULT_BLOCK_CONFIG = {
    "increase_channels_method": "conv",
    "subsample_mode": "conv",
    "activation": "relu",
    "kw_activation": {"inplace": True},
    "kernel_initializer": "he_normal",
    "kw_initializer": {},
    "bias": False,
}


class ResNetBasicBlock(nn.Module, SizeMixin):
    """Building blocks for :class:`ResNet`.

    Parameters
    ----------
    in_channels : int
        Number of features (channels) of the input tensor.
    num_filters : int
        Number of filters for the convolutional layers.
    filter_length : int
        Length (size) of the filter kernels.
    subsample_lengths : int
        Subsample length, including pool size for short cut,
        and stride for the top convolutional layer.
    groups : int, default 1
        Pattern of connections between inputs and outputs.
        For more details, ref. :class:`torch.nn.Conv1d`.
    dilation : int, default 1
        Not used.
    attn : dict, optional
        Attention mechanism for the neck conv layer.
        If is None, no attention mechanism is used.
        keys:

            - name: str,
              can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
            - pos: int,
              position of the attention mechanism.

        Other keys are specific to the attention mechanism.
    config : dict
        Other hyper-parameters, including
        increase channel method, subsample method, dropouts,
        activation choices, weight initializer, and short cut patterns, etc.

    """

    __name__ = "ResNetBasicBlock"
    expansion = 1  # not used
    __DEFAULT_BASE_WIDTH__ = 12 * 4  # not used
    __DEFAULT_CONFIG__ = _DEFAULT_BLOCK_CONFIG.copy()

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
    ) -> None:
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
        self.config = CFG(self.__DEFAULT_CONFIG__.copy())
        self.config.update(config)

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
        dropout_config = self.config.get("dropout", 0)
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
            if i < self.__num_convs - 1:
                if isinstance(dropout_config, dict):
                    if dropout_config["type"] == "1d" and dropout_config["p"] > 0:
                        self.main_stream.add_module(
                            f"dropout_{i}", nn.Dropout1d(dropout_config["p"])
                        )
                    elif dropout_config["type"] is None and dropout_config["p"] > 0:
                        self.main_stream.add_module(
                            f"dropout_{i}", nn.Dropout(dropout_config["p"])
                        )
                elif dropout_config > 0:  # float
                    self.main_stream.add_module(
                        f"dropout_{i}", nn.Dropout(dropout_config)
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

        if isinstance(dropout_config, dict):
            if dropout_config["type"] == "1d" and dropout_config["p"] > 0:
                self.out_dropout = nn.Dropout1d(dropout_config["p"])
            elif dropout_config["type"] is None and dropout_config["p"] > 0:
                self.out_dropout = nn.Dropout(dropout_config["p"])
        elif dropout_config > 0:  # float
            self.out_dropout = nn.Dropout(self.config.dropout)
        else:
            self.out_dropout = nn.Identity()

    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
        """Make shortcut layer for residual connection."""
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
            shortcut = nn.Identity()
        return shortcut

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the block.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        out : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        identity = input

        out = self.main_stream(input)
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
        out = self.out_dropout(out)

        return out

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the block.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            Batch size of the input tensor.

        Returns
        -------
        output_shape : sequenc
            Output shape of the block.

        """
        _seq_len = seq_len
        for module in self.main_stream:
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Identity)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class ResNetBottleNeck(nn.Module, SizeMixin):
    """Bottleneck blocks for :class:`ResNet`.

    As for 1D ECG, should be of the "baby-giant-baby" pattern?

    Parameters
    ----------
    in_channels : int
        Number of features (channels) of the input tensor.
    num_filters : Sequence[int]
        Number of filters for the neck conv layer.
    filter_length : int
        Lengths (sizes) of the filter kernels for the neck conv layer.
    subsample_length : int
        Subsample length, including pool size for short cut,
        and stride for the (top or neck) conv layer.
    groups : int, default 1
        Pattern of connections between inputs and outputs of the neck conv layer.
        For more details, ref. :class:`torch.nn.Conv1d`.
    dilation : int, default 1
        Dilation of the conv layers.
    base_width : numbers.Real, default 12*4
        Number of filters per group for the neck conv layer.
        Usually number of filters of the initial conv layer
        of the whole ResNet model.
    base_groups : int, default 1
        Pattern of connections between inputs and outputs of conv layers at the two ends,
        which should divide `groups`.
    base_filter_length : int, default 1
        Lengths (sizes) of the filter kernels for conv layers at the two ends.
    attn : dict, optional
        Attention mechanism for the neck conv layer.
        If is  None, no attention mechanism is used.
        Keys:

            - name: str,
              can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
            - pos: int,
              position of the attention mechanism.

        Other keys are specific to the attention mechanism.
    config : dict
        Other hyper-parameters, including
        increase channel method, subsample method, dropout,
        activation choices, weight initializer, and short cut patterns, etc.

    """

    __name__ = "ResNetBottleNeck"
    expansion = 4
    __DEFAULT_BASE_WIDTH__ = 12 * 4
    __DEFAULT_CONFIG__ = _DEFAULT_BLOCK_CONFIG.copy()
    __DEFAULT_CONFIG__["subsample_at"] = 0

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int,
        subsample_length: int,
        groups: int = 1,
        dilation: int = 1,
        base_width: Real = 12 * 4,
        base_groups: int = 1,
        base_filter_length: int = 1,
        attn: Optional[dict] = None,
        **config,
    ) -> None:
        super().__init__()
        self.__num_convs = 3
        self.config = CFG(self.__DEFAULT_CONFIG__.copy())
        self.config.update(config)
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
        dropout_config = self.config.get("dropout", 0)
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
            if i < self.__num_convs - 1:
                if isinstance(dropout_config, dict):
                    if dropout_config["type"] == "1d" and dropout_config["p"] > 0:
                        self.main_stream.add_module(
                            f"dropout_{i}", nn.Dropout1d(dropout_config["p"])
                        )
                    elif dropout_config["type"] is None and dropout_config["p"] > 0:
                        self.main_stream.add_module(
                            f"dropout_{i}", nn.Dropout(dropout_config["p"])
                        )
                elif dropout_config > 0:  # float
                    self.main_stream.add_module(
                        f"dropout_{i}", nn.Dropout(dropout_config)
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

        if isinstance(dropout_config, dict):
            if dropout_config["type"] == "1d" and dropout_config["p"] > 0:
                self.out_dropout = nn.Dropout1d(dropout_config["p"])
            elif dropout_config["type"] is None and dropout_config["p"] > 0:
                self.out_dropout = nn.Dropout(dropout_config["p"])
        elif dropout_config > 0:  # float
            self.out_dropout = nn.Dropout(self.config.dropout)
        else:
            self.out_dropout = nn.Identity()

    def _make_shortcut_layer(self) -> Union[nn.Module, None]:
        """Make shortcut layer for residual connection."""
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
            shortcut = nn.Identity()
        return shortcut

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the block.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        out : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        identity = input

        out = self.main_stream(input)
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
        out = self.out_dropout(out)

        return out

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the block.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            Batch size of the input tensor.

        Returns
        -------
        output_shape : sequenc
            Output shape of the block.

        """
        _seq_len = seq_len
        for module in self.main_stream:
            if isinstance(module, (nn.Identity, nn.Dropout, nn.Dropout1d)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class ResNetStem(nn.Sequential, SizeMixin):
    """The input stem of ResNet.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int or Sequence[int]
        Number of output channels.
    filter_lengths : int or Sequence[int]
        Length of the filter, or equivalently,
        kernel size(s) of the convolutions.
    conv_stride : int
        Stride of the convolution.
    pool_size : int
        Size of the pooling window.
    pool_stride : int
        Stride of the pooling window.
    subsample_mode : str
        Mode of subsampling, can be one of
        {:class:`DownSample`.__MODES__},
        or "s2d" (with aliases "space_to_depth", "SpaceToDepth").
    groups : int
        Number of groups for the convolution
    config : dict
        Other configs for convolution and pooling.

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
    ) -> None:
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

    @add_docstring(
        textwrap.indent(compute_sequential_output_shape_docstring, " " * 4),
        mode="append",
    )
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the stem."""
        return compute_sequential_output_shape(self, seq_len, batch_size)


class ResNet(nn.Sequential, SizeMixin, CitationMixin):
    """ResNet model.

    The ResNet model is used for ECG classification by a team from
    the Stanford University [1]_, from which the application of
    deep learning to ECG analysis becomes more widely known and accepted.

    This implementation bases much on the torchvision implementation [2]_,
    although which is for image tasks.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal tensor.
    config : dict
        Hyper-parameters of the Module, ref. corr. config file.
        keyword arguments that must be set:

            - bias: bool,
              if True, each convolution will have a bias term.
            - num_blocks: sequence of int,
              number of building blocks in each macro block.
            - filter_lengths: int or sequence of int or sequence of sequences of int,
              filter length(s) (kernel size(s)) of the convolutions,
              with granularity to the whole network, to each macro block,
              or to each building block.
            - subsample_lengths: int or sequence of int or sequence of sequences of int,
              subsampling length(s) (ratio(s)) of all blocks,
              with granularity to the whole network, to each macro block,
              or to each building block,
              the former 2 subsample at the first building block.
            - groups: int,
              connection pattern (of channels) of the inputs and outputs.
            - stem: dict,
              other parameters that can be set for the input stem.
            - block: dict,
              other parameters that can be set for the building blocks.

        For a full list of configurable parameters, ref. corr. config file.

    References
    ----------
    .. [1] https://github.com/awni/ecg
    .. [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    TODO
    ----
    1. Check performances of activations other than :class:`~torch.nn.ReLU`,
       especially :class:`~torch.nn.Mish`, etc.
    2. Add functionality of `replace_stride_with_dilation`.

    """

    __name__ = "ResNet"
    building_block = ResNetBasicBlock
    __DEFAULT_CONFIG__ = CFG(
        activation="relu",
        kw_activation={"inplace": True},
        kernel_initializer="he_normal",
        kw_initializer={},
        base_groups=1,
        dropouts=0,
    )

    def __init__(self, in_channels: int, **config) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
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

        stem_config = CFG(self.config.stem)
        stem_config.pop("num_filters", None)
        self.add_module(
            "input_stem",
            ResNetStem(
                in_channels=self.__in_channels,
                out_channels=self.config.stem.num_filters,
                # bottleneck use "base_groups"
                groups=self.config.get("base_groups", self.config.groups),
                activation=self.config.activation,
                **stem_config,
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
        if isinstance(self.config.dropouts, (Real, dict)):
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
                dropout = bb_config.pop("dropout", self.__dropouts[macro_idx])
                self.add_module(
                    f"{bb.__name__}_{macro_idx}_{block_idx}",
                    bb(
                        in_channels=block_in_channels,
                        num_filters=block_num_filters,
                        filter_length=block_filter_lengths[block_idx],
                        subsample_length=block_subsample_lengths[block_idx],
                        groups=self.config.groups,
                        dilation=1,
                        dropout=dropout,
                        **(bb_kw),
                        **(bb_config),
                    ),
                )
                block_in_channels = block_num_filters * bb.expansion
            macro_in_channels = macro_num_filters * bb.expansion

    @add_docstring(
        textwrap.indent(compute_sequential_output_shape_docstring, " " * 4),
        mode="append",
    )
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model."""
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def doi(self) -> List[str]:
        return list(
            set(
                self.config.get("doi", [])
                + [
                    "10.1109/cvpr.2016.90",
                    "10.1038/s41591-018-0268-3",
                    "10.1038/s41467-020-15432-4",
                    "10.1088/1361-6579/ac6aa3",
                ]
            )
        )
