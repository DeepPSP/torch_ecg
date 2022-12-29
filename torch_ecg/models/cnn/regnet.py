"""
Designing Network Design Spaces
"""

import math
import warnings
from collections import Counter
from itertools import repeat
from numbers import Real
from typing import Optional, Sequence, Union, List

import torch
from torch import nn

from ...cfg import CFG
from ...models._nets import (  # noqa: F401
    Activations,
    Initializers,
    Conv_Bn_Activation,
    DownSample,
    MultiConv,
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
from .resnet import ResNetBottleNeck, ResNetBasicBlock


class AnyStage(nn.Sequential, SizeMixin):
    """
    AnyStage of RegNet

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
    num_blocks: int,
        number of blocks in the stage
    group_width: int,
        group width for the bottleneck block
    stage_index: int,
        the index of the stage
    block_config: dict,
        (optional) configs for the blocks, including
        "block": str or nn.Module,
            the block class, can be one of
            "bottleneck", "bottle_neck", ResNetBottleNeck, ...,
        "expansion": int,
            the expansion factor for the bottleneck block
        "increase_channels_method": str,
            the method to increase the number of channels,
            can be one of {"conv", "zero_padding"}
        "subsample_mode": str,
            the mode of subsampling, can be one of
            {DownSample.__MODES__},
        "activation": str or nn.Module,
            the activation function, can be one of
            {list(Activations)},
        "kw_activation": dict,
            the keyword arguments for the activation function
        "kernel_initializer": str,
            the kernel initializer, can be one of
            {list(Initializers)},
        "kw_initializer": dict,
            the keyword arguments for the kernel initializer
        "bias": bool,
            whether to use bias in the convolution
        "dilation": int,
            the dilation factor for the convolution
        "base_width": int,
            number of filters per group for the neck conv layer
            usually number of filters of the initial conv layer of the whole ResNet
        "base_groups": int,
            pattern of connections between inputs and outputs of conv layers at the two ends,
            should divide `groups`
        "base_filter_length": int,
            lengths (sizes) of the filter kernels for conv layers at the two ends
        "attn": dict,
            attention mechanism for the neck conv layer,
            if None, no attention mechanism is used,
            keys:
                "name": str, can be "se", "gc", "nl" (alias "nonlocal", "non-local"), etc.
                "pos": int, position of the attention mechanism,
                other keys are specific to the attention mechanism

    """

    __name__ = "AnyStage"
    __DEFAULT_BLOCK_CONFIG__ = {
        "block": "bottleneck",
        "expansion": 1,
        "increase_channels_method": "conv",
        "subsample_mode": "conv",
        "activation": "relu",
        "kw_activation": {"inplace": True},
        "kernel_initializer": "he_normal",
        "kw_initializer": {},
        "bias": False,
    }

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        filter_length: int,
        subsample_length: int,
        num_blocks: int,
        group_width: int,
        stage_index: int,
        **block_config,
    ) -> None:
        """ """
        super().__init__()

        self.block_config = CFG(self.__DEFAULT_BLOCK_CONFIG__.copy())
        self.block_config.update(block_config)

        block_cls = self.get_building_block_cls(self.block_config)

        # adjust num_filters based on group_width
        if num_filters % group_width != 0:
            _num_filters = num_filters // group_width * group_width
            if _num_filters < 0.9 * num_filters:
                _num_filters += group_width
            num_filters = _num_filters
        groups = num_filters // group_width
        base_width = block_cls.__DEFAULT_BASE_WIDTH__ / groups

        block_in_channels = in_channels
        for i in range(num_blocks):
            block = block_cls(
                in_channels=block_in_channels,
                num_filters=num_filters,
                filter_length=filter_length,
                subsample_length=subsample_length if i == 0 else 1,
                groups=groups,
                base_width=base_width,
                **self.block_config,
            )
            block_in_channels = block.compute_output_shape()[1]
            self.add_module(f"block_{stage_index}_{i}", block)

    @staticmethod
    def get_building_block_cls(config: CFG) -> nn.Module:
        """ """
        block_cls = config.get("block")
        if isinstance(block_cls, str):
            if block_cls.lower() in ["bottleneck", "bottle_neck"]:
                block_cls = ResNetBottleNeck
            else:
                block_cls = ResNetBasicBlock
        return block_cls

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class RegNetStem(nn.Sequential, SizeMixin):
    """
    the input stem of RegNet

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
        """ """
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


class RegNet(nn.Sequential, SizeMixin, CitationMixin):
    """
    References
    ----------
    [1] https://arxiv.org/abs/2003.13678
    [2] https://github.com/pytorch/vision/blob/master/torchvision/models/regnet.py

    """

    __name__ = "RegNet"
    __DEFAULT_CONFIG__ = dict(
        activation="relu",
        kw_activation={"inplace": True},
        kernel_initializer="he_normal",
        kw_initializer={},
        base_groups=1,
        dropouts=0,
    )

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(self.__DEFAULT_CONFIG__.copy())
        self.config.update(config)

        stem_config = CFG(self.config.stem)
        stem_config.pop("num_filters", None)
        self.add_module(
            "input_stem",
            RegNetStem(
                in_channels=self.__in_channels,
                out_channels=self.config.stem.num_filters,
                groups=self.config.base_groups,
                activation=self.config.activation,
                **stem_config,
            ),
        )

        stage_configs = self._get_stage_configs()
        in_channels = self.input_stem.compute_output_shape()[1]
        for idx, stage_config in enumerate(stage_configs):
            stage_block = AnyStage(in_channels=in_channels, **stage_config)
            self.add_module(f"stage_{idx}", stage_block)
            in_channels = stage_block.compute_output_shape()[1]

    def _get_stage_configs(self) -> List[CFG]:
        """ """
        stage_configs = []
        if self.config.get("num_blocks", None) is not None:
            if isinstance(self.config.filter_lengths, int):
                self.__filter_lengths = list(
                    repeat(self.config.filter_lengths, len(self.config.num_blocks))
                )
            else:
                self.__filter_lengths = self.config.filter_lengths
            assert len(self.__filter_lengths) == len(self.config.num_blocks), (
                f"`config.filter_lengths` indicates {len(self.__filter_lengths)} stages, "
                f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
            )
            if isinstance(self.config.subsample_lengths, int):
                self.__subsample_lengths = list(
                    repeat(self.config.subsample_lengths, len(self.config.num_blocks))
                )
            else:
                self.__subsample_lengths = self.config.subsample_lengths
            assert len(self.__subsample_lengths) == len(self.config.num_blocks), (
                f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)} stages, "
                f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
            )
            self.__num_filters = self.config.num_filters
            assert len(self.__num_filters) == len(self.config.num_blocks), (
                f"`config.num_filters` indicates {len(self.__num_filters)} stages, "
                f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
            )
            if isinstance(self.config.dropouts, Real):
                self.__dropouts = list(
                    repeat(self.config.dropouts, len(self.config.num_blocks))
                )
            else:
                self.__dropouts = self.config.dropouts
            assert len(self.__dropouts) == len(self.config.num_blocks), (
                f"`config.dropouts` indicates {len(self.__dropouts)} stages, "
                f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
            )
            if isinstance(self.config.group_widths, int):
                self.__group_widths = list(
                    repeat(self.config.group_widths, len(self.config.num_blocks))
                )
            else:
                self.__group_widths = self.config.group_widths
            assert len(self.__group_widths) == len(self.config.num_blocks), (
                f"`config.group_widths` indicates {len(self.__group_widths)} stages, "
                f"while `config.num_blocks` indicates {len(self.config.num_blocks)}"
            )

            block_config = CFG(self.config.get("block", {}))
            block_config.pop("dropout", None)

            stage_configs = [
                CFG(
                    dict(
                        num_blocks=self.config.num_blocks[idx],
                        num_filters=self.__num_filters[idx],
                        filter_length=self.__filter_lengths[idx],
                        subsample_length=self.__subsample_lengths[idx],
                        dropout=self.__dropouts[idx],
                        group_width=self.__group_widths[idx],
                        stage_index=idx,
                        **block_config,
                    )
                )
                for idx in range(len(self.config.num_blocks))
            ]
            return stage_configs

        if self.config.get("num_filters", None) is not None:
            warnings.warn(
                "num_filters are computed from config.w_a, config.w_0, config.w_m, "
                "if config.num_blocks is not provided. "
                "This may not be the intended behavior.",
                RuntimeWarning,
            )
            assert {"w_a", "w_0", "w_m", "tot_blocks"}.issubset(
                set(self.config.keys())
            ), (
                "If `num_blocks` is not provided, then `w_a`, `w_0`, `w_m`, "
                "and `tot_blocks` must be provided."
            )
        QUANT = 8
        if (
            self.config.w_a < 0
            or self.config.w_0 <= 0
            or self.config.w_m <= 1
            or self.config.w_0 % QUANT != 0
        ):
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = (
            torch.arange(self.config.tot_blocks) * self.config.w_a + self.config.w_0
        )
        block_capacity = torch.round(
            torch.log(widths_cont / self.config.w_0) / math.log(self.config.w_m)
        )
        block_widths = (
            (
                torch.round(
                    torch.divide(
                        self.config.w_0 * torch.pow(self.config.w_m, block_capacity),
                        QUANT,
                    )
                )
                * QUANT
            )
            .int()
            .tolist()
        )
        counter = Counter(block_widths)
        num_stages = len(counter)

        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = list(repeat(self.config.filter_lengths, num_stages))
        else:
            self.__filter_lengths = self.config.filter_lengths
        assert len(self.__filter_lengths) == num_stages, (
            f"`config.filter_lengths` indicates {len(self.__filter_lengths)} stages, "
            f"while there are {num_stages} computed from "
            "`config.w_a`, `config.w_0`, `config.w_m`, `config.tot_blocks`"
        )
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(self.config.subsample_lengths, num_stages)
            )
        else:
            self.__subsample_lengths = self.config.subsample_lengths
        assert len(self.__subsample_lengths) == num_stages, (
            f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)} stages, "
            f"while there are {num_stages} computed from "
            "`config.w_a`, `config.w_0`, `config.w_m`, `config.tot_blocks`"
        )
        if isinstance(self.config.dropouts, Real):
            self.__dropouts = list(repeat(self.config.dropouts, num_stages))
        else:
            self.__dropouts = self.config.dropouts
        assert len(self.__dropouts) == num_stages, (
            f"`config.dropouts` indicates {len(self.__dropouts)} stages, "
            f"while there are {num_stages} computed from "
            "`config.w_a`, `config.w_0`, `config.w_m`, `config.tot_blocks`"
        )
        if isinstance(self.config.group_widths, int):
            self.__group_widths = list(repeat(self.config.group_widths, num_stages))
        else:
            self.__group_widths = self.config.group_widths
        assert len(self.__group_widths) == num_stages, (
            f"`config.group_widths` indicates {len(self.__group_widths)} stages, "
            f"while there are {num_stages} computed from "
            "`config.w_a`, `config.w_0`, `config.w_m`, `config.tot_blocks`"
        )
        block_config = CFG(self.config.get("block", {}))
        block_config.pop("dropout", None)

        for idx, num_filters in enumerate(sorted(counter)):
            stage_configs.append(
                CFG(
                    dict(
                        num_blocks=counter[num_filters],
                        num_filters=num_filters,
                        filter_length=self.__filter_lengths[idx],
                        subsample_length=self.__subsample_lengths[idx],
                        group_width=self.__group_widths[idx],
                        dropout=self.__dropouts[idx],
                        stage_index=idx,
                        **block_config,
                    )
                )
            )
        return stage_configs

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.48550/ARXIV.2003.13678"]))
