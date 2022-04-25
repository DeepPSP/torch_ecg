"""
famous for its use of separable convolutions,
usually the SOTA image classifier,
however seems not have been used in physiological signal processing tasks
"""

from copy import deepcopy
from itertools import repeat
from numbers import Real
from typing import NoReturn, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    MultiConv,
    NonLocalBlock,
    SEBlock,
    SeparableConv,
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
    "Xception",
    "XceptionEntryFlow",
    "XceptionMiddleFlow",
    "XceptionExitFlow",
    "XceptionMultiConv",
]


_DEFAULT_CONV_CONFIGS = CFG(
    ordering="acb",
    conv_type="separable",
    batch_norm=True,
    subsample_mode="max",
    activation="relu",
    kw_activation={"inplace": True},
    kernel_initializer="he_normal",
    kw_initializer={},
)


class XceptionMultiConv(nn.Module, SizeMixin):
    """

    -> n(2 or 3) x (activation -> norm -> sep_conv) (-> optional sub-sample) ->
    |-------------------------------- shortcut ------------------------------|
    """

    __DEBUG__ = False
    __name__ = "XceptionMultiConv"

    def __init__(
        self,
        in_channels: int,
        num_filters: Sequence[int],
        filter_lengths: Union[Sequence[int], int],
        subsample_length: int = 1,
        subsample_kernel: Optional[int] = None,
        dilations: Union[Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        num_filters: sequence of int,
            number of channels produced by the main stream convolutions,
            the length of `num_filters` also indicates the number of convolutions
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_length: int,
            stride of the main stream subsample layer
        subsample_kernel: int, optional,
            kernel size of the main stream subsample layer,
            if not set, defaults to `subsample_length`,
        dilations: int or sequence of int, default 1,
            dilation(s) of the convolutions
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.,
            for the convolutional layers,
            and subsampling modes for subsampling layers, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_convs = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_convs == len(
            self.__filter_lengths
        ), f"the main stream has {self.__num_convs} convolutions, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        self.__subsample_length = subsample_length
        self.__subsample_kernel = subsample_kernel or subsample_length
        self.__groups = groups
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_convs))
        else:
            self.__dilations = list(dilations)
        assert self.__num_convs == len(
            self.__dilations
        ), f"the main stream has {self.__num_convs} convolutions, while `dilations` indicates {len(self.__dilations)}"
        if isinstance(dropouts, Real):
            self.__dropouts = list(repeat(dropouts, self.__num_convs))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_convs == len(
            self.__dropouts
        ), f"the main stream has {self.__num_convs} convolutions, while `dropouts` indicates {len(self.__dropouts)}"
        self.config = CFG(deepcopy(_DEFAULT_CONV_CONFIGS))
        self.config.update(deepcopy(config))

        self.main_stream_conv = MultiConv(
            in_channels=self.__in_channels,
            out_channels=self.__num_filters,
            filter_lengths=self.__filter_lengths,
            subsample_lengths=1,
            dilations=self.__dilations,
            groups=self.__groups,
            dropouts=self.__dropouts,
            **self.config,
        )
        if self.__subsample_length > 1:
            self.subsample = DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__num_filters[-1],
                kernel_size=self.__subsample_kernel,
                groups=self.__groups,
                padding=(self.__subsample_kernel - 1) // 2,
                mode=self.config.subsample_mode,
            )
            self.shortcut = DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__in_channels,
                out_channels=self.__num_filters[-1],
                groups=self.__groups,
                kernel_size=1,
                norm=self.config.get("norm", self.config.get("batch_norm")),
                mode="conv",
            )
        else:
            self.subsample = None
            self.shortcut = None

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
        main_out = self.main_stream_conv(input)
        if self.subsample:
            main_out = self.subsample(main_out)
        residue = input
        if self.shortcut:
            residue = self.shortcut(residue)
        output = residue + main_out
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
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        output_shape = self.main_stream_conv.compute_output_shape(seq_len, batch_size)
        if self.subsample is not None:
            output_shape = self.subsample.compute_output_shape(
                output_shape[-1], output_shape[0]
            )
        return output_shape


class XceptionEntryFlow(nn.Sequential, SizeMixin):
    """

    Entry flow of the Xception model,
    consisting of 2 initial convolutions which subsamples at the first one,
    followed by several Xception blocks of 2 convolutions and of sub-sampling size 2
    """

    __DEBUG__ = False
    __name__ = "XceptionEntryFlow"

    def __init__(
        self,
        in_channels: int,
        init_num_filters: Sequence[int],
        init_filter_lengths: Union[int, Sequence[int]],
        init_subsample_lengths: Union[int, Sequence[int]],
        num_filters: Union[Sequence[int], Sequence[Sequence[int]]],
        filter_lengths: Union[int, Sequence[int], Sequence[Sequence[int]]],
        subsample_lengths: Union[int, Sequence[int]],
        subsample_kernels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Union[int, Sequence[int], Sequence[Sequence[int]]] = 1,
        groups: int = 1,
        dropouts: Union[float, Sequence[float], Sequence[Sequence[float]]] = 0.0,
        block_dropouts: Union[float, Sequence[float]] = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        init_num_filters: sequence of int,
            number of filters (output channels) of the initial convolutions
        init_filter_lengths: int or sequence of int,
            filter length(s) (kernel size(s)) of the initial convolutions
        init_subsample_lengths: int or sequence of int,
            subsampling length(s) (stride(s)) of the initial convolutions
        num_filters: sequence of int or sequence of sequences of int,
            number of filters of the convolutions of Xception blocks
        filter_lengths: int or sequence of int or sequence of sequences of int,
            filter length(s) of the convolutions of Xception blocks
        subsample_lengths: int or sequence of int,
            subsampling length(s) of the Xception blocks
        subsample_kernels: int or sequence of int, optional,
            subsampling kernel size(s) of the Xception blocks
        dilations: int or sequence of int or sequence of sequences of int, default 1,
            dilation(s) of the convolutions of Xception blocks
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float or sequence of sequences of float, default 0.0,
            dropout(s) after each `Conv_Bn_Activation` blocks in the Xception blocks
        block_dropouts: float or sequence of float, default 0.0,
            dropout(s) after the Xception blocks
        config: dict,
            other parameters, Xception blocks and initial convolutions, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers,
            and subsampling modes for subsampling layers, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_blocks = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_blocks == len(
            self.__filter_lengths
        ), f"the entry flow has {self.__num_blocks} blocks, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(subsample_lengths, self.__num_blocks)
            )
        else:
            self.__subsample_lengths = list(subsample_lengths)
        assert self.__num_blocks == len(
            self.__subsample_lengths
        ), f"the entry flow has {self.__num_blocks} blocks, while `subsample_lengths` indicates {len(self.__subsample_lengths)}"
        if subsample_kernels is None:
            self.__subsample_kernels = deepcopy(self.__subsample_lengths)
        elif isinstance(subsample_kernels, int):
            self.__subsample_kernels = list(
                repeat(subsample_kernels, self.__num_blocks)
            )
        else:
            self.__subsample_kernels = list(subsample_kernels)
        assert self.__num_blocks == len(
            self.__subsample_kernels
        ), f"the entry flow has {self.__num_blocks} blocks, while `subsample_kernels` indicates {len(self.__subsample_kernels)}"
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(
            self.__dilations
        ), f"the entry flow has {self.__num_blocks} blocks, while `dilations` indicates {len(self.__dilations)}"
        if isinstance(dropouts, Real):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(
            self.__dropouts
        ), f"the entry flow has {self.__num_blocks} blocks, while `dropouts` indicates {len(self.__dropouts)}"
        if isinstance(block_dropouts, Real):
            self.__block_dropouts = list(repeat(block_dropouts, self.__num_blocks))
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks == len(
            self.__block_dropouts
        ), f"the entry flow has {self.__num_blocks} blocks, except the initial convolutions, while `block_dropouts` indicates {len(self.__block_dropouts)}"
        self.__groups = groups
        self.config = CFG(deepcopy(_DEFAULT_CONV_CONFIGS))
        self.config.update(deepcopy(config))

        self.add_module(
            "init_convs",
            MultiConv(
                in_channels=self.__in_channels,
                out_channels=init_num_filters,
                filter_lengths=init_filter_lengths,
                subsample_lengths=init_subsample_lengths,
                groups=groups,
                activation=self.config.activation,
            ),
        )

        block_in_channels = init_num_filters[-1]
        for idx, nf in enumerate(self.__num_filters):
            # in the case of ordering of "acb",
            # `out_activation` is indeed `in_activation`
            if idx == 0:
                out_activation = False
            else:
                out_activation = True
            # number of main stream convolution defaults to 2
            if isinstance(nf, int):
                block_out_channels = list(repeat(nf, 2))
            else:
                block_out_channels = list(nf)
            self.add_module(
                f"entry_flow_conv_block_{idx}",
                XceptionMultiConv(
                    in_channels=block_in_channels,
                    num_filters=block_out_channels,
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    subsample_kernel=self.__subsample_kernels[idx],
                    dilations=self.__dilations[idx],
                    groups=self.__groups,
                    dropouts=self.__dropouts[idx],
                    out_activation=out_activation,
                    **self.config,
                ),
            )
            block_in_channels = block_out_channels[-1]
            if self.__block_dropouts[idx] > 0:
                self.add_module(
                    f"entry_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx])
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
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if type(module).__name__ == "Dropout":
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class XceptionMiddleFlow(nn.Sequential, SizeMixin):
    """

    Middle flow of the Xception model,
    consisting of several Xception blocks of 3 convolutions and without sub-sampling
    """

    __DEBUG__ = False
    __name__ = "XceptionMiddleFlow"

    def __init__(
        self,
        in_channels: int,
        num_filters: Union[Sequence[int], Sequence[Sequence[int]]],
        filter_lengths: Union[int, Sequence[int], Sequence[Sequence[int]]],
        dilations: Union[int, Sequence[int], Sequence[Sequence[int]]] = 1,
        groups: int = 1,
        dropouts: Union[float, Sequence[float], Sequence[Sequence[float]]] = 0.0,
        block_dropouts: Union[float, Sequence[float]] = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        num_filters: sequence of int or sequence of sequences of int,
            number of filters (output channels) of the convolutions of Xception blocks
        filter_lengths: int or sequence of int or sequence of sequences of int,
            filter length(s) of the convolutions of Xception blocks
        dilations: int or sequence of int or sequence of sequences of int, default 1,
            dilation(s) of the convolutions of Xception blocks
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float or sequence of sequences of float, default 0.0,
            dropout(s) after each `Conv_Bn_Activation` blocks in the Xception blocks
        block_dropouts: float or sequence of float, default 0.0,
            dropout(s) after the Xception blocks
        config: dict,
            other parameters for Xception blocks, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers,
            and subsampling modes for subsampling layers, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_blocks = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_blocks == len(
            self.__filter_lengths
        ), f"the middle flow has {self.__num_blocks} blocks, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(
            self.__dilations
        ), f"the middle flow has {self.__num_blocks} blocks, while `dilations` indicates {len(self.__dilations)}"
        if isinstance(dropouts, Real):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(
            self.__dropouts
        ), f"the middle flow has {self.__num_blocks} blocks, while `dropouts` indicates {len(self.__dropouts)}"
        if isinstance(block_dropouts, Real):
            self.__block_dropouts = list(repeat(block_dropouts, self.__num_blocks))
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks == len(
            self.__block_dropouts
        ), f"the middle flow has {self.__num_blocks} blocks, while `block_dropouts` indicates {len(self.__block_dropouts)}"
        self.__groups = groups
        self.config = CFG(deepcopy(_DEFAULT_CONV_CONFIGS))
        self.config.update(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx, nf in enumerate(self.__num_filters):
            # number of main stream convolution defaults to 3
            if isinstance(nf, int):
                block_out_channels = list(repeat(nf, 3))
            else:
                block_out_channels = list(nf)
            self.add_module(
                f"middle_flow_conv_block_{idx}",
                XceptionMultiConv(
                    in_channels=block_in_channels,
                    num_filters=block_out_channels,
                    filter_lengths=self.__filter_lengths[idx],
                    dilations=self.__dilations[idx],
                    groups=self.__groups,
                    dropouts=self.__dropouts[idx],
                    **self.config,
                ),
            )
            block_in_channels = block_out_channels[-1]
            if self.__block_dropouts[idx] > 0:
                self.add_module(
                    f"middle_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx])
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
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if type(module).__name__ == "Dropout":
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class XceptionExitFlow(nn.Sequential, SizeMixin):
    """

    Exit flow of the Xception model,
    consisting of several Xception blocks of 2 convolutions,
    followed by several separable convolutions
    """

    __DEBUG__ = False
    __name__ = "XceptionExitFlow"

    def __init__(
        self,
        in_channels: int,
        final_num_filters: Sequence[int],
        final_filter_lengths: Union[int, Sequence[int]],
        num_filters: Union[Sequence[int], Sequence[Sequence[int]]],
        filter_lengths: Union[int, Sequence[int], Sequence[Sequence[int]]],
        subsample_lengths: Union[int, Sequence[int]],
        subsample_kernels: Optional[Union[int, Sequence[int]]] = None,
        dilations: Union[int, Sequence[int], Sequence[Sequence[int]]] = 1,
        groups: int = 1,
        dropouts: Union[float, Sequence[float], Sequence[Sequence[float]]] = 0.0,
        block_dropouts: Union[float, Sequence[float]] = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        final_num_filters: sequence of int,
            number of filters (output channels) of the final convolutions
        final_filter_lengths: int or sequence of int,
            filter length(s) of the convolutions of the final convolutions
        final_subsample_lengths: int or sequence of int,
            subsampling length(s) (stride(s)) of the final convolutions
        num_filters: sequence of int or sequence of sequences of int,
            number of filters of the convolutions of Xception blocks
        filter_lengths: int or sequence of int or sequence of sequences of int,
            filter length(s) of the convolutions of Xception blocks
        subsample_lengths: int or sequence of int,
            subsampling length(s) of the Xception blocks
        subsample_kernels: int or sequence of int, optional,
            subsampling kernel size(s) of the Xception blocks
        dilations: int or sequence of int or sequence of sequences of int, default 1,
            dilation(s) of the convolutions of Xception blocks
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float or sequence of sequences of float, default 0.0,
            dropout(s) after each `Conv_Bn_Activation` blocks in the Xception blocks
        block_dropouts: float or sequence of float, default 0.0,
            dropout(s) after each of the Xception blocks and each of the final convolutions
        config: dict,
            other parameters for Xception blocks and final convolutions, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers,
            and subsampling modes for subsampling layers, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_blocks = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_blocks == len(
            self.__filter_lengths
        ), f"the exit flow has {self.__num_blocks} blocks, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(subsample_lengths, self.__num_blocks)
            )
        else:
            self.__subsample_lengths = list(subsample_lengths)
        assert self.__num_blocks == len(
            self.__subsample_lengths
        ), f"the exit flow has {self.__num_blocks} blocks, while `subsample_lengths` indicates {len(self.__subsample_lengths)}"
        if subsample_kernels is None:
            self.__subsample_kernels = deepcopy(self.__subsample_lengths)
        elif isinstance(subsample_kernels, int):
            self.__subsample_kernels = list(
                repeat(subsample_kernels, self.__num_blocks)
            )
        else:
            self.__subsample_kernels = list(subsample_kernels)
        assert self.__num_blocks == len(
            self.__subsample_kernels
        ), f"the exit flow has {self.__num_blocks} blocks, while `subsample_kernels` indicates {len(self.__subsample_kernels)}"
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(
            self.__dilations
        ), f"the exit flow has {self.__num_blocks} blocks, while `dilations` indicates {len(self.__dilations)}"
        if isinstance(dropouts, Real):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(
            self.__dropouts
        ), f"the exit flow has {self.__num_blocks} blocks, while `dropouts` indicates {len(self.__dropouts)}"
        if isinstance(block_dropouts, Real):
            self.__block_dropouts = list(
                repeat(block_dropouts, self.__num_blocks + len(final_num_filters))
            )
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks + len(final_num_filters) == len(
            self.__block_dropouts
        ), f"the exit flow has {self.__num_blocks + len(final_num_filters)} blocks, including the final convolutions, while `block_dropouts` indicates {len(self.__block_dropouts)}"
        self.__groups = groups
        self.config = CFG(deepcopy(_DEFAULT_CONV_CONFIGS))
        self.config.update(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx, nf in enumerate(self.__num_filters):
            # number of main stream convolution defaults to 2
            if isinstance(nf, int):
                block_out_channels = list(repeat(nf, 2))
            else:
                block_out_channels = list(nf)
            self.add_module(
                f"exit_flow_conv_block_{idx}",
                XceptionMultiConv(
                    in_channels=block_in_channels,
                    num_filters=block_out_channels,
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    subsample_kernel=self.__subsample_kernels[idx],
                    dilations=self.__dilations[idx],
                    groups=self.__groups,
                    dropouts=self.__dropouts[idx],
                    **self.config,
                ),
            )
            block_in_channels = block_out_channels[-1]
            if self.__block_dropouts[idx] > 0:
                self.add_module(
                    f"exit_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx])
                )

        self.add_module(
            "final_convs",
            MultiConv(
                in_channels=block_in_channels,
                out_channels=final_num_filters,
                filter_lengths=final_filter_lengths,
                groups=groups,
                conv_type="separable",
                activation=self.config.activation,
            ),
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
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if type(module).__name__ == "Dropout":
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class Xception(nn.Sequential, SizeMixin):
    """

    References
    ----------
    [1] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
    [2] https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
    [3] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
    """

    __DEBUG__ = False
    __name__ = "Xception"

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set in 3 sub-dict,
            namely in "entry_flow", "middle_flow", and "exit_flow",
            ref. corresponding docstring of each class
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        entry_flow_in_channels = self.__in_channels
        entry_flow = XceptionEntryFlow(
            in_channels=entry_flow_in_channels, **(self.config.entry_flow)
        )
        self.add_module("entry_flow", entry_flow)

        _, middle_flow_in_channels, _ = entry_flow.compute_output_shape()
        middle_flow = XceptionMiddleFlow(
            in_channels=middle_flow_in_channels, **(self.config.middle_flow)
        )
        self.add_module("middle_flow", middle_flow)

        _, exit_flow_in_channels, _ = middle_flow.compute_output_shape()
        exit_flow = XceptionExitFlow(
            in_channels=exit_flow_in_channels, **(self.config.exit_flow)
        )
        self.add_module(
            "exit_flow",
            exit_flow,
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
        output = super().forward(input)
        return output

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels
