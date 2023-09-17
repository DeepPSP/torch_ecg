"""
famous for its use of separable convolutions,
usually the SOTA image classifier,
however seems not have been used in physiological signal processing tasks
"""

import textwrap
from copy import deepcopy
from itertools import repeat
from numbers import Real
from typing import List, Optional, Sequence, Union

from torch import Tensor, nn

from ...cfg import CFG
from ...models._nets import DownSample, MultiConv
from ...utils.misc import CitationMixin, add_docstring
from ...utils.utils_nn import SizeMixin, compute_sequential_output_shape, compute_sequential_output_shape_docstring

__all__ = [
    "Xception",
    "XceptionEntryFlow",
    "XceptionMiddleFlow",
    "XceptionExitFlow",
    "XceptionMultiConv",
]


if not hasattr(nn, "Dropout1d"):
    nn.Dropout1d = nn.Dropout  # added in pytorch 1.12


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


class XceptionMultiConv(nn.Module, SizeMixin, CitationMixin):
    """Xception multi-convolutional block.

    -> n(2 or 3) x (activation -> norm -> sep_conv) (-> optional sub-sample) ->
    |-------------------------------- shortcut ------------------------------|

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    num_filters : Sequence[int]
        Number of channels produced by the main stream convolutions.
        The length of `num_filters` indicates the number of convolutions.
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size).
    subsample_length : int, default 1
        Stride of the main stream subsample layer.
    subsample_kernel : int, optional
        Kernel size of the main stream subsample layer.
        If not set, defaults to `subsample_length`.
    dilations : int or Sequence[int], default 1
        Dilation(s) of the convolutions.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[float] or Sequence[dict], default 0.0
        If is dictionary, it should contain the keys ``"p"`` and ``"type"``,
        where ``"p"`` is the dropout rate and ``"type"`` is the type of dropout,
        which can be either ``"1d"`` (:class:`torch.nn.Dropout1d`) or
        ``None`` (:class:`torch.nn.Dropout`).
    config : dict, optional
        Other parameters, including
        activation choices, weight initializer, batch normalization choices, etc.,
        for the convolutional layers,
        and subsampling modes for subsampling layers, etc.

    """

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
    ) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_convs = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_convs == len(self.__filter_lengths), (
            f"the main stream has {self.__num_convs} convolutions, "
            f"while `filter_lengths` indicates {len(self.__filter_lengths)}"
        )
        self.__subsample_length = subsample_length
        self.__subsample_kernel = subsample_kernel or subsample_length
        self.__groups = groups
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_convs))
        else:
            self.__dilations = list(dilations)
        assert self.__num_convs == len(self.__dilations), (
            f"the main stream has {self.__num_convs} convolutions, " f"while `dilations` indicates {len(self.__dilations)}"
        )
        if isinstance(dropouts, (Real, dict)):
            self.__dropouts = list(repeat(dropouts, self.__num_convs))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_convs == len(self.__dropouts), (
            f"the main stream has {self.__num_convs} convolutions, " f"while `dropouts` indicates {len(self.__dropouts)}"
        )
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
            self.subsample = nn.Identity()
            self.shortcut = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        main_out = self.main_stream_conv(input)
        main_out = self.subsample(main_out)
        residue = input
        residue = self.shortcut(residue)
        output = residue + main_out
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the module.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            Batch size of the input tensor.

        Returns
        -------
        output_shape : sequence
            The output shape of the module.

        """
        output_shape = self.main_stream_conv.compute_output_shape(seq_len, batch_size)
        if not isinstance(self.subsample, nn.Identity):
            output_shape = self.subsample.compute_output_shape(output_shape[-1], output_shape[0])
        return output_shape


class XceptionEntryFlow(nn.Sequential, SizeMixin):
    """Entry flow of the Xception model.

    The entry flow is consisting of
    2 initial convolutions which subsamples at the first one,
    followed by several Xception blocks of 2 convolutions
    and of sub-sampling size 2.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input signal.
    init_num_filters : Sequence[int]
        Number of filters (output channels) of the initial convolutions.
    init_filter_lengths : int or Sequence[int]
        Filter length(s) (kernel size(s)) of the initial convolutions.
    init_subsample_lengths : int or Sequence[int]
        Subsampling length(s) (stride(s)) of the initial convolutions.
    num_filters : Sequence[int] or Sequence[Sequence[int]]
        Number of filters of the convolutions of Xception blocks.
    filter_lengths : int or Sequence[int] or Sequence[Sequence[int]]
        Filter length(s) of the convolutions of Xception blocks.
    subsample_lengths : int or Sequence[int]
        Subsampling length(s) of the Xception blocks.
    subsample_kernels : int or Sequence[int], optional
        Subsampling kernel size(s) of the Xception blocks.
    dilations : int or Sequence[int] or Sequence[Sequence[int]], default 1
        Dilation(s) of the convolutions of Xception blocks.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]] or Sequence[Sequence[Union[float, dict]]], default 0.0
        Dropout(s) after each :class:`Conv_Bn_Activation` blocks
        in the Xception blocks.
    block_dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout(s) after the Xception blocks.
    config : dict, optional
        Other parameters, Xception blocks and initial convolutions, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers,
        and subsampling modes for subsampling layers, etc.

    """

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
        dropouts: Union[
            float,
            dict,
            Sequence[Union[float, dict]],
            Sequence[Sequence[Union[float, dict]]],
        ] = 0.0,
        block_dropouts: Union[float, dict, Sequence[Union[float, dict]]] = 0.0,
        **config,
    ) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_blocks = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_blocks == len(self.__filter_lengths), (
            f"the entry flow has {self.__num_blocks} blocks, " f"while `filter_lengths` indicates {len(self.__filter_lengths)}"
        )
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(repeat(subsample_lengths, self.__num_blocks))
        else:
            self.__subsample_lengths = list(subsample_lengths)
        assert self.__num_blocks == len(self.__subsample_lengths), (
            f"the entry flow has {self.__num_blocks} blocks, "
            f"while `subsample_lengths` indicates {len(self.__subsample_lengths)}"
        )
        if subsample_kernels is None:
            self.__subsample_kernels = deepcopy(self.__subsample_lengths)
        elif isinstance(subsample_kernels, int):
            self.__subsample_kernels = list(repeat(subsample_kernels, self.__num_blocks))
        else:
            self.__subsample_kernels = list(subsample_kernels)
        assert self.__num_blocks == len(self.__subsample_kernels), (
            f"the entry flow has {self.__num_blocks} blocks, "
            f"while `subsample_kernels` indicates {len(self.__subsample_kernels)}"
        )
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(self.__dilations), (
            f"the entry flow has {self.__num_blocks} blocks, " f"while `dilations` indicates {len(self.__dilations)}"
        )
        if isinstance(dropouts, (Real, dict)):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(self.__dropouts), (
            f"the entry flow has {self.__num_blocks} blocks, " f"while `dropouts` indicates {len(self.__dropouts)}"
        )
        if isinstance(block_dropouts, (Real, dict)):
            self.__block_dropouts = list(repeat(block_dropouts, self.__num_blocks))
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks == len(self.__block_dropouts), (
            f"the entry flow has {self.__num_blocks} blocks, except the initial convolutions, "
            f"while `block_dropouts` indicates {len(self.__block_dropouts)}"
        )
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
            if isinstance(self.__block_dropouts[idx], dict):
                if self.__block_dropouts[idx]["type"] == "1d" and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"entry_flow_dropout_{idx}",
                        nn.Dropout1d(self.__block_dropouts[idx]["p"]),
                    )
                elif self.__block_dropouts[idx]["type"] is None and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"entry_flow_dropout_{idx}",
                        nn.Dropout(self.__block_dropouts[idx]["p"]),
                    )
            elif self.__block_dropouts[idx] > 0:
                self.add_module(f"entry_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx]))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the entry flow.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the entry flow.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensors.
        batch_size : int, optional
            Batch size of the input tensors.

        Returns
        -------
        output_shape : sequence
            The output shape of the entry flow.

        """
        _seq_len = seq_len
        for module in self:
            if isinstance(module, (nn.Dropout, nn.Dropout1d)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class XceptionMiddleFlow(nn.Sequential, SizeMixin):
    """Middle flow of the Xception model.

    Middle flow consists of several Xception blocks of 3 convolutions
    and without sub-sampling.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    num_filters: Sequence[int] or Sequence[Sequence[int]]
        Number of filters (output channels) of the convolutions
        of Xception blocks.
    filter_lengths : int or Sequence[int] or Sequence[Sequence[int]]
        Filter length(s) of the convolutions of Xception blocks.
    dilations : int or Sequence[int] or Sequence[Sequence[int]], default 1
        Dilation(s) of the convolutions of Xception blocks.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]] or Sequence[Sequence[Union[float, dict]]], default 0.0
        Dropout(s) after each :class:`Conv_Bn_Activation` blocks
        in the Xception blocks.
    block_dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout(s) after the Xception blocks
    config : dict, optional
        Other parameters for Xception blocks, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers,
        and subsampling modes for subsampling layers, etc.

    """

    __name__ = "XceptionMiddleFlow"

    def __init__(
        self,
        in_channels: int,
        num_filters: Union[Sequence[int], Sequence[Sequence[int]]],
        filter_lengths: Union[int, Sequence[int], Sequence[Sequence[int]]],
        dilations: Union[int, Sequence[int], Sequence[Sequence[int]]] = 1,
        groups: int = 1,
        dropouts: Union[
            float,
            dict,
            Sequence[Union[float, dict]],
            Sequence[Sequence[Union[float, dict]]],
        ] = 0.0,
        block_dropouts: Union[float, dict, Sequence[Union[float, dict]]] = 0.0,
        **config,
    ) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.__num_filters = list(num_filters)
        self.__num_blocks = len(self.__num_filters)
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_blocks))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert self.__num_blocks == len(self.__filter_lengths), (
            f"the middle flow has {self.__num_blocks} blocks, " f"while `filter_lengths` indicates {len(self.__filter_lengths)}"
        )
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(self.__dilations), (
            f"the middle flow has {self.__num_blocks} blocks, " f"while `dilations` indicates {len(self.__dilations)}"
        )
        if isinstance(dropouts, (Real, dict)):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(self.__dropouts), (
            f"the middle flow has {self.__num_blocks} blocks, " f"while `dropouts` indicates {len(self.__dropouts)}"
        )
        if isinstance(block_dropouts, (Real, dict)):
            self.__block_dropouts = list(repeat(block_dropouts, self.__num_blocks))
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks == len(self.__block_dropouts), (
            f"the middle flow has {self.__num_blocks} blocks, " f"while `block_dropouts` indicates {len(self.__block_dropouts)}"
        )
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
            if isinstance(self.__block_dropouts[idx], dict):
                if self.__block_dropouts[idx]["type"] == "1d" and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"middle_flow_dropout_{idx}",
                        nn.Dropout1d(self.__block_dropouts[idx]["p"]),
                    )
                elif self.__block_dropouts[idx]["type"] is None and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"middle_flow_dropout_{idx}",
                        nn.Dropout(self.__block_dropouts[idx]["p"]),
                    )
            elif self.__block_dropouts[idx] > 0:
                self.add_module(f"middle_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx]))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the middle flow.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the middle flow.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensors.
        batch_size : int, optional
            Batch size of the input tensors.

        Returns
        -------
        output_shape : sequence
            The output shape of the middle flow.

        """
        _seq_len = seq_len
        for module in self:
            if isinstance(module, (nn.Dropout, nn.Dropout1d)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class XceptionExitFlow(nn.Sequential, SizeMixin):
    """Exit flow of the Xception model.

    Exit flow consists of several Xception blocks of 2 convolutions,
    followed by several separable convolutions.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    final_num_filters : Sequence[int]
        Number of filters (output channels) of the final convolutions.
    final_filter_lengths : int or Sequence[int]
        Filter length(s) of the convolutions of the final convolutions.
    final_subsample_lengths : int or Sequence[int]
        Subsampling length(s) (stride(s)) of the final convolutions.
    num_filters : Sequence[int] or Sequence[Sequence[int]]
        Number of filters of the convolutions of Xception blocks.
    filter_lengths : int or Sequence[int] or Sequence[Sequence[int]]
        Filter length(s) of the convolutions of Xception blocks.
    subsample_lengths : int or Sequence[int]
        Subsampling length(s) of the Xception blocks.
    subsample_kernels : int or Sequence[int], optional
        Subsampling kernel size(s) of the Xception blocks.
    dilations: int or Sequence[int] or Sequence[Sequence[int]], default 1
        Dilation(s) of the convolutions of Xception blocks.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]] or Sequence[Sequence[Union[float, dict]]], default 0.0
        Dropout(s) after each :class:`Conv_Bn_Activation` blocks
        in the Xception blocks.
    block_dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout(s) after each of the Xception blocks
        and each of the final convolutions.
    config : dict, optional
        Other parameters for Xception blocks and final convolutions, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers,
        and subsampling modes for subsampling layers, etc.

    """

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
        dropouts: Union[
            float,
            dict,
            Sequence[Union[float, dict]],
            Sequence[Sequence[Union[float, dict]]],
        ] = 0.0,
        block_dropouts: Union[float, dict, Sequence[Union[float, dict]]] = 0.0,
        **config,
    ) -> None:
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
            self.__subsample_lengths = list(repeat(subsample_lengths, self.__num_blocks))
        else:
            self.__subsample_lengths = list(subsample_lengths)
        assert self.__num_blocks == len(self.__subsample_lengths), (
            f"the exit flow has {self.__num_blocks} blocks, "
            f"while `subsample_lengths` indicates {len(self.__subsample_lengths)}"
        )
        if subsample_kernels is None:
            self.__subsample_kernels = deepcopy(self.__subsample_lengths)
        elif isinstance(subsample_kernels, int):
            self.__subsample_kernels = list(repeat(subsample_kernels, self.__num_blocks))
        else:
            self.__subsample_kernels = list(subsample_kernels)
        assert self.__num_blocks == len(self.__subsample_kernels), (
            f"the exit flow has {self.__num_blocks} blocks, "
            f"while `subsample_kernels` indicates {len(self.__subsample_kernels)}"
        )
        if isinstance(dilations, int):
            self.__dilations = list(repeat(dilations, self.__num_blocks))
        else:
            self.__dilations = list(dilations)
        assert self.__num_blocks == len(self.__dilations), (
            f"the exit flow has {self.__num_blocks} blocks, " f"while `dilations` indicates {len(self.__dilations)}"
        )
        if isinstance(dropouts, (Real, dict)):
            self.__dropouts = list(repeat(dropouts, self.__num_blocks))
        else:
            self.__dropouts = list(dropouts)
        assert self.__num_blocks == len(self.__dropouts), (
            f"the exit flow has {self.__num_blocks} blocks, " f"while `dropouts` indicates {len(self.__dropouts)}"
        )
        if isinstance(block_dropouts, (Real, dict)):
            self.__block_dropouts = list(repeat(block_dropouts, self.__num_blocks + len(final_num_filters)))
        else:
            self.__block_dropouts = list(block_dropouts)
        assert self.__num_blocks + len(final_num_filters) == len(self.__block_dropouts), (
            f"the exit flow has {self.__num_blocks + len(final_num_filters)} blocks, "
            f"including the final convolutions, while `block_dropouts` indicates {len(self.__block_dropouts)}"
        )
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
            if isinstance(self.__block_dropouts[idx], dict):
                if self.__block_dropouts[idx]["type"] == "1d" and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"exit_flow_dropout_{idx}",
                        nn.Dropout1d(self.__block_dropouts[idx]["p"]),
                    )
                elif self.__block_dropouts[idx]["type"] is None and self.__block_dropouts[idx]["p"] > 0:
                    self.add_module(
                        f"exit_flow_dropout_{idx}",
                        nn.Dropout(self.__block_dropouts[idx]["p"]),
                    )
            elif self.__block_dropouts[idx] > 0:
                self.add_module(f"exit_flow_dropout_{idx}", nn.Dropout(self.__block_dropouts[idx]))

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
        """Forward pass of the exit flow.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the exit flow.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensors.
        batch_size : int, optional
            Batch size of the input tensors.

        Returns
        -------
        output_shape : sequence
            The output shape of the exit flow.

        """
        _seq_len = seq_len
        for module in self:
            if isinstance(module, (nn.Dropout, nn.Dropout1d)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape


class Xception(nn.Sequential, SizeMixin, CitationMixin):
    """Xception model.

    Xception is an architecture that uses depthwise separable convolutions
    to build light-weight deep neural networks, as described in [1]_.
    Its official implementation is available in [2]_, and a PyTorch
    implementation is available in [3]_. Xception is currently not widely
    used in the field of ECG analysis, but has the potential to be highly
    effective for this task.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input.
    config : dict
        Other hyper-parameters of the Module, ref. corr. config file.
        For keyword arguments that must be set in 3 sub-dict,
        namely in "entry_flow", "middle_flow", and "exit_flow",
        refer to corr. docstring of each class.

    References
    ----------
    .. [1] Chollet, François. "Xception: Deep learning with depthwise separable convolutions."
           Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
    .. [2] https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
    .. [3] https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

    """

    __name__ = "Xception"

    def __init__(self, in_channels: int, **config) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))

        entry_flow_in_channels = self.__in_channels
        entry_flow = XceptionEntryFlow(in_channels=entry_flow_in_channels, **(self.config.entry_flow))
        self.add_module("entry_flow", entry_flow)

        _, middle_flow_in_channels, _ = entry_flow.compute_output_shape()
        middle_flow = XceptionMiddleFlow(in_channels=middle_flow_in_channels, **(self.config.middle_flow))
        self.add_module("middle_flow", middle_flow)

        _, exit_flow_in_channels, _ = middle_flow.compute_output_shape()
        exit_flow = XceptionExitFlow(in_channels=exit_flow_in_channels, **(self.config.exit_flow))
        self.add_module(
            "exit_flow",
            exit_flow,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor.
            Input signal tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        output = super().forward(input)
        return output

    @add_docstring(
        textwrap.indent(compute_sequential_output_shape_docstring, " " * 4),
        mode="append",
    )
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape the model."""
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.1109/cvpr.2017.195"]))
