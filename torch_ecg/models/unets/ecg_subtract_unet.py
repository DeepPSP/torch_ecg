"""
2nd place (entry 0433) of CPSC2019

the main differences to a normal Unet are that

1. at the bottom, subtraction (and concatenation) is used
2. uses triple convolutions at each block, instead of double convolutions
3. dropout is used between certain convolutional layers ("cba" layers indeed)

"""

import textwrap
import warnings
from copy import deepcopy
from itertools import repeat
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn

from ...cfg import CFG
from ...model_configs import ECG_SUBTRACT_UNET_CONFIG
from ...models._nets import BranchedConv, Conv_Bn_Activation, DownSample, MultiConv
from ...utils.misc import add_docstring
from ...utils.utils_nn import (
    CkptMixin,
    SizeMixin,
    compute_deconv_output_shape,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)

__all__ = [
    "ECG_SUBTRACT_UNET",
]


class TripleConv(MultiConv):
    """Triple convolutional layer.

    CBA --> (Dropout) --> CBA --> (Dropout) --> CBA --> (Dropout).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int or Sequence[int]
        Number of channels produced by the (last) convolutional layer(s).
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size).
    subsample_lengths : int or Sequence[int], default 1
        Subsample length(s) (stride(s)) of the convolutions
    groups: int, default 1,
        connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    out_activation : bool, default True
        If True, the last mini-block of :class:`Conv_Bn_Activation`
        will have activation as in `config`; otherwise, no activation.
    config : dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers.

    """

    __name__ = "TripleConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: Union[Sequence[int], int],
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        out_activation: bool = True,
        **config,
    ) -> None:
        _num_convs = 3
        if isinstance(out_channels, int):
            _out_channels = list(repeat(out_channels, _num_convs))
        else:
            _out_channels = list(out_channels)
            assert _num_convs == len(_out_channels)

        super().__init__(
            in_channels=in_channels,
            out_channels=_out_channels,
            filter_lengths=filter_lengths,
            subsample_lengths=subsample_lengths,
            groups=groups,
            dropouts=dropouts,
            out_activation=out_activation,
            **config,
        )


class DownTripleConv(nn.Sequential, SizeMixin):
    """Down sampling block of the U-net architecture.

    Composed of a down sampling layer and 3 convolutional layers.

    Parameters
    ----------
    down_scale : int
        Down sampling scale.
    in_channels : int
        Number of channels in the input.
    out_channels : int or Sequence[int]
        Number of channels produced by the (last) convolutional layer(s).
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size).
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    mode : str, default "max"
        Down sampling mode, one of {:class:`DownSample`.__MODES__}.
    config : dict
        Other parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers.

    """

    __name__ = "DownTripleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: Union[Sequence[int], int],
        filter_lengths: Union[Sequence[int], int],
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        mode: str = "max",
        **config,
    ) -> None:
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.config = CFG(deepcopy(config))

        self.add_module(
            "down_sample",
            DownSample(
                down_scale=self.__down_scale,
                in_channels=self.__in_channels,
                norm=False,
                mode=mode,
            ),
        )
        self.add_module(
            "triple_conv",
            TripleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_lengths=filter_lengths,
                subsample_lengths=1,
                groups=groups,
                dropouts=dropouts,
                **(self.config),
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        out : torch.Tensor
            Output tensor,
            of shape ``(batch_size, channels, seq_len)``.

        """
        out = super().forward(input)
        return out

    @add_docstring(
        textwrap.indent(compute_sequential_output_shape_docstring, " " * 4),
        mode="append",
    )
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the down sampling block."""
        return compute_sequential_output_shape(self, seq_len, batch_size)


class DownBranchedDoubleConv(nn.Module, SizeMixin):
    """The bottom block of the encoder in the U-Net architecture.

    Parameters
    ----------
    down_scale : int
        Down sampling scale.
    in_channels : int
        Number of channels in the input tensor.
    out_channels : Sequence[Sequence[int]]
        Number of channels produced by the (last) convolutional layer(s).
    filter_lengths : int or Sequence[int] or Sequence[Sequence[int]]
        Length(s) of the filters (kernel size).
    dilations : int or Sequence[int] or Sequence[Sequence[int]], default 1
        Dilation(s) of the convolutions.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    mode : str, default "max"
        Down sampling mode, one of {:class:`DownSample`.__MODES__}.
    config: dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers.

    """

    __name__ = "DownBranchedDoubleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: Sequence[Sequence[int]],
        filter_lengths: Union[Sequence[Sequence[int]], Sequence[int], int],
        dilations: Union[Sequence[Sequence[int]], Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        mode: str = "max",
        **config,
    ) -> None:
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.config = CFG(deepcopy(config))

        self.down_sample = DownSample(
            down_scale=self.__down_scale,
            in_channels=self.__in_channels,
            norm=False,
            mode=mode,
        )
        self.branched_conv = BranchedConv(
            in_channels=self.__in_channels,
            out_channels=self.__out_channels,
            filter_lengths=filter_lengths,
            subsample_lengths=1,
            dilations=dilations,
            groups=groups,
            dropouts=dropouts,
            **(self.config),
        )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        out : torch.Tensor
            Output tensor,
            of shape ``(batch_size, channels, seq_len)``.

        """
        out = self.down_sample(input)
        out = self.branched_conv(out)
        # SUBTRACT
        # currently (micro scope) - (macro scope)
        # TODO: consider (macro scope) - (micro scope)
        out.append(out[0] - out[1])
        out = torch.cat(out, dim=1)  # concate along the channel axis
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
        output_shape : sequence
            Output shape of the block.

        """
        _seq_len = seq_len
        output_shape = self.down_sample.compute_output_shape(seq_len=_seq_len)
        _, _, _seq_len = output_shape
        output_shapes = self.branched_conv.compute_output_shape(seq_len=_seq_len)
        # output_shape = output_shapes[0][0], sum([s[1] for s in output_shapes]), output_shapes[0][-1]
        n_branches = len(output_shapes)
        output_shape = (
            output_shapes[0][0],
            (n_branches + 1) * output_shapes[0][1],
            output_shapes[0][-1],
        )
        return output_shape


class UpTripleConv(nn.Module, SizeMixin):
    """Up sampling block in the U-Net architecture.

    Upscaling then double conv, with input of corr. down layer concatenated
    up sampling --> conv (conv --> (dropout -->) conv --> (dropout -->) conv)
        ^
        |
    extra input

    Channels are shrinked after up sampling.

    Parameters
    ----------
    up_scale : int
        Scale of up sampling.
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the convolutional layers.
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size) of the convolutional layers.
    deconv_filter_length : int
        Length(s) of the filters (kernel size) of the
        deconvolutional upsampling layer,
        used when `mode` is ``"deconv"``.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
        Not used currently.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    mode : str, default "deconv"
        Mode of up sampling, case insensitive. Should be ``"deconv"``
        or methods supported by :class:`torch.nn.Upsample`.
    config : dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the deconvolutional layers.

    """

    __name__ = "UpTripleConv"
    __MODES__ = [
        "nearest",
        "linear",
        "area",
        "deconv",
    ]

    def __init__(
        self,
        up_scale: int,
        in_channels: int,
        out_channels: int,
        filter_lengths: Union[Sequence[int], int],
        deconv_filter_length: Optional[int] = None,
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        mode: str = "deconv",
        **config,
    ) -> None:
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__deconv_filter_length = deconv_filter_length
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.config = CFG(deepcopy(config))

        # the following has to be checked
        # if bilinear, use the normal convolutions to reduce the number of channels
        if self.__mode == "deconv":
            self.__deconv_padding = max(0, (self.__deconv_filter_length - self.__up_scale) // 2)
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
        self.conv = TripleConv(
            # `+ self.__out_channels` corr. to the output of the corr. down layer
            in_channels=self.__in_channels + self.__out_channels[-1],
            out_channels=self.__out_channels,
            filter_lengths=filter_lengths,
            subsample_lengths=1,
            groups=groups,
            dropouts=dropouts,
            **(self.config),
        )

    def forward(self, input: Tensor, down_output: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor from the previous up sampling block.
        down_output : torch.Tensor
            Input tensor of the last layer of corr. down sampling block.

        Returns
        -------
        output : torch.Tensor
            Output tensor of this block.

        """
        output = self.up(input)
        output = torch.cat([down_output, output], dim=1)  # concate along the channel axis
        output = self.conv(output)

        return output

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
        output_shape : sequence
            Output shape of the block.

        """
        _sep_len = seq_len
        if self.__mode == "deconv":
            output_shape = compute_deconv_output_shape(
                input_shape=[batch_size, self.__in_channels, _sep_len],
                num_filters=self.__in_channels,
                kernel_size=self.__deconv_filter_length,
                stride=self.__up_scale,
                padding=self.__deconv_padding,
            )
        else:
            output_shape = [batch_size, self.__in_channels, self.__up_scale * _sep_len]
        _, _, _seq_len = output_shape
        output_shape = self.conv.compute_output_shape(_seq_len, batch_size)
        return output_shape


class ECG_SUBTRACT_UNET(nn.Module, CkptMixin, SizeMixin):
    """U-Net for ECG wave delineation.

    Entry 0433 of CPSC2019, which is a modification of the U-Net
    using subtraction instead of addition in branched bottom block.

    Parameters
    ----------
    classes : Sequence[str]
        List of names of the classes.
    n_leads : int
        Number of input leads (number of input channels).
    config : CFG, optional
        Other hyper-parameters, including kernel sizes, etc.
        Refer to the corresponding config file.

    """

    __name__ = "ECG_SUBTRACT_UNET"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.__out_channels = len(classes)
        self.__in_channels = n_leads
        self.config = deepcopy(ECG_SUBTRACT_UNET_CONFIG)
        if not config:
            warnings.warn("No config is provided, using default config.", RuntimeWarning)
        self.config.update(deepcopy(config) or {})

        # TODO: an init batch normalization?
        if self.config.init_batch_norm:
            self.init_bn = nn.BatchNorm1d(
                num_features=self.__in_channels,
                eps=1e-5,  # default val
                momentum=0.1,  # default val
            )

        self.init_conv = TripleConv(
            in_channels=self.__in_channels,
            out_channels=self.config.init_num_filters,
            filter_lengths=self.config.init_filter_length,
            subsample_lengths=1,
            groups=self.config.groups,
            dropouts=self.config.init_dropouts,
            batch_norm=self.config.batch_norm,
            activation=self.config.activation,
            kw_activation=self.config.kw_activation,
            kernel_initializer=self.config.kernel_initializer,
            kw_initializer=self.config.kw_initializer,
        )

        self.down_blocks = nn.ModuleDict()
        in_channels = self.config.init_num_filters
        for idx in range(self.config.down_up_block_num - 1):
            self.down_blocks[f"down_{idx}"] = DownTripleConv(
                down_scale=self.config.down_scales[idx],
                in_channels=in_channels,
                out_channels=self.config.down_num_filters[idx],
                filter_lengths=self.config.down_filter_lengths[idx],
                groups=self.config.groups,
                dropouts=self.config.down_dropouts[idx],
                mode=self.config.down_mode,
                **(self.config.down_block),
            )
            in_channels = self.config.down_num_filters[idx][-1]

        self.bottom_block = DownBranchedDoubleConv(
            down_scale=self.config.down_scales[-1],
            in_channels=in_channels,
            out_channels=self.config.bottom_num_filters,
            filter_lengths=self.config.bottom_filter_lengths,
            dilations=self.config.bottom_dilations,
            groups=self.config.groups,
            dropouts=self.config.bottom_dropouts,
            mode=self.config.down_mode,
            **(self.config.down_block),
        )

        self.up_blocks = nn.ModuleDict()
        # in_channels = sum([branch[-1] for branch in self.config.bottom_num_filters])
        in_channels = self.bottom_block.compute_output_shape(None, None)[1]
        for idx in range(self.config.down_up_block_num):
            self.up_blocks[f"up_{idx}"] = UpTripleConv(
                up_scale=self.config.up_scales[idx],
                in_channels=in_channels,
                out_channels=self.config.up_num_filters[idx],
                filter_lengths=self.config.up_conv_filter_lengths[idx],
                deconv_filter_length=self.config.up_deconv_filter_lengths[idx],
                groups=self.config.groups,
                mode=self.config.up_mode,
                dropouts=self.config.up_dropouts[idx],
                **(self.config.up_block),
            )
            in_channels = self.config.up_num_filters[idx][-1]

        self.out_conv = Conv_Bn_Activation(
            in_channels=self.config.up_num_filters[-1][-1],
            out_channels=self.__out_channels,
            kernel_size=self.config.out_filter_length,
            stride=1,
            groups=self.config.groups,
            norm=self.config.get("out_norm", self.config.get("out_batch_norm")),
            activation=None,
            kernel_initializer=self.config.kernel_initializer,
            kw_initializer=self.config.kw_initializer,
        )

        # for inference
        # if background counted in `classes`, use softmax
        # otherwise use sigmoid
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        if self.config.init_batch_norm:
            x = self.init_bn(input)
        else:
            x = input

        # down
        to_concat = [self.init_conv(x)]
        for idx in range(self.config.down_up_block_num - 1):
            to_concat.append(self.down_blocks[f"down_{idx}"](to_concat[-1]))
        to_concat.append(self.bottom_block(to_concat[-1]))

        # up
        up_input = to_concat[-1]
        to_concat = to_concat[-2::-1]
        for idx in range(self.config.down_up_block_num):
            up_output = self.up_blocks[f"up_{idx}"](up_input, to_concat[idx])
            up_input = up_output

        # output
        output = self.out_conv(up_output)

        # to keep in accordance with other models
        # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
        output = output.permute(0, 2, 1)

        # TODO: consider adding CRF at the tail to make final prediction

        return output

    @torch.no_grad()
    def inference(self, input: Union[np.ndarray, Tensor], bin_pred_thr: float = 0.5) -> Tensor:
        """Method for making inference on a single input."""
        raise NotImplementedError("Implement a task-specific inference method.")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input signal tensor.
        batch_size : int, optional
            Batch size of the input signal tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the model.

        """
        output_shape = (batch_size, seq_len, self.n_classes)
        return output_shape

    @property
    def doi(self) -> List[str]:
        # TODO: add doi
        return list(set(self.config.get("doi", []) + []))
