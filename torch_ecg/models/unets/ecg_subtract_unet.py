"""
2nd place (entry 0433) of CPSC2019

the main differences to a normal Unet are that
1. at the bottom, subtraction (and concatenation) is used
2. uses triple convolutions at each block, instead of double convolutions
3. dropout is used between certain convolutional layers ("cba" layers indeed)

"""

from copy import deepcopy
from itertools import repeat
from typing import NoReturn, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    BranchedConv,
    Conv_Bn_Activation,
    DownSample,
    MultiConv,
)
from ...utils.misc import dict_to_str, add_docstring
from ...utils.utils_nn import (
    CkptMixin,
    SizeMixin,
    compute_deconv_output_shape,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_SUBTRACT_UNET",
]


class TripleConv(MultiConv):
    """

    CBA --> (Dropout) --> CBA --> (Dropout) --> CBA --> (Dropout)
    """

    __DEBUG__ = False
    __name__ = "TripleConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: Union[Sequence[int], int],
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
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
        out_channels: int, or sequence of int,
            number of channels produced by the (last) convolutional layer(s)
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        out_activation: bool, default True,
            if True, the last mini-block of `Conv_Bn_Activation` will have activation as in `config`,
            otherwise None
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
        """
        _num_convs = 3
        if isinstance(out_channels, int):
            _out_channels = list(repeat(out_channels, _num_convs))
        else:
            _out_channels = list(out_channels)
            assert _num_convs == len(_out_channels)
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(config)}"
            )

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
    """ """

    __DEBUG__ = False
    __name__ = "DownTripleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: Union[Sequence[int], int],
        filter_lengths: Union[Sequence[int], int],
        groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        mode: str = "max",
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        down_scale: int,
            down sampling scale
        in_channels: int,
            number of channels in the input
        out_channels: int, or sequence of int,
            number of channels produced by the (last) convolutional layer(s)
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
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
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

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
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        out: Tensor,
            of shape (batch_size, channels, seq_len)
        """
        out = super().forward(input)
        return out

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class DownBranchedDoubleConv(nn.Module, SizeMixin):
    """
    the bottom block of the `subtract_unet`
    """

    __DEBUG__ = False
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
        dropouts: Union[Sequence[float], float] = 0.0,
        mode: str = "max",
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        down_scale: int,
            down sampling scale
        in_channels: int,
            number of channels in the input
        out_channels: sequence of sequence of int,
            number of channels produced by the (last) convolutional layer(s)
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
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
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

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
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        out: Tensor,
            of shape (batch_size, channels, seq_len)
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
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
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
    """
    Upscaling then double conv, with input of corr. down layer concatenated
    up sampling --> conv (conv --> (dropout -->) conv --> (dropout -->) conv)
        ^
        |
    extra input

    channels are shrinked after up sampling
    """

    __DEBUG__ = False
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
        dropouts: Union[Sequence[float], float] = 0.0,
        mode: str = "deconv",
        **config,
    ) -> NoReturn:
        """finished, NOT checked,

        Parameters
        ----------
        up_scale: int,
            scale of up sampling
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size) of the convolutional layers
        deconv_filter_length: int,
            only used when `mode` == "deconv"
            length(s) of the filters (kernel size) of the deconvolutional upsampling layer
        groups: int, default 1, not used currently,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        mode: str, default "deconv", case insensitive,
            mode of up sampling
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the deconvolutional layers
        """
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__deconv_filter_length = deconv_filter_length
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        # the following has to be checked
        # if bilinear, use the normal convolutions to reduce the number of channels
        if self.__mode == "deconv":
            self.__deconv_padding = max(
                0, (self.__deconv_filter_length - self.__up_scale) // 2
            )
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
        """

        Parameters
        ----------
        input: Tensor,
            input tensor from the previous layer
        down_output:Tensor: Tensor,
            input tensor of the last layer of corr. down block
        """
        output = self.up(input)
        output = torch.cat(
            [down_output, output], dim=1
        )  # concate along the channel axis
        output = self.conv(output)

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
            the output shape of this `DownDoubleConv` layer, given `seq_len` and `batch_size`
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
    """

    entry 0433 of CPSC2019
    """

    __DEBUG__ = False
    __name__ = "ECG_SUBTRACT_UNET"

    def __init__(self, classes: Sequence[str], n_leads: int, config: dict) -> NoReturn:
        """

        Parameters
        ----------
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
        self.n_classes = len(classes)
        self.__out_channels = len(classes)
        self.__in_channels = n_leads
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
            __debug_seq_len = 5000

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
        if self.__DEBUG__:
            __debug_output_shape = self.init_conv.compute_output_shape(__debug_seq_len)
            print(
                f"given seq_len = {__debug_seq_len}, init_conv output shape = {__debug_output_shape}"
            )
            _, _, __debug_seq_len = __debug_output_shape

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
            if self.__DEBUG__:
                __debug_output_shape = self.down_blocks[
                    f"down_{idx}"
                ].compute_output_shape(__debug_seq_len)
                print(
                    f"given seq_len = {__debug_seq_len}, down_{idx} output shape = {__debug_output_shape}"
                )
                _, _, __debug_seq_len = __debug_output_shape

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
        if self.__DEBUG__:
            __debug_output_shape = self.bottom_block.compute_output_shape(
                __debug_seq_len
            )
            print(
                f"given seq_len = {__debug_seq_len}, bottom_block output shape = {__debug_output_shape}"
            )
            _, _, __debug_seq_len = __debug_output_shape

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
            if self.__DEBUG__:
                __debug_output_shape = self.up_blocks[f"up_{idx}"].compute_output_shape(
                    __debug_seq_len
                )
                print(
                    f"given seq_len = {__debug_seq_len}, up_{idx} output shape = {__debug_output_shape}"
                )
                _, _, __debug_seq_len = __debug_output_shape

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
        if self.__DEBUG__:
            __debug_output_shape = self.out_conv.compute_output_shape(__debug_seq_len)
            print(
                f"given seq_len = {__debug_seq_len}, out_conv output shape = {__debug_output_shape}"
            )

        # for inference
        # if background counted in `classes`, use softmax
        # otherwise use sigmoid
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

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
        if self.config.init_batch_norm:
            x = self.init_bn(input)
        else:
            x = input

        # down
        to_concat = [self.init_conv(x)]
        # if self.__DEBUG__:
        #     print(f"shape of the init conv block output = {to_concat[-1].shape}")
        for idx in range(self.config.down_up_block_num - 1):
            to_concat.append(self.down_blocks[f"down_{idx}"](to_concat[-1]))
            # if self.__DEBUG__:
            #     print(f"shape of the {idx}-th down block output = {to_concat[-1].shape}")
        to_concat.append(self.bottom_block(to_concat[-1]))
        # if self.__DEBUG__:
        #     print(f"shape of the bottom block output = {to_concat[-1].shape}")

        # up
        up_input = to_concat[-1]
        to_concat = to_concat[-2::-1]
        for idx in range(self.config.down_up_block_num):
            up_output = self.up_blocks[f"up_{idx}"](up_input, to_concat[idx])
            up_input = up_output
            # if self.__DEBUG__:
            #     print(f"shape of the {idx}-th up block output = {up_output.shape}")

        # output
        output = self.out_conv(up_output)
        # if self.__DEBUG__:
        #     print(f"shape of out_conv layer output = {output.shape}")

        # to keep in accordance with other models
        # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
        output = output.permute(0, 2, 1)

        # TODO: consider adding CRF at the tail to make final prediction

        return output

    @torch.no_grad()
    def inference(
        self, input: Union[np.ndarray, Tensor], bin_pred_thr: float = 0.5
    ) -> Tensor:
        """ """
        NotImplementedError("implement a task specific inference method")

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
            the output shape of this model, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, seq_len, self.n_classes)
        return output_shape
