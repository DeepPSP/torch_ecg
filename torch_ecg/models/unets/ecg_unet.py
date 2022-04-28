"""
UNet structure models,
mainly for ECG wave delineation

References
----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
[2] https://github.com/milesial/Pytorch-UNet/

"""

from copy import deepcopy
from typing import NoReturn, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...cfg import DEFAULTS, CFG
from ...models._nets import (
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
    "ECG_UNET",
]


class DoubleConv(MultiConv):
    """

    building blocks of UNet

    References
    ----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    """

    __DEBUG__ = False
    __name__ = "DoubleConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        out_activation: bool = True,
        mid_channels: Optional[int] = None,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the last convolutional layer
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
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers

        """
        _mid_channels = mid_channels if mid_channels is not None else out_channels
        _out_channels = [_mid_channels, out_channels]

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


class DownDoubleConv(nn.Sequential, SizeMixin):
    """
    Downscaling with maxpool then double conv
    down sample (maxpool) --> double conv (conv --> conv)

    channels are increased after down sampling

    References
    ----------
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    """

    __DEBUG__ = False
    __name__ = "DownDoubleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: int,
        filter_lengths: Union[Sequence[int], int],
        groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        mid_channels: Optional[int] = None,
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
        out_channels: int,
            number of channels produced by the last convolutional layer
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        mid_channels: int, optional,
            number of channels produced by the first convolutional layer,
            defaults to `out_channels`
        mode: str, default "max",
            mode for down sampling,
            can be one of `DownSample.__MODES__`
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
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
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
            "double_conv",
            DoubleConv(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                filter_lengths=filter_lengths,
                subsample_lengths=1,
                groups=groups,
                dropouts=dropouts,
                mid_channels=self.__mid_channels,
                **(self.config),
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
        out = super().forward(input)
        return out

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class UpDoubleConv(nn.Module, SizeMixin):
    """
    Upscaling then double conv, with input of corr. down layer concatenated
    up sampling --> conv (conv --> conv)
        ^
        |
    extra input

    channels are shrinked after up sampling

    """

    __DEBUG__ = False
    __name__ = "UpDoubleConv"
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
        deconv_groups: int = 1,
        dropouts: Union[Sequence[float], float] = 0.0,
        mode: str = "deconv",
        mid_channels: Optional[int] = None,
        **config,
    ) -> NoReturn:
        """

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
        deconv_groups: int, default 1,
            only used when `mode` == "deconv"
            connection pattern (of channels) of the deconvolutional upsampling layer
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        mode: str, default "deconv", case insensitive,
            mode of up sampling
        mid_channels: int, optional,
            number of channels produced by the first deconvolutional layer,
            defaults to `out_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the deconvolutional layers

        """
        super().__init__()
        self.__up_scale = up_scale
        self.__in_channels = in_channels
        self.__mid_channels = (
            mid_channels if mid_channels is not None else in_channels // 2
        )
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
                groups=deconv_groups,
            )
        else:
            self.up = nn.Upsample(
                scale_factor=self.__up_scale,
                mode=mode,
            )
        self.conv = DoubleConv(
            in_channels=self.__in_channels + self.__in_channels // 2,
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
            input tensor from the previous layer,
            of shape (batch_size, n_channels, seq_len)
        down_output:Tensor: Tensor,
            input tensor of the last layer of corr. down block,
            of shape (batch_size, n_channels', seq_len')

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels'', seq_len')

        """
        output = self.up(input)

        diff_sig_len = down_output.shape[-1] - output.shape[-1]
        output = F.pad(output, [diff_sig_len // 2, diff_sig_len - diff_sig_len // 2])

        # TODO: consider the case `groups` > 1 when concatenating
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
            the output shape of this `UpDoubleConv` layer, given `seq_len` and `batch_size`

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


class ECG_UNET(nn.Module, CkptMixin, SizeMixin):
    """

    UNet for (multi-lead) ECG wave delineation

    References
    ----------
    [1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
    [2] https://github.com/milesial/Pytorch-UNet/

    """

    __DEBUG__ = False
    __name__ = "ECG_UNET"

    def __init__(self, classes: Sequence[str], n_leads: int, config: dict) -> NoReturn:
        """

        Parameters
        ----------
        classes: sequence of int,
            name of the classes
        n_leads: int,
            number of input leads (number of input channels)
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)  # final out_channels
        self.__out_channels = self.n_classes
        self.__in_channels = n_leads
        self.config = CFG(deepcopy(config))
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
            __debug_seq_len = 4000

        self.init_conv = DoubleConv(
            in_channels=self.__in_channels,
            out_channels=self.config.init_num_filters,
            filter_lengths=self.config.init_filter_length,
            subsample_lengths=1,
            groups=self.config.groups,
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
        for idx in range(self.config.down_up_block_num):
            self.down_blocks[f"down_{idx}"] = DownDoubleConv(
                down_scale=self.config.down_scales[idx],
                in_channels=in_channels,
                out_channels=self.config.down_num_filters[idx],
                filter_lengths=self.config.down_filter_lengths[idx],
                groups=self.config.groups,
                mode=self.config.down_mode,
                **(self.config.down_block),
            )
            in_channels = self.config.down_num_filters[idx]
            if self.__DEBUG__:
                __debug_output_shape = self.down_blocks[
                    f"down_{idx}"
                ].compute_output_shape(__debug_seq_len)
                print(
                    f"given seq_len = {__debug_seq_len}, down_{idx} output shape = {__debug_output_shape}"
                )
                _, _, __debug_seq_len = __debug_output_shape

        self.up_blocks = nn.ModuleDict()
        in_channels = self.config.down_num_filters[-1]
        for idx in range(self.config.down_up_block_num):
            self.up_blocks[f"up_{idx}"] = UpDoubleConv(
                up_scale=self.config.up_scales[idx],
                in_channels=in_channels,
                out_channels=self.config.up_num_filters[idx],
                filter_lengths=self.config.up_conv_filter_lengths[idx],
                deconv_filter_length=self.config.up_deconv_filter_lengths[idx],
                groups=self.config.groups,
                mode=self.config.up_mode,
                **(self.config.up_block),
            )
            in_channels = self.config.up_num_filters[idx]
            if self.__DEBUG__:
                __debug_output_shape = self.up_blocks[f"up_{idx}"].compute_output_shape(
                    __debug_seq_len
                )
                print(
                    f"given seq_len = {__debug_seq_len}, up_{idx} output shape = {__debug_output_shape}"
                )
                _, _, __debug_seq_len = __debug_output_shape

        self.out_conv = Conv_Bn_Activation(
            in_channels=self.config.up_num_filters[-1],
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
        to_concat = [self.init_conv(input)]
        # if self.__DEBUG__:
        #     print(f"shape of init conv block output = {to_concat[-1].shape}")
        for idx in range(self.config.down_up_block_num):
            to_concat.append(self.down_blocks[f"down_{idx}"](to_concat[-1]))
            # if self.__DEBUG__:
            #     print(f"shape of {idx}-th down block output = {to_concat[-1].shape}")
        up_input = to_concat[-1]
        to_concat = to_concat[-2::-1]
        for idx in range(self.config.down_up_block_num):
            # if self.__DEBUG__:
            #     print(f"shape of {idx}-th up block 1st input = {up_input.shape}")
            #     print(f"shape of {idx}-th up block 2nd input (from down) = {to_concat[idx].shape}")
            up_output = self.up_blocks[f"up_{idx}"](up_input, to_concat[idx])
            up_input = up_output
            # if self.__DEBUG__:
            #     print(f"shape of {idx}-th up block output = {up_output.shape}")
        output = self.out_conv(up_output)
        # if self.__DEBUG__:
        #     print(f"shape of out_conv layer output = {output.shape}")

        # to keep in accordance with other models
        # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
        output = output.permute(0, 2, 1)

        # TODO: consider adding CRF at the tail to make final prediction

        return output

    @torch.no_grad()
    def inference(self, input: Tensor, bin_pred_thr: float = 0.5) -> Tensor:
        """ """
        raise NotImplementedError("implement a task specific inference method")

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
