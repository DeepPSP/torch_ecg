"""
UNet structure models,
mainly for ECG wave delineation

References
----------
1. Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
2. https://github.com/milesial/Pytorch-UNet/

"""

import textwrap
import warnings
from copy import deepcopy
from typing import Optional, Sequence, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...cfg import CFG
from ...models._nets import (
    Conv_Bn_Activation,
    DownSample,
    MultiConv,
)
from ...model_configs import ECG_UNET_VANILLA_CONFIG
from ...utils.misc import add_docstring, CitationMixin
from ...utils.utils_nn import (
    CkptMixin,
    SizeMixin,
    compute_deconv_output_shape,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)


__all__ = [
    "ECG_UNET",
]


class DoubleConv(MultiConv):
    """Buildings blocks for UNet.

    2 convolutions (conv --> conv) with the same number of channels.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the last convolutional layer.
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size).
    subsample_lengths : int or Sequence[int], default 1
        Subsample length(s) (stride(s)) of the convolutions.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    out_activation : bool, default True
        If True, the last mini-block of :class:`Conv_Bn_Activation`
        will have activation as in `config`; otherwise, no activation.
    mid_channels : int, optional
        Number of channels produced by the first convolutional layer,
        defaults to `out_channels`.
    config : dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers.

    """

    __name__ = "DoubleConv"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_lengths: Union[Sequence[int], int],
        subsample_lengths: Union[Sequence[int], int] = 1,
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        out_activation: bool = True,
        mid_channels: Optional[int] = None,
        **config,
    ) -> None:
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
    """Downsampling block for the U-Net architecture.

    Downscaling with maxpool then double conv
    down sample (maxpool) --> double conv (conv --> conv)

    Channels are increased after down sampling.

    Parameters
    ----------
    down_scale : int
        Down sampling scale.
    in_channels : int
        Number of channels in the input tensor.
    out_channels : int
        Number of channels produced by the last convolutional layer.
    filter_lengths : int or Sequence[int]
        Length(s) of the filters (kernel size).
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    mid_channels : int, optional
        Number of channels produced by the first convolutional layer,
        defaults to `out_channels`.
    mode : str, default "max"
        Mode for down sampling,
        can be one of {:class:`DownSample`.__MODES__}.
    config : dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the convolutional layers.

    """

    __name__ = "DownDoubleConv"
    __MODES__ = deepcopy(DownSample.__MODES__)

    def __init__(
        self,
        down_scale: int,
        in_channels: int,
        out_channels: int,
        filter_lengths: Union[Sequence[int], int],
        groups: int = 1,
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        mid_channels: Optional[int] = None,
        mode: str = "max",
        **config,
    ) -> None:
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__mid_channels = mid_channels if mid_channels is not None else out_channels
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
        """Forward pass of the down sampling block.

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


class UpDoubleConv(nn.Module, SizeMixin):
    """Upsampling block of the U-Net architecture.

    Upscaling then double conv, with input of corr. down layer concatenated
    up sampling --> conv (conv --> conv)
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
    deconv_filter_length : int, optional
        Length(s) of the filters (kernel size) of the
        deconvolutional upsampling layer, used only when `mode` is "deconv".
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
        Not used currently.
    deconv_groups : int, default 1
        Connection pattern (of channels) of the deconvolutional upsampling layer,
        used only when `mode` is "deconv".
    dropouts : float or dict or Sequence[Union[float, dict]], default 0.0
        Dropout ratio after each :class:`Conv_Bn_Activation` block.
    mode : str, default "deconv"
        Mode for up sampling, can be one of {:class:`UpSample`.__MODES__}.
    mid_channels : int, optional
        Number of channels produced by the first deconvolutional layer,
        defaults to `out_channels`.
    config : dict
        Other hyper-parameters, including
        activation choices, weight initializer, batch normalization choices, etc.
        for the deconvolutional layers.

    """

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
        dropouts: Union[Sequence[Union[float, dict]], float, dict] = 0.0,
        mode: str = "deconv",
        mid_channels: Optional[int] = None,
        **config,
    ) -> None:
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
        """Forward pass of the up sampling block.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor from the previous layer,
            of shape ``(batch_size, n_channels, seq_len)``.
        down_output : torch.Tensor
            Input tensor of the last layer of corr. down sampling block,
            of shape ``(batch_size, n_channels', seq_len')``.

        Returns
        -------
        output : torch.Tensor
            Output tensor of the up sampling block,
            of shape ``(batch_size, n_channels'', seq_len')``.

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
        """Compute the output shape of the up sampling block.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            Batch size of the input tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the up sampling block.

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


class ECG_UNET(nn.Module, CkptMixin, SizeMixin, CitationMixin):
    """U-Net for (multi-lead) ECG wave delineation.

    The U-Net is a fully convolutional network originally
    proposed for biomedical image segmentation [1]_.
    This architecture is applied to ECG wave delineation in [2]_.
    This implementation is based on an open-source implementation
    on GitHub [3]_.

    Parameters
    ----------
    classes : Sequence[str]
        List of names of the classes.
    n_leads : int
        Number of input leads (number of input channels).
    config : CFG, optional,
        Other hyper-parameters, including kernel sizes, etc.
        Refer to the corresponding config file.

    References
    ----------
    .. [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox.
           "U-net: Convolutional networks for biomedical image segmentation."
           International Conference on Medical image computing and computer-assisted intervention. Springer, 2015.
    .. [2] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov.
           "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
    .. [3] https://github.com/milesial/Pytorch-UNet/

    """

    __name__ = "ECG_UNET"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)  # final out_channels
        self.__out_channels = self.n_classes
        self.__in_channels = n_leads
        self.config = deepcopy(ECG_UNET_VANILLA_CONFIG)
        if not config:
            warnings.warn(
                "No config is provided, using default config.", RuntimeWarning
            )
        self.config.update(deepcopy(config) or {})

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
        to_concat = [self.init_conv(input)]
        for idx in range(self.config.down_up_block_num):
            to_concat.append(self.down_blocks[f"down_{idx}"](to_concat[-1]))
        up_input = to_concat[-1]
        to_concat = to_concat[-2::-1]
        for idx in range(self.config.down_up_block_num):
            up_output = self.up_blocks[f"up_{idx}"](up_input, to_concat[idx])
            up_input = up_output
        output = self.out_conv(up_output)

        # to keep in accordance with other models
        # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
        output = output.permute(0, 2, 1)

        # TODO: consider adding CRF at the tail to make final prediction

        return output

    @torch.no_grad()
    def inference(self, input: Tensor, bin_pred_thr: float = 0.5) -> Tensor:
        """Method for making inference on a single input."""
        raise NotImplementedError("implement a task specific inference method")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            The length of the input signal tensor.
        batch_size : int, optional
            The batch size of the input signal tensor.

        Returns
        -------
        output_shape : sequence
            The output shape of the model.

        """
        output_shape = (batch_size, seq_len, self.n_classes)
        return output_shape

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.1007/978-3-030-30425-6_29"]))
