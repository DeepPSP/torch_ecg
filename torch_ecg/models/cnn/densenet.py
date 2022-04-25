"""
The core part of the SOTA model (framework) of CPSC2020

Its key points:
1. ECG signal pre-processing: filters out the high frequency noise and baseline drift in the ECG signal
2. QRS complex detection
3. noisy heartbeat recognition: transient noise and artifacts
4. atrial fibrillation recognition: remove SPB and carefully distinguish PVC and AF beats with aberrant ventricular conduction in episodes with atrial fibrillation
5. PVC and SPB model detection: DenseNet
6. post-processing with clinical rules: a set of clinical experiences and rules including rhythm and morphological rules to suppress false positives and search for false negatives of PVC and SPB detection

"""

import math
from copy import deepcopy
from itertools import repeat
from typing import NoReturn, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
)
from ...utils.misc import dict_to_str, list_sum, add_docstring
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "DenseNet",
    "DenseBasicBlock",
    "DenseBottleNeck",
    "DenseMacroBlock",
    "DenseTransition",
]


class DenseBasicBlock(nn.Module, SizeMixin):
    """

    the basic building block for DenseNet,
    consisting of normalization -> activation -> convolution (-> dropout (optional)),
    the output Tensor is the concatenation of old features (input) with new features

    """

    __DEBUG__ = True
    __name__ = "DenseBasicBlock"
    __DEFAULT_CONFIG__ = CFG(
        activation="relu",
        kw_activation={"inplace": True},
        memory_efficient=False,
    )

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        filter_length: int,
        groups: int = 1,
        bias: bool = False,
        dropout: float = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        growth_rate: int,
            number of features of (channels) output from the main stream,
            further concatenated to the shortcut,
            hence making the final number of output channels grow by this value
        filter_length: int,
            length (size) of the filter kernels
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            activation choices, memory_efficient choices, etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__growth_rate = growth_rate
        self.__kernel_size = filter_length
        self.__groups = groups
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        assert all([in_channels % groups == 0, growth_rate % groups == 0])

        self.bac = Conv_Bn_Activation(
            in_channels=self.__in_channels,
            out_channels=self.__growth_rate,
            kernel_size=self.__kernel_size,
            stride=1,
            dilation=1,
            groups=self.__groups,
            norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

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
        new_features = self.bac(input)
        if self.dropout:
            new_features = self.dropout(new_features)
        if self.__groups == 1:
            output = torch.cat([input, new_features], dim=1)
        else:  # see TODO of `DenseNet`
            # input width per group
            iw_per_group = self.__in_channels // self.__groups
            # new features width per group
            nfw_per_group = self.__growth_rate // self.__groups
            output = torch.cat(
                list_sum(
                    [
                        [
                            input[..., iw_per_group * i : iw_per_group * (i + 1), ...],
                            new_features[
                                ..., nfw_per_group * i : nfw_per_group * (i + 1), ...
                            ],
                        ]
                        for i in range(self.__groups)
                    ]
                ),
                1,
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
        out_channels = self.__in_channels + self.__growth_rate
        output_shape = (batch_size, out_channels, seq_len)
        return output_shape


class DenseBottleNeck(nn.Module, SizeMixin):
    """

    bottleneck modification of `DenseBasicBlock`,
    with an additional prefixed sequence of
    (normalization -> activation -> convolution of kernel size 1)
    """

    __DEBUG__ = True
    __name__ = "DenseBottleNeck"
    __DEFAULT_CONFIG__ = CFG(
        activation="relu",
        kw_activation={"inplace": True},
        memory_efficient=False,
    )

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        filter_length: int,
        groups: int = 1,
        bias: bool = False,
        dropout: float = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        growth_rate: int,
            number of features of (channels) output from the main stream,
            further concatenated to the shortcut,
            hence making the final number of output channels grow by this value
        bn_size: int,
            base width of intermediate layers (the bottleneck)
        filter_length: int,
            length (size) of the filter kernels of the second convolutional layer
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            activation choices, memory_efficient choices, etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__growth_rate = growth_rate
        self.__bn_size = bn_size
        self.__kernel_size = filter_length
        self.__groups = groups
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        bottleneck_channels = self.__bn_size * self.__growth_rate

        self.neck_conv = Conv_Bn_Activation(
            in_channels=self.__in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=groups,
            norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        self.main_conv = Conv_Bn_Activation(
            in_channels=bottleneck_channels,
            out_channels=self.__growth_rate,
            kernel_size=self.__kernel_size,
            stride=1,
            dilation=1,
            groups=self.__groups,
            norm=True,
            activation=self.config.activation.lower(),
            kw_activation=self.config.kw_activation,
            bias=bias,
            ordering="bac",
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def bn_function(self, input: Tensor) -> Tensor:
        """

        the `not memory_efficient` way

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        bottleneck_output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        bottleneck_output = self.neck_conv(input)
        return bottleneck_output

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
        if self.config.memory_efficient:
            raise NotImplementedError
        else:
            new_features = self.bn_function(input)
        new_features = self.main_conv(new_features)
        if self.dropout:
            new_features = self.dropout(new_features)
        if self.__groups == 1:
            output = torch.cat([input, new_features], dim=1)
        else:  # see TODO of `DenseNet`
            # input width per group
            iw_per_group = self.__in_channels // self.__groups
            # new features width per group
            nfw_per_group = self.__growth_rate // self.__groups
            output = torch.cat(
                list_sum(
                    [
                        [
                            input[..., iw_per_group * i : iw_per_group * (i + 1), ...],
                            new_features[
                                ..., nfw_per_group * i : nfw_per_group * (i + 1), ...
                            ],
                        ]
                        for i in range(self.__groups)
                    ]
                ),
                1,
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
        out_channels = self.__in_channels + self.__growth_rate
        output_shape = (batch_size, out_channels, seq_len)
        return output_shape


class DenseMacroBlock(nn.Sequential, SizeMixin):
    """

    macro blocks for `DenseNet`,
    stacked sequence of builing blocks of similar pattern

    """

    __DEBUG__ = True
    __name__ = "DenseMacroBlock"
    building_block = DenseBottleNeck

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rates: Union[Sequence[int], int],
        bn_size: int,
        filter_lengths: Union[Sequence[int], int],
        groups: int = 1,
        bias: bool = False,
        dropout: float = 0.0,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        num_layers: int,
            number of building block layers
        growth_rates: sequence of int, or int,
            growth rate(s) for each building block layers,
            if is sequence of int, should have length equal to `num_layers`
        bn_size: int,
            base width of intermediate layers for `DenseBottleNeck`,
            not used for `DenseBasicBlock`
        filter_lengths: sequence of int, or int,
            filter lengths(s) (kernel size(s)) for each building block layers,
            if is sequence of int, should have length equal to `num_layers`
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        dropout: float, default 0.0,
            dropout rate of the new features produced from the main stream
        config: dict,
            other hyper-parameters, including
            extra kw for `DenseBottleNeck`, and
            activation choices, memory_efficient choices, etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__num_layers = num_layers
        if isinstance(growth_rates, int):
            self.__growth_rates = list(repeat(growth_rates, num_layers))
        else:
            self.__growth_rates = list(growth_rates)
        assert len(self.__growth_rates) == self.__num_layers
        self.__bn_size = bn_size
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, num_layers))
        else:
            self.__filter_lengths = list(filter_lengths)
        assert len(self.__filter_lengths) == self.__num_layers
        self.__groups = groups
        self.config = deepcopy(config)
        if self.config.get("building_block", "").lower() in [
            "basic",
            "basic_block",
        ]:
            self.building_block = DenseBasicBlock

        for idx in range(self.__num_layers):
            self.add_module(
                f"dense_building_block_{idx}",
                self.building_block(
                    in_channels=self.__in_channels + idx * self.__growth_rates[idx],
                    growth_rate=self.__growth_rates[idx],
                    bn_size=self.__bn_size,
                    filter_length=self.__filter_lengths[idx],
                    bias=bias,
                    dropout=dropout,
                    **(self.config),
                ),
            )

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class DenseTransition(nn.Sequential, SizeMixin):
    """

    transition blocks between `DenseMacroBlock`s,
    used to perform sub-sampling,
    and compression of channels if specified

    """

    __DEBUG__ = True
    __name__ = "DenseTransition"
    __DEFAULT_CONFIG__ = CFG(
        activation="relu",
        kw_activation={"inplace": True},
        subsample_mode="avg",
    )

    def __init__(
        self,
        in_channels: int,
        compression: float = 1.0,
        subsample_length: int = 2,
        groups: int = 1,
        bias: bool = False,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        compression: float, default 1.0,
            compression factor,
            proportion of the number of output channels to the number of input channels
        subsample_length: int, default 2,
            subsampling length (size)
        groups: int, default 1,
            pattern of connections between inputs and outputs,
            for more details, ref. `nn.Conv1d`
        bias: bool, default False,
            if True, the convolutional layer has `bias` set `True`, otherwise `False`
        config: dict,
            other parameters, including
            activation choices, subsampling mode (method), etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__compression = compression
        self.__subsample_length = subsample_length
        self.__groups = groups
        assert 0 < self.__compression <= 1.0 and self.__in_channels % self.__groups == 0
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))

        # input width per group
        iw_per_group = self.__in_channels // self.__groups
        # new feature widths per group
        nfw_per_group = math.floor(iw_per_group * self.__compression)
        self.__out_channels = nfw_per_group * self.__groups

        self.add_module(
            "bac",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                groups=self.__groups,
                norm=True,
                activation=self.config.activation.lower(),
                kw_activation=self.config.kw_activation,
                bias=bias,
                ordering="bac",
            ),
        )
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels,
                mode=self.config.subsample_mode.lower(),
            ),
        )

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)


class DenseNet(nn.Sequential, SizeMixin):
    """

    The core part of the SOTA model (framework) of CPSC2020

    References
    ----------
    [1] G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.
    [2] G. Huang, Z. Liu, G. Pleiss, L. Van Der Maaten and K. Weinberger, "Convolutional Networks with Dense Connectivity," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2019.2918284.
    [3] https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    [4] https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
    [5] https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
    [6] https://github.com/liuzhuang13/DenseNet/tree/master/models

    NOTE
    ----
    the difference of forward output of [5] from others, however [5] doesnot support dropout

    TODO
    ----
    1. for `groups` > 1, the concatenated output should be re-organized in the channel dimension?
    2. memory-efficient mode, i.e. storing the `new_features` in a shared memory instead of stacking in newly created `Tensor`s after each mini-block

    """

    __DEBUG__ = True
    __name__ = "DenseNet"
    __DEFAULT_CONFIG__ = CFG(
        bias=False,
        activation="relu",
        kw_activation={"inplace": True},
        kernel_initializer="he_normal",
        kw_initializer={},
        init_subsample_mode="avg",
    )

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set:
            num_layers: sequence of int,
                number of building block layers of each dense (macro) block
            init_num_filters: sequence of int,
                number of filters of the first convolutional layer
            init_filter_length: sequence of int,
                filter length (kernel size) of the first convolutional layer
            init_conv_stride: int,
                stride of the first convolutional layer
            init_pool_size: int,
                pooling kernel size of the first pooling layer
            init_pool_stride: int,
                pooling stride of the first pooling layer
            growth_rates: int or sequence of int or sequence of sequences of int,
                growth rates of the building blocks,
                with granularity to the whole network, or to each dense (macro) block,
                or to each building block
            filter_lengths: int or sequence of int or sequence of sequences of int,
                filter length(s) (kernel size(s)) of the convolutions,
                with granularity to the whole network, or to each macro block,
                or to each building block
            subsample_lengths: int or sequence of int,
                subsampling length(s) (ratio(s)) of the transition blocks
            compression: float,
                compression factor of the transition blocks
            bn_size: int,
                bottleneck base width, used only when building block is `DenseBottleNeck`
            dropouts: int,
                dropout ratio of each building block
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for the building blocks
            for a full list of configurable parameters, ref. corr. config file

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(deepcopy(config))
        self.__num_blocks = len(self.config.num_layers)
        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        self.add_module(
            "init_cba",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.init_num_filters,
                kernel_size=self.config.init_filter_length,
                stride=1,
                dilation=1,
                groups=self.config.groups,
                norm=True,
                activation=self.config.activation.lower(),
                kw_activation=self.config.kw_activation,
                bias=self.config.bias,
            ),
        )
        self.add_module(
            "init_pool",
            DownSample(
                down_scale=self.config.init_pool_stride,
                in_channels=self.config.init_num_filters,
                kernel_size=self.config.init_pool_size,
                padding=(self.config.init_pool_size - 1) // 2,
                mode=self.config.init_subsample_mode.lower(),
            ),
        )

        if isinstance(self.config.growth_rates, int):
            self.__growth_rates = list(
                repeat(self.config.growth_rates, self.__num_blocks)
            )
        else:
            self.__growth_rates = list(self.config.growth_rates)
        assert (
            len(self.__growth_rates) == self.__num_blocks
        ), f"`config.growth_rates` indicates {len(self.__growth_rates)} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"
        if isinstance(self.config.filter_lengths, int):
            self.__filter_lengths = list(
                repeat(self.config.filter_lengths, self.__num_blocks)
            )
        else:
            self.__filter_lengths = list(self.config.filter_lengths)
            assert (
                len(self.__filter_lengths) == self.__num_blocks
            ), f"`config.filter_lengths` indicates {len(self.__filter_lengths)} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"
        if isinstance(self.config.subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(self.config.subsample_lengths, self.__num_blocks - 1)
            )
        else:
            self.__subsample_lengths = list(self.config.subsample_lengths)
            assert (
                len(self.__subsample_lengths) == self.__num_blocks - 1
            ), f"`config.subsample_lengths` indicates {len(self.__subsample_lengths)+1} macro blocks, while `config.num_layers` indicates {self.__num_blocks}"

        macro_in_channels = self.config.init_num_filters
        for idx, macro_num_layers in enumerate(self.config.num_layers):
            dmb = DenseMacroBlock(
                in_channels=macro_in_channels,
                num_layers=macro_num_layers,
                growth_rates=self.__growth_rates[idx],
                bn_size=self.config.bn_size,
                filter_lengths=self.__filter_lengths[idx],
                groups=self.config.groups,
                bias=self.config.bias,
                dropout=self.config.dropout,
                **(self.config.block),
            )
            _, transition_in_channels, _ = dmb.compute_output_shape()
            self.add_module(f"dense_macro_block_{idx}", dmb)
            if idx < self.__num_blocks - 1:
                dt = DenseTransition(
                    in_channels=transition_in_channels,
                    compression=self.config.compression,
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.config.groups,
                    bias=self.config.bias,
                    **(self.config.transition),
                )
                _, macro_in_channels, _ = dt.compute_output_shape()
                self.add_module(f"transition_{idx}", dt)

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    @property
    def in_channels(self) -> int:
        return self.__in_channels
