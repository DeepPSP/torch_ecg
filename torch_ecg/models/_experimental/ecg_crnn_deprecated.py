"""
deprecated (archived) crnn structure models
"""
from copy import deepcopy
from itertools import repeat
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real, Number

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg
from torch_ecg.model_configs.ati_cnn import ATI_CNN_CONFIG
from torch_ecg.model_configs.cpsc import CPSC_CONFIG
from .nets import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM,
    # AML_Attention, AML_GatedAttention,
    AttentionWithContext, MultiHeadAttention,
)
from .ecg_crnn import CPSCBlock, CPSCCNN
from torch_ecg.utils.utils_nn import compute_conv_output_shape, compute_module_size
from torch_ecg.utils.misc import dict_to_str


if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


@DeprecationWarning
class CPSC(nn.Sequential):
    """
    SOTA model of the CPSC2018 challenge
    """
    __DEBUG__ = True
    __name__ = "CPSC"
    
    def __init__(self, classes:list, input_len:int, **config) -> NoReturn:
        """

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        input_len: int,
            sequence length (last dim.) of the input
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.n_leads = 12
        self.input_len = input_len
        self.config = deepcopy(CPSC_CONFIG)
        self.config.update(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of CPSC is as follows\n{dict_to_str(self.config)}")

        cnn_choice = self.config.cnn.name.lower()
        if cnn_choice == 'cpsc_2018':
            cnn_config = self.config.cnn.cpsc
            self.cnn = CPSCCNN(
                filter_lengths=cnn_config.filter_lengths,
                subsample_lengths=cnn_config.subsample_lengths,
                dropouts=cnn_config.dropouts,
            )
        else:
            raise NotImplementedError

        cnn_output_shape = self.cnn.compute_output_shape(self.input_len)

        self.rnn = nn.Sequential()
        self.rnn.add_module(
            "bidirectional_gru",
            nn.GRU(input_size=12, hidden_size=12, bidirectional=True),
        )
        self.rnn.add_module(
            "leaky",
            Activations["leaky"](negative_slope=0.2),
        )
        self.rnn.add_module(
            "dropout",
            nn.Dropout(0.2),
        )
        self.rnn.add_module(
            "attention",
            AttentionWithContext(12, 12),
        )
        self.rnn.add_module(
            "batch_normalization",
            nn.BatchNorm1d(12),
        )
        self.rnn.add_module(
            "leaky",
            Activations["leaky"](negative_slope=0.2),
        )
        self.rnn.add_module(
            "dropout",
            nn.Dropout(0.2),
        )

        # self.clf = nn.Linear()  # TODO: set correct the in-and-out-features
        
    def forward(self, input:Tensor) -> Tensor:
        """
        """
        output = self.cnn(input)
        output = self.rnn(output)
        return output

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


@DeprecationWarning
class ResNetStanfordBlock(nn.Module):
    """
    building blocks of the CNN feature extractor `ResNetStanford`
    """
    __DEBUG__ = True
    __name__ = "ResNetStanfordBlock"
    
    def __init__(self, block_index:int, in_channels:int, num_filters:int, filter_length:int, subsample_length:int, dilation:int=1, **config) -> NoReturn:
        """ finished, checked,

        the main stream uses `subsample_length` as stride to perform down-sampling,
        the short cut uses `subsample_length` as pool size to perform down-sampling,

        Parameters:
        -----------
        block_index: int,
            index of the block in the whole sequence of `ResNetStanford`
        in_channels: int,
            number of features (channels) of the input
        num_filters: int,
            number of filters for the convolutional layers
        filter_length: int,
            length (size) of the filter kernels
        subsample_length: int,
            subsample length,
            including pool size for short cut, and stride for the top convolutional layer
        config: dict,
            other hyper-parameters, including
            filter length (kernel size), activation choices, weight initializer, dropout,
            and short cut patterns, etc.

        Issues:
        -------
        1. even kernel size would case mismatch of shapes of main stream and short cut
        """
        super().__init__()
        self.__block_index = block_index
        self.__in_channels = in_channels
        self.__out_channels = num_filters
        self.__kernel_size = filter_length
        self.__down_scale = subsample_length
        self.__stride = subsample_length
        self.config = ED(deepcopy(config))
        self.__num_convs = self.config.num_skip
        
        self.__increase_channels = (self.__out_channels > self.__in_channels)
        self.short_cut = self._make_short_cut_layer()
        
        self.main_stream = nn.Sequential()
        conv_in_channels = self.__in_channels
        for i in range(self.__num_convs):
            if not (block_index == 0 and i == 0):
                self.main_stream.add_module(
                    f"ba_{self.__block_index}_{i}",
                    Bn_Activation(
                        num_features=self.__in_channels,
                        activation=self.config.activation,
                        kw_activation=self.config.kw_activation,
                        dropout=self.config.dropout if i > 0 else 0,
                    ),
                )
            self.main_stream.add_module(
                f"conv_{self.__block_index}_{i}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels,
                    kernel_size=self.__kernel_size,
                    stride = (self.__stride if i == 0 else 1),
                    batch_norm=False,
                    activation=None,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                )
            )
            conv_in_channels = self.__out_channels

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        if self.__DEBUG__:
            print(f"forwarding in the {self.__block_index}-th `ResNetStanfordBlock`...")
            args = {k.split("__")[1]:v for k,v in self.__dict__.items() if isinstance(v, Number) and '__' in k}
            print(f"input arguments:\n{args}")
            print(f"input shape = {input.shape}")
        if self.short_cut:
            sc = self.short_cut(input)
        else:
            sc = input
        output = self.main_stream(input)
        if self.__DEBUG__:
            print(f"shape of short_cut output = {sc.shape}, shape of main stream output = {output.shape}")
        output = output +sc
        return output

    def _make_short_cut_layer(self) -> Union[nn.Module, type(None)]:
        """
        """
        if self.__down_scale > 1 or self.__increase_channels:
            if self.config.increase_channels_method.lower() == 'conv':
                short_cut = DownSample(
                    down_scale=self.__down_scale,
                    in_channels=self.__in_channels,
                    out_channels=self.__out_channels,
                    batch_norm=False,
                    mode=self.config.subsample_mode,
                )
            if self.config.increase_channels_method.lower() == 'zero_padding':
                short_cut = nn.Sequential(
                    DownSample(
                        down_scale=self.__down_scale,
                        in_channels=self.__in_channels,
                        out_channels=self.__in_channels,
                        batch_norm=False,
                        mode=self.config.subsample_mode,
                    ),
                    ZeroPadding(self.__in_channels, self.__out_channels),
                )
        else:
            short_cut = None
        return short_cut

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self.main_stream:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)


@DeprecationWarning
class ResNetStanford(nn.Sequential):
    """
    the model proposed in ref. [1] and implemented in ref. [2]

    References:
    -----------
    [1] Hannun, Awni Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65.
    [2] https://github.com/awni/ecg
    """
    __DEBUG__ = True
    __name__ = "ResNetStanford"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ NOT finished, NOT checked,
        
        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, including
            number of convolutional layers, number of filters for each layer, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))

        if self.__DEBUG__:
            print(f"configuration of ResNetStanford is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels=self.__in_channels,
                out_channels=self.config.num_filters_start,
                kernel_size=self.config.filter_lengths,
                stride=1,
                batch_norm=True,
                activation=self.config.block.activation,
                kw_activation=self.config.block.kw_activation,
                kernel_initializer=self.config.block.kernel_initializer,
                kw_initializer=self.config.block.kw_initializer,
            )
        )

        module_in_channels = self.config.num_filters_start
        for idx, subsample_length in enumerate(self.config.subsample_lengths):
            num_filters = self.get_num_filters_at_index(idx, self.config.num_filters_start)
            self.add_module(
                f"resnet_block_{idx}",
                ResNetStanfordBlock(
                    block_index=idx,
                    in_channels=module_in_channels,
                    num_filters=num_filters,
                    filter_length=self.config.filter_lengths,
                    subsample_length=subsample_length,
                    **(self.config.block),
                )
            )
            module_in_channels = num_filters
            # if idx % self.config.increase_channels_at == 0 and idx > 0:
            #     module_in_channels *= 2

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        output = super().forward(input)
        return output

    def get_num_filters_at_index(self, index:int, num_start_filters:int) -> int:
        """ finished, checked,

        Parameters:
        -----------
        index: int,
            index of a `ResNetStanfordBlock` in the sequence of such blocks in the whole network
        num_start_filters: int,
            number of filters of the first convolutional layer of the whole network

        Returns:
        --------
        num_filters: int,
            number of filters at the {index}-th `ResNetStanfordBlock`
        """
        num_filters = 2**int(index / self.config.increase_channels_at) * num_start_filters
        return num_filters

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this Module, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        return compute_module_size(self)
