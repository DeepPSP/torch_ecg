"""
Sequence labeling nets, for wave delineation,

the labeling granularity is the frequency of the input signal,
divided by the length (counted by the number of basic blocks) of each branch

pipeline:
multi-scopic cnn --> (bidi-lstm -->) "attention" (se block) --> seq linear

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg
from torch_ecg.utils.utils_nn import compute_conv_output_shape
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.model_configs import ECG_SEQ_LAB_NET_CONFIG
from .nets import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    # MultiConv,
    SEBlock, GlobalContextBlock,
    DownSample,
    StackedLSTM,
    AttentivePooling,
    SeqLin,
)

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_SEQ_LAB_NET",
]


class MultiScopicBasicBlock(nn.Sequential):
    """ finished, checked,

    basic building block of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)

    (conv -> activation) * N --> bn --> down_sample
    """
    __DEBUG__ = False
    __name__ = "MultiScopicBasicBlock"

    def __init__(self, in_channels:int, scopes:Sequence[int], num_filters:Union[int,Sequence[int]], filter_lengths:Union[int,Sequence[int]], subsample_length:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        scopes: sequence of int,
            scopes of the convolutional layers, via `dilation`
        num_filters: int or sequence of int,
        filter_lengths: int or sequence of int,
        subsample_length: int,
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_convs = len(self.__scopes)
        if isinstance(num_filters, int):
            self.__out_channels = list(repeat(num_filters, self.__num_convs))
        else:
            self.__out_channels = num_filters
            assert len(self.__out_channels) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `num_filters` indicates {len(self.__out_channels)}"
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = filter_lengths
            assert len(self.__filter_lengths) == self.__num_convs, \
                f"`scopes` indicates {self.__num_convs} convolutional layers, while `filter_lengths` indicates {len(self.__filter_lengths)}"
        self.__subsample_length = subsample_length
        self.__groups = groups
        self.config = ED(deepcopy(config))

        conv_in_channels = self.__in_channels
        for idx in range(self.__num_convs):
            self.add_module(
                f"ca_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels[idx],
                    kernel_size=self.__filter_lengths[idx],
                    stride=1,
                    dilation=self.__scopes[idx],
                    groups=self.__groups,
                    batch_norm=False,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                )
            )
            conv_in_channels = self.__out_channels[idx]
        self.add_module(
            "bn",
            nn.BatchNorm1d(self.__out_channels[-1])
        )
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels[-1],
                groups=self.__groups,
                # padding=
                batch_norm=False,
                mode=self.config.subsample_mode,
            )
        )
        if self.config.dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.config.dropout, inplace=False)
            )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
        for idx, module in enumerate(self):
            if idx == self.__num_convs:  # bn layer
                continue
            elif self.config.dropout > 0 and idx == len(self)-1:  # dropout layer
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiScopicBranch(nn.Sequential):
    """ finished, checked,
    
    branch path of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = False
    __name__ = "MultiScopicBranch"

    def __init__(self, in_channels:int, scopes:Sequence[Sequence[int]], num_filters:Union[Sequence[int],Sequence[Sequence[int]]], filter_lengths:Union[Sequence[int],Sequence[Sequence[int]]], subsample_lengths:Union[int,Sequence[int]], groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of features (channels) of the input
        scopes: sequence of sequence of int,
            scopes (in terms of `dilation`) for the convolutional layers,
            each sequence of int is for one branch
        num_filters: sequence of int, or sequence of sequence of int,
            number of filters for the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same number of filters
        filter_lengths: sequence of int, or sequence of sequence of int,
            filter length (kernel size) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same filter length
        subsample_lengths: int, or sequence of int,
            subsample length (stride) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same subsample length
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other hyper-parameters, including
            dropout, activation choices, weight initializer, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_blocks = len(self.__scopes)
        self.__num_filters = num_filters
        assert len(self.__num_filters) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `num_filters` indicates {len(self.__num_filters)}"
        self.__filter_lengths = filter_lengths
        assert len(self.__filter_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `filter_lengths` indicates {llen(self.__filter_lengths)}"
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(repeat(subsample_lengths, self.__num_blocks))
        else:
            self.__subsample_lengths = filter_lengths
            assert len(self.__subsample_lengths) == self.__num_blocks, \
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, while `subsample_lengths` indicates {llen(self.__subsample_lengths)}"
        self.__groups = groups
        self.config = ED(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx in range(self.__num_blocks):
            self.add_module(
                f"block_{idx}",
                MultiScopicBasicBlock(
                    in_channels=block_in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.__num_filters[idx],
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.__groups,
                    dropout=self.config.dropouts[idx],
                    **(self.config.block)
                )
            )
            block_in_channels = self.__num_filters[idx]

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
        for idx, module in enumerate(self):
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiScopicCNN(nn.Module):
    """ finished, checked,

    CNN part of the SOTA model from CPSC2019 challenge (entry 0416)
    """
    __DEBUG__ = False
    __name__ = "MultiScopicCNN"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = ED(deepcopy(config))
        self.__scopes = self.config.scopes
        self.__num_branches = len(self.__scopes)

        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.branches = nn.ModuleDict()
        for idx in range(self.__num_branches):
            self.branches[f"branch_{idx}"] = \
                MultiScopicBranch(
                    in_channels=self.__in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.config.num_filters[idx],
                    filter_lengths=self.config.filter_lengths[idx],
                    subsample_lengths=self.config.subsample_lengths[idx],
                    dropouts=self.config.dropouts[idx],
                    block=self.config.block,  # a dict
                )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,
        
        Parameters:
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        branch_out = OrderedDict()
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            branch_out[key] = self.branches[key].forward(input)
        output = torch.cat(
            [branch_out[f"branch_{idx}"] for idx in range(self.__num_branches)],
            dim=1,  # along channels
        )
        return output
    
    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
        out_channels = 0
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            _, _branch_oc, _seq_len = \
                self.branches[key].compute_output_shape(seq_len, batch_size)
            out_channels += _branch_oc
        return (batch_size, out_channels, _seq_len)

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ECG_SEQ_LAB_NET(nn.Module):
    """ finished, checked,

    SOTA model from CPSC2019 challenge (entry 0416)

    pipeline:
    multi-scopic cnn --> (bidi-lstm -->) "attention" --> seq linear

    References:
    -----------
    [1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
    """
    __DEBUG__ = False
    __name__ = "ECG_SEQ_LAB_NET"

    def __init__(self, classes:Sequence[str], n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for sequence labeling
        n_leads: int,
            number of leads (number of input channels)
        input_len: int, optional,
            sequence length (last dim.) of the input,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.__out_channels = self.n_classes
        # self.__out_channels = self.n_classes if self.n_classes > 2 else 1
        self.n_leads = n_leads
        self.input_len = input_len
        self.config = ED(deepcopy(ECG_SEQ_LAB_NET_CONFIG))
        self.config.update(config or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        __debug_seq_len = self.input_len or 4000
        
        # currently, the CNN part only uses `MultiScopicCNN`
        # can be 'multi_scopic' or 'multi_scopic_leadwise'
        cnn_choice = self.config.cnn.name.lower()
        self.cnn = MultiScopicCNN(self.n_leads, **(self.config.cnn[cnn_choice]))
        rnn_input_size = self.cnn.compute_output_shape(self.input_len, batch_size=None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(__debug_seq_len, batch_size=None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, given input seq_len = {__debug_seq_len}")
            __debug_seq_len = cnn_output_shape[-1]

        if self.config.rnn.name.lower() == 'none':
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == 'lstm':
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropout=self.config.rnn.lstm.dropout,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=True,
                nonlinearity=self.config.rnn.lstm.nonlinearity,
            )
            # rnn output shape (seq_len, batch_size, n_channels)
            attn_input_size = self.rnn.compute_output_shape(None,None)[-1]
        else:
            raise NotImplementedError

        if self.__DEBUG__:
            if self.rnn:
                rnn_output_shape = self.rnn.compute_output_shape(__debug_seq_len, batch_size=None)
                print(f"rnn output shape (seq_len, batch_size, features) = {rnn_output_shape}, given input seq_len = {__debug_seq_len}")

        # SEBlock already has `AdaptiveAvgPool1d`
        # self.pool = nn.AdaptiveAvgPool1d((1,))

        if self.config.attn.name.lower() == "se":
            self.attn = SEBlock(
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            clf_input_size = attn_input_size
        else:
            raise NotImplementedError(f"attention of {self.config.attn.name} not implemented yet")
        
        if self.__DEBUG__:
            print(f"configs of attn are {dict_to_str(self.config.attn)}")

        clf_out_channels = self.config.clf.out_channels + [self.__out_channels]
        self.clf = SeqLin(
            in_channels=clf_input_size,
            out_channels=clf_out_channels,
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            kernel_initializer=self.config.clf.kernel_initializer,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )
        
        # for inference
        # if background counted in `classes`, use softmax
        # otherwise use sigmoid
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)
        """
        # cnn
        cnn_output = self.cnn(input)  # (batch_size, channels, seq_len)

        # rnn or none
        if self.rnn:
            rnn_output = cnn_output.permute(2,0,1)  # (seq_len, batch_size, channels)
            rnn_output = self.rnn(rnn_output)  # (seq_len, batch_size, channels)
            rnn_output = rnn_output.permute(1,2,0)  # (batch_size, channels, seq_len)
        else:
            rnn_output = cnn_output

        # attention
        x = self.attn(rnn_output)  # (batch_size, channels, seq_len)
        x = x.permute(0,2,1)  # (batch_size, seq_len, channels)

        # classify
        output = self.clf(x)

        return output

    # inference will not be included in the model itself
    # as it is strongly related to the usage scenario
    # @torch.no_grad()
    # def inference(self, input:Union[np.ndarray,Tensor]) -> np.ndarray:

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
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
        output_shape = self.cnn.compute_output_shape(_seq_len, batch_size)
        _, _, _seq_len = output_shape
        if self.rnn:
            output_shape = self.rnn.compute_output_shape(_seq_len, batch_size)
            _seq_len, _, _ = output_shape
        output_shape = self.attn.compute_output_shape(_seq_len, batch_size)
        _, _, _seq_len = output_shape
        output_shape = self.clf.compute_output_shape(_seq_len, batch_size)
        return output_shape

    @property
    def module_size(self):
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params
