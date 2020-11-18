"""
AF (and perhaps other arrhythmias like preamature beats) detection
using rr time series as input and using lstm as model

References:
-----------
[1] https://github.com/al3xsh/rnn-based-af-detection
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
from torch_ecg.model_configs import RR_LSTM_CONFIG
from torch_ecg.utils.misc import dict_to_str
from torch_ecg.utils.utils_nn import compute_module_size
from torch_ecg.models.nets import (
    Mish, Swish, Activations,
    StackedLSTM,
    AttentionWithContext,
    SelfAttention, MultiHeadAttention,
    AttentivePooling,
    SeqLin, CRF,
)

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "RR_LSTM",
]


class RR_LSTM(nn.Module):
    """
    classification or sequence labeling using LSTM and using RR intervals as input
    """
    __DEBUG__ = True
    __name__ = "RR_LSTM"

    def __init__(self,classes:Sequence[str], n_leads:int, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = n_leads
        self.config = deepcopy(RR_LSTM_CONFIG)
        self.config.update(config or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        
        self.lstm = StackedLSTM(
            input_size=self.n_leads,
            hidden_sizes=self.config.lstm.hidden_sizes,
            bias=self.config.lstm.bias,
            dropouts=self.config.lstm.dropouts,
            bidirectional=self.config.lstm.bidirectional,
            return_sequences=self.config.lstm.retseq,
        )

        if self.__DEBUG__:
            print(f"lstm module has size {self.lstm.module_size}")

        attn_input_size = self.lstm.compute_output_shape(None, None)[-1]

        if not self.config.lstm.retseq:
            self.attn = None
        elif self.config.attn.name.lower() == "none":
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # non_local
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,
                filter_lengths=self.config.attn.nl.filter_lengths,
                subsample_length=self.config.attn.nl.subsample_length,
                batch_norm=self.config.attn.nl.batch_norm,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # squeeze_exitation
            self.attn = SEBlock(
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # global_context
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,
                ratio=self.config.attn.gc.ratio,
                reduction=self.config.attn.gc.reduction,
                pooling_type=self.config.attn.gc.pooling_type,
                fusion_types=self.config.attn.gc.fusion_types,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # self_attention
            # NOTE: this branch NOT tested
            self.attn = SelfAttention(
                in_features=attn_in_channels,
                head_num=self.config.attn.sa.head_num,
                dropout=self.config.attn.sa.dropout,
                bias=self.config.attn.sa.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        if self.__DEBUG__ and self.attn:
            print(f"attn module has size {self.attn.module_size}")

        if not self.config.lstm.retseq:
            self.pool = None
            self.clf = None
        elif self.config.clf.name.lower() == "linear":
            if self.config.global_pool.lower() == "max":
                self.pool = nn.AdaptiveMaxPool1d((1,))
                self.clf = SeqLin(
                    in_channels=clf_input_size,
                    out_channels=self.config.clf.linear.out_channels + [self.n_classes],
                    activation=self.config.clf.linear.activation,
                    bias=self.config.clf.linear.bias,
                    dropouts=self.config.clf.linear.dropouts,
                    skip_last_activation=True,
                )
            if self.__DEBUG__:
                print(f"linear clf module has size {self.clf.module_size}")
        elif self.config.clf.name.lower() == "crf":
            self.pool = None
            self.clf = nn.Sequential()
            proj = nn.Linear(
                in_features=clf_input_size,
                out_features=self.n_classes,
                bias=self.config.clf.crf.proj_bias,
            )
            crf = CRF(num_tags=self.n_classes, batch_first=True,)
            self.clf.add_module(
                name="proj",
                module=proj,
            )
            self.clf.add_module(
                name="crf",
                module=crf,
            )
            if self.__DEBUG__:
                print(f"for crf clf, proj module has size {compute_module_size(proj)}, crf module has size {crf.module_size}")

        # for inference, except for crf
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input:Tensor) -> Tensor:
        """ NOT finished, NOT checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, seq_len, n_classes) or (batch_size, n_classes)
        """
        x = self.lstm(input)  # (seq_len, batch_size, n_channels) or (batch_size, n_channels)
        if self.attn:
            # (seq_len, batch_size, n_channels) --> (batch_size, n_channels, seq_len)
            x = x.permute(1,2,0)
            x = self.attn(x)  # (batch_size, n_channels, seq_len)
        if self.pool:
            x = self.pool(x)  # (batch_size, n_channels, 1)
            x = x.squeeze(dim=-1)  # (batch_size, n_channels)
        elif x.ndim == 3:
            # (batch_size, n_channels, seq_len) --> (batch_size, seq_len, n_channels)
            x = x.permute(0,2,1)
        else:
            # x of shape (batch_size, n_channels), 
            # in the case where config.lstm.retseq = False
            pass
        if self.clf:
            x = self.clf(x)  # (batch_size, seq_len, n_classes) or (batch_size, n_classes)
        output = x

        return output
        

    @torch.no_grad()
    def inference(self, input:Tensor, bin_pred_thr:float=0.5) -> Tensor:
        """
        """
        raise NotImplementedError("implement a task specific inference method")


    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `CRF` layer, given `seq_len` and `batch_size`
        """
        if self.config.clf.name.lower() == "crf":
            output_shape = (batch_size, seq_len, self.n_classes)
        else:
            # clf is "linear" or lstm.retseq is False
            output_shape = (batch_size, self.n_classes)
        return output_shape


    @property
    def module_size(self) -> int:
        """
        """
        return compute_module_size(self)
