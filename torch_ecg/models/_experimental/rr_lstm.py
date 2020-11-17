"""
AF (and perhaps other arrhythmias like preamature beats) detection
using rr time series as input and using lstm as model
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
from torch_ecg.models.nets import (
    Mish, Swish, Activations,
    StackedLSTM,
    AttentionWithContext,
    SelfAttention, MultiHeadAttention,
    AttentivePooling,
    SeqLin,
)

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)



class RR_LSTM(nn.Module):
    """
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
        self.input_len = input_len
        self.config = deepcopy(RR_LSTM_CONFIG)
        self.config.update(config or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        
        self.lstm = StackedLSTM(
            input_size=rnn_input_size,
            hidden_sizes=self.config.lstm.hidden_sizes,
            bias=self.config.lstm.bias,
            dropouts=self.config.lstm.dropouts,
            bidirectional=self.config.lstm.bidirectional,
            return_sequences=self.config.lstm.retseq,
        )

        # TODO: add attn and clf module
        self.clf = None

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, input:Tensor, bin_pred_thr:float=0.5) -> Tensor:
        """
        """
        raise NotImplementedError("implement a task specific inference method")

    def compute_output_shape(self):
        """
        """
        raise NotImplementedError

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params