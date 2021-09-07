"""
Sequence labeling nets, for wave delineation,

the labeling granularity is the frequency of the input signal,
divided by the length (counted by the number of basic blocks) of each branch

pipeline:
multi-scopic cnn --> (bidi-lstm -->) "attention" (se block) --> seq linear

References
----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, List, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from ..cfg import Cfg
from ..utils.utils_nn import compute_conv_output_shape, compute_module_size
from ..utils.misc import dict_to_str
from ..model_configs.ecg_seq_lab_net import ECG_SEQ_LAB_NET_CONFIG
from ._nets import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    SEBlock,
    StackedLSTM,
    AttentivePooling,
    SeqLin,
)
from .cnn.multi_scopic import MultiScopicCNN


if Cfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_SEQ_LAB_NET",
]


class ECG_SEQ_LAB_NET(nn.Module):
    """ finished, checked,

    SOTA model from CPSC2019 challenge (entry 0416)

    pipeline
    --------
    multi-scopic cnn --> (bidi-lstm -->) "attention" --> seq linear

    References
    ----------
    [1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
    """
    __DEBUG__ = False
    __name__ = "ECG_SEQ_LAB_NET"

    def __init__(self, classes:Sequence[str], n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
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
        self.config.update(deepcopy(config) or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
        __debug_seq_len = self.input_len or 4000
        
        # currently, the CNN part only uses `MultiScopicCNN`
        # can be "multi_scopic" or "multi_scopic_leadwise"
        cnn_choice = self.config.cnn.name.lower()
        self.cnn = MultiScopicCNN(self.n_leads, **(self.config.cnn[cnn_choice]))
        rnn_input_size = self.cnn.compute_output_shape(self.input_len, batch_size=None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(__debug_seq_len, batch_size=None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, given input seq_len = {__debug_seq_len}")
            __debug_seq_len = cnn_output_shape[-1]

        if self.config.rnn.name.lower() == "none":
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropouts=self.config.rnn.lstm.dropouts,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=True,
                # nonlinearity=self.config.rnn.lstm.nonlinearity,
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

    def extract_features(self, input:Tensor) -> Tensor:
        """ finished, checked,

        extract feature map before the dense (linear) classifying layer(s)

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)
        
        Returns
        -------
        features: Tensor,
            of shape (batch_size, seq_len, channels)
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
        features = self.attn(rnn_output)  # (batch_size, channels, seq_len)
        features = features.permute(0,2,1)  # (batch_size, seq_len, channels)
        return features

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)
        
        Returns
        -------
        pred: Tensor,
            of shape (batch_size, seq_len)
        """
        features = self.extract_features(input)

        # classify
        pred = self.clf(features)

        return pred

    # inference will not be included in the model itself
    # as it is strongly related to the usage scenario
    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        """
        raise NotImplementedError("implement a task specific inference method")

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
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
    def module_size(self) -> int:
        """
        """
        return compute_module_size(self)


    @staticmethod
    def from_checkpoint(path:str, device:Optional[torch.device]=None) -> nn.Module:
        """
        """
        _device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        model = ECG_CRNN(
            classes=aux_config["classes"],
            n_leads=aux_config["n_leads"],
            config=ckpt["model_config"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model
