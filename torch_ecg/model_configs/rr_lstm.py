"""
"""
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "RR_AF_VANILLA",
    "RR_LSTM_CONFIG",
]



RR_AF_VANILLA = ED()

RR_AF_VANILLA.lstm = ED()
RR_AF_VANILLA.lstm.bias = True
RR_AF_VANILLA.lstm.dropouts = 0.1
RR_AF_VANILLA.lstm.bidirectional = True
RR_AF_VANILLA.lstm.retseq = True
RR_AF_VANILLA.lstm.hidden_sizes = [200]

RR_AF_VANILLA.attn = ED()
RR_AF_VANILLA.attn.name = "none"  # "gc", "nl", "se"

RR_AF_VANILLA.global_pool = "max"  # "avg", "attn"

RR_AF_VANILLA.clf = ED()
RR_AF_VANILLA.clf.name = "linear"  # crf
RR_AF_VANILLA.clf.linear = ED()
RR_AF_VANILLA.clf.linear.out_channels = [
  50,  # not including the last linear layer, with out channels equals n_classes
]
RR_AF_VANILLA.clf.linear.bias = True
RR_AF_VANILLA.clf.linear.dropouts = 0.1
RR_AF_VANILLA.clf.linear.activation = "relu"



RR_AF_CRF = ED()

RR_AF_CRF.lstm = ED()
RR_AF_CRF.lstm.bias = True
RR_AF_CRF.lstm.dropouts = 0.1
RR_AF_CRF.lstm.bidirectional = True
RR_AF_CRF.lstm.retseq = True
RR_AF_CRF.lstm.hidden_sizes = [200]

RR_AF_CRF.attn = ED()
RR_AF_CRF.attn.name = "none"  # "gc", "nl", "se"

RR_AF_CRF.clf = ED()
RR_AF_CRF.clf.name = "crf"  # crf
RR_AF_CRF.clf.crf = ED()
RR_AF_CRF.clf.crf.proj_bias = True



RR_LSTM_CONFIG = ED()

RR_LSTM_CONFIG.lstm = ED()
RR_LSTM_CONFIG.lstm.bias = True
RR_LSTM_CONFIG.lstm.dropouts = 0.1
RR_LSTM_CONFIG.lstm.bidirectional = True
RR_LSTM_CONFIG.lstm.retseq = True
RR_LSTM_CONFIG.lstm.hidden_sizes = [200]

RR_LSTM_CONFIG.attn = ED()
RR_LSTM_CONFIG.attn.name = "se"  # "gc", "nl", "none"
RR_LSTM_CONFIG.attn.se = ED()
RR_LSTM_CONFIG.attn.se.reduction = 8  # not including the last linear layer
RR_LSTM_CONFIG.attn.se.activation = "relu"
RR_LSTM_CONFIG.attn.se.kw_activation = ED(inplace=True)
RR_LSTM_CONFIG.attn.se.bias = True
RR_LSTM_CONFIG.attn.se.kernel_initializer = "he_normal"


RR_LSTM_CONFIG.global_pool = "avg"  # "avg", "attn", "none"


RR_LSTM_CONFIG.clf = ED()
RR_LSTM_CONFIG.clf.name = "crf"  # "linear"

RR_LSTM_CONFIG.clf.crf = ED()

RR_LSTM_CONFIG.clf.linear = ED()
