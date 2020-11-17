"""
"""
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "RR_LSTM_CONFIG",
]


RR_LSTM_CONFIG = ED()

RR_LSTM_CONFIG.lstm = ED()
RR_LSTM_CONFIG.lstm.bias = True
RR_LSTM_CONFIG.lstm.dropouts = 0.2
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
