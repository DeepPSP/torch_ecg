"""
"""

from copy import deepcopy

from ..cfg import CFG
from .attn import non_local, squeeze_excitation, global_context


__all__ = [
    "RR_AF_VANILLA_CONFIG",
    "RR_AF_CRF_CONFIG",
    "RR_LSTM_CONFIG",
]


RR_AF_VANILLA_CONFIG = CFG()

RR_AF_VANILLA_CONFIG.lstm = CFG()
RR_AF_VANILLA_CONFIG.lstm.bias = True
RR_AF_VANILLA_CONFIG.lstm.dropouts = 0.1
RR_AF_VANILLA_CONFIG.lstm.bidirectional = True
RR_AF_VANILLA_CONFIG.lstm.retseq = True
RR_AF_VANILLA_CONFIG.lstm.hidden_sizes = [200]

RR_AF_VANILLA_CONFIG.attn = CFG()
RR_AF_VANILLA_CONFIG.attn.name = "none"  # "gc", "nl", "se"
RR_AF_VANILLA_CONFIG.attn.nl = deepcopy(non_local)
RR_AF_VANILLA_CONFIG.attn.se = deepcopy(squeeze_excitation)
RR_AF_VANILLA_CONFIG.attn.gc = deepcopy(global_context)

RR_AF_VANILLA_CONFIG.global_pool = "max"  # "avg", "attn"
RR_AF_VANILLA_CONFIG.global_pool_size = 1

RR_AF_VANILLA_CONFIG.clf = CFG()
RR_AF_VANILLA_CONFIG.clf.name = "linear"  # crf
RR_AF_VANILLA_CONFIG.clf.linear = CFG()
RR_AF_VANILLA_CONFIG.clf.linear.out_channels = [
    50,  # not including the last linear layer, with out channels equals n_classes
]
RR_AF_VANILLA_CONFIG.clf.linear.bias = True
RR_AF_VANILLA_CONFIG.clf.linear.dropouts = 0.1
RR_AF_VANILLA_CONFIG.clf.linear.activation = "relu"


RR_AF_CRF_CONFIG = CFG()

RR_AF_CRF_CONFIG.lstm = CFG()
RR_AF_CRF_CONFIG.lstm.bias = True
RR_AF_CRF_CONFIG.lstm.dropouts = 0.1
RR_AF_CRF_CONFIG.lstm.bidirectional = True
RR_AF_CRF_CONFIG.lstm.retseq = True
RR_AF_CRF_CONFIG.lstm.hidden_sizes = [200]

RR_AF_CRF_CONFIG.attn = CFG()
RR_AF_CRF_CONFIG.attn.name = "none"  # "gc", "nl", "se"
RR_AF_CRF_CONFIG.attn.nl = deepcopy(non_local)
RR_AF_CRF_CONFIG.attn.se = deepcopy(squeeze_excitation)
RR_AF_CRF_CONFIG.attn.gc = deepcopy(global_context)

RR_AF_CRF_CONFIG.global_pool = "none"
RR_AF_CRF_CONFIG.global_pool_size = 1

RR_AF_CRF_CONFIG.clf = CFG()
RR_AF_CRF_CONFIG.clf.name = "crf"  # crf
RR_AF_CRF_CONFIG.clf.crf = CFG()
RR_AF_CRF_CONFIG.clf.crf.proj_bias = True


RR_LSTM_CONFIG = CFG()

RR_LSTM_CONFIG.lstm = CFG()
RR_LSTM_CONFIG.lstm.bias = True
RR_LSTM_CONFIG.lstm.dropouts = 0.1
RR_LSTM_CONFIG.lstm.bidirectional = True
RR_LSTM_CONFIG.lstm.retseq = True
RR_LSTM_CONFIG.lstm.hidden_sizes = [200]

RR_LSTM_CONFIG.attn = CFG()
RR_LSTM_CONFIG.attn.name = "se"  # "gc", "nl", "none"
RR_LSTM_CONFIG.attn.se = deepcopy(squeeze_excitation)
RR_LSTM_CONFIG.attn.se.bias = True
RR_LSTM_CONFIG.attn.nl = deepcopy(non_local)
RR_LSTM_CONFIG.attn.gc = deepcopy(global_context)

RR_LSTM_CONFIG.global_pool = "none"  # "avg", "attn", "none"
RR_LSTM_CONFIG.global_pool_size = 1

RR_LSTM_CONFIG.clf = CFG()
RR_LSTM_CONFIG.clf.name = "crf"  # "linear"
RR_LSTM_CONFIG.clf.crf = CFG()
RR_LSTM_CONFIG.clf.crf.proj_bias = True
RR_LSTM_CONFIG.clf.linear = CFG()
RR_LSTM_CONFIG.clf.linear.out_channels = [
    50,  # not including the last linear layer, with out channels equals n_classes
]
RR_LSTM_CONFIG.clf.linear.bias = True
RR_LSTM_CONFIG.clf.linear.dropouts = 0.1
RR_LSTM_CONFIG.clf.linear.activation = "relu"
