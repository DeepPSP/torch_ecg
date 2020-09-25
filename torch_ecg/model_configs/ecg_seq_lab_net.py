"""
configs of C(R)NN structure models, for ECG wave delineation
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
)


__all__ = [
    "ECG_SEQ_LAB_NET_CONFIG",
]


ECG_SEQ_LAB_NET_CONFIG = ED()


ECG_SEQ_LAB_NET_CONFIG.cnn = ED()
ECG_SEQ_LAB_NET_CONFIG.cnn.name = 'multi_scopic'  # 'multi_scopic_leadwise
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic = deepcopy(multi_scopic)
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic.block = deepcopy(multi_scopic_block)
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise = deepcopy(multi_scopic_leadwise)
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise.block = deepcopy(multi_scopic_block)


ECG_SEQ_LAB_NET_CONFIG.rnn = ED()
ECG_SEQ_LAB_NET_CONFIG.rnn.name = 'lstm'  # 'none'
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm = ED()
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.hidden_sizes = [256, 256]
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bias = True
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.dropout = 0
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bidirectional = True


ECG_SEQ_LAB_NET_CONFIG.attn = ED()
ECG_SEQ_LAB_NET_CONFIG.attn.out_channels = [64]  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.attn.activation = "relu"
ECG_SEQ_LAB_NET_CONFIG.attn.bias = True
ECG_SEQ_LAB_NET_CONFIG.attn.kernel_initializer = 'he_normal'
ECG_SEQ_LAB_NET_CONFIG.attn.dropouts = [0.2, 0.0]


ECG_SEQ_LAB_NET_CONFIG.clf = ED()
ECG_SEQ_LAB_NET_CONFIG.clf.out_channels = [256, 64]  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.clf.activation = "mish"
ECG_SEQ_LAB_NET_CONFIG.clf.bias = True
ECG_SEQ_LAB_NET_CONFIG.clf.kernel_initializer = 'he_normal'
ECG_SEQ_LAB_NET_CONFIG.clf.dropouts = [0.2, 0.2, 0.0]
