"""
configs of C(R)NN structure models, for ECG wave delineation
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
)
from .attn import (
    non_local,
    squeeze_excitation,
    global_context,
)


__all__ = [
    "ECG_SEQ_LAB_NET_CONFIG",
]


# vanilla config, for delineation using single-lead ECG in corresponding papers
ECG_SEQ_LAB_NET_CONFIG = ED()


ECG_SEQ_LAB_NET_CONFIG.cnn = ED()
ECG_SEQ_LAB_NET_CONFIG.cnn.name = "multi_scopic"
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic = deepcopy(multi_scopic)
_base_num_filters = 4
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic.block = deepcopy(multi_scopic_block)
# ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise = deepcopy(multi_scopic_leadwise)
# ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise.block = deepcopy(multi_scopic_block)


ECG_SEQ_LAB_NET_CONFIG.rnn = ED()
ECG_SEQ_LAB_NET_CONFIG.rnn.name = "lstm"  # "none"
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm = ED()
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.hidden_sizes = [256, 256]
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bias = True
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.dropout = 0
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bidirectional = True


ECG_SEQ_LAB_NET_CONFIG.attn = ED()
ECG_SEQ_LAB_NET_CONFIG.attn.name = "se"  # "gc"
ECG_SEQ_LAB_NET_CONFIG.attn.se = ED()
ECG_SEQ_LAB_NET_CONFIG.attn.se.reduction = 8  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.attn.se.activation = "relu"
ECG_SEQ_LAB_NET_CONFIG.attn.se.kw_activation = ED(inplace=True)
ECG_SEQ_LAB_NET_CONFIG.attn.se.bias = True
ECG_SEQ_LAB_NET_CONFIG.attn.se.kernel_initializer = "he_normal"
# ECG_SEQ_LAB_NET_CONFIG.attn.se.dropouts = [0.2, 0.0]


ECG_SEQ_LAB_NET_CONFIG.clf = ED()
ECG_SEQ_LAB_NET_CONFIG.clf.out_channels = [256, 64]  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.clf.activation = "mish"
ECG_SEQ_LAB_NET_CONFIG.clf.bias = True
ECG_SEQ_LAB_NET_CONFIG.clf.kernel_initializer = "he_normal"
ECG_SEQ_LAB_NET_CONFIG.clf.dropouts = [0.2, 0.2, 0.0]
