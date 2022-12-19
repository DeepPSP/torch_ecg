"""
configs of C(R)NN structure models, for ECG wave delineation
"""

from copy import deepcopy

from ..cfg import CFG
from ..utils.utils_nn import adjust_cnn_filter_lengths
from .attn import global_context, non_local
from .cnn import multi_scopic, multi_scopic_leadwise


__all__ = [
    "ECG_SEQ_LAB_NET_CONFIG",
]


# vanilla config, for delineation using single-lead ECG in corresponding papers
ECG_SEQ_LAB_NET_CONFIG = CFG()
ECG_SEQ_LAB_NET_CONFIG.fs = 500


ECG_SEQ_LAB_NET_CONFIG.cnn = CFG()
ECG_SEQ_LAB_NET_CONFIG.cnn.name = "multi_scopic"
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic = deepcopy(multi_scopic)
_base_num_filters = 4
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic = adjust_cnn_filter_lengths(
    ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic, ECG_SEQ_LAB_NET_CONFIG.fs
)

ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise = deepcopy(multi_scopic_leadwise)
ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise = adjust_cnn_filter_lengths(
    ECG_SEQ_LAB_NET_CONFIG.cnn.multi_scopic_leadwise, ECG_SEQ_LAB_NET_CONFIG.fs
)


ECG_SEQ_LAB_NET_CONFIG.rnn = CFG()
ECG_SEQ_LAB_NET_CONFIG.rnn.name = "lstm"  # "none"
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm = CFG()
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.hidden_sizes = [256, 256]
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bias = True
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.retseq = True
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.dropouts = 0
ECG_SEQ_LAB_NET_CONFIG.rnn.lstm.bidirectional = True


ECG_SEQ_LAB_NET_CONFIG.attn = CFG()
ECG_SEQ_LAB_NET_CONFIG.attn.name = "se"  # "gc", "nl", "none"
ECG_SEQ_LAB_NET_CONFIG.attn.se = CFG()
ECG_SEQ_LAB_NET_CONFIG.attn.se.reduction = 8  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.attn.se.activation = "relu"
ECG_SEQ_LAB_NET_CONFIG.attn.se.kw_activation = CFG(inplace=True)
ECG_SEQ_LAB_NET_CONFIG.attn.se.bias = True
ECG_SEQ_LAB_NET_CONFIG.attn.se.kernel_initializer = "he_normal"
# ECG_SEQ_LAB_NET_CONFIG.attn.se.dropouts = [0.2, 0.0]
ECG_SEQ_LAB_NET_CONFIG.attn.gc = deepcopy(global_context)
ECG_SEQ_LAB_NET_CONFIG.attn.nl = deepcopy(non_local)


ECG_SEQ_LAB_NET_CONFIG.clf = CFG()
ECG_SEQ_LAB_NET_CONFIG.clf.out_channels = [
    256,
    64,
]  # not including the last linear layer
ECG_SEQ_LAB_NET_CONFIG.clf.activation = "mish"
ECG_SEQ_LAB_NET_CONFIG.clf.bias = True
ECG_SEQ_LAB_NET_CONFIG.clf.kernel_initializer = "he_normal"
ECG_SEQ_LAB_NET_CONFIG.clf.dropouts = [0.2, 0.2, 0.0]
