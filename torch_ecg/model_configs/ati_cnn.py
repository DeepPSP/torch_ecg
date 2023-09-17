"""
the model of (attention-based) time-incremental CNN

the cnn layers of this model has a constant kernel size 3,
but keep increasing the number of channels
"""

from copy import deepcopy

from ..cfg import CFG
from ..utils.utils_nn import adjust_cnn_filter_lengths
from .cnn import resnet_nature_comm_bottle_neck_se, vgg16, vgg16_leadwise, vgg_block_basic, vgg_block_mish

__all__ = [
    "ATI_CNN_CONFIG",
]


ATI_CNN_CONFIG = CFG()
ATI_CNN_CONFIG.fs = 500


# cnn part
ATI_CNN_CONFIG.cnn = CFG()
ATI_CNN_CONFIG.cnn.name = "vgg16"

ATI_CNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)

ATI_CNN_CONFIG.cnn.vgg16_mish = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
ATI_CNN_CONFIG.cnn.vgg16_mish.block = deepcopy(vgg_block_mish)

ATI_CNN_CONFIG.cnn.vgg16_leadwise = adjust_cnn_filter_lengths(vgg16_leadwise, ATI_CNN_CONFIG.fs)
ATI_CNN_CONFIG.cnn.vgg16_leadwise.block = deepcopy(vgg_block_mish)

ATI_CNN_CONFIG.cnn.resnet = adjust_cnn_filter_lengths(resnet_nature_comm_bottle_neck_se, ATI_CNN_CONFIG.fs)


# rnn part
ATI_CNN_CONFIG.rnn = CFG()
ATI_CNN_CONFIG.rnn.name = "lstm"

ATI_CNN_CONFIG.rnn.lstm = CFG()
ATI_CNN_CONFIG.rnn.lstm.bias = True
ATI_CNN_CONFIG.rnn.lstm.dropout = 0.2
ATI_CNN_CONFIG.rnn.lstm.bidirectional = True
ATI_CNN_CONFIG.rnn.lstm.retseq = False
ATI_CNN_CONFIG.rnn.lstm.hidden_sizes = [128, 32]

ATI_CNN_CONFIG.attn = CFG()
ATI_CNN_CONFIG.attn.name = "none"

ATI_CNN_CONFIG.global_pool = "none"
