"""
the model of (attention-based) time-incremental CNN

the cnn layers of this model has a constant kernel size 3,
but keep increasing the number of channels
"""

from copy import deepcopy

from ..cfg import CFG
from ..utils.utils_nn import adjust_cnn_filter_lengths
from .cnn import (  # noqa: F401
    resnet_block_basic,
    resnet_block_stanford,
    resnet_bottle_neck,
    resnet_cpsc2018,
    resnet_cpsc2018_leadwise,
    resnet_stanford,
    vgg16,
    vgg16_leadwise,
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
)

__all__ = [
    "ATI_CNN_CONFIG",
]


ATI_CNN_CONFIG = CFG()
ATI_CNN_CONFIG.fs = 500


# cnn part
ATI_CNN_CONFIG.cnn = CFG()
ATI_CNN_CONFIG.cnn.name = "vgg16"


if ATI_CNN_CONFIG.cnn.name == "vgg16":
    ATI_CNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_mish":
    ATI_CNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_mish)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_swish":
    ATI_CNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_swish)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_dilation":  # not finished
    ATI_CNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(vgg16, ATI_CNN_CONFIG.fs)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "resnet":
    ATI_CNN_CONFIG.cnn.resnet = adjust_cnn_filter_lengths(
        resnet_cpsc2018, ATI_CNN_CONFIG.fs
    )
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "resnet_bottleneck":
    ATI_CNN_CONFIG.cnn.resnet = adjust_cnn_filter_lengths(
        resnet_cpsc2018, ATI_CNN_CONFIG.fs
    )
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_bottle_neck)
elif ATI_CNN_CONFIG.cnn.name == "resnet_stanford":
    ATI_CNN_CONFIG.cnn.resnet = adjust_cnn_filter_lengths(
        resnet_stanford, ATI_CNN_CONFIG.fs
    )
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_stanford)
else:
    pass


# rnn part
ATI_CNN_CONFIG.rnn = CFG()
ATI_CNN_CONFIG.rnn.name = "lstm"

if ATI_CNN_CONFIG.rnn.name == "lstm":
    ATI_CNN_CONFIG.rnn.bias = True
    ATI_CNN_CONFIG.rnn.dropout = 0.2
    ATI_CNN_CONFIG.rnn.bidirectional = True
    ATI_CNN_CONFIG.rnn.retseq = False
    ATI_CNN_CONFIG.rnn.hidden_sizes = [128, 32]
elif ATI_CNN_CONFIG.rnn.name == "attention":
    pass
else:
    pass
