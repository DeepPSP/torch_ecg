"""
the model of (attention-based) time-incremental CNN

the cnn layers of this model has a constant kernel size 3,
but keep increasing the number of channels
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish,
    vgg16, vgg16_leadwise,
    resnet_block_stanford, resnet_stanford,
    resnet_block_basic, resnet_bottle_neck,
    resnet, resnet_leadwise,
)


__all__ = [
    "ATI_CNN_CONFIG",
]


ATI_CNN_CONFIG = ED()

# cnn part
ATI_CNN_CONFIG.cnn = ED()
ATI_CNN_CONFIG.cnn.name = "vgg16"


if ATI_CNN_CONFIG.cnn.name == "vgg16":
    ATI_CNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_mish":
    ATI_CNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_mish)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_swish":
    ATI_CNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_swish)
elif ATI_CNN_CONFIG.cnn.name == "vgg16_dilation":  # not finished
    ATI_CNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
    ATI_CNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "resnet":
    ATI_CNN_CONFIG.cnn.resnet = deepcopy(resnet)
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_basic)
elif ATI_CNN_CONFIG.cnn.name == "resnet_bottleneck":
    ATI_CNN_CONFIG.cnn.resnet = deepcopy(resnet)
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_bottle_neck)
elif ATI_CNN_CONFIG.cnn.name == "resnet_stanford":
    ATI_CNN_CONFIG.cnn.resnet = deepcopy(resnet_stanford)
    ATI_CNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_stanford)
else:
    pass


# rnn part
ATI_CNN_CONFIG.rnn = ED()
ATI_CNN_CONFIG.rnn.name = "lstm"

if ATI_CNN_CONFIG.rnn.name == "lstm":
    ATI_CNN_CONFIG.rnn.bias = True
    ATI_CNN_CONFIG.rnn.dropout = 0.2
    ATI_CNN_CONFIG.rnn.bidirectional = True
    ATI_CNN_CONFIG.rnn.retseq = False
    ATI_CNN_CONFIG.rnn.hidden_sizes = [128,32]
elif ATI_CNN_CONFIG.rnn.name == "attention":
    pass
else:
    pass
