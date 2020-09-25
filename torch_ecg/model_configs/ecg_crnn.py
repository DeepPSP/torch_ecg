"""
configs of models of CRNN structures, for classification
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish,
    vgg16, vgg16_leadwise,
    resnet_block_stanford, resnet_stanford,
    resnet_block_basic, resnet_bottle_neck,
    resnet, resnet_leadwise,
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
)
from .rnn import (
    lstm,
    attention,
    linear,
)


__all__ = [
    "ECG_CRNN_CONFIG",
]


ECG_CRNN_CONFIG = ED()

# cnn part
ECG_CRNN_CONFIG.cnn = ED()
ECG_CRNN_CONFIG.cnn.name = 'resnet_leadwise'


ECG_CRNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
ECG_CRNN_CONFIG.cnn.vgg16_mish = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16_mish.block = deepcopy(vgg_block_mish)
ECG_CRNN_CONFIG.cnn.vgg16_swish = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16_swish.block = deepcopy(vgg_block_swish)
ECG_CRNN_CONFIG.cnn.vgg16_leadwise = deepcopy(vgg16_leadwise)
ECG_CRNN_CONFIG.cnn.vgg16_leadwise.block = deepcopy(vgg_block_swish)
# ECG_CRNN_CONFIG.cnn.vgg16_dilation = deepcopy(vgg16)
# ECG_CRNN_CONFIG.cnn.vgg16_dilation.block = deepcopy(vgg_block_basic)

ECG_CRNN_CONFIG.cnn.resnet = deepcopy(resnet)
ECG_CRNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_basic)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck = deepcopy(resnet)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck.block = deepcopy(resnet_bottle_neck)
ECG_CRNN_CONFIG.cnn.resnet_leadwise = deepcopy(resnet_leadwise)
ECG_CRNN_CONFIG.cnn.resnet_leadwise.block = deepcopy(resnet_block_basic)

ECG_CRNN_CONFIG.cnn.resnet_stanford = deepcopy(resnet_stanford)
ECG_CRNN_CONFIG.cnn.resnet_stanford.block = deepcopy(resnet_block_stanford)

ECG_CRNN_CONFIG.cnn.multi_scopic = deepcopy(multi_scopic)
ECG_CRNN_CONFIG.cnn.multi_scopic.block = deepcopy(multi_scopic_block)
ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise = deepcopy(multi_scopic_leadwise)
ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise.block = deepcopy(multi_scopic_block)


# rnn part
ECG_CRNN_CONFIG.rnn = ED()
ECG_CRNN_CONFIG.rnn.name = 'linear'  # 'none', 'lstm', 'attention'

ECG_CRNN_CONFIG.rnn.lstm = deepcopy(lstm)
ECG_CRNN_CONFIG.rnn.attention = deepcopy(attention)
ECG_CRNN_CONFIG.rnn.linear = deepcopy(linear)


# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
ECG_CRNN_CONFIG.global_pool = 'max'  # 'avg', 'attentive'
