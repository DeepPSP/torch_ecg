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
    resnet_cpsc2018, resnet_cpsc2018_leadwise,
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
    dense_net_vanilla, dense_net_leadwise,
    xception_vanilla, xception_leadwise,
)
from .rnn import (
    lstm,
    attention,
)
from .mlp import linear
from .attn import (
    non_local,
    squeeze_excitation,
    global_context,
)
from ..utils.utils_nn import adjust_cnn_filter_lengths


__all__ = [
    "ECG_CRNN_CONFIG",
]


ECG_CRNN_CONFIG = ED()
ECG_CRNN_CONFIG.fs = 500

# cnn part
ECG_CRNN_CONFIG.cnn = ED()
# ECG_CRNN_CONFIG.cnn.name = "resnet_leadwise"
ECG_CRNN_CONFIG.cnn.name = "multi_scopic_leadwise"


ECG_CRNN_CONFIG.cnn.vgg16 = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16.block = deepcopy(vgg_block_basic)
ECG_CRNN_CONFIG.cnn.vgg16 = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.vgg16, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.vgg16_mish = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16_mish.block = deepcopy(vgg_block_mish)
ECG_CRNN_CONFIG.cnn.vgg16_mish = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.vgg16_mish, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.vgg16_swish = deepcopy(vgg16)
ECG_CRNN_CONFIG.cnn.vgg16_swish.block = deepcopy(vgg_block_swish)
ECG_CRNN_CONFIG.cnn.vgg16_swish = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.vgg16_swish, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.vgg16_leadwise = deepcopy(vgg16_leadwise)
ECG_CRNN_CONFIG.cnn.vgg16_leadwise.block = deepcopy(vgg_block_swish)
ECG_CRNN_CONFIG.cnn.vgg16_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.vgg16_leadwise, ECG_CRNN_CONFIG.fs)
# ECG_CRNN_CONFIG.cnn.vgg16_dilation = deepcopy(vgg16)
# ECG_CRNN_CONFIG.cnn.vgg16_dilation.block = deepcopy(vgg_block_basic)

ECG_CRNN_CONFIG.cnn.resnet = deepcopy(resnet_cpsc2018)
ECG_CRNN_CONFIG.cnn.resnet.block = deepcopy(resnet_block_basic)
ECG_CRNN_CONFIG.cnn.resnet = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck = deepcopy(resnet_cpsc2018)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck.block = deepcopy(resnet_bottle_neck)
ECG_CRNN_CONFIG.cnn.resnet_bottleneck = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_bottleneck, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_leadwise = deepcopy(resnet_cpsc2018_leadwise)
ECG_CRNN_CONFIG.cnn.resnet_leadwise.block = deepcopy(resnet_block_basic)
ECG_CRNN_CONFIG.cnn.resnet_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_leadwise, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.resnet_stanford = deepcopy(resnet_stanford)
ECG_CRNN_CONFIG.cnn.resnet_stanford.block = deepcopy(resnet_block_stanford)
ECG_CRNN_CONFIG.cnn.resnet_stanford = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_stanford, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.multi_scopic = deepcopy(multi_scopic)
ECG_CRNN_CONFIG.cnn.multi_scopic.block = deepcopy(multi_scopic_block)
ECG_CRNN_CONFIG.cnn.multi_scopic = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.multi_scopic, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise = deepcopy(multi_scopic_leadwise)
ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise.block = deepcopy(multi_scopic_block)
ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.multi_scopic_leadwise, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.xception_vanilla = deepcopy(xception_vanilla)
ECG_CRNN_CONFIG.cnn.xception_leadwise = deepcopy(xception_leadwise)
ECG_CRNN_CONFIG.cnn.xception_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.xception_leadwise, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.dense_net_vanilla = deepcopy(dense_net_vanilla)
ECG_CRNN_CONFIG.cnn.dense_net_leadwise = deepcopy(dense_net_leadwise)
ECG_CRNN_CONFIG.cnn.dense_net_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.dense_net_leadwise, ECG_CRNN_CONFIG.fs)



# rnn part
ECG_CRNN_CONFIG.rnn = ED()
ECG_CRNN_CONFIG.rnn.name = "linear"  # "none", "lstm", "linear"

ECG_CRNN_CONFIG.rnn.lstm = deepcopy(lstm)
ECG_CRNN_CONFIG.rnn.linear = deepcopy(linear)



# attention part
ECG_CRNN_CONFIG.attn = ED()
ECG_CRNN_CONFIG.attn.name = "se"  # "none", "se", "gc", "nl"

ECG_CRNN_CONFIG.attn.se = deepcopy(squeeze_excitation)

ECG_CRNN_CONFIG.attn.gc = deepcopy(global_context)

ECG_CRNN_CONFIG.attn.nl = deepcopy(non_local)



# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
ECG_CRNN_CONFIG.global_pool = "max"  # "avg", "attn"



ECG_CRNN_CONFIG.clf = ED()
ECG_CRNN_CONFIG.clf.out_channels = [
    # 12 * 32,
  # not including the last linear layer, whose out channels equals n_classes
]
ECG_CRNN_CONFIG.clf.bias = True
ECG_CRNN_CONFIG.clf.dropouts = 0.0
ECG_CRNN_CONFIG.clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored
