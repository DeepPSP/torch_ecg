"""
configs of models of CRNN structures, for classification
"""
from copy import deepcopy

from .cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish,
    vgg16, vgg16_leadwise,
    resnet_block_stanford, resnet_stanford,
    resnet_block_basic, resnet_bottle_neck,
    resnet_cpsc2018, resnet_cpsc2018_leadwise,
    resnet_vanilla_18, resnet_vanilla_34,
    resnet_vanilla_50, resnext_vanilla_50_32x4d,
    resnet_nature_comm,
    resnet_nature_comm_se, resnet_nature_comm_gc, resnet_nature_comm_nl,
    resnet_nature_comm_bottle_neck, resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_bottle_neck_gc, resnet_nature_comm_bottle_neck_nl,
    resnetN, resnetNB, resnetNS, resnetNBS,
    tresnetF, tresnetP, tresnetN, tresnetS, tresnetM,
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
    densenet_vanilla, densenet_leadwise,
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
    transformer,
)
from ..utils.utils_nn import adjust_cnn_filter_lengths
from ..cfg import CFG


__all__ = [
    "ECG_CRNN_CONFIG",
]


ECG_CRNN_CONFIG = CFG()
ECG_CRNN_CONFIG.fs = 500

# cnn part
ECG_CRNN_CONFIG.cnn = CFG()
# ECG_CRNN_CONFIG.cnn.name = "resnet_leadwise"
ECG_CRNN_CONFIG.cnn.name = "resnet_nature_comm_bottle_neck"


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

# ECG_CRNN_CONFIG.cnn.resnet_cpsc2018 = deepcopy(resnet_cpsc2018)
# ECG_CRNN_CONFIG.cnn.resnet_cpsc2018 = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_18 = deepcopy(resnet_vanilla_18)
ECG_CRNN_CONFIG.cnn.resnet_18 = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_18, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_34 = deepcopy(resnet_vanilla_34)
ECG_CRNN_CONFIG.cnn.resnet_34 = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_34, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_50 = deepcopy(resnet_vanilla_50)
ECG_CRNN_CONFIG.cnn.resnet_50 = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_50, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnext_50_32x4d = deepcopy(resnext_vanilla_50_32x4d)
ECG_CRNN_CONFIG.cnn.resnext_50_32x4d = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnext_50_32x4d, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.resnet_nature_comm = deepcopy(resnet_nature_comm)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_se = deepcopy(resnet_nature_comm_se)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_se = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_se, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_gc = deepcopy(resnet_nature_comm_gc)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_gc = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_gc, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_nl = deepcopy(resnet_nature_comm_nl)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_nl = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_nl, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck = deepcopy(resnet_nature_comm_bottle_neck)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_se = deepcopy(resnet_nature_comm_bottle_neck_se)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_se = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_se, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_gc = deepcopy(resnet_nature_comm_bottle_neck_gc)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_gc = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_gc, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_nl = deepcopy(resnet_nature_comm_bottle_neck_nl)
ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_nl = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnet_nature_comm_bottle_neck_nl, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.resnetN = deepcopy(resnetN)
ECG_CRNN_CONFIG.cnn.resnetN = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnetN, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnetNB = deepcopy(resnetNB)
ECG_CRNN_CONFIG.cnn.resnetNB = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnetNB, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnetNS = deepcopy(resnetNS)
ECG_CRNN_CONFIG.cnn.resnetNS = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnetNS, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.resnetNBS = deepcopy(resnetNBS)
ECG_CRNN_CONFIG.cnn.resnetNBS = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.resnetNBS, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.tresnetF = deepcopy(tresnetF)
ECG_CRNN_CONFIG.cnn.tresnetF = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.tresnetF, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.tresnetP = deepcopy(tresnetP)
ECG_CRNN_CONFIG.cnn.tresnetP = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.tresnetP, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.tresnetN = deepcopy(tresnetN)
ECG_CRNN_CONFIG.cnn.tresnetN = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.tresnetN, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.tresnetS = deepcopy(tresnetS)
ECG_CRNN_CONFIG.cnn.tresnetS = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.tresnetS, ECG_CRNN_CONFIG.fs)
ECG_CRNN_CONFIG.cnn.tresnetM = deepcopy(tresnetM)
ECG_CRNN_CONFIG.cnn.tresnetM = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.tresnetM, ECG_CRNN_CONFIG.fs)

ECG_CRNN_CONFIG.cnn.resnet_leadwise = deepcopy(resnet_cpsc2018_leadwise)
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

ECG_CRNN_CONFIG.cnn.densenet_vanilla = deepcopy(densenet_vanilla)
ECG_CRNN_CONFIG.cnn.densenet_leadwise = deepcopy(densenet_leadwise)
ECG_CRNN_CONFIG.cnn.densenet_leadwise = adjust_cnn_filter_lengths(ECG_CRNN_CONFIG.cnn.densenet_leadwise, ECG_CRNN_CONFIG.fs)



# rnn part
ECG_CRNN_CONFIG.rnn = CFG()
ECG_CRNN_CONFIG.rnn.name = "none"  # "none", "lstm", "linear"

ECG_CRNN_CONFIG.rnn.lstm = deepcopy(lstm)
ECG_CRNN_CONFIG.rnn.linear = deepcopy(linear)



# attention part
ECG_CRNN_CONFIG.attn = CFG()
ECG_CRNN_CONFIG.attn.name = "se"  # "none", "se", "gc", "nl"

ECG_CRNN_CONFIG.attn.se = deepcopy(squeeze_excitation)

ECG_CRNN_CONFIG.attn.gc = deepcopy(global_context)

ECG_CRNN_CONFIG.attn.nl = deepcopy(non_local)

ECG_CRNN_CONFIG.attn.transformer = deepcopy(transformer)



# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
ECG_CRNN_CONFIG.global_pool = "max"  # "avg", "attn"


ECG_CRNN_CONFIG.clf = CFG()
ECG_CRNN_CONFIG.clf.out_channels = [
    1024,
    # not including the last linear layer, whose out channels equals n_classes
]
ECG_CRNN_CONFIG.clf.activation = "mish"
ECG_CRNN_CONFIG.clf.bias = True
ECG_CRNN_CONFIG.clf.kernel_initializer = "he_normal"
ECG_CRNN_CONFIG.clf.dropouts = 0.2
