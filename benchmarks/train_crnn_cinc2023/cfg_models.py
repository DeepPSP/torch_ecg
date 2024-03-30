"""
"""

from copy import deepcopy

from torch_ecg.cfg import CFG
from torch_ecg.model_configs import (  # noqa: F401; cnn bankbone; lstm; mlp; attn; the whole model config
    ECG_CRNN_CONFIG,
    attention,
    densenet_leadwise,
    global_context,
    linear,
    lstm,
    multi_scopic,
    multi_scopic_block,
    multi_scopic_leadwise,
    non_local,
    resnet_block_basic,
    resnet_block_basic_gc,
    resnet_block_basic_se,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_se,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_se,
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    squeeze_excitation,
    tresnetF,
    tresnetM,
    tresnetN,
    tresnetP,
    tresnetS,
    vgg16,
    vgg16_leadwise,
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    xception_leadwise,
)

__all__ = [
    "ModelArchCfg",
]


# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = CFG()

ModelArchCfg.classification = CFG()
ModelArchCfg.classification.crnn = deepcopy(ECG_CRNN_CONFIG)
