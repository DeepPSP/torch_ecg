"""
configs for the basic cnn layers and blocks
"""

from .cpsc import cpsc_2018_leadwise  # cpsc2018 SOTA
from .cpsc import cpsc_2018, cpsc_block_basic, cpsc_block_mish, cpsc_block_swish
from .densenet import densenet_leadwise  # vanilla densenet; custom densenet
from .densenet import densenet_vanilla
from .mobilenet import (
    mobilenet_v1_vanilla,
    mobilenet_v2_vanilla,
    mobilenet_v3_small,
)  # vanilla mobilenets
from .multi_scopic import multi_scopic, multi_scopic_block, multi_scopic_leadwise
from .resnet import (  # building blocks; vanilla resnet; cpsc2018 resnet; smaller resnets; stanford resnet; ResNet Nature Communications; TResNet
    resnet_block_basic,
    resnet_block_basic_gc,
    resnet_block_basic_nl,
    resnet_block_basic_se,
    resnet_block_stanford,
    resnet_bottle_neck,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_nl,
    resnet_bottle_neck_se,
    resnet_cpsc2018,
    resnet_cpsc2018_leadwise,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_gc,
    resnet_nature_comm_bottle_neck_nl,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_nl,
    resnet_nature_comm_se,
    resnet_stanford,
    resnet_vanilla_18,
    resnet_vanilla_34,
    resnet_vanilla_50,
    resnet_vanilla_101,
    resnet_vanilla_152,
    resnet_vanilla_wide_50_2,
    resnet_vanilla_wide_101_2,
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    resnext_vanilla_50_32x4d,
    resnext_vanilla_101_32x8d,
    tresnetF,
    tresnetL,
    tresnetM,
    tresnetM_V2,
    tresnetN,
    tresnetP,
    tresnetS,
    tresnetXL,
)
from .vgg import vgg16, vgg16_leadwise, vgg_block_basic, vgg_block_mish, vgg_block_swish
from .xception import xception_leadwise  # vanilla xception; custom xception
from .xception import xception_vanilla

__all__ = [
    # vgg
    "vgg_block_basic",
    "vgg_block_mish",
    "vgg_block_swish",
    "vgg16",
    "vgg16_leadwise",
    # resnet building blocks
    "resnet_block_basic",
    "resnet_bottle_neck",
    "resnet_bottle_neck_B",
    "resnet_bottle_neck_D",
    "resnet_block_basic_se",
    "resnet_bottle_neck_se",
    "resnet_block_basic_nl",
    "resnet_bottle_neck_nl",
    "resnet_block_basic_gc",
    "resnet_bottle_neck_gc",
    # smaller resnets
    "resnetN",
    "resnetNB",
    "resnetNS",
    "resnetNBS",
    # vanilla resnet
    "resnet_vanilla_18",
    "resnet_vanilla_34",
    "resnet_vanilla_50",
    "resnet_vanilla_101",
    "resnet_vanilla_152",
    "resnext_vanilla_50_32x4d",
    "resnext_vanilla_101_32x8d",
    "resnet_vanilla_wide_50_2",
    "resnet_vanilla_wide_101_2",
    # cpsc2018 resnet
    "resnet_cpsc2018",
    "resnet_cpsc2018_leadwise",
    # stanford resnet
    "resnet_block_stanford",
    "resnet_stanford",
    # ResNet Nature Communications
    "resnet_nature_comm",
    "resnet_nature_comm_se",
    "resnet_nature_comm_nl",
    "resnet_nature_comm_gc",
    "resnet_nature_comm_bottle_neck",
    "resnet_nature_comm_bottle_neck_se",
    "resnet_nature_comm_bottle_neck_gc",
    "resnet_nature_comm_bottle_neck_nl",
    # TresNet
    "tresnetF",
    "tresnetP",
    "tresnetN",
    "tresnetS",
    "tresnetM",
    "tresnetL",
    "tresnetXL",
    "tresnetM_V2",
    # cpsc2018 SOTA
    "cpsc_block_basic",
    "cpsc_block_mish",
    "cpsc_block_swish",
    "cpsc_2018",
    "cpsc_2018_leadwise",
    # multi_scopic
    "multi_scopic_block",
    "multi_scopic",
    "multi_scopic_leadwise",
    # vanilla densenet
    "densenet_vanilla",
    # custom densenet
    "densenet_leadwise",
    # vanilla xception
    "xception_vanilla",
    # custom xception
    "xception_leadwise",
    # vanilla mobilenets
    "mobilenet_v1_vanilla",
    "mobilenet_v2_vanilla",
    "mobilenet_v3_small",
]
