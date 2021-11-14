"""
vanilla resnet, and various variants

Reference
---------
1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
2. He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019). Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 558-567).
3. Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nature medicine, 25(1), 65-69.
4. Ribeiro, A. H., Ribeiro, M. H., Paixão, G. M., Oliveira, D. M., Gomes, P. R., Canazart, J. A., ... & Ribeiro, A. L. P. (2020). Automatic diagnosis of the 12-lead ECG using a deep neural network. Nature communications, 11(1), 1-9.
5. Ridnik, T., Lawen, H., Noy, A., Ben Baruch, E., Sharir, G., & Friedman, I. (2021). Tresnet: High performance gpu-dedicated architecture. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1400-1409).
"""

from itertools import repeat
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED

from ..attn import (
    squeeze_excitation,
    non_local,
    global_context,
)


__all__ = [
    # building blocks
    "resnet_block_basic", "resnet_bottle_neck",
    "resnet_bottle_neck_B", "resnet_bottle_neck_D",
    "resnet_block_basic_se", "resnet_bottle_neck_se",
    "resnet_block_basic_nl", "resnet_bottle_neck_nl",
    "resnet_block_basic_gc", "resnet_bottle_neck_gc",
    # vanilla resnet
    "resnet_vanilla_18", "resnet_vanilla_34",
    "resnet_vanilla_50", "resnet_vanilla_101", "resnet_vanilla_152",
    "resnext_vanilla_50_32x4d", "resnext_vanilla_101_32x8d",
    "resnet_vanilla_wide_50_2", "resnet_vanilla_wide_101_2",
    # custom resnet
    "resnet_cpsc2018", "resnet_cpsc2018_leadwise",
    # stanford resnet
    "resnet_block_stanford", "resnet_stanford",
    # ResNet Nature Communications
    "resnet_nature_comm",
    "resnet_nature_comm_se", "resnet_nature_comm_nl", "resnet_nature_comm_gc",
    # TresNet
    "tresnetM", "tresnetL", "tresnetXL",
]


# building blocks

resnet_block_basic = ED()
resnet_block_basic.increase_channels_method = "conv"  # or "zero_padding"
resnet_block_basic.subsample_mode = "conv"  # or "max", "avg", "nearest", "linear", "bilinear"
resnet_block_basic.kernel_initializer = "he_normal"
resnet_block_basic.kw_initializer = {}
resnet_block_basic.activation = "relu"  # "mish", "swish"
resnet_block_basic.kw_activation = {"inplace": True}
resnet_block_basic.bias = False

resnet_bottle_neck = ED()
resnet_bottle_neck.expansion = 4
resnet_bottle_neck.increase_channels_method = "conv"  # or "zero_padding"
resnet_bottle_neck.subsample_mode = "conv"  # or "max", "avg", "nearest", "linear", "bilinear"
resnet_bottle_neck.subsample_at = 0  # or 0  # 0 is for the original paper, 1 for ResNet-B
resnet_bottle_neck.kernel_initializer = "he_normal"
resnet_bottle_neck.kw_initializer = {}
resnet_bottle_neck.activation = "relu"  # "mish", "swish"
resnet_bottle_neck.kw_activation = {"inplace": True}
resnet_bottle_neck.bias = False

resnet_bottle_neck_B = deepcopy(resnet_bottle_neck)
resnet_bottle_neck_B.subsample_at = 1

resnet_bottle_neck_D = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_D.subsample_mode = "avg"

# one can add more variants by changing the attention modules
resnet_block_basic_se = deepcopy(resnet_block_basic)
resnet_block_basic_se.attn = deepcopy(squeeze_excitation)
resnet_block_basic_se.attn.name = "se"
resnet_block_basic_se.attn.pos = -1

resnet_bottle_neck_se = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_se.attn = deepcopy(squeeze_excitation)
resnet_bottle_neck_se.attn.name = "se"
resnet_bottle_neck_se.attn.pos = -1

resnet_block_basic_nl = deepcopy(resnet_block_basic)
resnet_block_basic_nl.attn = deepcopy(non_local)
resnet_block_basic_nl.attn.name = "nl"
resnet_block_basic_nl.attn.pos = -1

resnet_bottle_neck_nl = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_nl.attn = deepcopy(non_local)
resnet_bottle_neck_nl.attn.name = "nl"
resnet_bottle_neck_nl.attn.pos = -1

resnet_block_basic_gc = deepcopy(resnet_block_basic)
resnet_block_basic_gc.attn = deepcopy(global_context)
resnet_block_basic_gc.attn.name = "gc"
resnet_block_basic_gc.attn.pos = -1

resnet_bottle_neck_gc = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_gc.attn = deepcopy(global_context)
resnet_bottle_neck_gc.attn.name = "gc"
resnet_bottle_neck_gc.attn.pos = -1


# vanilla ResNets
resnet_vanilla_common = ED()
resnet_vanilla_common.fs = 500
resnet_vanilla_common.subsample_lengths = [
    1, 2, 2, 2,
]
resnet_vanilla_common.filter_lengths = 15
resnet_vanilla_common.groups = 1
resnet_vanilla_common.increase_channels_method = "conv"
resnet_vanilla_common.init_num_filters = 64
resnet_vanilla_common.init_filter_length = 29
resnet_vanilla_common.init_conv_stride = 2
resnet_vanilla_common.init_pool_size = 3
resnet_vanilla_common.init_pool_stride = 2
resnet_vanilla_common.kernel_initializer = "he_normal"
resnet_vanilla_common.kw_initializer = {}
resnet_vanilla_common.activation = "relu"  # "mish", "swish"
resnet_vanilla_common.kw_activation = {"inplace": True}
resnet_vanilla_common.bias = False


resnet_vanilla_18 = ED()
resnet_vanilla_18.num_blocks = [
    2, 2, 2, 2,
]
resnet_vanilla_18.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_18.building_block = "basic"
resnet_vanilla_18.block = deepcopy(resnet_block_basic)
resnet_vanilla_18.block.kernel_initializer = resnet_vanilla_18.kernel_initializer
resnet_vanilla_18.block.kw_initializer = resnet_vanilla_18.kw_initializer
resnet_vanilla_18.block.activation = resnet_vanilla_18.activation
resnet_vanilla_18.block.kw_activation = resnet_vanilla_18.kw_activation
resnet_vanilla_18.block.bias = resnet_vanilla_18.bias

resnet_vanilla_34 = ED()
resnet_vanilla_34.num_blocks = [
    3, 4, 6, 3,
]
resnet_vanilla_34.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_34.building_block = "basic"
resnet_vanilla_34.block = deepcopy(resnet_block_basic)
resnet_vanilla_34.block.kernel_initializer = resnet_vanilla_34.kernel_initializer
resnet_vanilla_34.block.kw_initializer = resnet_vanilla_34.kw_initializer
resnet_vanilla_34.block.activation = resnet_vanilla_34.activation
resnet_vanilla_34.block.kw_activation = resnet_vanilla_34.kw_activation
resnet_vanilla_34.block.bias = resnet_vanilla_34.bias

resnet_vanilla_50 = ED()  # uses bottleneck
resnet_vanilla_50.num_blocks = [
    3, 4, 6, 3,
]
resnet_vanilla_50.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_50.base_groups = 1
resnet_vanilla_50.base_width = 64
resnet_vanilla_50.building_block = "bottleneck"
resnet_vanilla_50.block = deepcopy(resnet_bottle_neck_B)
resnet_vanilla_50.block.kernel_initializer = resnet_vanilla_50.kernel_initializer
resnet_vanilla_50.block.kw_initializer = resnet_vanilla_50.kw_initializer
resnet_vanilla_50.block.activation = resnet_vanilla_50.activation
resnet_vanilla_50.block.kw_activation = resnet_vanilla_50.kw_activation
resnet_vanilla_50.block.bias = resnet_vanilla_50.bias

resnet_vanilla_101 = ED()  # uses bottleneck
resnet_vanilla_101.num_blocks = [
    3, 4, 23, 3,
]
resnet_vanilla_101.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_101.base_groups = 1
resnet_vanilla_101.base_width = 64
resnet_vanilla_101.building_block = "bottleneck"
resnet_vanilla_101.block = deepcopy(resnet_bottle_neck_B)
resnet_vanilla_101.block.kernel_initializer = resnet_vanilla_101.kernel_initializer
resnet_vanilla_101.block.kw_initializer = resnet_vanilla_101.kw_initializer
resnet_vanilla_101.block.activation = resnet_vanilla_101.activation
resnet_vanilla_101.block.kw_activation = resnet_vanilla_101.kw_activation
resnet_vanilla_101.block.bias = resnet_vanilla_101.bias

resnet_vanilla_152 = ED()  # uses bottleneck
resnet_vanilla_152.num_blocks = [
    3, 8, 36, 3,
]
resnet_vanilla_152.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_152.base_groups = 1
resnet_vanilla_152.base_width = 64
resnet_vanilla_152.building_block = "bottleneck"
resnet_vanilla_152.block = deepcopy(resnet_bottle_neck_B)
resnet_vanilla_152.block.kernel_initializer = resnet_vanilla_152.kernel_initializer
resnet_vanilla_152.block.kw_initializer = resnet_vanilla_152.kw_initializer
resnet_vanilla_152.block.activation = resnet_vanilla_152.activation
resnet_vanilla_152.block.kw_activation = resnet_vanilla_152.kw_activation
resnet_vanilla_152.block.bias = resnet_vanilla_152.bias

resnext_vanilla_50_32x4d = ED()  # uses bottleneck
resnext_vanilla_50_32x4d.num_blocks = [
    3, 4, 6, 3,
]
resnext_vanilla_50_32x4d.update(deepcopy(resnet_vanilla_common))
resnext_vanilla_50_32x4d.groups = 32
resnext_vanilla_50_32x4d.base_groups = 1
resnext_vanilla_50_32x4d.base_width = 4
resnext_vanilla_50_32x4d.building_block = "bottleneck"
resnext_vanilla_50_32x4d.block = deepcopy(resnet_bottle_neck_B)
resnext_vanilla_50_32x4d.block.kernel_initializer = resnext_vanilla_50_32x4d.kernel_initializer
resnext_vanilla_50_32x4d.block.kw_initializer = resnext_vanilla_50_32x4d.kw_initializer
resnext_vanilla_50_32x4d.block.activation = resnext_vanilla_50_32x4d.activation
resnext_vanilla_50_32x4d.block.kw_activation = resnext_vanilla_50_32x4d.kw_activation
resnext_vanilla_50_32x4d.block.bias = resnext_vanilla_50_32x4d.bias

resnext_vanilla_101_32x8d = ED()  # uses bottleneck
resnext_vanilla_101_32x8d.num_blocks = [
    3, 4, 23, 3,
]
resnext_vanilla_101_32x8d.update(deepcopy(resnet_vanilla_common))
resnext_vanilla_101_32x8d.groups = 32
resnext_vanilla_101_32x8d.base_groups = 1
resnext_vanilla_101_32x8d.base_width = 8
resnext_vanilla_101_32x8d.building_block = "bottleneck"
resnext_vanilla_101_32x8d.block = deepcopy(resnet_bottle_neck_B)
resnext_vanilla_101_32x8d.block.kernel_initializer = resnext_vanilla_101_32x8d.kernel_initializer
resnext_vanilla_101_32x8d.block.kw_initializer = resnext_vanilla_101_32x8d.kw_initializer
resnext_vanilla_101_32x8d.block.activation = resnext_vanilla_101_32x8d.activation
resnext_vanilla_101_32x8d.block.kw_activation = resnext_vanilla_101_32x8d.kw_activation
resnext_vanilla_101_32x8d.block.bias = resnext_vanilla_101_32x8d.bias

resnet_vanilla_wide_50_2 = ED()  # uses bottleneck
resnet_vanilla_wide_50_2.num_blocks = [
    3, 4, 6, 3,
]
resnet_vanilla_wide_50_2.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_wide_50_2.base_groups = 1
resnet_vanilla_wide_50_2.base_width = 64 * 2
resnet_vanilla_wide_50_2.building_block = "bottleneck"
resnet_vanilla_wide_50_2.block = deepcopy(resnet_bottle_neck_B)
resnet_vanilla_wide_50_2.block.kernel_initializer = resnet_vanilla_wide_50_2.kernel_initializer
resnet_vanilla_wide_50_2.block.kw_initializer = resnet_vanilla_wide_50_2.kw_initializer
resnet_vanilla_wide_50_2.block.activation = resnet_vanilla_wide_50_2.activation
resnet_vanilla_wide_50_2.block.kw_activation = resnet_vanilla_wide_50_2.kw_activation
resnet_vanilla_wide_50_2.block.bias = resnet_vanilla_wide_50_2.bias

resnet_vanilla_wide_101_2 = ED()  # uses bottleneck
resnet_vanilla_wide_101_2.num_blocks = [
    3, 4, 23, 3,
]
resnet_vanilla_wide_101_2.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_wide_101_2.base_groups = 1
resnet_vanilla_wide_101_2.base_width = 64 * 2
resnet_vanilla_wide_101_2.building_block = "bottleneck"
resnet_vanilla_wide_101_2.block = deepcopy(resnet_bottle_neck_B)
resnet_vanilla_wide_101_2.block.kernel_initializer = resnet_vanilla_wide_101_2.kernel_initializer
resnet_vanilla_wide_101_2.block.kw_initializer = resnet_vanilla_wide_101_2.kw_initializer
resnet_vanilla_wide_101_2.block.activation = resnet_vanilla_wide_101_2.activation
resnet_vanilla_wide_101_2.block.kw_activation = resnet_vanilla_wide_101_2.kw_activation
resnet_vanilla_wide_101_2.block.bias = resnet_vanilla_wide_101_2.bias


# custom ResNets
resnet_cpsc2018 = ED()
resnet_cpsc2018.fs = 500
resnet_cpsc2018.building_block = "basic"  # "bottleneck"
resnet_cpsc2018.expansion = 1
resnet_cpsc2018.subsample_lengths = [
    1, 2, 2, 2,
]
# resnet_cpsc2018.num_blocks = [
#     2, 2, 2, 2, 2,
# ]
# resnet_cpsc2018.filter_lengths = 3
# resnet_cpsc2018.num_blocks = [
#     3, 4, 6, 3,
# ]
# resnet_cpsc2018.filter_lengths = [
#     [5, 5, 13],
#     [5, 5, 5, 13],
#     [5, 5, 5, 5, 5, 13],
#     [5, 5, 25],
# ]
resnet_cpsc2018.num_blocks = [
    3, 4, 6, 3,
]
resnet_cpsc2018.filter_lengths = [
    [5, 5, 25],
    [5, 5, 5, 25],
    [5, 5, 5, 5, 5, 25],
    [5, 5, 49],
]
resnet_cpsc2018.groups = 1
_base_num_filters = 12 * 4
resnet_cpsc2018.init_num_filters = _base_num_filters
resnet_cpsc2018.init_filter_length = 15  # corr. to 30 ms
resnet_cpsc2018.init_conv_stride = 2
resnet_cpsc2018.init_pool_size = 3
resnet_cpsc2018.init_pool_stride = 2
resnet_cpsc2018.kernel_initializer = "he_normal"
resnet_cpsc2018.kw_initializer = {}
resnet_cpsc2018.activation = "relu"  # "mish", "swish"
resnet_cpsc2018.kw_activation = {"inplace": True}
resnet_cpsc2018.bias = False


resnet_cpsc2018_leadwise = deepcopy(resnet_cpsc2018)
resnet_cpsc2018_leadwise.groups = 12
resnet_cpsc2018_leadwise.init_num_filters = 12 * 8


# set default building block
resnet_cpsc2018.building_block = "basic"
resnet_cpsc2018.block = deepcopy(resnet_block_basic)
resnet_cpsc2018_leadwise.building_block = "basic"
resnet_cpsc2018_leadwise.block = deepcopy(resnet_block_basic)



# ResNet Stanford
resnet_stanford = ED()
resnet_stanford.fs = 500
resnet_stanford.groups = 1
resnet_stanford.num_blocks = [
    2, 2, 2, 2,
]
resnet_stanford.subsample_lengths = 2
resnet_stanford.filter_lengths = 17
_base_num_filters = 36
resnet_stanford.init_num_filters = _base_num_filters*2
resnet_stanford.init_filter_length = 17
resnet_stanford.init_conv_stride = 2
resnet_stanford.init_pool_size = 3
resnet_stanford.init_pool_stride = 2
resnet_stanford.kernel_initializer = "he_normal"
resnet_stanford.kw_initializer = {}
resnet_stanford.activation = "relu"
resnet_stanford.kw_activation = {"inplace": True}
resnet_stanford.bias = False


resnet_block_stanford = ED()
resnet_block_stanford.increase_channels_at = 4
resnet_block_stanford.increase_channels_method = "conv"  # or "zero_padding"
resnet_block_stanford.num_skip = 2
resnet_block_stanford.subsample_mode = "conv"  # "max", "avg"
resnet_block_stanford.kernel_initializer = resnet_stanford.kernel_initializer
resnet_block_stanford.kw_initializer = deepcopy(resnet_stanford.kw_initializer)
resnet_block_stanford.activation = resnet_stanford.activation
resnet_block_stanford.kw_activation = deepcopy(resnet_stanford.kw_activation)
resnet_block_stanford.dropout = 0.2
resnet_block_stanford.bias = False

resnet_stanford.building_block = "basic"
resnet_stanford.block = deepcopy(resnet_block_stanford)


# ResNet Nature Communications
# a modified version of resnet 34
resnet_nature_comm = deepcopy(resnet_vanilla_34)
resnet_nature_comm.init_filter_length = 17  # originally 16, we make it odd
resnet_nature_comm.filter_lengths = 17
resnet_nature_comm.init_conv_stride = 1
resnet_nature_comm.init_pool_stride = 1  # if < 2, init pool will not be added
resnet_nature_comm.dropouts = 0.2
resnet_nature_comm.subsample_lengths = [
    4, 4, 4, 4,
]
resnet_nature_comm.num_filters = [
    2 * resnet_nature_comm.init_num_filters,
    3 * resnet_nature_comm.init_num_filters,
    4 * resnet_nature_comm.init_num_filters,
    5 * resnet_nature_comm.init_num_filters,
]
resnet_nature_comm.block.subsample_mode = "max"


# variant with attention of ResNet Nature Communications
resnet_nature_comm_se = deepcopy(resnet_nature_comm)
resnet_nature_comm_se.block = deepcopy(resnet_block_basic_se)

resnet_nature_comm_nl = deepcopy(resnet_nature_comm)
resnet_nature_comm_nl.block = deepcopy(resnet_block_basic_nl)

resnet_nature_comm_gc = deepcopy(resnet_nature_comm)
resnet_nature_comm_gc.block = deepcopy(resnet_block_basic_gc)



# TResNet
# NOTE: TResNet is not finished!
tresnet_common = ED()
tresnet_common.fs = 500
tresnet_common.subsample_lengths = [
    1, 2, 2, 2,
]
tresnet_common.filter_lengths = 15
tresnet_common.groups = 1
tresnet_common.increase_channels_method = "conv"
tresnet_common.init_num_filters = [48, 64,]
tresnet_common.init_filter_length = [19, 19]
tresnet_common.init_conv_stride = 1
tresnet_common.init_pool_size = 1
tresnet_common.init_pool_stride = 1
tresnet_common.kernel_initializer = "he_normal"
tresnet_common.kw_initializer = {}
tresnet_common.activation = "relu"  # "mish", "swish"
tresnet_common.kw_activation = {"inplace": True}
tresnet_common.bias = False

tresnet_common.building_block = ["basic", "basic", "bottleneck", "bottleneck"]
tresnet_common.block = [
    deepcopy(resnet_block_basic_se),
    deepcopy(resnet_block_basic_se),
    deepcopy(resnet_bottle_neck_se),
    deepcopy(resnet_bottle_neck_B),
]
tresnet_common.block[0].reduction = 4
tresnet_common.block[1].reduction = 4
tresnet_common.block[2].pos = 2
for b in tresnet_common.block:
    b.kernel_initializer = tresnet_common.kernel_initializer
    b.kw_initializer = tresnet_common.kw_initializer
    b.activation = tresnet_common.activation
    b.kw_activation = tresnet_common.kw_activation
    b.bias = tresnet_common.bias
    b.subsample_mode = "blur"
    b.filt_size = 7  # for blur subsampling

# TResNet-M
tresnetM = deepcopy(tresnet_common)
tresnetM.num_blocks = [
    3, 4, 11, 3,
]
tresnetM.init_num_filters = [48, 64,]
tresnetM.num_filters = [
    tresnetM.init_num_filters[-1],
    2 * tresnetM.init_num_filters[-1],
    16 * tresnetM.init_num_filters[-1],
    32 * tresnetM.init_num_filters[-1],
]

# TResNet-L
tresnetL = deepcopy(tresnet_common)
tresnetL.num_blocks = [
    4, 5, 18, 3,
]
tresnetM.init_num_filters = [48, 76,]
tresnetM.num_filters = [
    tresnetM.init_num_filters[-1],
    2 * tresnetM.init_num_filters[-1],
    16 * tresnetM.init_num_filters[-1],
    32 * tresnetM.init_num_filters[-1],
]

# TResNet-XL
tresnetXL = deepcopy(tresnet_common)
tresnetXL.num_blocks = [
    4, 5, 24, 3,
]
tresnetM.init_num_filters = [48, 76,]
tresnetM.num_filters = [
    tresnetM.init_num_filters[-1],
    2 * tresnetM.init_num_filters[-1],
    16 * tresnetM.init_num_filters[-1],
    32 * tresnetM.init_num_filters[-1],
]
