"""
"""

from itertools import repeat
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED


__all__ = [
    # building blocks
    "resnet_block_basic", "resnet_bottle_neck",
    "resnet_bottle_neck_B", "resnet_bottle_neck_D",
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
resnet_nature_comm.init_filter_length = 17
resnet_nature_comm.filter_lengths = 17
resnet_nature_comm.init_conv_stride = 1
resnet_nature_comm.init_pool_stride = 1  # if < 2, init pool will not be added
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
