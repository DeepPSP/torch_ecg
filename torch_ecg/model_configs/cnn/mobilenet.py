"""
"""

from itertools import repeat
from copy import deepcopy

import numpy as np

from ...cfg import CFG


__all__ = [
    # vanilla mobilenets
    "mobilenet_v1_vanilla",
    "mobilenet_v2_vanilla",
]


mobilenet_v1_vanilla = CFG()
mobilenet_v1_vanilla.fs = 500
mobilenet_v1_vanilla.groups = 1
mobilenet_v1_vanilla.batch_norm = True
mobilenet_v1_vanilla.activation = "relu6"
mobilenet_v1_vanilla.depth_multiplier = 1  # multiplier of number of depthwise convolution output channels
mobilenet_v1_vanilla.width_multiplier = 1.0  # controls the width (number of filters) of the network
mobilenet_v1_vanilla.bias = True
mobilenet_v1_vanilla.ordering = "cba"

_base_num_filters = 12 * 3
mobilenet_v1_vanilla.init_num_filters = _base_num_filters
mobilenet_v1_vanilla.init_filter_lengths = 27
mobilenet_v1_vanilla.init_subsample_lengths = 2

mobilenet_v1_vanilla.entry_flow = CFG()
mobilenet_v1_vanilla.entry_flow.out_channels = [
    # 64, 128, 128, 256, 256
    _base_num_filters * 2,
    _base_num_filters * 4, _base_num_filters * 4,
    _base_num_filters * 8, _base_num_filters * 8,
    _base_num_filters * 16,
]
mobilenet_v1_vanilla.entry_flow.filter_lengths = 15
mobilenet_v1_vanilla.entry_flow.subsample_lengths = [
    1, 2, 1, 2, 1, 2,
]
mobilenet_v1_vanilla.entry_flow.groups = mobilenet_v1_vanilla.groups
mobilenet_v1_vanilla.entry_flow.batch_norm = mobilenet_v1_vanilla.batch_norm
mobilenet_v1_vanilla.entry_flow.activation = mobilenet_v1_vanilla.activation

mobilenet_v1_vanilla.middle_flow = CFG()
mobilenet_v1_vanilla.middle_flow.out_channels = list(repeat(_base_num_filters * 16, 5))
mobilenet_v1_vanilla.middle_flow.filter_lengths = 13
mobilenet_v1_vanilla.middle_flow.subsample_lengths = 1
mobilenet_v1_vanilla.middle_flow.groups = mobilenet_v1_vanilla.groups
mobilenet_v1_vanilla.middle_flow.batch_norm = mobilenet_v1_vanilla.batch_norm
mobilenet_v1_vanilla.middle_flow.activation = mobilenet_v1_vanilla.activation

mobilenet_v1_vanilla.exit_flow = CFG()
mobilenet_v1_vanilla.exit_flow.out_channels = [
    _base_num_filters * 32, _base_num_filters * 32,
]
mobilenet_v1_vanilla.exit_flow.filter_lengths = 17
mobilenet_v1_vanilla.exit_flow.subsample_lengths = [
    2, 1
]
mobilenet_v1_vanilla.exit_flow.groups = mobilenet_v1_vanilla.groups
mobilenet_v1_vanilla.exit_flow.batch_norm = mobilenet_v1_vanilla.batch_norm
mobilenet_v1_vanilla.exit_flow.activation = mobilenet_v1_vanilla.activation


mobilenet_v2_vanilla = CFG()
mobilenet_v2_vanilla.fs = 500
mobilenet_v2_vanilla.groups = 1
mobilenet_v2_vanilla.batch_norm = True
mobilenet_v2_vanilla.activation = "relu6"
mobilenet_v2_vanilla.depth_multiplier = 1  # multiplier of number of depthwise convolution output channels
mobilenet_v2_vanilla.width_multiplier = 1.0  # controls the width (number of filters) of the network
mobilenet_v2_vanilla.bias = True
mobilenet_v2_vanilla.ordering = "cba"

_base_num_filters = 12
mobilenet_v2_vanilla.init_num_filters = _base_num_filters * 4
mobilenet_v2_vanilla.init_filter_lengths = 27
mobilenet_v2_vanilla.init_subsample_lengths = 2

_inverted_residual_setting = np.array([
    # t, c, n, s, k
    [1, _base_num_filters*2, 1, 1, 15],
    [6, _base_num_filters*3, 2, 2, 15],
    [6, _base_num_filters*4, 3, 2, 15],
    [6, _base_num_filters*6, 4, 2, 15],
    [6, _base_num_filters*8, 3, 1, 15],
    [6, _base_num_filters*20, 3, 2, 15],
    [6, _base_num_filters*40, 1, 1, 15],
    # t: expansion
    # c: output channels
    # n: number of blocks
    # s: stride
    # k: kernel size
])
mobilenet_v2_vanilla.inv_res = CFG()
mobilenet_v2_vanilla.inv_res.expansions = _inverted_residual_setting[...,0]
mobilenet_v2_vanilla.inv_res.out_channels = _inverted_residual_setting[...,1]
mobilenet_v2_vanilla.inv_res.n_blocks = _inverted_residual_setting[...,2]
mobilenet_v2_vanilla.inv_res.strides = _inverted_residual_setting[...,3]
mobilenet_v2_vanilla.inv_res.filter_lengths = _inverted_residual_setting[...,4]

mobilenet_v2_vanilla.final_num_filters = _base_num_filters * 128
mobilenet_v2_vanilla.final_filter_lengths = 19
mobilenet_v2_vanilla.final_subsample_lengths = 2
