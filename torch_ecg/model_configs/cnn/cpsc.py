"""
"""

from copy import deepcopy
from itertools import repeat

from ...cfg import CFG

__all__ = [
    "cpsc_block_basic",
    "cpsc_block_mish",
    "cpsc_block_swish",
    "cpsc_2018",
    "cpsc_2018_leadwise",
]


cpsc_2018 = CFG()
cpsc_2018.fs = 500
# cpsc_2018.num_filters = [  # original
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
# ]
_base_num_filters = 36
cpsc_2018.num_filters = [
    list(repeat(_base_num_filters * 2, 3)),
    list(repeat(_base_num_filters * 4, 3)),
    list(repeat(_base_num_filters * 8, 3)),
    list(repeat(_base_num_filters * 16, 3)),
    list(repeat(_base_num_filters * 32, 3)),
]
cpsc_2018.filter_lengths = [
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 24],
    [3, 3, 48],
]
cpsc_2018.subsample_lengths = [
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
]
cpsc_2018.dropouts = [0.2, 0.2, 0.2, 0.2, 0.2]
cpsc_2018.groups = 1
cpsc_2018.activation = "leaky"
cpsc_2018.kw_activation = CFG(negative_slope=0.3, inplace=True)
cpsc_2018.kernel_initializer = "he_normal"
cpsc_2018.kw_initializer = {}

cpsc_2018_leadwise = deepcopy(cpsc_2018)
cpsc_2018_leadwise.groups = 12


cpsc_block_basic = CFG()
cpsc_block_basic.activation = cpsc_2018.activation
cpsc_block_basic.kw_activation = deepcopy(cpsc_2018.kw_activation)
cpsc_block_basic.kernel_initializer = cpsc_2018.kernel_initializer
cpsc_block_basic.kw_initializer = deepcopy(cpsc_2018.kw_initializer)
cpsc_block_basic.batch_norm = False

cpsc_block_mish = deepcopy(cpsc_block_basic)
cpsc_block_mish.activation = "mish"
del cpsc_block_mish.kw_activation

cpsc_block_swish = deepcopy(cpsc_block_basic)
cpsc_block_swish.activation = "swish"
del cpsc_block_swish.kw_activation
