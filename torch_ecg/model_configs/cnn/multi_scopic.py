"""
"""

from copy import deepcopy
from itertools import repeat

from ...cfg import CFG

__all__ = [
    "multi_scopic_block",
    "multi_scopic",
    "multi_scopic_leadwise",
]


# configs of multi_scopic cnn net are set by path, not by level
multi_scopic = CFG()
multi_scopic.fs = 500
multi_scopic.groups = 1
multi_scopic.scopes = [
    [
        [
            1,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    ],
    [
        [
            2,
        ],
        [
            2,
            4,
        ],
        [
            8,
            8,
            8,
        ],
    ],
    [
        [
            4,
        ],
        [
            4,
            8,
        ],
        [
            16,
            32,
            64,
        ],
    ],
]
multi_scopic.filter_lengths = [
    [
        11,
        7,
        5,
    ],
    [
        11,
        7,
        5,
    ],
    [
        11,
        7,
        5,
    ],
]
# subsample_lengths for each branch
multi_scopic.subsample_lengths = list(repeat(2, len(multi_scopic.scopes)))
_base_num_filters = 12 * 2
multi_scopic.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
multi_scopic.dropouts = [
    [0, 0.2, 0],
    [0, 0.2, 0],
    [0, 0.2, 0],
]
multi_scopic.bias = True
multi_scopic.kernel_initializer = "he_normal"
multi_scopic.kw_initializer = {}
multi_scopic.activation = "relu"
multi_scopic.kw_activation = {"inplace": True}
# multi_scopic.batch_norm = False
# multi_scopic.kw_bn = {}

multi_scopic_leadwise = deepcopy(multi_scopic)
multi_scopic_leadwise.groups = 12
# multi_scopic_leadwise.batch_norm = False  # consider using "group_norm"
_base_num_filters = 12 * 4
multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]


multi_scopic_block = CFG()
multi_scopic_block.subsample_mode = (
    "max"  # or "conv", "avg", "nearest", "linear", "bilinear"
)
multi_scopic_block.bias = multi_scopic.bias
multi_scopic_block.kernel_initializer = multi_scopic.kernel_initializer
multi_scopic_block.kw_initializer = deepcopy(multi_scopic.kw_initializer)
multi_scopic_block.activation = multi_scopic.activation
multi_scopic_block.kw_activation = deepcopy(multi_scopic.kw_activation)
multi_scopic_block.batch_norm = False  # consider using "group_norm"
multi_scopic_block.kw_bn = {}


# set default building block
multi_scopic.block = deepcopy(multi_scopic_block)
multi_scopic_leadwise.block = deepcopy(multi_scopic_block)
