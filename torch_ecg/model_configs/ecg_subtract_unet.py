"""
configs of the model of (subtract) UNET structures
"""

from copy import deepcopy
from itertools import repeat

from ..cfg import CFG

__all__ = [
    "ECG_SUBTRACT_UNET_CONFIG",
]


ECG_SUBTRACT_UNET_CONFIG = CFG()
ECG_SUBTRACT_UNET_CONFIG.fs = 500


ECG_SUBTRACT_UNET_CONFIG.groups = 1
ECG_SUBTRACT_UNET_CONFIG.init_batch_norm = False


# in triple conv
ECG_SUBTRACT_UNET_CONFIG.init_num_filters = 16
ECG_SUBTRACT_UNET_CONFIG.init_filter_length = 21
ECG_SUBTRACT_UNET_CONFIG.init_dropouts = [0.0, 0.15, 0.0]
ECG_SUBTRACT_UNET_CONFIG.batch_norm = True
ECG_SUBTRACT_UNET_CONFIG.kernel_initializer = "he_normal"
ECG_SUBTRACT_UNET_CONFIG.kw_initializer = {}
ECG_SUBTRACT_UNET_CONFIG.activation = "relu"
ECG_SUBTRACT_UNET_CONFIG.kw_activation = {"inplace": True}


_num_convs = 3  # TripleConv

# down, triple conv
ECG_SUBTRACT_UNET_CONFIG.down_up_block_num = 3

ECG_SUBTRACT_UNET_CONFIG.down_mode = "max"
ECG_SUBTRACT_UNET_CONFIG.down_scales = [10, 5, 2]
init_down_num_filters = 24
ECG_SUBTRACT_UNET_CONFIG.down_num_filters = [
    list(repeat(init_down_num_filters * (2**idx), _num_convs))
    for idx in range(0, ECG_SUBTRACT_UNET_CONFIG.down_up_block_num - 1)
]
ECG_SUBTRACT_UNET_CONFIG.down_filter_lengths = [11, 5]
ECG_SUBTRACT_UNET_CONFIG.down_dropouts = list(
    repeat([0.0, 0.15, 0.0], ECG_SUBTRACT_UNET_CONFIG.down_up_block_num - 1)
)


# bottom, double conv
ECG_SUBTRACT_UNET_CONFIG.bottom_num_filters = [
    # branch 1
    list(
        repeat(
            init_down_num_filters
            * (2 ** (ECG_SUBTRACT_UNET_CONFIG.down_up_block_num - 1)),
            2,
        )
    ),
    # branch 2
    list(
        repeat(
            init_down_num_filters
            * (2 ** (ECG_SUBTRACT_UNET_CONFIG.down_up_block_num - 1)),
            2,
        )
    ),
    # branch 1 and branch 2 should have the same `num_filters`,
    # otherwise `subtraction` would be infeasible
]
ECG_SUBTRACT_UNET_CONFIG.bottom_filter_lengths = [
    list(repeat(5, 2)),  # branch 1
    list(repeat(5, 2)),  # branch 2
]
ECG_SUBTRACT_UNET_CONFIG.bottom_dilations = [
    # the ordering matters
    list(repeat(1, 2)),  # branch 1
    list(repeat(10, 2)),  # branch 2
]
ECG_SUBTRACT_UNET_CONFIG.bottom_dropouts = [
    [0.15, 0.0],  # branch 1
    [0.15, 0.0],  # branch 2
]


# up, triple conv
ECG_SUBTRACT_UNET_CONFIG.up_mode = "nearest"
ECG_SUBTRACT_UNET_CONFIG.up_scales = [2, 5, 10]
ECG_SUBTRACT_UNET_CONFIG.up_num_filters = [
    list(repeat(48, _num_convs)),
    list(repeat(24, _num_convs)),
    list(repeat(16, _num_convs)),
]
ECG_SUBTRACT_UNET_CONFIG.up_deconv_filter_lengths = list(
    repeat(9, ECG_SUBTRACT_UNET_CONFIG.down_up_block_num)
)
ECG_SUBTRACT_UNET_CONFIG.up_conv_filter_lengths = [5, 11, 21]
ECG_SUBTRACT_UNET_CONFIG.up_dropouts = [
    [0.15, 0.15, 0.0],
    [0.15, 0.15, 0.0],
    [0.15, 0.15, 0.0],
]


unet_down_block = CFG()
unet_down_block.batch_norm = ECG_SUBTRACT_UNET_CONFIG.batch_norm
unet_down_block.kernel_initializer = ECG_SUBTRACT_UNET_CONFIG.kernel_initializer
unet_down_block.kw_initializer = deepcopy(ECG_SUBTRACT_UNET_CONFIG.kw_initializer)
unet_down_block.activation = ECG_SUBTRACT_UNET_CONFIG.activation
unet_down_block.kw_activation = deepcopy(ECG_SUBTRACT_UNET_CONFIG.kw_activation)


unet_up_block = CFG()
unet_up_block.batch_norm = ECG_SUBTRACT_UNET_CONFIG.batch_norm
unet_up_block.kernel_initializer = ECG_SUBTRACT_UNET_CONFIG.kernel_initializer
unet_up_block.kw_initializer = deepcopy(ECG_SUBTRACT_UNET_CONFIG.kw_initializer)
unet_up_block.activation = ECG_SUBTRACT_UNET_CONFIG.activation
unet_up_block.kw_activation = deepcopy(ECG_SUBTRACT_UNET_CONFIG.kw_activation)


ECG_SUBTRACT_UNET_CONFIG.down_block = deepcopy(unet_down_block)
ECG_SUBTRACT_UNET_CONFIG.up_block = deepcopy(unet_up_block)


# out conv
ECG_SUBTRACT_UNET_CONFIG.out_filter_length = 1
ECG_SUBTRACT_UNET_CONFIG.out_batch_norm = True  # False
