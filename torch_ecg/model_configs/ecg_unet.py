"""
configs of the model of UNET structures
"""

from copy import deepcopy
from itertools import repeat

from ..cfg import CFG

__all__ = [
    "ECG_UNET_VANILLA_CONFIG",
]


ECG_UNET_VANILLA_CONFIG = CFG()
ECG_UNET_VANILLA_CONFIG.fs = 500

ECG_UNET_VANILLA_CONFIG.groups = 1

ECG_UNET_VANILLA_CONFIG.init_num_filters = 4  # keep the same with n_classes
ECG_UNET_VANILLA_CONFIG.init_filter_length = 25
ECG_UNET_VANILLA_CONFIG.batch_norm = True
ECG_UNET_VANILLA_CONFIG.kernel_initializer = "he_normal"
ECG_UNET_VANILLA_CONFIG.kw_initializer = {}
ECG_UNET_VANILLA_CONFIG.activation = "relu"
ECG_UNET_VANILLA_CONFIG.kw_activation = {"inplace": True}

ECG_UNET_VANILLA_CONFIG.down_up_block_num = 4

_base_filter_length = 15

ECG_UNET_VANILLA_CONFIG.down_mode = "max"
ECG_UNET_VANILLA_CONFIG.down_scales = list(
    repeat(2, ECG_UNET_VANILLA_CONFIG.down_up_block_num)
)
ECG_UNET_VANILLA_CONFIG.down_num_filters = [
    ECG_UNET_VANILLA_CONFIG.init_num_filters * (2**idx)
    for idx in range(1, ECG_UNET_VANILLA_CONFIG.down_up_block_num + 1)
]
ECG_UNET_VANILLA_CONFIG.down_filter_lengths = list(
    repeat(_base_filter_length, ECG_UNET_VANILLA_CONFIG.down_up_block_num)
)

ECG_UNET_VANILLA_CONFIG.up_mode = "nearest"
ECG_UNET_VANILLA_CONFIG.up_scales = list(
    repeat(2, ECG_UNET_VANILLA_CONFIG.down_up_block_num)
)
ECG_UNET_VANILLA_CONFIG.up_num_filters = [
    ECG_UNET_VANILLA_CONFIG.init_num_filters * (2**idx)
    for idx in range(ECG_UNET_VANILLA_CONFIG.down_up_block_num - 1, -1, -1)
]
ECG_UNET_VANILLA_CONFIG.up_deconv_filter_lengths = list(
    repeat(_base_filter_length, ECG_UNET_VANILLA_CONFIG.down_up_block_num)
)
ECG_UNET_VANILLA_CONFIG.up_conv_filter_lengths = list(
    repeat(_base_filter_length, ECG_UNET_VANILLA_CONFIG.down_up_block_num)
)


unet_down_block = CFG()
unet_down_block.batch_norm = ECG_UNET_VANILLA_CONFIG.batch_norm
unet_down_block.kernel_initializer = ECG_UNET_VANILLA_CONFIG.kernel_initializer
unet_down_block.kw_initializer = deepcopy(ECG_UNET_VANILLA_CONFIG.kw_initializer)
unet_down_block.activation = ECG_UNET_VANILLA_CONFIG.activation
unet_down_block.kw_activation = deepcopy(ECG_UNET_VANILLA_CONFIG.kw_activation)


unet_up_block = CFG()
unet_up_block.batch_norm = ECG_UNET_VANILLA_CONFIG.batch_norm
unet_up_block.kernel_initializer = ECG_UNET_VANILLA_CONFIG.kernel_initializer
unet_up_block.kw_initializer = deepcopy(ECG_UNET_VANILLA_CONFIG.kw_initializer)
unet_up_block.activation = ECG_UNET_VANILLA_CONFIG.activation
unet_up_block.kw_activation = deepcopy(ECG_UNET_VANILLA_CONFIG.kw_activation)


ECG_UNET_VANILLA_CONFIG.down_block = deepcopy(unet_down_block)
ECG_UNET_VANILLA_CONFIG.up_block = deepcopy(unet_up_block)


# out conv
ECG_UNET_VANILLA_CONFIG.out_filter_length = 9
ECG_UNET_VANILLA_CONFIG.out_batch_norm = True  # False
