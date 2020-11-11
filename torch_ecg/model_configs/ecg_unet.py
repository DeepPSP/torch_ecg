"""
configs of the model of UNET structures
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "ECG_UNET_CONFIG",
]


ECG_UNET_CONFIG = ED()

ECG_UNET_CONFIG.groups = 1

ECG_UNET_CONFIG.init_num_filters = 4  # keep the same with n_classes
ECG_UNET_CONFIG.init_filter_length = 9
ECG_UNET_CONFIG.out_filter_length = 9
ECG_UNET_CONFIG.batch_norm = True
ECG_UNET_CONFIG.kernel_initializer = "he_normal"
ECG_UNET_CONFIG.kw_initializer = {}
ECG_UNET_CONFIG.activation = "relu"
ECG_UNET_CONFIG.kw_activation = {"inplace": True}

ECG_UNET_CONFIG.down_up_block_num = 4

ECG_UNET_CONFIG.down_mode = "max"
ECG_UNET_CONFIG.down_scales = list(repeat(2, ECG_UNET_CONFIG.down_up_block_num))
ECG_UNET_CONFIG.down_num_filters = [
    ECG_UNET_CONFIG.init_num_filters * (2**idx) \
        for idx in range(1, ECG_UNET_CONFIG.down_up_block_num+1)
]
ECG_UNET_CONFIG.down_filter_lengths = \
    list(repeat(ECG_UNET_CONFIG.init_filter_length, ECG_UNET_CONFIG.down_up_block_num))

ECG_UNET_CONFIG.up_mode = "nearest"
ECG_UNET_CONFIG.up_scales = list(repeat(2, ECG_UNET_CONFIG.down_up_block_num))
ECG_UNET_CONFIG.up_num_filters = [
    ECG_UNET_CONFIG.init_num_filters * (2**idx) \
        for idx in range(ECG_UNET_CONFIG.down_up_block_num-1,-1,-1)
]
ECG_UNET_CONFIG.up_deconv_filter_lengths = \
    list(repeat(9, ECG_UNET_CONFIG.down_up_block_num))
ECG_UNET_CONFIG.up_conv_filter_lengths = \
    list(repeat(ECG_UNET_CONFIG.init_filter_length, ECG_UNET_CONFIG.down_up_block_num))


unet_down_block = ED()
unet_down_block.batch_norm = ECG_UNET_CONFIG.batch_norm
unet_down_block.kernel_initializer = ECG_UNET_CONFIG.kernel_initializer 
unet_down_block.kw_initializer = deepcopy(ECG_UNET_CONFIG.kw_initializer)
unet_down_block.activation = ECG_UNET_CONFIG.activation
unet_down_block.kw_activation = deepcopy(ECG_UNET_CONFIG.kw_activation)


unet_up_block = ED()
unet_up_block.batch_norm = ECG_UNET_CONFIG.batch_norm
unet_up_block.kernel_initializer = ECG_UNET_CONFIG.kernel_initializer 
unet_up_block.kw_initializer = deepcopy(ECG_UNET_CONFIG.kw_initializer)
unet_up_block.activation = ECG_UNET_CONFIG.activation
unet_up_block.kw_activation = deepcopy(ECG_UNET_CONFIG.kw_activation)


ECG_UNET_CONFIG.down_block = deepcopy(unet_down_block)
ECG_UNET_CONFIG.up_block = deepcopy(unet_up_block)
