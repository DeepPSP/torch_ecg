"""
configs of the model of UNET structures
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "ecg_unet_vanilla_config",
]


# vanilla config, for delineation using single-lead ECG
ecg_unet_vanilla_config = ED()

ecg_unet_vanilla_config.groups = 1

ecg_unet_vanilla_config.init_num_filters = 4  # keep the same with n_classes
ecg_unet_vanilla_config.init_filter_length = 9
ecg_unet_vanilla_config.out_filter_length = 9
ecg_unet_vanilla_config.batch_norm = True
ecg_unet_vanilla_config.kernel_initializer = "he_normal"
ecg_unet_vanilla_config.kw_initializer = {}
ecg_unet_vanilla_config.activation = "relu"
ecg_unet_vanilla_config.kw_activation = {"inplace": True}

ecg_unet_vanilla_config.down_up_block_num = 4

ecg_unet_vanilla_config.down_mode = "max"
ecg_unet_vanilla_config.down_scales = list(repeat(2, ecg_unet_vanilla_config.down_up_block_num))
ecg_unet_vanilla_config.down_num_filters = [
    ecg_unet_vanilla_config.init_num_filters * (2**idx) \
        for idx in range(1, ecg_unet_vanilla_config.down_up_block_num+1)
]
ecg_unet_vanilla_config.down_filter_lengths = \
    list(repeat(ecg_unet_vanilla_config.init_filter_length, ecg_unet_vanilla_config.down_up_block_num))

ecg_unet_vanilla_config.up_mode = "nearest"
ecg_unet_vanilla_config.up_scales = list(repeat(2, ecg_unet_vanilla_config.down_up_block_num))
ecg_unet_vanilla_config.up_num_filters = [
    ecg_unet_vanilla_config.init_num_filters * (2**idx) \
        for idx in range(ecg_unet_vanilla_config.down_up_block_num-1,-1,-1)
]
ecg_unet_vanilla_config.up_deconv_filter_lengths = \
    list(repeat(9, ecg_unet_vanilla_config.down_up_block_num))
ecg_unet_vanilla_config.up_conv_filter_lengths = \
    list(repeat(ecg_unet_vanilla_config.init_filter_length, ecg_unet_vanilla_config.down_up_block_num))


unet_down_block = ED()
unet_down_block.batch_norm = ecg_unet_vanilla_config.batch_norm
unet_down_block.kernel_initializer = ecg_unet_vanilla_config.kernel_initializer 
unet_down_block.kw_initializer = deepcopy(ecg_unet_vanilla_config.kw_initializer)
unet_down_block.activation = ecg_unet_vanilla_config.activation
unet_down_block.kw_activation = deepcopy(ecg_unet_vanilla_config.kw_activation)


unet_up_block = ED()
unet_up_block.batch_norm = ecg_unet_vanilla_config.batch_norm
unet_up_block.kernel_initializer = ecg_unet_vanilla_config.kernel_initializer 
unet_up_block.kw_initializer = deepcopy(ecg_unet_vanilla_config.kw_initializer)
unet_up_block.activation = ecg_unet_vanilla_config.activation
unet_up_block.kw_activation = deepcopy(ecg_unet_vanilla_config.kw_activation)


ecg_unet_vanilla_config.down_block = deepcopy(unet_down_block)
ecg_unet_vanilla_config.up_block = deepcopy(unet_up_block)
