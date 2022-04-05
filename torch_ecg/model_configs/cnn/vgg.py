"""
"""

from copy import deepcopy

from ...cfg import CFG

__all__ = [
    "vgg_block_basic",
    "vgg_block_mish",
    "vgg_block_swish",
    "vgg16",
    "vgg16_leadwise",
]


vgg16 = CFG()
vgg16.fs = 500
vgg16.num_convs = [2, 2, 3, 3, 3]
_base_num_filters = 12
vgg16.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
vgg16.groups = 1
vgg16.kernel_initializer = "he_normal"
vgg16.kw_initializer = {}
vgg16.activation = "relu"
vgg16.kw_activation = {}

vgg16_leadwise = deepcopy(vgg16)
vgg16_leadwise.groups = 12
_base_num_filters = 12 * 4
vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]


vgg_block_basic = CFG()
vgg_block_basic.filter_length = 15
vgg_block_basic.subsample_length = 1
vgg_block_basic.dilation = 1
vgg_block_basic.batch_norm = True
vgg_block_basic.pool_size = 3
vgg_block_basic.pool_stride = 2  # 2
vgg_block_basic.kernel_initializer = vgg16.kernel_initializer
vgg_block_basic.kw_initializer = deepcopy(vgg16.kw_initializer)
vgg_block_basic.activation = vgg16.activation
vgg_block_basic.kw_activation = deepcopy(vgg16.kw_activation)

vgg_block_mish = deepcopy(vgg_block_basic)
vgg_block_mish.activation = "mish"
del vgg_block_mish.kw_activation

vgg_block_swish = deepcopy(vgg_block_basic)
vgg_block_swish.activation = "swish"
del vgg_block_swish.kw_activation


# set default building block
vgg16.block = deepcopy(vgg_block_basic)
vgg16_leadwise.block = deepcopy(vgg_block_basic)
