"""
configs for the basic cnn layers and blocks
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "vgg_block_basic", "vgg_block_mish", "vgg_block_swish",
    "vgg16", "vgg16_leadwise",
    "resnet_block_basic", "resnet_bottle_neck",
    "resnet", "resnet_leadwise",
    "resnet_block_stanford",
    "resnet_stanford",
    "cpsc_block_basic", "cpsc_block_mish", "cpsc_block_swish",
    "cpsc_2018", "cpsc_2018_leadwise",
    "multi_scopic_block",
    "multi_scopic", "multi_scopic_leadwise",
]


# VGG
vgg16 = ED()
vgg16.num_convs = [2, 2, 3, 3, 3]
_base_num_filters = 36
vgg16.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]
vgg16.groups = 1
vgg16.kernel_initializer = "he_normal"
vgg16.kw_initializer = {}
vgg16.activation = "relu"
vgg16.kw_activation = {}

vgg16_leadwise = deepcopy(vgg16)
vgg16_leadwise.groups = 12
_base_num_filters = 96
vgg16_leadwise.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]


vgg_block_basic = ED()
vgg_block_basic.filter_length = 3
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



# ResNet
resnet = ED()
resnet.subsample_lengths = 2
# resnet.num_blocks = [
#     2, 2, 2, 2, 2,
# ]
# resnet.filter_lengths = 3
# resnet.num_blocks = [
#     3, 4, 6, 3,
# ]
# resnet.filter_lengths = [
#     [5, 5, 13],
#     [5, 5, 5, 13],
#     [5, 5, 5, 5, 5, 13],
#     [5, 5, 25],
# ]
resnet.num_blocks = [
    3, 4, 6, 3,
]
resnet.filter_lengths = [
    [5, 5, 25],
    [5, 5, 5, 25],
    [5, 5, 5, 5, 5, 25],
    [5, 5, 49],
]
resnet.groups = 1
_base_num_filters = 36
resnet.init_num_filters = _base_num_filters
resnet.init_filter_length = 15  # corr. to 30 ms
resnet.init_conv_stride = 2
resnet.init_pool_size = 3
resnet.init_pool_stride = 2
resnet.kernel_initializer = "he_normal"
resnet.kw_initializer = {}
resnet.activation = "relu"  # "mish", "swish"
resnet.kw_activation = {"inplace": True}
resnet.bias = False

resnet_leadwise = deepcopy(resnet)
resnet_leadwise.groups = 12
resnet_leadwise.init_num_filters = 96


resnet_block_basic = ED()
resnet_block_basic.increase_channels_method = 'conv'  # or 'zero_padding'
resnet_block_basic.subsample_mode = 'conv'  # or 'max', 'avg', 'nearest', 'linear', 'bilinear'
resnet_block_basic.kernel_initializer = resnet.kernel_initializer
resnet_block_basic.kw_initializer = deepcopy(resnet.kw_initializer)
resnet_block_basic.activation = resnet.activation
resnet_block_basic.kw_activation = deepcopy(resnet.kw_activation)
resnet_block_basic.bias = False

resnet_bottle_neck = ED()
resnet_bottle_neck.increase_channels_method = 'conv'  # or 'zero_padding'
resnet_bottle_neck.subsample_mode = 'conv'  # or 'max', 'avg', 'nearest', 'linear', 'bilinear'
resnet_bottle_neck.subsample_at = 1  # or 0
resnet_bottle_neck.kernel_initializer = resnet.kernel_initializer
resnet_bottle_neck.kw_initializer = deepcopy(resnet.kw_initializer)
resnet_bottle_neck.activation = resnet.activation
resnet_bottle_neck.kw_activation = deepcopy(resnet.kw_activation)
resnet_bottle_neck.bias = False



# ResNet Stanford
resnet_stanford = ED()
resnet_stanford.subsample_lengths = [
    1, 2, 1, 2,
    1, 2, 1, 2,
    1, 2, 1, 2,
    1, 2, 1, 2,
]
resnet_stanford.filter_lengths = 17
_base_num_filters = 36
resnet_stanford.num_filters_start = _base_num_filters*2
resnet_stanford.kernel_initializer = "he_normal"
resnet_stanford.kw_initializer = {}
resnet_stanford.activation = "relu"
resnet_stanford.kw_activation = {"inplace": True}


resnet_block_stanford = ED()
resnet_block_stanford.increase_channels_at = 4
resnet_block_stanford.increase_channels_method = 'conv'  # or 'zero_padding'
resnet_block_stanford.num_skip = 2
resnet_block_stanford.subsample_mode = 'conv'  # 'max', 'avg'
resnet_block_stanford.kernel_initializer = resnet_stanford.kernel_initializer
resnet_block_stanford.kw_initializer = deepcopy(resnet_stanford.kw_initializer)
resnet_block_stanford.activation = resnet_stanford.activation
resnet_block_stanford.kw_activation = deepcopy(resnet_stanford.kw_activation)
resnet_block_stanford.dropout = 0.2



# CPSC
cpsc_2018 = ED()
# cpsc_2018.num_filters = [  # original
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
#     [12, 12, 12],
# ]
_base_num_filters = 36
cpsc_2018.num_filters = [
    list(repeat(_base_num_filters*2, 3)),
    list(repeat(_base_num_filters*4, 3)),
    list(repeat(_base_num_filters*8, 3)),
    list(repeat(_base_num_filters*16, 3)),
    list(repeat(_base_num_filters*32, 3)),
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
cpsc_2018.kw_activation = ED(negative_slope=0.3, inplace=True)
cpsc_2018.kernel_initializer = "he_normal"
cpsc_2018.kw_initializer = {}

cpsc_2018_leadwise = deepcopy(cpsc_2018)
cpsc_2018_leadwise.groups = 12


cpsc_block_basic = ED()
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


# TODO: add more

# configs of multi_scopic cnn net are set by path, not by level
multi_scopic = ED()
multi_scopic.groups = 1
multi_scopic.scopes = [
    [
        [1,],
        [1,1,],
        [1,1,1,],
    ],
    [
        [2,],
        [2,4,],
        [8,8,8,],
    ],
    [
        [4,],
        [4,8,],
        [16,32,64,],
    ],
]
multi_scopic.filter_lengths = [
    [11, 7, 5,],
    [11, 7, 5,],
    [11, 7, 5,],
]
# subsample_lengths for each branch
multi_scopic.subsample_lengths = list(repeat(2, len(multi_scopic.scopes)))
_base_num_filters = 36
multi_scopic.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
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

multi_scopic_leadwise = deepcopy(multi_scopic)
multi_scopic_leadwise.groups = 12
_base_num_filters = 96
multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]


multi_scopic_block = ED()
multi_scopic_block.subsample_mode = 'max'  # or 'conv', 'avg', 'nearest', 'linear', 'bilinear'
multi_scopic_block.bias = multi_scopic.bias
multi_scopic_block.kernel_initializer = multi_scopic.kernel_initializer
multi_scopic_block.kw_initializer = deepcopy(multi_scopic.kw_initializer)
multi_scopic_block.activation = multi_scopic.activation
multi_scopic_block.kw_activation = deepcopy(multi_scopic.kw_activation)
