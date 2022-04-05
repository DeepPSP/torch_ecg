"""
vanilla resnet, and various variants

Reference
---------
1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
2. He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019). Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 558-567).
3. Hannun, A. Y., Rajpurkar, P., Haghpanahi, M., Tison, G. H., Bourn, C., Turakhia, M. P., & Ng, A. Y. (2019). Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network. Nature medicine, 25(1), 65-69.
4. Ribeiro, A. H., Ribeiro, M. H., Paixão, G. M., Oliveira, D. M., Gomes, P. R., Canazart, J. A., ... & Ribeiro, A. L. P. (2020). Automatic diagnosis of the 12-lead ECG using a deep neural network. Nature communications, 11(1), 1-9.
5. Ridnik, T., Lawen, H., Noy, A., Ben Baruch, E., Sharir, G., & Friedman, I. (2021). Tresnet: High performance gpu-dedicated architecture. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1400-1409).
"""

from copy import deepcopy

from ...cfg import CFG
from ..attn import global_context, non_local, squeeze_excitation

__all__ = [
    # building blocks
    "resnet_block_basic",
    "resnet_bottle_neck",
    "resnet_bottle_neck_B",
    "resnet_bottle_neck_D",
    "resnet_block_basic_se",
    "resnet_bottle_neck_se",
    "resnet_block_basic_nl",
    "resnet_bottle_neck_nl",
    "resnet_block_basic_gc",
    "resnet_bottle_neck_gc",
    # smaller resnets
    "resnetN",
    "resnetNB",
    "resnetNS",
    "resnetNBS",
    # vanilla resnet
    "resnet_vanilla_18",
    "resnet_vanilla_34",
    "resnet_vanilla_50",
    "resnet_vanilla_101",
    "resnet_vanilla_152",
    "resnext_vanilla_50_32x4d",
    "resnext_vanilla_101_32x8d",
    "resnet_vanilla_wide_50_2",
    "resnet_vanilla_wide_101_2",
    # cpsc2018 resnet
    "resnet_cpsc2018",
    "resnet_cpsc2018_leadwise",
    # stanford resnet
    "resnet_block_stanford",
    "resnet_stanford",
    # ResNet Nature Communications
    "resnet_nature_comm",
    "resnet_nature_comm_se",
    "resnet_nature_comm_nl",
    "resnet_nature_comm_gc",
    "resnet_nature_comm_bottle_neck",
    "resnet_nature_comm_bottle_neck_se",
    "resnet_nature_comm_bottle_neck_gc",
    "resnet_nature_comm_bottle_neck_nl",
    # TresNet
    "tresnetF",
    "tresnetP",
    "tresnetN",
    "tresnetS",
    "tresnetM",
    "tresnetL",
    "tresnetXL",
    "tresnetM_V2",
]


# building blocks

resnet_block_basic = CFG()
resnet_block_basic.increase_channels_method = "conv"  # or "zero_padding"
resnet_block_basic.subsample_mode = (
    "conv"  # or "max", "avg", "nearest", "linear", "bilinear"
)
resnet_block_basic.kernel_initializer = "he_normal"
resnet_block_basic.kw_initializer = {}
resnet_block_basic.activation = "relu"  # "mish", "swish"
resnet_block_basic.kw_activation = {"inplace": True}
resnet_block_basic.bias = False

resnet_bottle_neck = CFG()
resnet_bottle_neck.expansion = 4
resnet_bottle_neck.increase_channels_method = "conv"  # or "zero_padding"
resnet_bottle_neck.subsample_mode = (
    "conv"  # or "max", "avg", "nearest", "linear", "bilinear"
)
resnet_bottle_neck.subsample_at = (
    0  # or 0  # 0 is for the original paper, 1 for ResNet-B
)
resnet_bottle_neck.kernel_initializer = "he_normal"
resnet_bottle_neck.kw_initializer = {}
resnet_bottle_neck.activation = "relu"  # "mish", "swish"
resnet_bottle_neck.kw_activation = {"inplace": True}
resnet_bottle_neck.bias = False

resnet_bottle_neck_B = deepcopy(resnet_bottle_neck)
resnet_bottle_neck_B.subsample_at = 1

resnet_bottle_neck_D = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_D.subsample_mode = "avg"

# one can add more variants by changing the attention modules
resnet_block_basic_se = deepcopy(resnet_block_basic)
resnet_block_basic_se.attn = deepcopy(squeeze_excitation)
resnet_block_basic_se.attn.name = "se"
resnet_block_basic_se.attn.pos = -1

resnet_bottle_neck_se = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_se.attn = deepcopy(squeeze_excitation)
resnet_bottle_neck_se.attn.name = "se"
resnet_bottle_neck_se.attn.pos = -1

resnet_block_basic_nl = deepcopy(resnet_block_basic)
resnet_block_basic_nl.attn = deepcopy(non_local)
resnet_block_basic_nl.attn.name = "nl"
resnet_block_basic_nl.attn.pos = -1

resnet_bottle_neck_nl = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_nl.attn = deepcopy(non_local)
resnet_bottle_neck_nl.attn.name = "nl"
resnet_bottle_neck_nl.attn.pos = -1

resnet_block_basic_gc = deepcopy(resnet_block_basic)
resnet_block_basic_gc.attn = deepcopy(global_context)
resnet_block_basic_gc.attn.name = "gc"
resnet_block_basic_gc.attn.pos = -1

resnet_bottle_neck_gc = deepcopy(resnet_bottle_neck_B)
resnet_bottle_neck_gc.attn = deepcopy(global_context)
resnet_bottle_neck_gc.attn.name = "gc"
resnet_bottle_neck_gc.attn.pos = -1


resnet_stem = CFG()
resnet_stem.num_filters = 64
resnet_stem.filter_lengths = 25
resnet_stem.conv_stride = 2
resnet_stem.pool_size = 3
resnet_stem.pool_stride = 2

resnet_stem_C = deepcopy(resnet_stem)
resnet_stem_C.num_filters = [
    32,
    32,
    64,
]
resnet_stem_C.filter_lengths = [
    15,
    15,
    15,
]


# vanilla ResNets
resnet_vanilla_common = CFG()
resnet_vanilla_common.fs = 500
resnet_vanilla_common.subsample_lengths = [
    1,
    2,
    2,
    2,
]
resnet_vanilla_common.filter_lengths = 15
resnet_vanilla_common.groups = 1
resnet_vanilla_common.increase_channels_method = "conv"
resnet_vanilla_common.kernel_initializer = "he_normal"
resnet_vanilla_common.kw_initializer = {}
resnet_vanilla_common.activation = "relu"  # "mish", "swish"
resnet_vanilla_common.kw_activation = {"inplace": True}
resnet_vanilla_common.bias = False
resnet_vanilla_common.stem = deepcopy(resnet_stem)


resnet_vanilla_18 = CFG()
resnet_vanilla_18.num_blocks = [
    2,
    2,
    2,
    2,
]
resnet_vanilla_18.update(deepcopy(resnet_vanilla_common))
resnet_vanilla_18.building_block = "basic"
resnet_vanilla_18.block = deepcopy(resnet_block_basic)
resnet_vanilla_18.block.kernel_initializer = resnet_vanilla_18.kernel_initializer
resnet_vanilla_18.block.kw_initializer = resnet_vanilla_18.kw_initializer
resnet_vanilla_18.block.activation = resnet_vanilla_18.activation
resnet_vanilla_18.block.kw_activation = resnet_vanilla_18.kw_activation
resnet_vanilla_18.block.bias = resnet_vanilla_18.bias

resnet_vanilla_34 = deepcopy(resnet_vanilla_18)
resnet_vanilla_34.num_blocks = [
    3,
    4,
    6,
    3,
]

resnet_vanilla_50 = CFG()  # uses bottleneck
resnet_vanilla_50.num_blocks = [
    3,
    4,
    6,
    3,
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

resnet_vanilla_101 = deepcopy(resnet_vanilla_50)  # uses bottleneck
resnet_vanilla_101.num_blocks = [
    3,
    4,
    23,
    3,
]
resnet_vanilla_101.filter_lengths = 11

resnet_vanilla_152 = deepcopy(resnet_vanilla_101)  # uses bottleneck
resnet_vanilla_152.num_blocks = [
    3,
    8,
    36,
    3,
]

resnext_vanilla_50_32x4d = deepcopy(resnet_vanilla_152)  # uses bottleneck
resnext_vanilla_50_32x4d.num_blocks = [
    3,
    4,
    6,
    3,
]
resnext_vanilla_50_32x4d.groups = 32
resnext_vanilla_50_32x4d.base_groups = 1
resnext_vanilla_50_32x4d.base_width = 4

resnext_vanilla_101_32x8d = deepcopy(resnext_vanilla_50_32x4d)  # uses bottleneck
resnext_vanilla_101_32x8d.num_blocks = [
    3,
    4,
    23,
    3,
]
resnext_vanilla_101_32x8d.base_width = 8

resnet_vanilla_wide_50_2 = deepcopy(resnet_vanilla_50)  # uses bottleneck
resnet_vanilla_wide_50_2.base_groups = 1
resnet_vanilla_wide_50_2.base_width = 64 * 2

resnet_vanilla_wide_101_2 = deepcopy(resnet_vanilla_101)  # uses bottleneck
resnet_vanilla_wide_101_2.base_groups = 1
resnet_vanilla_wide_101_2.base_width = 64 * 2


# smaller ResNets
resnetN = deepcopy(resnet_vanilla_18)
resnetN.num_blocks = [
    1,
    1,
    1,
    1,
]
resnetN.filter_lengths = [
    19,
    15,
    11,
    7,
]

resnetNB = deepcopy(resnetN)
resnetNB.building_block = "bottleneck"
resnetNB.block = deepcopy(resnet_bottle_neck_B)

resnetNS = deepcopy(resnetN)
resnetNS.block.conv_type = "separable"

resnetNBS = deepcopy(resnetNB)
resnetNBS.block.conv_type = "separable"


# custom ResNets
resnet_cpsc2018 = CFG()
resnet_cpsc2018.fs = 500
resnet_cpsc2018.building_block = "basic"  # "bottleneck"
resnet_cpsc2018.expansion = 1
resnet_cpsc2018.subsample_lengths = [
    1,
    2,
    2,
    2,
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
    3,
    4,
    6,
    3,
]
resnet_cpsc2018.filter_lengths = [
    [5, 5, 25],
    [5, 5, 5, 25],
    [5, 5, 5, 5, 5, 25],
    [5, 5, 49],
]
resnet_cpsc2018.groups = 1
_base_num_filters = 12 * 4
resnet_cpsc2018.stem = deepcopy(resnet_stem)
resnet_cpsc2018.stem.num_filters = _base_num_filters
resnet_cpsc2018.stem.filter_lengths = 15  # corr. to 30 ms
resnet_cpsc2018.kernel_initializer = "he_normal"
resnet_cpsc2018.kw_initializer = {}
resnet_cpsc2018.activation = "relu"  # "mish", "swish"
resnet_cpsc2018.kw_activation = {"inplace": True}
resnet_cpsc2018.bias = False


resnet_cpsc2018_leadwise = deepcopy(resnet_cpsc2018)
resnet_cpsc2018_leadwise.groups = 12
resnet_cpsc2018_leadwise.stem.num_filters = 12 * 8


# set default building block
resnet_cpsc2018.building_block = "basic"
resnet_cpsc2018.block = deepcopy(resnet_block_basic)
resnet_cpsc2018_leadwise.building_block = "basic"
resnet_cpsc2018_leadwise.block = deepcopy(resnet_block_basic)


# ResNet Stanford
resnet_stanford = CFG()
resnet_stanford.fs = 500
resnet_stanford.groups = 1
resnet_stanford.num_blocks = [
    2,
    2,
    2,
    2,
]
resnet_stanford.subsample_lengths = 2
resnet_stanford.filter_lengths = 17
_base_num_filters = 36
resnet_stanford.stem = deepcopy(resnet_stem)
resnet_stanford.stem.num_filters = _base_num_filters * 2
resnet_stanford.stem.filter_lengths = 17
resnet_stanford.kernel_initializer = "he_normal"
resnet_stanford.kw_initializer = {}
resnet_stanford.activation = "relu"
resnet_stanford.kw_activation = {"inplace": True}
resnet_stanford.bias = False


resnet_block_stanford = CFG()
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
# a modified version of resnet [1,1,1,1]
resnet_nature_comm = deepcopy(resnet_vanilla_18)
resnet_nature_comm.num_blocks = [
    1,
    1,
    1,
    1,
]
resnet_nature_comm.stem.filter_lengths = 17  # originally 16, we make it odd
resnet_nature_comm.filter_lengths = 17
resnet_nature_comm.stem.conv_stride = 1
resnet_nature_comm.stem.pool_stride = 1  # if < 2, init pool will not be added
resnet_nature_comm.dropouts = 0.2
resnet_nature_comm.subsample_lengths = [
    4,
    4,
    4,
    4,
]
resnet_nature_comm.num_filters = [
    2 * resnet_nature_comm.stem.num_filters,
    3 * resnet_nature_comm.stem.num_filters,
    4 * resnet_nature_comm.stem.num_filters,
    5 * resnet_nature_comm.stem.num_filters,
]
resnet_nature_comm.block.subsample_mode = "max"


# variant with attention of ResNet Nature Communications
resnet_nature_comm_se = deepcopy(resnet_nature_comm)
resnet_nature_comm_se.block = deepcopy(resnet_block_basic_se)

resnet_nature_comm_nl = deepcopy(resnet_nature_comm)
resnet_nature_comm_nl.block = deepcopy(resnet_block_basic_nl)

resnet_nature_comm_gc = deepcopy(resnet_nature_comm)
resnet_nature_comm_gc.block = deepcopy(resnet_block_basic_gc)

resnet_nature_comm_bottle_neck = deepcopy(resnet_nature_comm)
resnet_nature_comm_bottle_neck.building_block = "bottleneck"
resnet_nature_comm_bottle_neck.block = deepcopy(resnet_bottle_neck_B)

resnet_nature_comm_bottle_neck_se = deepcopy(resnet_nature_comm_bottle_neck)
resnet_nature_comm_bottle_neck_se.block = deepcopy(resnet_bottle_neck_se)

resnet_nature_comm_bottle_neck_gc = deepcopy(resnet_nature_comm_bottle_neck)
resnet_nature_comm_bottle_neck_gc.block = deepcopy(resnet_bottle_neck_gc)

resnet_nature_comm_bottle_neck_nl = deepcopy(resnet_nature_comm_bottle_neck)
resnet_nature_comm_bottle_neck_nl.block = deepcopy(resnet_bottle_neck_nl)


# TResNet
tresnet_common = CFG()
tresnet_common.fs = 500
tresnet_common.subsample_lengths = [
    1,
    2,
    2,
    2,
]
tresnet_common.filter_lengths = 11
tresnet_common.groups = 1
tresnet_common.increase_channels_method = "conv"
tresnet_common.stem = CFG()
tresnet_common.stem.subsample_mode = "s2d"
tresnet_common.stem.num_filters = 64
tresnet_common.stem.filter_lengths = 1
tresnet_common.stem.conv_stride = 1
tresnet_common.stem.pool_size = 1
tresnet_common.stem.pool_stride = 1
tresnet_common.kernel_initializer = "he_normal"
tresnet_common.kw_initializer = {}
tresnet_common.activation = "leaky"  # "mish", "swish"
tresnet_common.kw_activation = {"inplace": True}
tresnet_common.bias = False

tresnet_common.building_block = ["basic", "basic", "bottleneck", "bottleneck"]
tresnet_common.block = [
    deepcopy(resnet_block_basic_se),
    deepcopy(resnet_block_basic_se),
    deepcopy(resnet_bottle_neck_se),
    deepcopy(resnet_bottle_neck_B),
]
tresnet_common.block[0].reduction = 4
tresnet_common.block[1].reduction = 4
tresnet_common.block[2].pos = 2
for b in tresnet_common.block:
    b.kernel_initializer = tresnet_common.kernel_initializer
    b.kw_initializer = tresnet_common.kw_initializer
    b.activation = tresnet_common.activation
    b.kw_activation = tresnet_common.kw_activation
    b.bias = tresnet_common.bias
    b.subsample_mode = "blur"
    b.filt_size = 7  # for blur subsampling
    b.conv_type = "aa"

# TResNet-F, femto
tresnetF = deepcopy(tresnet_common)
tresnetF.filter_lengths = 17
tresnetF.num_blocks = [
    1,
    1,
    1,
    1,
]
tresnetF.stem.num_filters = 32
for b in tresnetF.block:
    b.conv_type = "separable"

# TResNet-P, pico
tresnetP = deepcopy(tresnet_common)
tresnetP.filter_lengths = 17
tresnetP.num_blocks = [
    1,
    1,
    1,
    1,
]
tresnetP.stem.num_filters = 56

# TResNet-N
tresnetN = deepcopy(tresnet_common)
tresnetN.filter_lengths = 15
tresnetN.num_blocks = [
    2,
    2,
    2,
    2,
]
tresnetN.stem.num_filters = 56

# TResNet-S
tresnetS = deepcopy(tresnet_common)
tresnetS.filter_lengths = 13
tresnetS.num_blocks = [
    3,
    4,
    6,
    3,
]
tresnetS.stem.num_filters = 56

# TResNet-M
tresnetM = deepcopy(tresnet_common)
tresnetM.num_blocks = [
    3,
    4,
    11,
    3,
]
tresnetM.stem.num_filters = 64

# TResNet-L
tresnetL = deepcopy(tresnet_common)
tresnetL.filter_lengths = [
    11,
    11,
    9,
    7,
]
tresnetL.num_blocks = [
    4,
    5,
    18,
    3,
]
tresnetL.stem.num_filters = 76

# TResNet-XL
tresnetXL = deepcopy(tresnet_common)
tresnetXL.filter_lengths = [
    11,
    9,
    9,
    7,
]
tresnetXL.num_blocks = [
    4,
    5,
    24,
    3,
]
tresnetXL.stem.num_filters = 84

# TResNet V2
tresnetM_V2 = deepcopy(tresnetM)
tresnetM_V2.building_block = ["bottleneck", "bottleneck", "bottleneck", "bottleneck"]
tresnetM_V2.block = [
    deepcopy(resnet_bottle_neck_se),
    deepcopy(resnet_bottle_neck_se),
    deepcopy(resnet_bottle_neck_se),
    deepcopy(resnet_bottle_neck_B),
]
tresnetM_V2.block[0].pos = 2
tresnetM_V2.block[1].pos = 2
tresnetM_V2.block[2].pos = 2
