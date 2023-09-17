"""
Designing Network Design Spaces
"""

from copy import deepcopy

from ...cfg import CFG
from .resnet import (
    resnet_bottle_neck,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_nl,
    resnet_bottle_neck_se,
    resnet_stem,
    resnet_stem_C,
)

__all__ = [
    "regnet_16_8",
    "regnet_27_24",
    "regnet_23_168",
    "regnet_S",
    "regnet_bottle_neck",
    "regnet_bottle_neck_B",
    "regnet_bottle_neck_D",
    "regnet_bottle_neck_se",
    "regnet_bottle_neck_nl",
    "regnet_bottle_neck_gc",
]


regnet_stem = deepcopy(resnet_stem)
regnet_stem_C = deepcopy(resnet_stem_C)
regnet_bottle_neck = deepcopy(resnet_bottle_neck)
regnet_bottle_neck_B = deepcopy(resnet_bottle_neck_B)
regnet_bottle_neck_D = deepcopy(resnet_bottle_neck_D)
regnet_bottle_neck_se = deepcopy(resnet_bottle_neck_se)
regnet_bottle_neck_nl = deepcopy(resnet_bottle_neck_nl)
regnet_bottle_neck_gc = deepcopy(resnet_bottle_neck_gc)


regnet_16_8 = CFG()
regnet_16_8.fs = 500
regnet_16_8.stem = deepcopy(resnet_stem)
regnet_16_8.filter_lengths = 11
regnet_16_8.subsample_lengths = 2

regnet_16_8.tot_blocks = 16
regnet_16_8.group_widths = 8
regnet_16_8.w_a = 27.89
regnet_16_8.w_0 = 48
regnet_16_8.w_m = 2.09

regnet_16_8.block = deepcopy(resnet_bottle_neck_se)
regnet_16_8.block.expansion = 1


regnet_27_24 = deepcopy(regnet_16_8)
regnet_27_24.tot_blocks = 27
regnet_27_24.group_widths = 24
regnet_27_24.w_a = 20.71
regnet_27_24.w_0 = 48
regnet_27_24.w_m = 2.65


regnet_23_168 = deepcopy(regnet_16_8)
regnet_23_168.tot_blocks = 23
regnet_23_168.group_widths = 168
regnet_23_168.w_a = 69.86
regnet_23_168.w_0 = 320
regnet_23_168.w_m = 2.0


regnet_S = CFG()
regnet_S.fs = 500
regnet_S.stem = deepcopy(resnet_stem)

regnet_S.num_blocks = [2, 2, 2, 2]
regnet_S.filter_lengths = [13, 11, 7, 5]
regnet_S.subsample_lengths = [2, 2, 2, 2]
regnet_S.num_filters = [32, 64, 128, 256]
regnet_S.group_widths = [8, 16, 32, 64]
regnet_S.block = deepcopy(resnet_bottle_neck_se)
regnet_S.block.expansion = 4
