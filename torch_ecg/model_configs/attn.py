"""
configs for the attention modules
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "non_local",
    "squeeze_excitation",
    "global_context",
]


non_local = ED()
non_local.filter_lengths=1,
non_local.subsample_length={
    "g":2, "phi":2, 'theta':2, "W":2,
}
non_local.batch_norm=True


squeeze_excitation = ED()
squeeze_excitation.reduction = 8  # not including the last linear layer
squeeze_excitation.activation = "relu"
squeeze_excitation.kw_activation = ED(inplace=True)
squeeze_excitation.bias = True
squeeze_excitation.kernel_initializer = "he_normal"


global_context = ED()
global_context.ratio = 8
global_context.reduction = False
global_context.pooling_type = "attn"
global_context.fusion_types = ["mul",]
