"""
configs for the attention modules
"""

from ..cfg import CFG

__all__ = [
    "non_local",
    "squeeze_excitation",
    "global_context",
    "transformer",
]


non_local = CFG()
non_local.filter_lengths = {
    "g": 1,
    "phi": 1,
    "theta": 1,
    "W": 1,
}
non_local.subsample_length = 2
non_local.batch_norm = True


squeeze_excitation = CFG()
squeeze_excitation.reduction = 8  # not including the last linear layer
squeeze_excitation.activation = "relu"
squeeze_excitation.kw_activation = CFG(inplace=True)
squeeze_excitation.bias = True
squeeze_excitation.kernel_initializer = "he_normal"


global_context = CFG()
global_context.ratio = 4
global_context.reduction = True
global_context.pooling_type = "attn"
global_context.fusion_types = [
    "mul",
]


transformer = CFG()
transformer.hidden_size = 1024
transformer.num_heads = 8
transformer.num_layers = 1
transformer.dropout = 0.1
transformer.activation = "relu"
