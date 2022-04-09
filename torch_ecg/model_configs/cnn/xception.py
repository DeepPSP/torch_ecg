"""
"""

from itertools import repeat

from ...cfg import CFG

__all__ = [
    # vanilla xception
    "xception_vanilla",
    # custom xception
    "xception_leadwise",
]


xception_vanilla = CFG()
xception_vanilla.fs = 500
xception_vanilla.groups = 1
_base_num_filters = 8
xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=31,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=15,
    subsample_lengths=2,
    subsample_kernels=3,
)
xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=13,
)
xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=17,
    subsample_lengths=2,
    subsample_kernels=3,
)

xception_leadwise = CFG()
xception_leadwise.fs = 500
xception_leadwise.groups = 12
_base_num_filters = 12 * 2
xception_leadwise.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=31,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=15,
    subsample_lengths=2,
    subsample_kernels=3,
)
xception_leadwise.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=13,
)
xception_leadwise.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=17,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
