"""
"""

from copy import deepcopy
from itertools import repeat

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from torch_ecg.cfg import CFG
from torch_ecg.model_configs import (  # noqa: F401
    ECG_CRNN_CONFIG,
    attention,
    densenet_leadwise,
    global_context,
    linear,
    lstm,
    multi_scopic,
    multi_scopic_block,
    multi_scopic_leadwise,
    non_local,
    resnet_block_basic,
    resnet_block_basic_gc,
    resnet_block_basic_se,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_se,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_se,
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    squeeze_excitation,
    tresnetF,
    tresnetM,
    tresnetN,
    tresnetP,
    tresnetS,
    vgg16,
    vgg16_leadwise,
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    xception_leadwise,
)

__all__ = [
    "ModelArchCfg",
]


_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)
_BASE_MODEL_CONFIG.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"  # False

# detailed configs for 12-lead, 6-lead, 4-lead, 3-lead, 2-lead models
# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = CFG()

ModelArchCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)

# TODO: add adjustifications for "leadwise" configs for 6,4,3,2 leads models
ModelArchCfg.six_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.six_leads.cnn.vgg16_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.six_leads.cnn.resnet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.resnet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.six_leads.cnn.densenet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.densenet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.xception_leadwise.groups = 6
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.six_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.four_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.four_leads.cnn.vgg16_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.four_leads.cnn.resnet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.resnet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.four_leads.cnn.densenet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.densenet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.xception_leadwise.groups = 4
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.four_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.three_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.three_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.three_leads.cnn.resnet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.resnet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.three_leads.cnn.densenet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.densenet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 3 * 4  # 12 * 2
ModelArchCfg.three_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.two_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.two_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 2 * 12  # 12 * 4
ModelArchCfg.two_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.two_leads.cnn.resnet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.resnet_leadwise.init_num_filters = 2 * 16  # 12 * 8
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.groups = 2
_base_num_filters = 2 * 8  # 12 * 4
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.two_leads.cnn.densenet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.densenet_leadwise.init_num_filters = 2 * 12  # 12 * 8
ModelArchCfg.two_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 2 * 6  # 12 * 2
ModelArchCfg.two_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)


nas_list = []
cnn_list = [
    vgg16,
    vgg16_leadwise,
    resnet_nature_comm,
    resnet_nature_comm_gc,
    resnet_nature_comm_bottle_neck,
    resnetN,
    resnetNB,
    resnetNS,
    resnetNBS,
    tresnetM,
    resnet_nature_comm_se,
    resnet_nature_comm_bottle_neck_se,
    tresnetF,
    tresnetP,
    tresnetN,
    tresnetS,
    multi_scopic,
    multi_scopic_leadwise,
]
for item in cnn_list:
    pass
