"""Configs for the CINC2021 benchmark example.

The model configs for the non-12-lead Challenge lead sets are generated
systematically from the lead count.  The original hand-tuned entries
(for 4/3/2 leads) contained inconsistent arithmetic (e.g. ``6*4`` vs
``n_leads*6`` style bases mixed between lead sets); this rewrite uses
one systematic rule throughout, documented per block below.
"""

from copy import deepcopy
from typing import Dict

from torch_ecg.cfg import CFG
from torch_ecg.databases.datasets.cinc2021 import CINC2021TrainCfg
from torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG

__all__ = [
    "TrainCfg",
    "ModelCfg",
    "LEAD_SETS",
]


TrainCfg = deepcopy(CINC2021TrainCfg)

_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)
_BASE_MODEL_CONFIG.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"


def _make_model_config(n_leads: int) -> CFG:
    """Systematic per-lead-set scaling of the leadwise backbone variants.

    Scaling rules (following the original 6-lead entry, which was the
    internally consistent one):

    - ``groups = n_leads`` for all leadwise backbones;
    - vgg16_leadwise / multi_scopic_leadwise filter base = ``n_leads * 6``;
    - resnet_leadwise / densenet_leadwise ``init_num_filters = n_leads * 8``;
    - xception_vanilla (leadwise) filter base = ``n_leads * 2``.
    """
    cfg = deepcopy(_BASE_MODEL_CONFIG)

    vgg_base = n_leads * 6
    cfg.cnn.vgg16_leadwise.groups = n_leads
    cfg.cnn.vgg16_leadwise.num_filters = [
        vgg_base * 4,
        vgg_base * 8,
        vgg_base * 16,
        vgg_base * 32,
        vgg_base * 32,
    ]

    cfg.cnn.resnet_leadwise.groups = n_leads
    cfg.cnn.resnet_leadwise.init_num_filters = n_leads * 8

    ms_base = n_leads * 6
    cfg.cnn.multi_scopic_leadwise.groups = n_leads
    cfg.cnn.multi_scopic_leadwise.num_filters = [
        [ms_base * 4, ms_base * 8, ms_base * 16],
    ] * 3

    cfg.cnn.densenet_leadwise.groups = n_leads
    cfg.cnn.densenet_leadwise.init_num_filters = n_leads * 8

    xcp_base = n_leads * 2
    cfg.cnn.xception_leadwise.groups = n_leads
    cfg.cnn.xception_vanilla.entry_flow = CFG(
        init_num_filters=[xcp_base * 4, xcp_base * 8],
        init_filter_lengths=3,
        init_subsample_lengths=[2, 1],
        num_filters=[xcp_base * 16, xcp_base * 32, xcp_base * 91],
        filter_lengths=3,
        subsample_lengths=2,
        subsample_kernels=3,
    )
    cfg.cnn.xception_vanilla.middle_flow = CFG(
        num_filters=[xcp_base * 91] * 8,
        filter_lengths=3,
    )
    cfg.cnn.xception_vanilla.exit_flow = CFG(
        final_num_filters=[xcp_base * 182, xcp_base * 256],
        final_filter_lengths=3,
        num_filters=[[xcp_base * 91, xcp_base * 128]],
        filter_lengths=3,
        subsample_lengths=2,
        subsample_kernels=3,
    )
    return cfg


# the Challenge lead sets
LEAD_SETS: Dict[str, tuple] = {
    "twelve_leads": ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"),
    "six_leads": ("I", "II", "III", "aVR", "aVL", "aVF"),
    "four_leads": ("I", "II", "III", "V2"),
    "three_leads": ("I", "II", "V2"),
    "two_leads": ("I", "II"),
}

ModelCfg = CFG()
# the 12-lead config is the (adjusted) stock ECG_CRNN_CONFIG; the others
# follow the systematic per-lead scaling of `_make_model_config`
ModelCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)
for _name, _leads in LEAD_SETS.items():
    if _name != "twelve_leads":
        ModelCfg[_name] = _make_model_config(len(_leads))

ModelCfg.bin_pred_nsr_thr = 0.5
ModelCfg.bin_pred_look_again_tol = 0.0

del _name, _leads
