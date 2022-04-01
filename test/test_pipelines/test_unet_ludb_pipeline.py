"""
"""

from typing import NoReturn

import torch

try:
    import torch_ecg
except:
    import sys
    from pathlib import Path

    sys.path.append(Path(__file__).absolute().parent.parent)
    import torch_ecg

from torch_ecg.databases.datasets.ludb import LUDBDataset, LUDBTrainCfg
from torch_ecg.databases.physionet_databases.ludb import (
    compute_metrics as compute_ludb_metrics,
)
from torch_ecg.cfg import CFG
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.trainer import BaseTrainer


def test_unet_ludb_pipeline() -> NoReturn:
    """ """
    pass
