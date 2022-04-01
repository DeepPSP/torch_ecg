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

from torch_ecg.databases.datasets.cpsc2019 import CPSC2019Dataset, CPSC2019TrainCfg
from torch_ecg.databases.physionet_databases.cpsc2019 import (
    compute_metrics as compute_cpsc2019_metrics,
)
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.trainer import BaseTrainer


def test_seq_lab_cpsc2019_pipeline() -> NoReturn:
    """ """
    pass
