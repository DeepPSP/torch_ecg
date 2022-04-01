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

from torch_ecg.databases.datasets.cinc2021 import CINC2021Dataset, CINC2021TrainCfg
from torch_ecg.databases.physionet_databases.cinc2021 import (
    compute_metrics as compute_cinc2021_metrics,
)
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.trainer import BaseTrainer


def test_crnn_cinc2021_pipeline() -> NoReturn:
    """ """
    pass
