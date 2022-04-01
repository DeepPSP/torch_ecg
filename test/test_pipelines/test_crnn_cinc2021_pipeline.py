"""
"""

import shutil
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, Optional, Any, Sequence, Union, Tuple, Dict, List

import pytest
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP

try:
    import torch_ecg
except:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))
    import torch_ecg

from torch_ecg.databases.datasets.cinc2021 import CINC2021Dataset, CINC2021TrainCfg
from torch_ecg.databases.physionet_databases.cinc2021 import (
    compute_metrics as compute_cinc2021_metrics,
)
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.components.trainer import BaseTrainer


def test_crnn_cinc2021_pipeline() -> NoReturn:
    """ """
    pass
