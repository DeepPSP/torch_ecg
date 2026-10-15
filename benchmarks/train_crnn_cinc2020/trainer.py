"""The CINC2020 trainer: the CINC2021 trainer with 2020 dataset/metric."""

from torch_ecg.databases.datasets.cinc2020 import CINC2020Dataset
from torch_ecg.databases.physionet_databases.cinc2020 import compute_metrics as compute_cinc2020_metrics

from ..train_crnn_cinc2021.trainer import CINC2021Trainer

__all__ = [
    "CINC2020Trainer",
]


class CINC2020Trainer(CINC2021Trainer):
    """Identical to `CINC2021Trainer` except for the dataset and the
    official metric (the 2020 challenge metric has no load-balancing
    multiplier)."""

    __name__ = "CINC2020Trainer"

    _dataset_cls = CINC2020Dataset
    _compute_metrics = staticmethod(compute_cinc2020_metrics)
