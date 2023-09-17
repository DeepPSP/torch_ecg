"""
torch_ecg.databases.datasets
============================

This module contains PyTorch ECG datasets for training neural networks.

.. contents:: torch_ecg.databases.datasets
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.databases.datasets

.. autosummary::
    :toctree: generated/
    :recursive:

    CINC2020Dataset
    CINC2021Dataset
    CPSC2019Dataset
    CPSC2021Dataset
    LUDBDataset
    MITDBDataset

"""

# fmt: off
from .cinc2020.cinc2020_cfg import CINC2020TrainCfg
from .cinc2020.cinc2020_dataset import CINC2020Dataset
from .cinc2021.cinc2021_cfg import CINC2021TrainCfg
from .cinc2021.cinc2021_dataset import CINC2021Dataset
from .cpsc2019.cpsc2019_cfg import CPSC2019TrainCfg
from .cpsc2019.cpsc2019_dataset import CPSC2019Dataset

# from .cpsc2020.cpsc2020_cfg import CPSC2020TrainCfg
# from .cpsc2020.cpsc2020_dataset import CPSC2020Dataset
from .cpsc2021.cpsc2021_cfg import CPSC2021TrainCfg
from .cpsc2021.cpsc2021_dataset import CPSC2021Dataset
from .ludb.ludb_cfg import LUDBTrainCfg
from .ludb.ludb_dataset import LUDBDataset
from .mitdb.mitdb_cfg import MITDBTrainCfg
from .mitdb.mitdb_dataset import MITDBDataset

__all__ = [
    "CINC2020TrainCfg", "CINC2020Dataset",
    "CINC2021TrainCfg", "CINC2021Dataset",
    "CPSC2019TrainCfg", "CPSC2019Dataset",
    # "CPSC2020TrainCfg", "CPSC2020Dataset",
    "CPSC2021TrainCfg", "CPSC2021Dataset",
    "LUDBTrainCfg", "LUDBDataset",
    "MITDBTrainCfg", "MITDBDataset",
    "list_datasets",
]
# fmt: on


def list_datasets() -> list:
    return [item for item in __all__ if item.endswith("Dataset")]
