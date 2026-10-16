"""Configs for the LUDB wave-delineation benchmark example.

Reference
----------
[1] Moskalenko, Zolotykh, Osipov. "Deep Learning for ECG Segmentation."
    Neuroinformatics, 2019.
"""

from copy import deepcopy

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.databases.datasets.ludb import LUDBTrainCfg
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG

__all__ = [
    "TrainCfg",
    "ModelCfg",
]


TrainCfg = deepcopy(LUDBTrainCfg)

ModelCfg = CFG()
ModelCfg.torch_dtype = DEFAULTS.DTYPE.TORCH
ModelCfg.fs = TrainCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs

ModelCfg.classes = deepcopy(TrainCfg.classes)
ModelCfg.class_map = deepcopy(TrainCfg.class_map)
ModelCfg.mask_classes = deepcopy(TrainCfg.mask_classes)
ModelCfg.mask_class_map = deepcopy(TrainCfg.mask_class_map)

ModelCfg.n_leads = TrainCfg.n_leads

ModelCfg.skip_dist = TrainCfg.skip_dist

ModelCfg.model_name = "unet"

ModelCfg.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)
