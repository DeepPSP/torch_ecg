"""
"""

import os

import numpy as np
import torch
from easydict import EasyDict as ED


__all__ = ["DEFAULTS"]


_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_PROJECT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "torch_ecg")
os.makedirs(name=_PROJECT_CACHE, exist_ok=True)


DEFAULTS = ED()

DEFAULTS.log_dir = os.path.join(_PROJECT_CACHE, "log")
DEFAULTS.checkpoints = os.path.join(_PROJECT_CACHE, "checkpoints")
DEFAULTS.prefix = "TorchECG"

DEFAULTS.torch_dtype = torch.float32  # torch.float64, torch.float16
DEFAULTS.np_dtype = np.float32  # np.float64, np.float16
