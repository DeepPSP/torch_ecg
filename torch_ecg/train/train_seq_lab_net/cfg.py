"""
"""
import os
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED

from ...cfg import Cfg as BaseCfg


__all__ = [
    "TrainCfg",
]


ModelCfg = ED()
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs

TrainCfg = ED()
