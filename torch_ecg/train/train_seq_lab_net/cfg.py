"""
"""
import os
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED

from ...cfg import Cfg as BaseCfg


__all__ = [
    "ModelCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.classes = ["i", "N"]  # N for qrs, i for other parts
ModelCfg.class_map = {c:i for i,c in enumerate(ModelCfg.classes)}
ModelCfg.n_leads = 1


TrainCfg = ED()
TrainCfg.fs = ModelCfg.fs
TrainCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2019/train/"
TrainCfg.log_dir = os.path.join(_BASE_DIR, 'log')
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 20

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd"

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

TrainCfg.momentum = 0.949
TrainCfg.decay = 0.0005

TrainCfg.input_len = int(TrainCfg.fs * 10)  # 10 s
TrainCfg.classes = ModelCfg.classes
TrainCfg.class_map = ModelCfg.class_map
TrainCfg.n_leads = ModelCfg.n_leads

# configs of data aumentation
TrainCfg.normalize_data = True
# TODO: add more data aumentation
