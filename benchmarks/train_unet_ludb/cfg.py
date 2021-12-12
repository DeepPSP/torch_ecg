"""
References
----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.cfg import DEFAULTS
from torch_ecg.model_configs import (
    ECG_SUBTRACT_UNET_CONFIG, ECG_UNET_VANILLA_CONFIG,
)


__all__ = [
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = ED()
BaseCfg.fs = 500  # Hz, LUDB data fs
BaseCfg.classes = [
    "p",  # pwave
    "N",  # qrs complex
    "t",  # twave
    "i",  # isoelectric
]
BaseCfg.class_map = ED(p=1, N=2, t=3, i=0)
BaseCfg.db_dir = None
BaseCfg.bias_thr = int(0.075 * BaseCfg.fs)  # TODO: renew this const
# detected waves that are within `skip_dist` from two ends of the signal will be ignored,
BaseCfg.skip_dist = int(0.5 * BaseCfg.fs)
BaseCfg.torch_dtype = DEFAULTS.torch_dtype



ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
# NOTE(update): "background" now do not count as a class
ModelCfg.classes = deepcopy(BaseCfg.classes)
# ModelCfg.classes = ["i", "N"]  # N for qrs, i for other parts
# ModelCfg.class_map = {c:i for i,c in enumerate(ModelCfg.classes)}
ModelCfg.n_leads = 1  # or 12
ModelCfg.skip_dist = BaseCfg.skip_dist

ModelCfg.model_name = "unet"

ModelCfg.update(deepcopy(ECG_UNET_VANILLA_CONFIG))

# TODO: add detailed ModelCfg


TrainCfg = ED()

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = os.path.join(_BASE_DIR, "log")
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 20
TrainCfg.torch_dtype = BaseCfg.torch_dtype

TrainCfg.fs = 500
TrainCfg.train_ratio = 0.8
TrainCfg.classes = BaseCfg.classes
TrainCfg.class_map = deepcopy(BaseCfg.class_map)

TrainCfg.skip_dist = BaseCfg.skip_dist

TrainCfg.leads_ordering = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]
TrainCfg.lead = "II"  # the lead to tain model, None --> all leads
TrainCfg.use_single_lead = True  # use single lead as input or use all 12 leads. used only when `TrainCfg.lead` = None

# as for `start_from` and `end_at`, see ref. [1] section 3.1
TrainCfg.start_from = int(2 * TrainCfg.fs)
TrainCfg.end_at = int(2 * TrainCfg.fs)
TrainCfg.input_len = int(4 * TrainCfg.fs)

TrainCfg.over_sampling = 2

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd", "rmsprop",

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

# configs of loss function
TrainCfg.loss = "CrossEntropyLoss"
TrainCfg.eval_every = 20
