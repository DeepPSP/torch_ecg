"""
References
----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
"""

from copy import deepcopy
from pathlib import Path

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.model_configs import (  # noqa: F401
    ECG_SUBTRACT_UNET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
)
from torch_ecg.utils import EAK

__all__ = [
    "TrainCfg",
]


_BASE_DIR = Path(__file__).absolute().parent


BaseCfg = CFG()
BaseCfg.fs = 500  # Hz, LUDB data fs
BaseCfg.classes = [
    "p",  # pwave
    "N",  # qrs complex
    "t",  # twave
    "i",  # isoelectric
]
# BaseCfg.mask_classes = [
#     "p",  # pwave
#     "N",  # qrs complex
#     "t",  # twave
# ]
BaseCfg.mask_classes = deepcopy(BaseCfg.classes)
BaseCfg.class_map = CFG(p=1, N=2, t=3, i=0)
# BaseCfg.mask_class_map = CFG({k:v-1 for k,v in BaseCfg.class_map.items() if k!="i"})
BaseCfg.mask_class_map = deepcopy(BaseCfg.class_map)
BaseCfg.db_dir = None
BaseCfg.bias_thr = int(0.075 * BaseCfg.fs)  # TODO: renew this const
# detected waves that are within `skip_dist` from two ends of the signal will be ignored,
BaseCfg.skip_dist = int(0.5 * BaseCfg.fs)
BaseCfg.torch_dtype = DEFAULTS.torch_dtype
BaseCfg.np_dtype = DEFAULTS.np_dtype


TrainCfg = CFG()

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = _BASE_DIR / "log"
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
TrainCfg.keep_checkpoint_max = 20
TrainCfg.torch_dtype = BaseCfg.torch_dtype
TrainCfg.np_dtype = BaseCfg.np_dtype

TrainCfg.fs = 500
TrainCfg.train_ratio = 0.8
TrainCfg.classes = deepcopy(BaseCfg.classes)
TrainCfg.class_map = deepcopy(BaseCfg.class_map)
TrainCfg.mask_classes = deepcopy(BaseCfg.mask_classes)
TrainCfg.mask_class_map = deepcopy(BaseCfg.mask_class_map)

TrainCfg.skip_dist = BaseCfg.skip_dist

TrainCfg.leads = (
    EAK.Standard12Leads
)  # ["II",]  # the lead to tain model, None --> all leads
TrainCfg.use_single_lead = (
    False  # use single lead as input or use all leads in `TrainCfg.leads`
)

if TrainCfg.use_single_lead:
    TrainCfg.n_leads = 1
else:
    TrainCfg.n_leads = len(TrainCfg.leads)

# as for `start_from` and `end_at`, see ref. [1] section 3.1
TrainCfg.start_from = int(2 * TrainCfg.fs)
TrainCfg.end_at = int(2 * TrainCfg.fs)
TrainCfg.input_len = int(4 * TrainCfg.fs)

TrainCfg.over_sampling = 1

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 32
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-3  # 1e-4
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 30

# configs of loss function
TrainCfg.loss = "FocalLoss"  # "BCEWithLogitsLoss", "AsymmetricLoss", "CrossEntropyLoss"
TrainCfg.loss_kw = CFG()  # "BCEWithLogitsLoss", "AsymmetricLoss"
TrainCfg.flooding_level = 0.0  # flooding performed if positive

TrainCfg.log_every = 1
TrainCfg.monitor = "f1_score"

TrainCfg.model_name = "unet"


ModelCfg = CFG()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs

ModelCfg.classes = deepcopy(BaseCfg.classes)
ModelCfg.class_map = deepcopy(BaseCfg.class_map)
ModelCfg.mask_classes = deepcopy(BaseCfg.mask_classes)
ModelCfg.mask_class_map = deepcopy(BaseCfg.mask_class_map)

ModelCfg.n_leads = TrainCfg.n_leads

ModelCfg.skip_dist = BaseCfg.skip_dist

ModelCfg.model_name = TrainCfg.model_name

ModelCfg.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)

# TODO: add detailed ModelCfg
