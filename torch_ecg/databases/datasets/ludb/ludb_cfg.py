"""
References
----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.
"""

from copy import deepcopy

from ....cfg import CFG, DEFAULTS
from ....utils import ecg_arrhythmia_knowledge as EAK

__all__ = [
    "LUDBTrainCfg",
]


_NAME = "ludb"


LUDBTrainCfg = CFG()
LUDBTrainCfg.fs = 500  # Hz, LUDB data fs
LUDBTrainCfg.classes = [
    "p",  # pwave
    "N",  # qrs complex
    "t",  # twave
    "i",  # isoelectric
]
# LUDBTrainCfg.mask_classes = [
#     "p",  # pwave
#     "N",  # qrs complex
#     "t",  # twave
# ]
LUDBTrainCfg.mask_classes = deepcopy(LUDBTrainCfg.classes)
LUDBTrainCfg.class_map = CFG(p=1, N=2, t=3, i=0)
# LUDBTrainCfg.mask_class_map = CFG({k:v-1 for k,v in LUDBTrainCfg.class_map.items() if k!="i"})
LUDBTrainCfg.mask_class_map = deepcopy(LUDBTrainCfg.class_map)

LUDBTrainCfg.db_dir = None
LUDBTrainCfg.log_dir = DEFAULTS.log_dir / _NAME
LUDBTrainCfg.model_dir = DEFAULTS.model_dir / _NAME
LUDBTrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
LUDBTrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
LUDBTrainCfg.model_dir.mkdir(parents=True, exist_ok=True)
LUDBTrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
LUDBTrainCfg.keep_checkpoint_max = 20

LUDBTrainCfg.bias_thr = int(0.075 * LUDBTrainCfg.fs)  # TODO: renew this const
# detected waves that are within `skip_dist` from two ends of the signal will be ignored,
LUDBTrainCfg.skip_dist = int(0.5 * LUDBTrainCfg.fs)
LUDBTrainCfg.torch_dtype = DEFAULTS.torch_dtype

LUDBTrainCfg.fs = 500
LUDBTrainCfg.train_ratio = 0.8

LUDBTrainCfg.leads = (
    EAK.Standard12Leads
)  # ["II",]  # the lead to tain model, None --> all leads
LUDBTrainCfg.use_single_lead = (
    False  # use single lead as input or use all leads in `LUDBTrainCfg.leads`
)

if LUDBTrainCfg.use_single_lead:
    LUDBTrainCfg.n_leads = 1
else:
    LUDBTrainCfg.n_leads = len(LUDBTrainCfg.leads)

# as for `start_from` and `end_at`, see ref. [1] section 3.1
LUDBTrainCfg.start_from = int(2 * LUDBTrainCfg.fs)
LUDBTrainCfg.end_at = int(2 * LUDBTrainCfg.fs)
LUDBTrainCfg.input_len = int(4 * LUDBTrainCfg.fs)

LUDBTrainCfg.over_sampling = 1

# configs of training epochs, batch, etc.
LUDBTrainCfg.n_epochs = 300
LUDBTrainCfg.batch_size = 32
# LUDBTrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
LUDBTrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
LUDBTrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
LUDBTrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
LUDBTrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

LUDBTrainCfg.learning_rate = 1e-3  # 1e-4
LUDBTrainCfg.lr = LUDBTrainCfg.learning_rate

LUDBTrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
LUDBTrainCfg.lr_step_size = 50
LUDBTrainCfg.lr_gamma = 0.1
LUDBTrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

LUDBTrainCfg.burn_in = 400
LUDBTrainCfg.steps = [5000, 10000]

LUDBTrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
LUDBTrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
LUDBTrainCfg.early_stopping.patience = 10

# configs of loss function
LUDBTrainCfg.loss = (
    "FocalLoss"  # "BCEWithLogitsLoss", "AsymmetricLoss", "CrossEntropyLoss"
)
LUDBTrainCfg.loss_kw = CFG()  # "BCEWithLogitsLoss", "AsymmetricLoss"
LUDBTrainCfg.flooding_level = 0.0  # flooding performed if positive

LUDBTrainCfg.log_every = 1
LUDBTrainCfg.monitor = "f1_score"

LUDBTrainCfg.model_name = "unet"
