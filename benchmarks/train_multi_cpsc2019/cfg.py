"""
"""
import os
from copy import deepcopy
from itertools import repeat

import numpy as np

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.model_configs import (
    ECG_SEQ_LAB_NET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
    ECG_SUBTRACT_UNET_CONFIG,
)
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths


__all__ = [
    "ModelCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = CFG()
BaseCfg.fs = 500  # Hz, CPSC2019 data fs
BaseCfg.classes = ["N",]
BaseCfg.n_leads = 1
# BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")
BaseCfg.db_dir = None
BaseCfg.bias_thr = 0.075 * BaseCfg.fs  # keep the same with `THR` in `cpsc2019_score.py`
# detected rpeaks that are within `skip_dist` from two ends of the signal will be ignored,
# as in the official entry function
BaseCfg.skip_dist = 0.5 * BaseCfg.fs
BaseCfg.torch_dtype = DEFAULTS.torch_dtype




_COMMON_MODEL_CONFIGS = CFG()
_COMMON_MODEL_CONFIGS.skip_dist = BaseCfg.skip_dist
_COMMON_MODEL_CONFIGS.torch_dtype = BaseCfg.torch_dtype
_COMMON_MODEL_CONFIGS.fs = BaseCfg.fs
_COMMON_MODEL_CONFIGS.spacing = 1000 / BaseCfg.fs
# NOTE(update): "background" now do not count as a class
_COMMON_MODEL_CONFIGS.classes = deepcopy(BaseCfg.classes)
# _COMMON_MODEL_CONFIGS.classes = ["i", "N"]  # N for qrs, i for other parts
# _COMMON_MODEL_CONFIGS.class_map = {c:i for i,c in enumerate(BaseCfg.classes)}
_COMMON_MODEL_CONFIGS.n_leads = BaseCfg.n_leads
_COMMON_MODEL_CONFIGS.skip_dist = BaseCfg.skip_dist


ModelCfg = deepcopy(_COMMON_MODEL_CONFIGS)
ModelCfg.seq_lab_crnn = adjust_cnn_filter_lengths(
    deepcopy(ECG_SEQ_LAB_NET_CONFIG), BaseCfg.fs,
)
ModelCfg.seq_lab_crnn.reduction = 2**3
ModelCfg.seq_lab_crnn.recover_length = True
ModelCfg.seq_lab_crnn.update(deepcopy(_COMMON_MODEL_CONFIGS))

# NOTE: one can adjust any of the cnn, rnn, attn, clf part of ModelCfg.seq_lab_crnn like ModelCfg.seq_lab_cnn


ModelCfg.seq_lab_cnn = deepcopy(ModelCfg.seq_lab_crnn)

ModelCfg.seq_lab_cnn.rnn = CFG()
ModelCfg.seq_lab_cnn.rnn.name = "none"  # "lstm"


ModelCfg.unet = adjust_cnn_filter_lengths(
    deepcopy(ECG_UNET_VANILLA_CONFIG), BaseCfg.fs,
)
ModelCfg.unet.reduction = 1
ModelCfg.unet.recover_length = True
ModelCfg.unet.update(deepcopy(_COMMON_MODEL_CONFIGS))
ModelCfg.unet.cnn = CFG(name="none")


ModelCfg.subtract_unet = adjust_cnn_filter_lengths(
    deepcopy(ECG_SUBTRACT_UNET_CONFIG), BaseCfg.fs,
)
ModelCfg.subtract_unet.reduction = 1
ModelCfg.subtract_unet.recover_length = True
ModelCfg.subtract_unet.update(deepcopy(_COMMON_MODEL_CONFIGS))
ModelCfg.subtract_unet.cnn = CFG(name="none")




TrainCfg = CFG()
TrainCfg.torch_dtype = BaseCfg.torch_dtype
TrainCfg.fs = BaseCfg.fs
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = os.path.join(_BASE_DIR, "log")
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.model_dir = os.path.join(_BASE_DIR, "saved_models")
os.makedirs(TrainCfg.log_dir, exist_ok=True)
os.makedirs(TrainCfg.checkpoints, exist_ok=True)
os.makedirs(TrainCfg.model_dir, exist_ok=True)
TrainCfg.final_model_name = None
TrainCfg.keep_checkpoint_max = 20
TrainCfg.train_ratio = 0.8

TrainCfg.input_len = int(TrainCfg.fs * 10)  # 10 s
TrainCfg.classes = deepcopy(BaseCfg.classes)
# TrainCfg.class_map = ModelCfg.class_map
TrainCfg.n_leads = BaseCfg.n_leads
TrainCfg.bias_thr = BaseCfg.bias_thr
TrainCfg.skip_dist = BaseCfg.skip_dist

# configs of signal preprocessing
TrainCfg.normalize = False
# frequency band of the filter to apply, should be chosen very carefully
TrainCfg.bandpass = False
# TrainCfg.bandpass = CFG(
#     lowcut=0.5,
#     highcut=60,
# )

# configs of data aumentation
# NOTE: compared to data augmentation of CPSC2020,
# `stretch_compress` and `label_smoothing` are not used in CPSC2019
TrainCfg.label_smooth = False
TrainCfg.random_masking = False
TrainCfg.stretch_compress = False  # stretch or compress in time axis
TrainCfg.mixup = False
# TrainCfg.baseline_wander = CFG(  # too slow!
#     prob = 0.5,
#     bw_fs = np.array([0.33, 0.1, 0.05, 0.01]),
#     ampl_ratio = np.array([
#         [0.01, 0.01, 0.02, 0.03],  # low
#         [0.01, 0.02, 0.04, 0.05],  # low
#         [0.1, 0.06, 0.04, 0.02],  # low
#         [0.02, 0.04, 0.07, 0.1],  # low
#         [0.05, 0.1, 0.16, 0.25],  # medium
#         [0.1, 0.15, 0.25, 0.3],  # high
#         [0.25, 0.25, 0.3, 0.35],  # extremely high
#     ]),
#     gaussian = np.array([  # default gaussian, mean and std, in terms of ratio
#         [0.0, 0.001],
#         [0.0, 0.003],
#         [0.0, 0.01],
#     ]),
# )
TrainCfg.random_flip = CFG(
    prob = 0.5,
)

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 150
TrainCfg.batch_size = 32

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
TrainCfg.early_stopping.min_delta = 0.0001  # should be non-negative
TrainCfg.early_stopping.patience = 15

# configs of loss function
TrainCfg.loss = "BCEWithLogitsLoss"
# TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.flooding_level = 0.0  # flooding performed if positive

TrainCfg.log_step = 2

# model selection
TrainCfg.model_name = "seq_lab_crnn"  # "seq_lab_cnn", "unet", "subtract_unet"
TrainCfg.cnn_name = "multi_scopic"
TrainCfg.rnn_name = "lstm"
TrainCfg.attn_name = "se"

TrainCfg.reduction = 2**3  # TODO: automatic adjust via model config
TrainCfg.recover_length = True

TrainCfg.monitor = "qrs_score"
