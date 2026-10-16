"""
Configurations for the CPSC2020 benchmark (premature beat detection
from long-term single-lead ECGs).
"""

from copy import deepcopy
from itertools import repeat
from pathlib import Path

import numpy as np

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS

__all__ = [
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = Path(__file__).absolute().parent


BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.fs = 400  # Hz, CPSC2020 data fs
BaseCfg.classes = ["N", "S", "V"]  # normal, supraventricular premature beat, premature ventricular contraction
BaseCfg.class_map = {c: idx for idx, c in enumerate(BaseCfg.classes)}
BaseCfg.bias_thr = 0.15 * BaseCfg.fs  # keep the same with the official challenge scoring function
BaseCfg.torch_dtype = DEFAULTS.DTYPE.TORCH


# two model families:
# - "crnn": segment-level classification (N / S / V)
# - "seq_lab": beat-level sequence labelling (S / V localization)
ModelCfg = CFG()
ModelCfg.fs = BaseCfg.fs
ModelCfg.n_leads = 1
ModelCfg.torch_dtype = BaseCfg.torch_dtype

ModelCfg.crnn = CFG()
ModelCfg.crnn.fs = BaseCfg.fs
ModelCfg.crnn.n_leads = ModelCfg.n_leads
ModelCfg.crnn.torch_dtype = BaseCfg.torch_dtype
ModelCfg.crnn.classes = deepcopy(BaseCfg.classes)
ModelCfg.crnn.class_map = deepcopy(BaseCfg.class_map)

ModelCfg.crnn.cnn = CFG()
ModelCfg.crnn.cnn.name = "multi_scopic"  # resnet, resnet_gc, vgg, cpsc2018, etc.
ModelCfg.crnn.cnn.multi_scopic = CFG()
ModelCfg.crnn.cnn.multi_scopic.groups = 1
ModelCfg.crnn.cnn.multi_scopic.scopes = [
    [  # branch 0
        [
            1,
        ],
        [
            1,
            1,
        ],
        [
            1,
            1,
            1,
        ],
    ],
    [  # branch 1
        [
            2,
        ],
        [
            2,
            4,
        ],
        [
            8,
            8,
            8,
        ],
    ],
    [  # branch 2
        [
            4,
        ],
        [
            4,
            8,
        ],
        [
            16,
            32,
            64,
        ],
    ],
]
# TODO:
# as sampling frequencies of CPSC2019 and CINC2020 are 500Hz
# while CPSC2020 is 400 Hz
# should the filter_lengths be adjusted?
ModelCfg.crnn.cnn.multi_scopic.filter_lengths = [
    [
        11,
        7,
        5,
    ],  # branch 0
    [
        11,
        7,
        5,
    ],  # branch 1
    [
        11,
        7,
        5,
    ],  # branch 2
]
ModelCfg.crnn.cnn.multi_scopic.subsample_lengths = list(repeat(2, len(ModelCfg.crnn.cnn.multi_scopic.scopes)))
_base_num_filters = 8
ModelCfg.crnn.cnn.multi_scopic.num_filters = [
    [  # branch 0
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [  # branch 1
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [  # branch 2
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelCfg.crnn.cnn.multi_scopic.dropouts = [
    [0, 0.2, 0],  # branch 0
    [0, 0.2, 0],  # branch 1
    [0, 0.2, 0],  # branch 2
]
ModelCfg.crnn.cnn.multi_scopic.bias = True
ModelCfg.crnn.cnn.multi_scopic.kernel_initializer = "he_normal"
ModelCfg.crnn.cnn.multi_scopic.kw_initializer = {}
ModelCfg.crnn.cnn.multi_scopic.activation = "relu"
ModelCfg.crnn.cnn.multi_scopic.kw_activation = {"inplace": True}
ModelCfg.crnn.cnn.multi_scopic.block = CFG()
ModelCfg.crnn.cnn.multi_scopic.block.subsample_mode = "max"  # or "conv", "avg", "nearest", "linear", "bilinear"
ModelCfg.crnn.cnn.multi_scopic.block.bias = ModelCfg.crnn.cnn.multi_scopic.bias
ModelCfg.crnn.cnn.multi_scopic.block.kernel_initializer = ModelCfg.crnn.cnn.multi_scopic.kernel_initializer
ModelCfg.crnn.cnn.multi_scopic.block.kw_initializer = deepcopy(ModelCfg.crnn.cnn.multi_scopic.kw_initializer)
ModelCfg.crnn.cnn.multi_scopic.block.activation = ModelCfg.crnn.cnn.multi_scopic.activation
ModelCfg.crnn.cnn.multi_scopic.block.kw_activation = deepcopy(ModelCfg.crnn.cnn.multi_scopic.kw_activation)

# rnn part
# abuse of notation
ModelCfg.crnn.rnn = CFG()
ModelCfg.crnn.rnn.name = "linear"  # "none", "lstm", "attention"
ModelCfg.crnn.rnn.linear = CFG()
ModelCfg.crnn.rnn.linear.out_channels = [
    256,
    64,
]
ModelCfg.crnn.rnn.linear.bias = True
ModelCfg.crnn.rnn.linear.dropouts = 0.2
ModelCfg.crnn.rnn.linear.activation = "mish"

# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
ModelCfg.crnn.global_pool = "max"  # "avg", "attentive"


ModelCfg.seq_lab = CFG()
ModelCfg.seq_lab.fs = BaseCfg.fs
ModelCfg.seq_lab.n_leads = 1
ModelCfg.seq_lab.torch_dtype = BaseCfg.torch_dtype
ModelCfg.seq_lab.classes = [c for c in BaseCfg.classes if c != "N"]
ModelCfg.seq_lab.class_map = {c: v for c, v in BaseCfg.class_map.items() if c != "N"}

ModelCfg.seq_lab.cnn = ModelCfg.crnn.cnn.copy()

ModelCfg.seq_lab.rnn = CFG()
ModelCfg.seq_lab.rnn.name = "none"  # "lstm"
ModelCfg.seq_lab.attn = CFG()
ModelCfg.seq_lab.attn.name = "se"  # "gc"
ModelCfg.seq_lab.attn.se = CFG()
ModelCfg.seq_lab.attn.se.reduction = 16  # not including the last linear layer
ModelCfg.seq_lab.attn.se.activation = "relu"
ModelCfg.seq_lab.attn.se.kw_activation = CFG(inplace=True)
ModelCfg.seq_lab.attn.se.bias = True
ModelCfg.seq_lab.attn.se.kernel_initializer = "he_normal"

ModelCfg.seq_lab.clf = CFG()
ModelCfg.seq_lab.clf.out_channels = [256, 64]  # not including the last linear layer
ModelCfg.seq_lab.clf.activation = "mish"
ModelCfg.seq_lab.clf.bias = True
ModelCfg.seq_lab.clf.kernel_initializer = "he_normal"
ModelCfg.seq_lab.clf.dropouts = [0.2, 0.2, 0.0]


TrainCfg = CFG()
TrainCfg.fs = ModelCfg.fs
TrainCfg.n_leads = 1
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = _BASE_DIR / "log"
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.model_dir = _BASE_DIR / "saved_models"
TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)
TrainCfg.keep_checkpoint_max = 50
TrainCfg.input_len = int(10 * TrainCfg.fs)  # 10 s
TrainCfg.overlap_len = int(6 * TrainCfg.fs)  # 6 s
TrainCfg.bias_thr = BaseCfg.bias_thr

TrainCfg.test_rec_num = 1

# data augmentation (on the fly)
TrainCfg.label_smoothing = 0.1
TrainCfg.random_mask = int(TrainCfg.fs * 0.0)  # 1.0s, 0 for no masking
# NOTE: named `stretch_compress_ratio` (not `stretch_compress`) since the
# augmentation here is done inside the dataset, not by the library
# AugmenterManager (whose config keys collide with registered names)
TrainCfg.stretch_compress_ratio = 5  # stretch or compress in time axis, units in percentage (0 - inf), offline
TrainCfg.random_normalize = True  # (re-)normalize to random mean and std
# valid segments has
# median of mean appr. 0, mean of mean 0.038
# median of std 0.13, mean of std 0.18
TrainCfg.random_normalize_mean = [-0.05, 0.1]
TrainCfg.random_normalize_std = [0.08, 0.32]
TrainCfg.bw = True  # randomly shifting the baseline (done inside the dataset)
TrainCfg.bw_fs = np.array([0.33, 0.1, 0.05, 0.01])
TrainCfg.bw_ampl_ratio = np.array(
    [
        [0.01, 0.01, 0.02, 0.03],  # low
        [0.01, 0.02, 0.04, 0.05],  # low
        [0.1, 0.06, 0.04, 0.02],  # low
        [0.02, 0.04, 0.07, 0.1],  # low
        [0.05, 0.1, 0.16, 0.25],  # medium
        [0.1, 0.15, 0.25, 0.3],  # high
        [0.25, 0.25, 0.3, 0.35],  # extremely high
    ]
)
TrainCfg.bw_gaussian = np.array(
    [  # mean and std, ratio
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],  # ensure at least one with no gaussian noise
        [0.0, 0.003],
        [0.0, 0.01],
    ]
)
TrainCfg.flip = [-1] + [1] * 4  # making the signal upside down, with probability 1/(1+4)

TrainCfg.seq_lab_reduction = 2**3  # TODO: automatic adjust via model config

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = None  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 1e-2  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 6

# model selection
TrainCfg.model_name = "crnn"  # "seq_lab"
TrainCfg.classes = deepcopy(ModelCfg[TrainCfg.model_name].classes)
TrainCfg.class_map = deepcopy(ModelCfg[TrainCfg.model_name].class_map)

# configs of loss function
TrainCfg.loss = "BCEWithLogitsLoss"
TrainCfg.loss_kw = CFG()
TrainCfg.flooding_level = 0.0  # flooding performed if positive

TrainCfg.log_step = 20

# monitor for determining the best model:
# "f_beta_measure" for the crnn model, "challenge_score" for the seq_lab model
TrainCfg.monitor = "f_beta_measure"
