"""
"""
import os
from copy import deepcopy
from itertools import repeat

import numpy as np
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg


__all__ = [
    "ModelCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = ED()
BaseCfg.fs = 500  # Hz, CPSC2019 data fs
BaseCfg.classes = ["N",]
# BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")
BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2019/train/"
BaseCfg.bias_thr = 0.075 * BaseCfg.fs  # keep the same with `THR` in `cpsc2019_score.py`
# detected rpeaks that are within `skip_dist` from two ends of the signal will be ignored,
# as in the official entry function
BaseCfg.skip_dist = 0.5 * BaseCfg.fs
BaseCfg.torch_dtype = Cfg.torch_dtype



ModelCfg = ED()
ModelCfg.skip_dist = BaseCfg.skip_dist
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
# NOTE(update): "background" now do not count as a class
ModelCfg.classes = deepcopy(BaseCfg.classes)
# ModelCfg.classes = ["i", "N"]  # N for qrs, i for other parts
# ModelCfg.class_map = {c:i for i,c in enumerate(ModelCfg.classes)}
ModelCfg.n_leads = 1
ModelCfg.skip_dist = BaseCfg.skip_dist


ModelCfg.seq_lab_crnn = ED()
ModelCfg.seq_lab_crnn.fs = ModelCfg.fs
ModelCfg.seq_lab_crnn.classes = deepcopy(ModelCfg.classes)
ModelCfg.seq_lab_crnn.n_leads = ModelCfg.n_leads
ModelCfg.seq_lab_crnn.skip_dist = ModelCfg.skip_dist
ModelCfg.seq_lab_crnn.torch_dtype = ModelCfg.torch_dtype

ModelCfg.seq_lab_crnn.cnn = ED()
ModelCfg.seq_lab_crnn.cnn.name = 'multi_scopic'  # resnet, resnet_gc, vgg, cpsc2018, etc.
ModelCfg.seq_lab_crnn.cnn.multi_scopic = ED()
ModelCfg.seq_lab_crnn.cnn.multi_scopic.groups = 1
ModelCfg.seq_lab_crnn.cnn.multi_scopic.scopes = [
    [ # branch 0
        [1,],
        [1,1,],
        [1,1,1,],
    ],
    [ # branch 1
        [2,],
        [2,4,],
        [8,8,8,],
    ],
    [ # branch 2
        [4,],
        [4,8,],
        [16,32,64,],
    ],
]
ModelCfg.seq_lab_crnn.cnn.multi_scopic.filter_lengths = [
    [11, 7, 5,],  # branch 0
    [11, 7, 5,],  # branch 1
    [11, 7, 5,],  # branch 2
]
ModelCfg.seq_lab_crnn.cnn.multi_scopic.subsample_lengths = \
    list(repeat(2, len(ModelCfg.seq_lab_crnn.cnn.multi_scopic.scopes)))
_base_num_filters = 8
ModelCfg.seq_lab_crnn.cnn.multi_scopic.num_filters = [
    [ # branch 0
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [ # branch 1
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [ # branch 2
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ModelCfg.seq_lab_crnn.cnn.multi_scopic.dropouts = [
    [0, 0.2, 0],  # branch 0
    [0, 0.2, 0],  # branch 1
    [0, 0.2, 0],  # branch 2
]
ModelCfg.seq_lab_crnn.cnn.multi_scopic.bias = True
ModelCfg.seq_lab_crnn.cnn.multi_scopic.kernel_initializer = "he_normal"
ModelCfg.seq_lab_crnn.cnn.multi_scopic.kw_initializer = {}
ModelCfg.seq_lab_crnn.cnn.multi_scopic.activation = "relu"
ModelCfg.seq_lab_crnn.cnn.multi_scopic.kw_activation = {"inplace": True}
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block = ED()
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.subsample_mode = 'max'  # or 'conv', 'avg', 'nearest', 'linear', 'bilinear'
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.bias = \
    ModelCfg.seq_lab_crnn.cnn.multi_scopic.bias
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.kernel_initializer = \
    ModelCfg.seq_lab_crnn.cnn.multi_scopic.kernel_initializer
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.kw_initializer = \
    deepcopy(ModelCfg.seq_lab_crnn.cnn.multi_scopic.kw_initializer)
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.activation = \
    ModelCfg.seq_lab_crnn.cnn.multi_scopic.activation
ModelCfg.seq_lab_crnn.cnn.multi_scopic.block.kw_activation = \
    deepcopy(ModelCfg.seq_lab_crnn.cnn.multi_scopic.kw_activation)

# rnn part
# abuse of notation
ModelCfg.seq_lab_crnn.rnn = ED()
ModelCfg.seq_lab_crnn.rnn.name = 'lstm'  # 'none', 'lstm', 'attention'
ModelCfg.seq_lab_crnn.rnn.lstm = ED()
ModelCfg.seq_lab_crnn.rnn.lstm.out_channels = [
    256, 64,
]
ModelCfg.seq_lab_crnn.rnn.lstm.bias = True
ModelCfg.seq_lab_crnn.rnn.lstm.dropouts = 0.2
ModelCfg.seq_lab_crnn.rnn.lstm.activation = 'mish'

ModelCfg.seq_lab_crnn.attn = ED()
ModelCfg.seq_lab_crnn.attn.name = 'se'  # 'gc'
ModelCfg.seq_lab_crnn.attn.se = ED()
ModelCfg.seq_lab_crnn.attn.se.reduction = 8  # not including the last linear layer
ModelCfg.seq_lab_crnn.attn.se.activation = "relu"
ModelCfg.seq_lab_crnn.attn.se.kw_activation = ED(inplace=True)
ModelCfg.seq_lab_crnn.attn.se.bias = True
ModelCfg.seq_lab_crnn.attn.se.kernel_initializer = 'he_normal'

ModelCfg.seq_lab_crnn.clf = ED()
ModelCfg.seq_lab_crnn.clf.out_channels = [256, 64]  # not including the last linear layer
ModelCfg.seq_lab_crnn.clf.activation = "mish"
ModelCfg.seq_lab_crnn.clf.bias = True
ModelCfg.seq_lab_crnn.clf.kernel_initializer = 'he_normal'
ModelCfg.seq_lab_crnn.clf.dropouts = [0.2, 0.2, 0.0]

# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
# ModelCfg.seq_lab_crnn.global_pool = 'max'  # 'avg', 'attentive'


ModelCfg.seq_lab_cnn = ED()
ModelCfg.seq_lab_cnn.fs = ModelCfg.fs
ModelCfg.seq_lab_cnn.classes = deepcopy(ModelCfg.classes)
ModelCfg.seq_lab_cnn.n_leads = ModelCfg.n_leads
ModelCfg.seq_lab_cnn.skip_dist = ModelCfg.skip_dist
ModelCfg.seq_lab_cnn.torch_dtype = ModelCfg.torch_dtype

ModelCfg.seq_lab_cnn.cnn = ModelCfg.seq_lab_crnn.cnn.copy()

ModelCfg.seq_lab_cnn.rnn = ED()
ModelCfg.seq_lab_cnn.rnn.name = 'none'  # 'lstm'
ModelCfg.seq_lab_cnn.attn = ED()
ModelCfg.seq_lab_cnn.attn.name = 'se'  # 'gc'
ModelCfg.seq_lab_cnn.attn.se = ED()
ModelCfg.seq_lab_cnn.attn.se.reduction = 8  # not including the last linear layer
ModelCfg.seq_lab_cnn.attn.se.activation = "relu"
ModelCfg.seq_lab_cnn.attn.se.kw_activation = ED(inplace=True)
ModelCfg.seq_lab_cnn.attn.se.bias = True
ModelCfg.seq_lab_cnn.attn.se.kernel_initializer = 'he_normal'

ModelCfg.seq_lab_cnn.clf = ED()
ModelCfg.seq_lab_cnn.clf.out_channels = [256, 64]  # not including the last linear layer
ModelCfg.seq_lab_cnn.clf.activation = "mish"
ModelCfg.seq_lab_cnn.clf.bias = True
ModelCfg.seq_lab_cnn.clf.kernel_initializer = 'he_normal'
ModelCfg.seq_lab_cnn.clf.dropouts = [0.2, 0.2, 0.0]


TrainCfg = ED()
TrainCfg.fs = ModelCfg.fs
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = os.path.join(_BASE_DIR, 'log')
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 50
TrainCfg.train_ratio = 0.8

TrainCfg.input_len = int(TrainCfg.fs * 10)  # 10 s
TrainCfg.classes = deepcopy(BaseCfg.classes)
# TrainCfg.class_map = ModelCfg.class_map
TrainCfg.n_leads = ModelCfg.n_leads
TrainCfg.bias_thr = BaseCfg.bias_thr
TrainCfg.skip_dist = BaseCfg.skip_dist

# configs of data aumentation
# NOTE: compared to data augmentation of CPSC2020,
# `stretch_compress` and `label_smoothing` are not used in CPSC2019
TrainCfg.label_smoothing = 0.0
TrainCfg.random_normalize = True  # (re-)normalize to random mean and std
TrainCfg.random_normalize_mean = [-0.05, 0.1]
TrainCfg.random_normalize_std = [0.08, 0.32]
TrainCfg.baseline_wander = True  # randomly shifting the baseline
TrainCfg.bw = TrainCfg.baseline_wander  # alias
TrainCfg.bw_fs = np.array([0.33, 0.1, 0.05, 0.01])
TrainCfg.bw_ampl_ratio = np.array([
    [0.01, 0.01, 0.02, 0.03],  # low
    [0.01, 0.02, 0.04, 0.05],  # low
    [0.1, 0.06, 0.04, 0.02],  # low
    [0.02, 0.04, 0.07, 0.1],  # low
    [0.05, 0.1, 0.16, 0.25],  # medium
    [0.1, 0.15, 0.25, 0.3],  # high
    [0.25, 0.25, 0.3, 0.35],  # extremely high
])
TrainCfg.bw_gaussian = np.array([  # mean and std, ratio
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],  # ensure at least one with no gaussian noise
    [0.0, 0.003],
    [0.0, 0.01],
])
TrainCfg.flip = [-1] + [1]*4  # making the signal upside down, with probability 1/(1+4)

TrainCfg.seq_lab_reduction = 2**3  # TODO: automatic adjust via model config

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 256
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd"

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

TrainCfg.momentum = 0.949
TrainCfg.decay = 0.0005

# configs of loss function
TrainCfg.loss = 'BCEWithLogitsLoss'
# TrainCfg.loss = 'BCEWithLogitsWithClassWeightLoss'
TrainCfg.eval_every = 20

# model selection
TrainCfg.model_name = "crnn"  # "seq_lab", "unet"
