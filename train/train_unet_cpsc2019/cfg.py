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
BaseCfg.classes = ["N", "i"]   # N for qrs, i for other parts
# BaseCfg.class_map = {c:i for i,c in enumerate(BaseCfg.classes)}
# BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")
BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2019/train/"
BaseCfg.bias_thr = int(0.075 * BaseCfg.fs)  # keep the same with `THR` in `cpsc2019_score.py`
# detected rpeaks that are within `skip_dist` from two ends of the signal will be ignored,
# as in the official entry function
BaseCfg.skip_dist = int(0.5 * BaseCfg.fs)
BaseCfg.torch_dtype = Cfg.torch_dtype



ModelCfg = ED()
ModelCfg.skip_dist = BaseCfg.skip_dist
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
# NOTE(update): "background" now do not count as a class
ModelCfg.classes = deepcopy(BaseCfg.classes)
# ModelCfg.class_map = deepcopy(BaseCfg.class_map)
# ModelCfg.classes = ["i", "N"]  # N for qrs, i for other parts
ModelCfg.n_leads = 1
ModelCfg.skip_dist = BaseCfg.skip_dist

ModelCfg.model_name = "subtract_unet"


# subtract unet, from CPSC2019 entry 0433
ModelCfg.subtract_unet = ED()
ModelCfg.subtract_unet.fs = ModelCfg.fs
ModelCfg.subtract_unet.classes = deepcopy(ModelCfg.classes)
ModelCfg.subtract_unet.n_leads = ModelCfg.n_leads
ModelCfg.subtract_unet.skip_dist = ModelCfg.skip_dist
ModelCfg.subtract_unet.torch_dtype = ModelCfg.torch_dtype

ModelCfg.subtract_unet.groups = 1
ModelCfg.subtract_unet.init_batch_norm = False

# in triple conv
ModelCfg.subtract_unet.init_num_filters = 16
ModelCfg.subtract_unet.init_filter_length = 21
ModelCfg.subtract_unet.init_dropouts = [0.0, 0.15, 0.0]
ModelCfg.subtract_unet.batch_norm = True
ModelCfg.subtract_unet.kernel_initializer = "he_normal"
ModelCfg.subtract_unet.kw_initializer = {}
ModelCfg.subtract_unet.activation = "relu"
ModelCfg.subtract_unet.kw_activation = {"inplace": True}

# down, triple conv
ModelCfg.subtract_unet.down_up_block_num = 3
ModelCfg.subtract_unet.down_mode = "max"
ModelCfg.subtract_unet.down_scales = [10, 5, 2]
init_down_num_filters = 24
_num_convs = 3  # TripleConv
ModelCfg.subtract_unet.down_num_filters = [
    list(repeat(init_down_num_filters * (2**idx), _num_convs)) \
        for idx in range(0, ModelCfg.subtract_unet.down_up_block_num-1)
]
ModelCfg.subtract_unet.down_filter_lengths = [11, 5]
ModelCfg.subtract_unet.down_dropouts = \
    list(repeat([0.0, 0.15, 0.0], ModelCfg.subtract_unet.down_up_block_num-1))

# bottom, double conv
ModelCfg.subtract_unet.bottom_num_filters = [
    # branch 1
    list(repeat(init_down_num_filters*(2**(ModelCfg.subtract_unet.down_up_block_num-1)), 2)),
    # branch 2
    list(repeat(init_down_num_filters*(2**(ModelCfg.subtract_unet.down_up_block_num-1)), 2)),
    # branch 1 and branch 2 should have the same `num_filters`,
    # otherwise `subtraction` would be infeasible
]
ModelCfg.subtract_unet.bottom_filter_lengths = [
    list(repeat(5, 2)),  # branch 1
    list(repeat(5, 2)),  # branch 2
]
ModelCfg.subtract_unet.bottom_dilations = [
    # the ordering matters
    list(repeat(1, 2)),  # branch 1
    list(repeat(10, 2)),  # branch 2
]
ModelCfg.subtract_unet.bottom_dropouts = [
    [0.15, 0.0],  # branch 1
    [0.15, 0.0],  # branch 2
]

# up, triple conv
ModelCfg.subtract_unet.up_mode = "nearest"
ModelCfg.subtract_unet.up_scales = [2, 5, 10]
ModelCfg.subtract_unet.up_num_filters = [
    list(repeat(48, _num_convs)),
    list(repeat(24, _num_convs)),
    list(repeat(16, _num_convs)),
]
ModelCfg.subtract_unet.up_deconv_filter_lengths = \
    list(repeat(9, ModelCfg.subtract_unet.down_up_block_num))
ModelCfg.subtract_unet.up_conv_filter_lengths = [5, 11, 21]
ModelCfg.subtract_unet.up_dropouts = [
    [0.15, 0.15, 0.0],
    [0.15, 0.15, 0.0],
    [0.15, 0.15, 0.0],
]

# out conv
ModelCfg.subtract_unet.out_filter_length = 1

ModelCfg.subtract_unet.down_block = ED()
ModelCfg.subtract_unet.down_block.batch_norm = ModelCfg.subtract_unet.batch_norm
ModelCfg.subtract_unet.down_block.kernel_initializer = ModelCfg.subtract_unet.kernel_initializer 
ModelCfg.subtract_unet.down_block.kw_initializer = deepcopy(ModelCfg.subtract_unet.kw_initializer)
ModelCfg.subtract_unet.down_block.activation = ModelCfg.subtract_unet.activation
ModelCfg.subtract_unet.down_block.kw_activation = deepcopy(ModelCfg.subtract_unet.kw_activation)

ModelCfg.subtract_unet.up_block = ED()
ModelCfg.subtract_unet.up_block.batch_norm = ModelCfg.subtract_unet.batch_norm
ModelCfg.subtract_unet.up_block.kernel_initializer = ModelCfg.subtract_unet.kernel_initializer 
ModelCfg.subtract_unet.up_block.kw_initializer = deepcopy(ModelCfg.subtract_unet.kw_initializer)
ModelCfg.subtract_unet.up_block.activation = ModelCfg.subtract_unet.activation
ModelCfg.subtract_unet.up_block.kw_activation = deepcopy(ModelCfg.subtract_unet.kw_activation)


# vanilla unet
ModelCfg.unet = ED()
ModelCfg.unet.fs = ModelCfg.fs
ModelCfg.unet.classes = deepcopy(ModelCfg.classes)
ModelCfg.unet.n_leads = ModelCfg.n_leads
ModelCfg.unet.skip_dist = ModelCfg.skip_dist
ModelCfg.unet.torch_dtype = ModelCfg.torch_dtype

ModelCfg.unet.groups = 1

# ModelCfg.unet.init_num_filters = len(ModelCfg.unet.classes)  # keep the same with n_classes
ModelCfg.unet.init_num_filters = 16
ModelCfg.unet.init_filter_length = 9
ModelCfg.unet.out_filter_length = 9
ModelCfg.unet.batch_norm = True
ModelCfg.unet.kernel_initializer = "he_normal"
ModelCfg.unet.kw_initializer = {}
ModelCfg.unet.activation = "relu"
ModelCfg.unet.kw_activation = {"inplace": True}

ModelCfg.unet.down_up_block_num = 4

ModelCfg.unet.down_mode = "max"
ModelCfg.unet.down_scales = list(repeat(2, ModelCfg.unet.down_up_block_num))
ModelCfg.unet.down_num_filters = [
    ModelCfg.unet.init_num_filters * (2**idx) \
        for idx in range(1, ModelCfg.unet.down_up_block_num+1)
]
ModelCfg.unet.down_filter_lengths = \
    list(repeat(ModelCfg.unet.init_filter_length, ModelCfg.unet.down_up_block_num))

ModelCfg.unet.up_mode = "nearest"
ModelCfg.unet.up_scales = list(repeat(2, ModelCfg.unet.down_up_block_num))
ModelCfg.unet.up_num_filters = [
    ModelCfg.unet.init_num_filters * (2**idx) \
        for idx in range(ModelCfg.unet.down_up_block_num-1,-1,-1)
]
ModelCfg.unet.up_deconv_filter_lengths = \
    list(repeat(9, ModelCfg.unet.down_up_block_num))
ModelCfg.unet.up_conv_filter_lengths = \
    list(repeat(ModelCfg.unet.init_filter_length, ModelCfg.unet.down_up_block_num))

ModelCfg.unet.down_block = ED()
ModelCfg.unet.down_block.batch_norm = ModelCfg.unet.batch_norm
ModelCfg.unet.down_block.kernel_initializer = ModelCfg.unet.kernel_initializer 
ModelCfg.unet.down_block.kw_initializer = deepcopy(ModelCfg.unet.kw_initializer)
ModelCfg.unet.down_block.activation = ModelCfg.unet.activation
ModelCfg.unet.down_block.kw_activation = deepcopy(ModelCfg.unet.kw_activation)

ModelCfg.unet.up_block = ED()
ModelCfg.unet.up_block.batch_norm = ModelCfg.unet.batch_norm
ModelCfg.unet.up_block.kernel_initializer = ModelCfg.unet.kernel_initializer 
ModelCfg.unet.up_block.kw_initializer = deepcopy(ModelCfg.unet.kw_initializer)
ModelCfg.unet.up_block.activation = ModelCfg.unet.activation
ModelCfg.unet.up_block.kw_activation = deepcopy(ModelCfg.unet.kw_activation)



TrainCfg = ED()
TrainCfg.fs = ModelCfg.fs
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

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 256
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-3  # 1e-4
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = None  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 1e-2  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.early_stopping = ED()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 6

# configs of loss function
TrainCfg.loss = "BCEWithLogitsLoss"
# TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.eval_every = 20
