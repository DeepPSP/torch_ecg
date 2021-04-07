"""
NOTE:

pipeline under this config file is NOT tested!
"""
import os
from itertools import repeat
from copy import deepcopy

import pywt
import numpy as np
from easydict import EasyDict as ED

from torch_ecg.cfg import Cfg


__all__ = [
    "BaseCfg",
    "PreprocCfg",
    "FeatureCfg",
    "ModelCfg",
    "TrainCfg",
    "PlotCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = ED()
BaseCfg.fs = 400  # Hz, CPSC2020 data fs
BaseCfg.classes = ["N", "S", "V"]
BaseCfg.class_map = {c: idx for idx, c in enumerate(BaseCfg.classes)}
# BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")
BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2020_30S/TrainingSet/"

BaseCfg.bias_thr = 0.15 * BaseCfg.fs  # keep the same with `THR` in `CPSC202_score.py`
BaseCfg.beat_ann_bias_thr = 0.1 * BaseCfg.fs  # half width of broad qrs complex
BaseCfg.beat_winL = 250 * BaseCfg.fs // 1000  # corr. to 250 ms
BaseCfg.beat_winR = 250 * BaseCfg.fs // 1000  # corr. to 250 ms

BaseCfg.torch_dtype = Cfg.torch_dtype



PreprocCfg = ED()
PreprocCfg.fs = BaseCfg.fs
# sequential, keep correct ordering, to add 'motion_artefact'
PreprocCfg.preproc = ['bandpass',]  # 'baseline',
# for 200 ms and 600 ms, ref. (`ecg_classification` in `reference`)
PreprocCfg.baseline_window1 = int(0.2*PreprocCfg.fs)  # 200 ms window
PreprocCfg.baseline_window2 = int(0.6*PreprocCfg.fs)  # 600 ms window
PreprocCfg.filter_band = [0.5, 45]
PreprocCfg.parallel_epoch_len = 600  # second
PreprocCfg.parallel_epoch_overlap = 10  # second
PreprocCfg.parallel_keep_tail = True
PreprocCfg.rpeaks = 'seq_lab'  # 'xqrs'
# or 'gqrs', or 'pantompkins', 'hamilton', 'ssf', 'christov', 'engzee', 'gamboa'
# or empty string '' if not detecting rpeaks
"""
for qrs detectors:
    `xqrs` sometimes detects s peak (valley) as r peak,
    but according to Jeethan, `xqrs` has the best performance
"""
# least distance of an valid R peak to two ends of ECG signals
PreprocCfg.rpeaks_skip_dist = int(0.5 * PreprocCfg.fs)  # 0.5s


# FeatureCfg only for ML models, deprecated
FeatureCfg = ED()
FeatureCfg.fs = BaseCfg.fs
FeatureCfg.features = ['wavelet', 'rr', 'morph',]

FeatureCfg.wt_family = 'db1'
FeatureCfg.wt_level = 3
FeatureCfg.beat_winL = BaseCfg.beat_winL
FeatureCfg.beat_winR = BaseCfg.beat_winR
FeatureCfg.wt_feature_len = pywt.wavedecn_shapes(
    shape=(1+FeatureCfg.beat_winL+FeatureCfg.beat_winR,), 
    wavelet=FeatureCfg.wt_family,
    level=FeatureCfg.wt_level
)[0][0]

FeatureCfg.rr_local_range = 10  # 10 r peaks
FeatureCfg.rr_global_range = 5 * 60 * FeatureCfg.fs  # 5min, units in number of points
FeatureCfg.rr_normalize_radius = 30  # number of beats (rpeaks)

FeatureCfg.morph_intervals = [[0,45], [85,95], [110,120], [170,200]]



ModelCfg = ED()
ModelCfg.fs = BaseCfg.fs
ModelCfg.n_leads = 1
ModelCfg.torch_dtype = BaseCfg.torch_dtype

ModelCfg.crnn = ED()
ModelCfg.crnn.fs = BaseCfg.fs
ModelCfg.crnn.n_leads = 1
ModelCfg.crnn.torch_dtype = BaseCfg.torch_dtype
ModelCfg.crnn.classes = deepcopy(BaseCfg.classes)
ModelCfg.crnn.class_map = deepcopy(BaseCfg.class_map)

ModelCfg.crnn.cnn = ED()
ModelCfg.crnn.cnn.name = 'multi_scopic'  # resnet, resnet_gc, vgg, cpsc2018, etc.
ModelCfg.crnn.cnn.multi_scopic = ED()
ModelCfg.crnn.cnn.multi_scopic.groups = 1
ModelCfg.crnn.cnn.multi_scopic.scopes = [
    [ # branch 0
        [1,],
        [1,1,],
        [1,1,1,],
        [1,1,1,1],
    ],
    [ # branch 1
        [2,],
        [2,4,],
        [4,8,8,],
        [8,16,16,16],
    ],
    [ # branch 2
        [4,],
        [4,8,],
        [8,16,32,],
        [16,32,64,128],
    ],
]
# TODO:
# as sampling frequencies of CPSC2019 and CINC2020 are 500Hz
# while CPSC2020 is 400 Hz
# should the filter_lengths be adjusted?
ModelCfg.crnn.cnn.multi_scopic.filter_lengths = [
    [11, 7, 5, 5],  # branch 0
    [11, 7, 5, 5],  # branch 1
    [11, 7, 5, 5],  # branch 2
]
ModelCfg.crnn.cnn.multi_scopic.subsample_lengths = \
    list(repeat(2, len(ModelCfg.crnn.cnn.multi_scopic.scopes)))
_base_num_filters = 8
ModelCfg.crnn.cnn.multi_scopic.num_filters = [
    [ # branch 0
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
        _base_num_filters*32,
    ],
    [ # branch 1
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
        _base_num_filters*32,
    ],
    [ # branch 2
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
        _base_num_filters*32,
    ],
]
ModelCfg.crnn.cnn.multi_scopic.dropouts = [
    [0, 0.2, 0.2, 0],  # branch 0
    [0, 0.2, 0.2, 0],  # branch 1
    [0, 0.2, 0.2, 0],  # branch 2
]
ModelCfg.crnn.cnn.multi_scopic.bias = True
ModelCfg.crnn.cnn.multi_scopic.kernel_initializer = "he_normal"
ModelCfg.crnn.cnn.multi_scopic.kw_initializer = {}
ModelCfg.crnn.cnn.multi_scopic.activation = "relu"
ModelCfg.crnn.cnn.multi_scopic.kw_activation = {"inplace": True}
ModelCfg.crnn.cnn.multi_scopic.block = ED()
ModelCfg.crnn.cnn.multi_scopic.block.subsample_mode = 'max'  # or 'conv', 'avg', 'nearest', 'linear', 'bilinear'
ModelCfg.crnn.cnn.multi_scopic.block.bias = \
    ModelCfg.crnn.cnn.multi_scopic.bias
ModelCfg.crnn.cnn.multi_scopic.block.kernel_initializer = \
    ModelCfg.crnn.cnn.multi_scopic.kernel_initializer
ModelCfg.crnn.cnn.multi_scopic.block.kw_initializer = \
    deepcopy(ModelCfg.crnn.cnn.multi_scopic.kw_initializer)
ModelCfg.crnn.cnn.multi_scopic.block.activation = \
    ModelCfg.crnn.cnn.multi_scopic.activation
ModelCfg.crnn.cnn.multi_scopic.block.kw_activation = \
    deepcopy(ModelCfg.crnn.cnn.multi_scopic.kw_activation)

# rnn part
# abuse of notation
ModelCfg.crnn.rnn = ED()
ModelCfg.crnn.rnn.name = 'linear'  # 'none', 'lstm', 'attention'
ModelCfg.crnn.rnn.linear = ED()
ModelCfg.crnn.rnn.linear.out_channels = [
    256, 64,
]
ModelCfg.crnn.rnn.linear.bias = True
ModelCfg.crnn.rnn.linear.dropouts = 0.2
ModelCfg.crnn.rnn.linear.activation = 'mish'

# ModelCfg.crnn.rnn.lstm = deepcopy(lstm)
# ModelCfg.crnn.rnn.attention = deepcopy(attention)
# ModelCfg.crnn.rnn.linear = deepcopy(linear)

# global pooling
# currently is fixed using `AdaptiveMaxPool1d`
ModelCfg.crnn.global_pool = 'max'  # 'avg', 'attentive'


ModelCfg.seq_lab = ED()
ModelCfg.seq_lab.fs = BaseCfg.fs
ModelCfg.seq_lab.n_leads = 1
ModelCfg.seq_lab.torch_dtype = BaseCfg.torch_dtype
ModelCfg.seq_lab.classes = [c for c in BaseCfg.classes if c != "N"]
ModelCfg.seq_lab.class_map = {c:v for c,v in BaseCfg.class_map.items() if c != "N"}

ModelCfg.seq_lab.cnn = ModelCfg.crnn.cnn.copy()

ModelCfg.seq_lab.rnn = ED()
# ModelCfg.seq_lab.rnn.name = 'lstm'  # 'none'
ModelCfg.seq_lab.rnn.name = 'none'  # 'lstm'
# ModelCfg.seq_lab.rnn.lstm = ED()
# ModelCfg.seq_lab.rnn.lstm.hidden_sizes = [256, 256]
# ModelCfg.seq_lab.rnn.lstm.bias = True
# ModelCfg.seq_lab.rnn.lstm.dropout = 0
# ModelCfg.seq_lab.rnn.lstm.bidirectional = True
ModelCfg.seq_lab.attn = ED()
ModelCfg.seq_lab.attn.name = 'se'  # 'gc'
ModelCfg.seq_lab.attn.se = ED()
ModelCfg.seq_lab.attn.se.reduction = 16  # not including the last linear layer
ModelCfg.seq_lab.attn.se.activation = "relu"
ModelCfg.seq_lab.attn.se.kw_activation = ED(inplace=True)
ModelCfg.seq_lab.attn.se.bias = True
ModelCfg.seq_lab.attn.se.kernel_initializer = 'he_normal'

ModelCfg.seq_lab.clf = ED()
ModelCfg.seq_lab.clf.out_channels = [256, 64]  # not including the last linear layer
ModelCfg.seq_lab.clf.activation = "mish"
ModelCfg.seq_lab.clf.bias = True
ModelCfg.seq_lab.clf.kernel_initializer = 'he_normal'
ModelCfg.seq_lab.clf.dropouts = [0.2, 0.2, 0.0]



TrainCfg = ED()
TrainCfg.fs = ModelCfg.fs
TrainCfg.n_leads = 1
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = os.path.join(_BASE_DIR, 'log')
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 50
TrainCfg.input_len = int(30 * TrainCfg.fs)  # 30 s
TrainCfg.overlap_len = int(20 * TrainCfg.fs)  # 20 s
TrainCfg.bias_thr = BaseCfg.bias_thr

TrainCfg.test_rec_num = 1

TrainCfg.normalize_data = True

# data augmentation
TrainCfg.label_smoothing = 0.1
TrainCfg.random_mask = int(TrainCfg.fs * 0.0)  # 1.0s, 0 for no masking
TrainCfg.stretch_compress = 5  # stretch or compress in time axis, units in percentage (0 - inf)
TrainCfg.random_normalize = True  # (re-)normalize to random mean and std
# valid segments has
# median of mean appr. 0, mean of mean 0.038
# median of std 0.13, mean of std 0.18
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
# TODO: explore and add more data augmentations

TrainCfg.seq_lab_reduction = 2**4  # TODO: automatic adjust via model config

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 200
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd"

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

TrainCfg.lr_scheduler = None  # 'plateau', 'burn_in', 'step', None

# model selection
TrainCfg.model_name = "crnn"  # "seq_lab", "unet"
TrainCfg.classes = deepcopy(ModelCfg[TrainCfg.model_name].classes)
TrainCfg.class_map = deepcopy(ModelCfg[TrainCfg.model_name].class_map)

# configs of loss function
TrainCfg.loss = 'BCEWithLogitsLoss'
# TrainCfg.loss = 'BCEWithLogitsWithClassWeightLoss'

TrainCfg.log_step = 20
TrainCfg.eval_every = 20



PlotCfg = ED()
PlotCfg.winL = 0.06  # second
PlotCfg.winR = 0.08  # second
