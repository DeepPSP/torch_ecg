"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants

all classes are treated using the same (deep learning) method uniformly, i.e. no special classes

NEW in CinC2021 compared to CinC2020
"""
import os
from copy import deepcopy
from itertools import repeat

import numpy as np
from easydict import EasyDict as ED

from .scoring_aux_data import (
    equiv_class_dict,
    get_class_weight,
)
from torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG
from torch_ecg.model_configs.cnn import (
    vgg_block_basic, vgg_block_mish, vgg_block_swish,
    vgg16, vgg16_leadwise,
    resnet_block_stanford, resnet_stanford,
    resnet_block_basic, resnet_bottle_neck,
    resnet, resnet_leadwise,
    multi_scopic_block,
    multi_scopic, multi_scopic_leadwise,
    dense_net_leadwise,
    xception_leadwise,
)
from torch_ecg.model_configs.rnn import (
    lstm,
    attention,
    linear,
)
from torch_ecg.model_configs.attn import (
    non_local,
    squeeze_excitation,
    global_context,
)


__all__ = [
    "BaseCfg",
    "PlotCfg",
    "SpecialDetectorCfg",
    "ModelCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONE_MINUTE_IN_MS = 60 * 1000


# names of the 12 leads
Standard12Leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]
InferiorLeads = ["II", "III", "aVF",]
LateralLeads = ["I", "aVL",] + [f"V{i}" for i in range(5,7)]
SeptalLeads = ["aVR", "V1",]
AnteriorLeads = [f"V{i}" for i in range(2,5)]
ChestLeads = [f"V{i}" for i in range(1, 7)]
PrecordialLeads = ChestLeads
LimbLeads = ["I", "II", "III", "aVR", "aVL", "aVF",]


# settings from official repo
twelve_leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
four_leads = ("I", "II", "III", "V2")
three_leads = ("I", "II", "V2")
two_leads = ("I", "II")
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)



BaseCfg = ED()
# BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2021/"
# BaseCfg.db_dir = "/home/taozi/Data/CinC2021/All_training_WFDB/"
BaseCfg.db_dir = "/home/wenh06/Jupyter/data/CinC2021/"
BaseCfg.log_dir = os.path.join(_BASE_DIR, "log")
BaseCfg.model_dir = os.path.join(_BASE_DIR, "saved_models")
os.makedirs(BaseCfg.log_dir, exist_ok=True)
os.makedirs(BaseCfg.model_dir, exist_ok=True)
BaseCfg.fs = 500
BaseCfg.torch_dtype = "float"  # "double"



SpecialDetectorCfg = ED()
SpecialDetectorCfg.leads_ordering = deepcopy(Standard12Leads)
SpecialDetectorCfg.pr_fs_lower_bound = 47  # Hz
SpecialDetectorCfg.pr_spike_mph_ratio = 15  # ratio to the average amplitude of the signal
SpecialDetectorCfg.pr_spike_mpd = 300  # ms
SpecialDetectorCfg.pr_spike_prominence = 0.3
SpecialDetectorCfg.pr_spike_prominence_wlen = 120  # ms
SpecialDetectorCfg.pr_spike_inv_density_threshold = 2500  # inverse density (1/density), one spike per 2000 ms
SpecialDetectorCfg.pr_spike_leads_threshold = 7 / 12  # proportion
SpecialDetectorCfg.axis_qrs_mask_radius = 70  # ms
SpecialDetectorCfg.axis_method = "2-lead"  # can also be "3-lead"
SpecialDetectorCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
SpecialDetectorCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm
SpecialDetectorCfg.lqrsv_qrs_mask_radius = 60  # ms
SpecialDetectorCfg.lqrsv_ampl_bias = 0.02  # mV, TODO: should be further determined by resolution, etc.
SpecialDetectorCfg.lqrsv_ratio_threshold = 0.8

# special classes using special detectors
# _SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]
_SPECIAL_CLASSES = []



# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60



# training configurations for machine learning and deep learning
TrainCfg = ED()

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir
TrainCfg.final_model_name = None
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
os.makedirs(TrainCfg.checkpoints, exist_ok=True)
TrainCfg.keep_checkpoint_max = 20

TrainCfg.leads = deepcopy(twelve_leads)

# configs of training data
TrainCfg.fs = BaseCfg.fs
TrainCfg.data_format = "channel_first"
TrainCfg.special_classes = deepcopy(_SPECIAL_CLASSES)
TrainCfg.normalize_data = False  # should be False considering the existence of LQRSV
TrainCfg.train_ratio = 0.8
TrainCfg.min_class_weight = 0.5
TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F", "G"

TrainCfg.tranche_class_weights = ED({
    t: get_class_weight(
        t,
        exclude_classes=TrainCfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=TrainCfg.min_class_weight,
    ) for t in ["A", "B", "AB", "E", "F", "G",]
})
TrainCfg.tranche_classes = ED({
    t: sorted(list(t_cw.keys())) \
        for t, t_cw in TrainCfg.tranche_class_weights.items()
})

TrainCfg.class_weights = get_class_weight(
    tranches="ABEFG",
    exclude_classes=TrainCfg.special_classes,
    scored_only=True,
    threshold=20,
    min_weight=TrainCfg.min_class_weight,
)
TrainCfg.classes = sorted(list(TrainCfg.class_weights.keys()))

# configs of signal preprocessing
# frequency band of the filter to apply, should be chosen very carefully
# TrainCfg.bandpass = None  # [-np.inf, 45]
# TrainCfg.bandpass = [-np.inf, 45]
TrainCfg.bandpass = [0.5, 60]
TrainCfg.bandpass_order = 5

# configs of data aumentation
TrainCfg.label_smoothing = 0.1
TrainCfg.random_mask = int(TrainCfg.fs * 0.0)  # 1.0s, 0 for no masking
TrainCfg.stretch_compress = 1.0  # stretch or compress in time axis

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 40
TrainCfg.batch_size = 64
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
TrainCfg.early_stopping.patience = 8

# configs of loss function
# TrainCfg.loss = "BCEWithLogitsLoss"
TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.flooding_level = 0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2021?

TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise"
TrainCfg.cnn_name = "multi_scopic_leadwise"
TrainCfg.rnn_name = "none"  # "none", "lstm"
TrainCfg.attn_name = "se"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `TrainCfg.input_len`
TrainCfg.input_len_tol = int(0.2 * TrainCfg.input_len)
TrainCfg.siglen = TrainCfg.input_len


# constants for model inference
_bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
_bin_pred_look_again_tol = 0.03
_bin_pred_nsr_thr = 0.1


TrainCfg.bin_pred_thr = _bin_pred_thr
TrainCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
TrainCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = BaseCfg.fs
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = _bin_pred_thr
ModelCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
ModelCfg.bin_pred_nsr_thr =_bin_pred_nsr_thr
ModelCfg.special_classes = deepcopy(_SPECIAL_CLASSES)

ModelCfg.dl_classes = deepcopy(TrainCfg.classes)
ModelCfg.dl_siglen = TrainCfg.siglen
ModelCfg.tranche_classes = deepcopy(TrainCfg.tranche_classes)
ModelCfg.full_classes = ModelCfg.dl_classes + ModelCfg.special_classes

ModelCfg.cnn_name = TrainCfg.cnn_name
ModelCfg.rnn_name = TrainCfg.rnn_name
ModelCfg.attn_name = TrainCfg.attn_name


_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)
_BASE_MODEL_CONFIG.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"  # False

# detailed configs for 12-lead, 6-lead, 4-lead, 3-lead, 2-lead models
# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn

ModelCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.twelve_leads.cnn.name = ModelCfg.cnn_name
ModelCfg.twelve_leads.rnn.name = ModelCfg.rnn_name
ModelCfg.twelve_leads.attn.name = ModelCfg.attn_name

# TODO: add adjustifications for "leadwise" configs for 6,4,3,2 leads models
ModelCfg.six_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.six_leads.cnn.name = ModelCfg.cnn_name
ModelCfg.six_leads.rnn.name = ModelCfg.rnn_name
ModelCfg.six_leads.attn.name = ModelCfg.attn_name
ModelCfg.six_leads.cnn.vgg16_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelCfg.six_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]
ModelCfg.six_leads.cnn.resnet_leadwise.groups = 6
ModelCfg.six_leads.cnn.resnet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelCfg.six_leads.cnn.multi_scopic_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelCfg.six_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ModelCfg.six_leads.cnn.dense_net_leadwise.groups = 6
ModelCfg.six_leads.cnn.dense_net_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelCfg.six_leads.cnn.xception_leadwise.groups = 6
_base_num_filters = 6 * 2  # 12 * 2
ModelCfg.six_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters*4, _base_num_filters*8],
    init_filter_lengths=3,
    init_subsample_lengths=[2,1],
    num_filters=[_base_num_filters*16, _base_num_filters*32, _base_num_filters*91],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelCfg.six_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters*91, 8)),
    filter_lengths=3,
)
ModelCfg.six_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters*182, _base_num_filters*256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters*91, _base_num_filters*128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelCfg.four_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.four_leads.cnn.name = ModelCfg.cnn_name
ModelCfg.four_leads.rnn.name = ModelCfg.rnn_name
ModelCfg.four_leads.attn.name = ModelCfg.attn_name
ModelCfg.four_leads.cnn.vgg16_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelCfg.four_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]
ModelCfg.four_leads.cnn.resnet_leadwise.groups = 4
ModelCfg.four_leads.cnn.resnet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelCfg.four_leads.cnn.multi_scopic_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelCfg.four_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ModelCfg.four_leads.cnn.dense_net_leadwise.groups = 4
ModelCfg.four_leads.cnn.dense_net_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelCfg.four_leads.cnn.xception_leadwise.groups = 4
_base_num_filters = 6 * 2  # 12 * 2
ModelCfg.four_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters*4, _base_num_filters*8],
    init_filter_lengths=3,
    init_subsample_lengths=[2,1],
    num_filters=[_base_num_filters*16, _base_num_filters*32, _base_num_filters*91],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelCfg.four_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters*91, 8)),
    filter_lengths=3,
)
ModelCfg.four_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters*182, _base_num_filters*256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters*91, _base_num_filters*128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelCfg.three_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.three_leads.cnn.name = ModelCfg.cnn_name
ModelCfg.three_leads.rnn.name = ModelCfg.rnn_name
ModelCfg.three_leads.attn.name = ModelCfg.attn_name
ModelCfg.three_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelCfg.three_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]
ModelCfg.three_leads.cnn.resnet_leadwise.groups = 3
ModelCfg.three_leads.cnn.resnet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelCfg.three_leads.cnn.multi_scopic_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelCfg.three_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ModelCfg.three_leads.cnn.dense_net_leadwise.groups = 3
ModelCfg.three_leads.cnn.dense_net_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelCfg.three_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 3 * 4  # 12 * 2
ModelCfg.three_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters*4, _base_num_filters*8],
    init_filter_lengths=3,
    init_subsample_lengths=[2,1],
    num_filters=[_base_num_filters*16, _base_num_filters*32, _base_num_filters*91],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelCfg.three_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters*91, 8)),
    filter_lengths=3,
)
ModelCfg.three_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters*182, _base_num_filters*256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters*91, _base_num_filters*128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelCfg.two_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelCfg.two_leads.cnn.name = ModelCfg.cnn_name
ModelCfg.two_leads.rnn.name = ModelCfg.rnn_name
ModelCfg.two_leads.attn.name = ModelCfg.attn_name
ModelCfg.two_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 2 * 12  # 12 * 4
ModelCfg.two_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters*4,
    _base_num_filters*8,
    _base_num_filters*16,
    _base_num_filters*32,
    _base_num_filters*32,
]
ModelCfg.two_leads.cnn.resnet_leadwise.groups = 2
ModelCfg.two_leads.cnn.resnet_leadwise.init_num_filters = 2 * 16  # 12 * 8
ModelCfg.two_leads.cnn.multi_scopic_leadwise.groups = 2
_base_num_filters = 2 * 8  # 12 * 4
ModelCfg.two_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
    [
        _base_num_filters*4,
        _base_num_filters*8,
        _base_num_filters*16,
    ],
]
ModelCfg.two_leads.cnn.dense_net_leadwise.groups = 2
ModelCfg.two_leads.cnn.dense_net_leadwise.init_num_filters = 2 * 12  # 12 * 8
ModelCfg.two_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 2 * 6  # 12 * 2
ModelCfg.two_leads.cnn.xception_vanilla.entry_flow = ED(
    init_num_filters=[_base_num_filters*4, _base_num_filters*8],
    init_filter_lengths=3,
    init_subsample_lengths=[2,1],
    num_filters=[_base_num_filters*16, _base_num_filters*32, _base_num_filters*91],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelCfg.two_leads.cnn.xception_vanilla.middle_flow = ED(
    num_filters=list(repeat(_base_num_filters*91, 8)),
    filter_lengths=3,
)
ModelCfg.two_leads.cnn.xception_vanilla.exit_flow = ED(
    final_num_filters=[_base_num_filters*182, _base_num_filters*256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters*91, _base_num_filters*128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
