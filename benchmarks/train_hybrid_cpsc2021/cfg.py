"""
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
    ECG_SEQ_LAB_NET_CONFIG,
    ECG_SUBTRACT_UNET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
    RR_AF_CRF_CONFIG,
    RR_AF_VANILLA_CONFIG,
    RR_LSTM_CONFIG,
    attention,
    densenet_leadwise,
    global_context,
    linear,
    lstm,
    multi_scopic,
    multi_scopic_block,
    multi_scopic_leadwise,
    non_local,
    resnet_block_basic,
    resnet_block_basic_gc,
    resnet_block_basic_se,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_se,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_se,
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    squeeze_excitation,
    tresnetF,
    tresnetM,
    tresnetN,
    tresnetP,
    tresnetS,
    vgg16,
    vgg16_leadwise,
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    xception_leadwise,
)

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
    "PlotCfg",
]


_BASE_DIR = Path(__file__).parent.absolute()


BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.model_dir.mkdir(parents=True, exist_ok=True)
BaseCfg.test_data_dir = _BASE_DIR / "working_dir" / "sample_data"
BaseCfg.fs = 200
BaseCfg.n_leads = 2
BaseCfg.torch_dtype = DEFAULTS.torch_dtype

BaseCfg.class_fn2abbr = {  # fullname to abbreviation
    "non atrial fibrillation": "N",
    "paroxysmal atrial fibrillation": "AFp",
    "persistent atrial fibrillation": "AFf",
}
BaseCfg.class_abbr2fn = {v: k for k, v in BaseCfg.class_fn2abbr.items()}
BaseCfg.class_fn_map = {  # fullname to number
    "non atrial fibrillation": 0,
    "paroxysmal atrial fibrillation": 2,
    "persistent atrial fibrillation": 1,
}
BaseCfg.class_abbr_map = {
    k: BaseCfg.class_fn_map[v] for k, v in BaseCfg.class_abbr2fn.items()
}

BaseCfg.bias_thr = (
    0.15 * BaseCfg.fs
)  # rhythm change annotations onsets or offset of corresponding R peaks
BaseCfg.beat_ann_bias_thr = 0.1 * BaseCfg.fs  # half width of broad qrs complex
BaseCfg.beat_winL = 250 * BaseCfg.fs // 1000  # corr. to 250 ms
BaseCfg.beat_winR = 250 * BaseCfg.fs // 1000  # corr. to 250 ms


TrainCfg = CFG()

# common confis for all training tasks
TrainCfg.fs = BaseCfg.fs
TrainCfg.n_leads = BaseCfg.n_leads
TrainCfg.data_format = "channel_first"
TrainCfg.torch_dtype = BaseCfg.torch_dtype

TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.model_dir = BaseCfg.model_dir
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
TrainCfg.keep_checkpoint_max = 20

TrainCfg.debug = True

# least distance of an valid R peak to two ends of ECG signals
TrainCfg.rpeaks_dist2border = int(0.5 * TrainCfg.fs)  # 0.5s
TrainCfg.qrs_mask_bias = int(0.075 * TrainCfg.fs)  # bias to rpeaks

# configs of signal preprocessing
TrainCfg.normalize = CFG(
    method="z-score",
    per_channel=True,
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
# TrainCfg.bandpass = None
TrainCfg.bandpass = CFG(
    lowcut=0.5,
    highcut=45,
    filter_type="fir",
    filter_order=int(0.3 * TrainCfg.fs),
)

# configs of data aumentation
# TrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
TrainCfg.label_smooth = False
TrainCfg.random_masking = False
TrainCfg.stretch_compress = False  # stretch or compress in time axis
# TrainCfg.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )
TrainCfg.random_flip = CFG(
    per_channel=True,
    prob=[0.4, 0.5],
)

# TODO: explore and add more data augmentations

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 30
TrainCfg.batch_size = 64
TrainCfg.train_ratio = 0.8

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 1e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = 8

# configs of loss function
# "MaskedBCEWithLogitsLoss", "BCEWithLogitsWithClassWeightLoss"  # "BCELoss"
# TrainCfg.loss = "AsymmetricLoss"
# TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
TrainCfg.loss = "MaskedBCEWithLogitsLoss"
TrainCfg.loss_kw = CFG()
TrainCfg.flooding_level = 0.0  # flooding performed if positive

TrainCfg.log_step = 40

# tasks of training
TrainCfg.tasks = [
    "qrs_detection",
    "rr_lstm",
    "main",
]

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise", etc.

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

TrainCfg.qrs_detection.final_model_name = None
TrainCfg.qrs_detection.model_name = "seq_lab"  # "unet"
TrainCfg.qrs_detection.reduction = 1
TrainCfg.qrs_detection.cnn_name = "multi_scopic"
TrainCfg.qrs_detection.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.qrs_detection.attn_name = "se"  # "none", "se", "gc", "nl"
TrainCfg.qrs_detection.input_len = int(30 * TrainCfg.fs)
TrainCfg.qrs_detection.overlap_len = int(15 * TrainCfg.fs)
TrainCfg.qrs_detection.critical_overlap_len = int(25 * TrainCfg.fs)
TrainCfg.qrs_detection.classes = [
    "N",
]
TrainCfg.qrs_detection.monitor = "qrs_score"  # monitor for determining the best model
TrainCfg.qrs_detection.loss = "BCEWithLogitsLoss"  # "AsymmetricLoss"
TrainCfg.qrs_detection.loss_kw = CFG()

TrainCfg.rr_lstm.final_model_name = None
TrainCfg.rr_lstm.model_name = "lstm"  # "lstm", "lstm_crf"
TrainCfg.rr_lstm.input_len = 30  # number of rr intervals ( number of rpeaks - 1)
TrainCfg.rr_lstm.overlap_len = 15  # number of rr intervals ( number of rpeaks - 1)
TrainCfg.rr_lstm.critical_overlap_len = (
    25  # number of rr intervals ( number of rpeaks - 1)
)
TrainCfg.rr_lstm.classes = [
    "af",
]
TrainCfg.rr_lstm.monitor = "neg_masked_bce"  # "rr_score", "neg_masked_bce"  # monitor for determining the best model
TrainCfg.rr_lstm.loss = "MaskedBCEWithLogitsLoss"
TrainCfg.rr_lstm.loss_kw = CFG()

TrainCfg.main.final_model_name = None
TrainCfg.main.model_name = "seq_lab"  # "unet"
TrainCfg.main.reduction = 1
TrainCfg.main.cnn_name = "multi_scopic"
TrainCfg.main.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.main.attn_name = "se"  # "none", "se", "gc", "nl"
TrainCfg.main.input_len = int(30 * TrainCfg.fs)
TrainCfg.main.overlap_len = int(15 * TrainCfg.fs)
TrainCfg.main.critical_overlap_len = int(25 * TrainCfg.fs)
TrainCfg.main.classes = [
    "af",
]
TrainCfg.main.monitor = "neg_masked_bce"  # "main_score", "neg_masked_bce"  # monitor for determining the best model
# TrainCfg.main.loss = "AsymmetricLoss" # "MaskedBCEWithLogitsLoss"
# TrainCfg.main.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
TrainCfg.main.loss = "MaskedBCEWithLogitsLoss"  # "MaskedBCEWithLogitsLoss"
TrainCfg.main.loss_kw = CFG()


# Plan:
# R-peak detection using UNets, sequence labelling,
# main task via RR-LSTM using sequence of R peaks as input
# main task via UNets, sequence labelling using raw ECGs

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = BaseCfg.fs
_BASE_MODEL_CONFIG.n_leads = BaseCfg.n_leads

ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t


ModelCfg.qrs_detection.input_len = TrainCfg.qrs_detection.input_len
ModelCfg.qrs_detection.classes = TrainCfg.qrs_detection.classes
ModelCfg.qrs_detection.model_name = TrainCfg.qrs_detection.model_name
ModelCfg.qrs_detection.cnn_name = TrainCfg.qrs_detection.cnn_name
ModelCfg.qrs_detection.rnn_name = TrainCfg.qrs_detection.rnn_name
ModelCfg.qrs_detection.attn_name = TrainCfg.qrs_detection.attn_name

# the following is a comprehensive choices for different choices of qrs_detection task
ModelCfg.qrs_detection.seq_lab = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
ModelCfg.qrs_detection.seq_lab.fs = BaseCfg.fs
ModelCfg.qrs_detection.seq_lab.reduction = TrainCfg.qrs_detection.reduction
ModelCfg.qrs_detection.seq_lab.cnn.name = ModelCfg.qrs_detection.cnn_name
ModelCfg.qrs_detection.seq_lab.rnn.name = ModelCfg.qrs_detection.rnn_name
ModelCfg.qrs_detection.seq_lab.attn.name = ModelCfg.qrs_detection.attn_name

ModelCfg.qrs_detection.seq_lab.cnn.multi_scopic.filter_lengths = [
    [5, 5, 3],
    [7, 5, 3],
    [7, 5, 3],
]

ModelCfg.qrs_detection.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)
ModelCfg.qrs_detection.unet.fs = BaseCfg.fs


ModelCfg.rr_lstm.input_len = TrainCfg.rr_lstm.input_len
ModelCfg.rr_lstm.classes = TrainCfg.rr_lstm.classes
ModelCfg.rr_lstm.model_name = TrainCfg.rr_lstm.model_name

ModelCfg.rr_lstm.lstm = deepcopy(RR_AF_VANILLA_CONFIG)
ModelCfg.rr_lstm.lstm.global_pool = "none"
ModelCfg.rr_lstm.lstm.attn = CFG()
ModelCfg.rr_lstm.lstm.attn.name = "se"  # "gc"
ModelCfg.rr_lstm.lstm.attn.se = CFG()
ModelCfg.rr_lstm.lstm.attn.se.reduction = 8  # not including the last linear layer
ModelCfg.rr_lstm.lstm.attn.se.activation = "relu"
ModelCfg.rr_lstm.lstm.attn.se.kw_activation = CFG(inplace=True)
ModelCfg.rr_lstm.lstm.attn.se.bias = True
ModelCfg.rr_lstm.lstm.attn.se.kernel_initializer = "he_normal"

ModelCfg.rr_lstm.lstm_crf = deepcopy(RR_AF_CRF_CONFIG)
ModelCfg.rr_lstm.lstm_crf.attn = CFG()
ModelCfg.rr_lstm.lstm_crf.attn.name = "se"  # "gc"
ModelCfg.rr_lstm.lstm_crf.attn.se = CFG()
ModelCfg.rr_lstm.lstm_crf.attn.se.reduction = 8  # not including the last linear layer
ModelCfg.rr_lstm.lstm_crf.attn.se.activation = "relu"
ModelCfg.rr_lstm.lstm_crf.attn.se.kw_activation = CFG(inplace=True)
ModelCfg.rr_lstm.lstm_crf.attn.se.bias = True
ModelCfg.rr_lstm.lstm_crf.attn.se.kernel_initializer = "he_normal"

if ModelCfg.rr_lstm[ModelCfg.rr_lstm.model_name].clf.name == "crf":
    TrainCfg.rr_lstm.loss = "BCELoss"

# the following is a comprehensive choices for different choices of rr_lstm task


ModelCfg.main.input_len = TrainCfg.main.input_len
ModelCfg.main.classes = TrainCfg.main.classes
ModelCfg.main.model_name = TrainCfg.main.model_name
ModelCfg.main.cnn_name = TrainCfg.main.cnn_name
ModelCfg.main.rnn_name = TrainCfg.main.rnn_name
ModelCfg.main.attn_name = TrainCfg.main.attn_name

# the following is a comprehensive choices for different choices of main task
ModelCfg.main.seq_lab = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
ModelCfg.main.seq_lab.fs = BaseCfg.fs
ModelCfg.main.seq_lab.reduction = TrainCfg.main.reduction
ModelCfg.main.seq_lab.cnn.name = ModelCfg.main.cnn_name
ModelCfg.main.seq_lab.rnn.name = ModelCfg.main.rnn_name
ModelCfg.main.seq_lab.attn.name = ModelCfg.main.attn_name

ModelCfg.main.seq_lab.cnn.multi_scopic.filter_lengths = [
    [3, 3, 3],
    [5, 5, 3],
    [9, 7, 5],
]

ModelCfg.main.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)
ModelCfg.main.unet.fs = BaseCfg.fs
ModelCfg.main.unet.reduction = 1
ModelCfg.main.unet.init_num_filters = 16  # keep the same with n_classes
ModelCfg.main.unet.down_num_filters = [
    ModelCfg.main.unet.init_num_filters * (2**idx)
    for idx in range(1, ModelCfg.main.unet.down_up_block_num + 1)
]
ModelCfg.main.unet.up_num_filters = [
    ModelCfg.main.unet.init_num_filters * (2**idx)
    for idx in range(ModelCfg.main.unet.down_up_block_num - 1, -1, -1)
]
ModelCfg.main.unet.up_mode = "interp"  # "deconv"


# configurations for visualization
PlotCfg = CFG()
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
