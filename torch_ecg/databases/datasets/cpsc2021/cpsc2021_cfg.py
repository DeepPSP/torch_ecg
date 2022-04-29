"""
"""

from ....cfg import CFG, DEFAULTS
from ....model_configs import (  # noqa: F401
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
    "CPSC2021TrainCfg",
]


_NAME = "cpsc2021"


CPSC2021TrainCfg = CFG()

CPSC2021TrainCfg.db_dir = None
CPSC2021TrainCfg.log_dir = DEFAULTS.log_dir / _NAME
CPSC2021TrainCfg.model_dir = DEFAULTS.model_dir / _NAME
CPSC2021TrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
CPSC2021TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
CPSC2021TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)
CPSC2021TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
# CPSC2021TrainCfg.test_data_dir = _BASE_DIR / "working_dir" / "sample_data"

CPSC2021TrainCfg.fs = 200
CPSC2021TrainCfg.n_leads = 2
CPSC2021TrainCfg.torch_dtype = DEFAULTS.torch_dtype

CPSC2021TrainCfg.class_fn2abbr = {  # fullname to abbreviation
    "non atrial fibrillation": "N",
    "paroxysmal atrial fibrillation": "AFp",
    "persistent atrial fibrillation": "AFf",
}
CPSC2021TrainCfg.class_abbr2fn = {
    v: k for k, v in CPSC2021TrainCfg.class_fn2abbr.items()
}
CPSC2021TrainCfg.class_fn_map = {  # fullname to number
    "non atrial fibrillation": 0,
    "paroxysmal atrial fibrillation": 2,
    "persistent atrial fibrillation": 1,
}
CPSC2021TrainCfg.class_abbr_map = {
    k: CPSC2021TrainCfg.class_fn_map[v]
    for k, v in CPSC2021TrainCfg.class_abbr2fn.items()
}

CPSC2021TrainCfg.bias_thr = (
    0.15 * CPSC2021TrainCfg.fs
)  # rhythm change annotations onsets or offset of corresponding R peaks
CPSC2021TrainCfg.beat_ann_bias_thr = (
    0.1 * CPSC2021TrainCfg.fs
)  # half width of broad qrs complex
CPSC2021TrainCfg.beat_winL = 250 * CPSC2021TrainCfg.fs // 1000  # corr. to 250 ms
CPSC2021TrainCfg.beat_winR = 250 * CPSC2021TrainCfg.fs // 1000  # corr. to 250 ms


# common confis for all training tasks
CPSC2021TrainCfg.data_format = "channel_first"

CPSC2021TrainCfg.keep_checkpoint_max = 20

CPSC2021TrainCfg.debug = True

# least distance of an valid R peak to two ends of ECG signals
CPSC2021TrainCfg.rpeaks_dist2border = int(0.5 * CPSC2021TrainCfg.fs)  # 0.5s
CPSC2021TrainCfg.qrs_mask_bias = int(0.075 * CPSC2021TrainCfg.fs)  # bias to rpeaks

# configs of signal preprocessing
CPSC2021TrainCfg.normalize = CFG(
    method="z-score",
    per_channel=True,
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
# CPSC2021TrainCfg.bandpass = None
CPSC2021TrainCfg.bandpass = CFG(
    lowcut=0.5,
    highcut=45,
    filter_type="fir",
    filter_order=int(0.3 * CPSC2021TrainCfg.fs),
)

# configs of data aumentation
# CPSC2021TrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
CPSC2021TrainCfg.label_smooth = False
CPSC2021TrainCfg.random_masking = False
CPSC2021TrainCfg.stretch_compress = False  # stretch or compress in time axis
# CPSC2021TrainCfg.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )
CPSC2021TrainCfg.random_flip = CFG(
    per_channel=True,
    prob=[0.4, 0.5],
)

# TODO: explore and add more data augmentations

# configs of training epochs, batch, etc.
CPSC2021TrainCfg.n_epochs = 30
CPSC2021TrainCfg.batch_size = 64
CPSC2021TrainCfg.train_ratio = 0.8

# configs of optimizers and lr_schedulers
CPSC2021TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
CPSC2021TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
CPSC2021TrainCfg.betas = (
    0.9,
    0.999,
)  # default values for corresponding PyTorch optimizers
CPSC2021TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

CPSC2021TrainCfg.learning_rate = 1e-4  # 1e-3
CPSC2021TrainCfg.lr = CPSC2021TrainCfg.learning_rate

CPSC2021TrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
CPSC2021TrainCfg.lr_step_size = 50
CPSC2021TrainCfg.lr_gamma = 0.1
CPSC2021TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

CPSC2021TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
CPSC2021TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
CPSC2021TrainCfg.early_stopping.patience = 8

# configs of loss function
# "MaskedBCEWithLogitsLoss", "BCEWithLogitsWithClassWeightLoss"  # "BCELoss"
# CPSC2021TrainCfg.loss = "AsymmetricLoss"
# CPSC2021TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
CPSC2021TrainCfg.loss = "MaskedBCEWithLogitsLoss"
CPSC2021TrainCfg.loss_kw = CFG()
CPSC2021TrainCfg.flooding_level = 0.0  # flooding performed if positive

CPSC2021TrainCfg.log_step = 40

# tasks of training
CPSC2021TrainCfg.tasks = [
    "qrs_detection",
    "rr_lstm",
    "main",
]

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise", etc.

for t in CPSC2021TrainCfg.tasks:
    CPSC2021TrainCfg[t] = CFG()

CPSC2021TrainCfg.qrs_detection.final_model_name = None
CPSC2021TrainCfg.qrs_detection.model_name = "seq_lab"  # "unet"
CPSC2021TrainCfg.qrs_detection.reduction = 1
CPSC2021TrainCfg.qrs_detection.cnn_name = "multi_scopic"
CPSC2021TrainCfg.qrs_detection.rnn_name = "lstm"  # "none", "lstm"
CPSC2021TrainCfg.qrs_detection.attn_name = "se"  # "none", "se", "gc", "nl"
CPSC2021TrainCfg.qrs_detection.input_len = int(30 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.qrs_detection.overlap_len = int(15 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.qrs_detection.critical_overlap_len = int(25 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.qrs_detection.classes = [
    "N",
]
CPSC2021TrainCfg.qrs_detection.monitor = (
    "qrs_score"  # monitor for determining the best model
)
CPSC2021TrainCfg.qrs_detection.loss = "BCEWithLogitsLoss"  # "AsymmetricLoss"
CPSC2021TrainCfg.qrs_detection.loss_kw = CFG()

CPSC2021TrainCfg.rr_lstm.final_model_name = None
CPSC2021TrainCfg.rr_lstm.model_name = "lstm"  # "lstm", "lstm_crf"
CPSC2021TrainCfg.rr_lstm.input_len = (
    30  # number of rr intervals ( number of rpeaks - 1)
)
CPSC2021TrainCfg.rr_lstm.overlap_len = (
    15  # number of rr intervals ( number of rpeaks - 1)
)
CPSC2021TrainCfg.rr_lstm.critical_overlap_len = (
    25  # number of rr intervals ( number of rpeaks - 1)
)
CPSC2021TrainCfg.rr_lstm.classes = [
    "af",
]
CPSC2021TrainCfg.rr_lstm.monitor = "neg_masked_bce"  # "rr_score", "neg_masked_bce"  # monitor for determining the best model
CPSC2021TrainCfg.rr_lstm.loss = "MaskedBCEWithLogitsLoss"
CPSC2021TrainCfg.rr_lstm.loss_kw = CFG()

CPSC2021TrainCfg.main.final_model_name = None
CPSC2021TrainCfg.main.model_name = "seq_lab"  # "unet"
CPSC2021TrainCfg.main.reduction = 1
CPSC2021TrainCfg.main.cnn_name = "multi_scopic"
CPSC2021TrainCfg.main.rnn_name = "lstm"  # "none", "lstm"
CPSC2021TrainCfg.main.attn_name = "se"  # "none", "se", "gc", "nl"
CPSC2021TrainCfg.main.input_len = int(30 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.main.overlap_len = int(15 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.main.critical_overlap_len = int(25 * CPSC2021TrainCfg.fs)
CPSC2021TrainCfg.main.classes = [
    "af",
]
CPSC2021TrainCfg.main.monitor = "neg_masked_bce"  # "main_score", "neg_masked_bce"  # monitor for determining the best model
# CPSC2021TrainCfg.main.loss = "AsymmetricLoss" # "MaskedBCEWithLogitsLoss"
# CPSC2021TrainCfg.main.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
CPSC2021TrainCfg.main.loss = "MaskedBCEWithLogitsLoss"  # "MaskedBCEWithLogitsLoss"
CPSC2021TrainCfg.main.loss_kw = CFG()
