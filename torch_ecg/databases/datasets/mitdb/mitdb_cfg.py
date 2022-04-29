"""
"""

from copy import deepcopy

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
)  # noqa: F401


__all__ = [
    "MITDBTrainCfg",
]


_NAME = "mitdb"


MITDBTrainCfg = CFG()

MITDBTrainCfg.db_dir = None
MITDBTrainCfg.log_dir = DEFAULTS.log_dir / _NAME
MITDBTrainCfg.model_dir = DEFAULTS.model_dir / _NAME
MITDBTrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
MITDBTrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
MITDBTrainCfg.model_dir.mkdir(parents=True, exist_ok=True)
MITDBTrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
# MITDBTrainCfg.test_data_dir = _BASE_DIR / "working_dir" / "sample_data"

MITDBTrainCfg.fs = 360
MITDBTrainCfg.n_leads = 2  # or 1
MITDBTrainCfg.torch_dtype = DEFAULTS.torch_dtype

MITDBTrainCfg.beat_types = [
    "N",
    "L",
    "R",
    "V",
    "/",
    "A",
    "f",
    "F",
    "j",
    "a",
    "E",
    "J",
    "Q",
]  # "e", "S" not included, since each of them has only one record
MITDBTrainCfg.rhythm_types = [
    "AFIB",
    "AFL",
    "B",
    "IVR",
    "N",
    "NOD",
    "P",
    "SVTA",
    "T",
    "VT",
]  # "AB", "BII", "PREX", "SBR", "VFL" not included, since each of them has only one record


# common confis for all training tasks
MITDBTrainCfg.data_format = "channel_first"

MITDBTrainCfg.keep_checkpoint_max = 20

MITDBTrainCfg.debug = True

# least distance of an valid R peak to two ends of ECG signals
MITDBTrainCfg.rpeaks_dist2border = int(0.5 * MITDBTrainCfg.fs)  # 0.5s
MITDBTrainCfg.qrs_mask_bias = int(0.075 * MITDBTrainCfg.fs)  # bias to rpeaks

# configs of signal preprocessing
# MITDBTrainCfg.normalize = CFG(
#     method="z-score",
#     per_channel=True,
#     mean=0.0,
#     std=1.0,
# )
# frequency band of the filter to apply, should be chosen very carefully
# MITDBTrainCfg.bandpass = None
MITDBTrainCfg.bandpass = CFG(
    lowcut=0.5,
    highcut=45,
    filter_type="fir",
    filter_order=int(0.3 * MITDBTrainCfg.fs),
)

# configs of data aumentation
# MITDBTrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
MITDBTrainCfg.label_smooth = False
MITDBTrainCfg.random_masking = False
MITDBTrainCfg.stretch_compress = False  # stretch or compress in time axis
# MITDBTrainCfg.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )
MITDBTrainCfg.random_flip = CFG(
    per_channel=True,
    prob=[0.4, 0.5],
)

# TODO: explore and add more data augmentations

# configs of training epochs, batch, etc.
MITDBTrainCfg.n_epochs = 30
MITDBTrainCfg.batch_size = 64
MITDBTrainCfg.train_ratio = 0.8

# configs of optimizers and lr_schedulers
MITDBTrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
MITDBTrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
MITDBTrainCfg.betas = (
    0.9,
    0.999,
)  # default values for corresponding PyTorch optimizers
MITDBTrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

MITDBTrainCfg.learning_rate = 1e-4  # 1e-3
MITDBTrainCfg.lr = MITDBTrainCfg.learning_rate

MITDBTrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
MITDBTrainCfg.lr_step_size = 50
MITDBTrainCfg.lr_gamma = 0.1
MITDBTrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

MITDBTrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
MITDBTrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
MITDBTrainCfg.early_stopping.patience = 8

# configs of loss function
# "MaskedBCEWithLogitsLoss", "BCEWithLogitsWithClassWeightLoss"  # "BCELoss"
# MITDBTrainCfg.loss = "AsymmetricLoss"
# MITDBTrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
MITDBTrainCfg.loss = "MaskedBCEWithLogitsLoss"
MITDBTrainCfg.loss_kw = CFG()
MITDBTrainCfg.flooding_level = 0.0  # flooding performed if positive

MITDBTrainCfg.log_step = 40

# tasks of training
MITDBTrainCfg.tasks = [
    "beat_classification",
    "qrs_detection",
    "rhythm_segmentation",
    "af_event",  # segmentation of AF events
    "rr_lstm",
]

# configs of model selection
# "resnet_leadwise", "multi_scopic_leadwise", "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise", etc.

for t in MITDBTrainCfg.tasks:
    MITDBTrainCfg[t] = CFG()


MITDBTrainCfg.beat_classification.final_model_name = None
MITDBTrainCfg.beat_classification.model_name = "crnn"
MITDBTrainCfg.beat_classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
MITDBTrainCfg.beat_classification.rnn_name = "none"
MITDBTrainCfg.beat_classification.attn_name = "se"  # "none", "se", "gc", "nl"
MITDBTrainCfg.beat_classification.winL = int(0.8 * MITDBTrainCfg.fs)
MITDBTrainCfg.beat_classification.winR = int(1.2 * MITDBTrainCfg.fs)
MITDBTrainCfg.beat_classification.input_len = (
    MITDBTrainCfg.beat_classification.winL + MITDBTrainCfg.beat_classification.winR
)
MITDBTrainCfg.beat_classification.classes = deepcopy(MITDBTrainCfg.beat_types)
MITDBTrainCfg.beat_classification.class_map = {
    k: i for i, k in enumerate(MITDBTrainCfg.beat_classification.classes)
}
MITDBTrainCfg.beat_classification.monitor = (
    "f1_measure"  # monitor for determining the best model
)
MITDBTrainCfg.beat_classification.loss = "AsymmetricLoss"  # "BCEWithLogitsLoss"
MITDBTrainCfg.beat_classification.loss_kw = CFG(
    gamma_pos=0, gamma_neg=0.5, implementation="deep-psp"
)

MITDBTrainCfg.qrs_detection.final_model_name = None
MITDBTrainCfg.qrs_detection.model_name = "seq_lab"  # "unet"
MITDBTrainCfg.qrs_detection.reduction = 1
MITDBTrainCfg.qrs_detection.cnn_name = "multi_scopic"
MITDBTrainCfg.qrs_detection.rnn_name = "lstm"  # "none", "lstm"
MITDBTrainCfg.qrs_detection.attn_name = "se"  # "none", "se", "gc", "nl"
MITDBTrainCfg.qrs_detection.input_len = int(30 * MITDBTrainCfg.fs)
MITDBTrainCfg.qrs_detection.overlap_len = int(15 * MITDBTrainCfg.fs)
MITDBTrainCfg.qrs_detection.critical_overlap_len = int(25 * MITDBTrainCfg.fs)
MITDBTrainCfg.qrs_detection.classes = [
    "N",
]
MITDBTrainCfg.qrs_detection.monitor = (
    "qrs_score"  # monitor for determining the best model
)
MITDBTrainCfg.qrs_detection.loss = "BCEWithLogitsLoss"  # "AsymmetricLoss"
MITDBTrainCfg.qrs_detection.loss_kw = CFG()

MITDBTrainCfg.rr_lstm.final_model_name = None
MITDBTrainCfg.rr_lstm.model_name = "lstm"  # "lstm", "lstm_crf"
MITDBTrainCfg.rr_lstm.input_len = 30  # number of rr intervals ( number of rpeaks - 1)
MITDBTrainCfg.rr_lstm.overlap_len = 15  # number of rr intervals ( number of rpeaks - 1)
MITDBTrainCfg.rr_lstm.critical_overlap_len = (
    25  # number of rr intervals ( number of rpeaks - 1)
)
MITDBTrainCfg.rr_lstm.classes = [
    "AFIB",
]
MITDBTrainCfg.rr_lstm.monitor = "neg_masked_bce"  # "rr_score", "neg_masked_bce"  # monitor for determining the best model
MITDBTrainCfg.rr_lstm.loss = "MaskedBCEWithLogitsLoss"
MITDBTrainCfg.rr_lstm.loss_kw = CFG()

for t in ["rhythm_segmentation", "af_event"]:  # segmentation of AF events
    MITDBTrainCfg[t].final_model_name = None
    MITDBTrainCfg[t].model_name = "seq_lab"  # "unet"
    MITDBTrainCfg[t].reduction = 1
    MITDBTrainCfg[t].cnn_name = "multi_scopic"
    MITDBTrainCfg[t].rnn_name = "lstm"  # "none", "lstm"
    MITDBTrainCfg[t].attn_name = "se"  # "none", "se", "gc", "nl"
    MITDBTrainCfg[t].input_len = int(30 * MITDBTrainCfg.fs)
    MITDBTrainCfg[t].overlap_len = int(15 * MITDBTrainCfg.fs)
    MITDBTrainCfg[t].critical_overlap_len = int(25 * MITDBTrainCfg.fs)
    MITDBTrainCfg[
        t
    ].monitor = "neg_masked_bce"  # "main_score", "neg_masked_bce"  # monitor for determining the best model
    # MITDBTrainCfg[t].loss = "AsymmetricLoss" # "MaskedBCEWithLogitsLoss"
    # MITDBTrainCfg[t].loss_kw = CFG(gamma_pos=0, gamma_neg=1, implementation="deep-psp")
    MITDBTrainCfg[t].loss = "MaskedBCEWithLogitsLoss"  # "MaskedBCEWithLogitsLoss"
    MITDBTrainCfg[t].loss_kw = CFG()

MITDBTrainCfg.rhythm_segmentation.classes = deepcopy(MITDBTrainCfg.rhythm_types)
MITDBTrainCfg.rhythm_segmentation.class_map = {
    k: i for i, k in enumerate(MITDBTrainCfg.rhythm_segmentation.classes)
}
MITDBTrainCfg.af_event.classes = ["AFIB"]
