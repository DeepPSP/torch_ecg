"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants

"Brady", "LAD", "RAD", "PR", "LQRSV" are treated exceptionally, as special classes
"""

from copy import deepcopy
from typing import List, NoReturn

from ....cfg import CFG, DEFAULTS
from ...aux_data.cinc2021_aux_data import get_class_weight

__all__ = [
    "CINC2021TrainCfg",
]


# settings from official repo
twelve_leads = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
four_leads = ("I", "II", "III", "V2")
three_leads = ("I", "II", "V2")
two_leads = ("I", "II")


# special classes using special detectors
# _SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]
_SPECIAL_CLASSES = []
_NAME = "cinc2021"


def _assign_classes(cfg: CFG, special_classes: List[str]) -> NoReturn:
    """ """
    cfg.special_classes = deepcopy(special_classes)
    cfg.tranche_class_weights = CFG(
        {
            t: get_class_weight(
                t,
                exclude_classes=cfg.special_classes,
                scored_only=True,
                threshold=20,
                min_weight=cfg.min_class_weight,
            )
            for t in [
                "A",
                "B",
                "AB",
                "E",
                "F",
                "G",
            ]
        }
    )
    cfg.tranche_classes = CFG(
        {t: sorted(list(t_cw.keys())) for t, t_cw in cfg.tranche_class_weights.items()}
    )

    cfg.class_weights = get_class_weight(
        tranches="ABEFG",
        exclude_classes=cfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=cfg.min_class_weight,
    )
    cfg.classes = sorted(list(cfg.class_weights.keys()))


# training configurations for machine learning and deep learning

CINC2021TrainCfg = CFG()
CINC2021TrainCfg.torch_dtype = DEFAULTS.torch_dtype

# configs of files
CINC2021TrainCfg.db_dir = None
CINC2021TrainCfg.log_dir = DEFAULTS.log_dir / _NAME
CINC2021TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
CINC2021TrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
CINC2021TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
CINC2021TrainCfg.model_dir = DEFAULTS.model_dir / _NAME
CINC2021TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)

CINC2021TrainCfg.final_model_name = None
CINC2021TrainCfg.keep_checkpoint_max = 20

CINC2021TrainCfg.leads = deepcopy(twelve_leads)

# configs of training data
CINC2021TrainCfg.fs = 500
CINC2021TrainCfg.data_format = "channel_first"

CINC2021TrainCfg.train_ratio = 0.8
CINC2021TrainCfg.min_class_weight = 0.5
CINC2021TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F", "G"

# assign classes, class weights, tranche classes, etc.
_assign_classes(CINC2021TrainCfg, _SPECIAL_CLASSES)

# configs of signal preprocessing
CINC2021TrainCfg.normalize = CFG(
    method="z-score",
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
CINC2021TrainCfg.bandpass = None
# CINC2021TrainCfg.bandpass = CFG(
#     lowcut=0.5,
#     highcut=60,
# )

# configs of data aumentation
# CINC2021TrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
CINC2021TrainCfg.label_smooth = False
CINC2021TrainCfg.random_masking = False
CINC2021TrainCfg.stretch_compress = False  # stretch or compress in time axis
CINC2021TrainCfg.mixup = CFG(
    prob=0.6,
    alpha=0.3,
)

# configs of training epochs, batch, etc.
CINC2021TrainCfg.n_epochs = 50
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
CINC2021TrainCfg.batch_size = 64
# CINC2021TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
CINC2021TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
CINC2021TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
CINC2021TrainCfg.betas = (
    0.9,
    0.999,
)  # default values for corresponding PyTorch optimizers
CINC2021TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

CINC2021TrainCfg.learning_rate = 1e-4  # 1e-3
CINC2021TrainCfg.lr = CINC2021TrainCfg.learning_rate

CINC2021TrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
CINC2021TrainCfg.lr_step_size = 50
CINC2021TrainCfg.lr_gamma = 0.1
CINC2021TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

CINC2021TrainCfg.burn_in = 400
CINC2021TrainCfg.steps = [5000, 10000]

CINC2021TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
CINC2021TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
CINC2021TrainCfg.early_stopping.patience = 10

# configs of loss function
# CINC2021TrainCfg.loss = "BCEWithLogitsLoss"
# CINC2021TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
CINC2021TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
CINC2021TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
CINC2021TrainCfg.flooding_level = (
    0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2021?
)

CINC2021TrainCfg.monitor = "challenge_metric"

CINC2021TrainCfg.log_step = 20
CINC2021TrainCfg.eval_every = 20

# configs of model selection
# "resnet_nature_comm_se", "multi_scopic_leadwise", "vgg16", "vgg16_leadwise",
CINC2021TrainCfg.cnn_name = "resnet_nature_comm_bottle_neck_se"
CINC2021TrainCfg.rnn_name = "none"  # "none", "lstm"
CINC2021TrainCfg.attn_name = "none"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
CINC2021TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `CINC2021TrainCfg.input_len`
CINC2021TrainCfg.input_len_tol = int(0.2 * CINC2021TrainCfg.input_len)
CINC2021TrainCfg.sig_slice_tol = 0.4  # None, do no slicing
CINC2021TrainCfg.siglen = CINC2021TrainCfg.input_len


# constants for model inference
CINC2021TrainCfg.bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
CINC2021TrainCfg.bin_pred_look_again_tol = 0.03
CINC2021TrainCfg.bin_pred_nsr_thr = 0.1
