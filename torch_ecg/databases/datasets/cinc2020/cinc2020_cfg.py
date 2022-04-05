"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants

"Brady", "LAD", "RAD", "PR", "LQRSV" are treated exceptionally, as special classes

"""

from copy import deepcopy
from typing import List, NoReturn

from ....cfg import CFG, DEFAULTS
from ....utils import ecg_arrhythmia_knowledge as EAK
from ...aux_data.cinc2020_aux_data import get_class_weight

__all__ = [
    "CINC2020TrainCfg",
]


# special classes using special detectors
# _SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]
_SPECIAL_CLASSES = []
_NAME = "cinc2020"


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
            ]
        }
    )
    cfg.tranche_classes = CFG(
        {t: sorted(list(t_cw.keys())) for t, t_cw in cfg.tranche_class_weights.items()}
    )

    cfg.class_weights = get_class_weight(
        tranches="ABEF",
        exclude_classes=cfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=cfg.min_class_weight,
    )
    cfg.classes = sorted(list(cfg.class_weights.keys()))


# training configurations for machine learning and deep learning
CINC2020TrainCfg = CFG()
CINC2020TrainCfg.torch_dtype = DEFAULTS.torch_dtype

# configs of files
CINC2020TrainCfg.db_dir = None
CINC2020TrainCfg.log_dir = DEFAULTS.log_dir / _NAME
CINC2020TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
CINC2020TrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
CINC2020TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
CINC2020TrainCfg.model_dir = DEFAULTS.model_dir / _NAME
CINC2020TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)

CINC2020TrainCfg.final_model_name = None
CINC2020TrainCfg.keep_checkpoint_max = 20

CINC2020TrainCfg.leads = deepcopy(EAK.Standard12Leads)

# configs of training data
CINC2020TrainCfg.fs = 500
CINC2020TrainCfg.data_format = "channel_first"

CINC2020TrainCfg.train_ratio = 0.8
CINC2020TrainCfg.min_class_weight = 0.5
CINC2020TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F", "G"

# assign classes, class weights, tranche classes, etc.
_assign_classes(CINC2020TrainCfg, _SPECIAL_CLASSES)

# configs of signal preprocessing
CINC2020TrainCfg.normalize = CFG(
    method="z-score",
    mean=0.0,
    std=1.0,
)
# frequency band of the filter to apply, should be chosen very carefully
CINC2020TrainCfg.bandpass = None
# CINC2020TrainCfg.bandpass = CFG(
#     lowcut=0.5,
#     highcut=60,
# )

# configs of data aumentation
# CINC2020TrainCfg.label_smooth = CFG(
#     prob=0.8,
#     smoothing=0.1,
# )
CINC2020TrainCfg.label_smooth = False
CINC2020TrainCfg.random_masking = False
CINC2020TrainCfg.stretch_compress = False  # stretch or compress in time axis
CINC2020TrainCfg.mixup = CFG(
    prob=0.6,
    alpha=0.3,
)

# configs of training epochs, batch, etc.
CINC2020TrainCfg.n_epochs = 50
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
CINC2020TrainCfg.batch_size = 64
# CINC2020TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
CINC2020TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
CINC2020TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
CINC2020TrainCfg.betas = (
    0.9,
    0.999,
)  # default values for corresponding PyTorch optimizers
CINC2020TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

CINC2020TrainCfg.learning_rate = 1e-4  # 1e-3
CINC2020TrainCfg.lr = CINC2020TrainCfg.learning_rate

CINC2020TrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
CINC2020TrainCfg.lr_step_size = 50
CINC2020TrainCfg.lr_gamma = 0.1
CINC2020TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

CINC2020TrainCfg.burn_in = 400
CINC2020TrainCfg.steps = [5000, 10000]

CINC2020TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
CINC2020TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
CINC2020TrainCfg.early_stopping.patience = 10

# configs of loss function
# CINC2020TrainCfg.loss = "BCEWithLogitsLoss"
# CINC2020TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
CINC2020TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss"
CINC2020TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
CINC2020TrainCfg.flooding_level = (
    0.0  # flooding performed if positive, typically 0.45-0.55 for cinc2020?
)

CINC2020TrainCfg.monitor = "challenge_metric"

CINC2020TrainCfg.log_step = 20
CINC2020TrainCfg.eval_every = 20

# configs of model selection
# "resnet_nature_comm_se", "multi_scopic_leadwise", "vgg16", "vgg16_leadwise",
CINC2020TrainCfg.cnn_name = "resnet_nature_comm_bottle_neck_se"
CINC2020TrainCfg.rnn_name = "none"  # "none", "lstm"
CINC2020TrainCfg.attn_name = "none"  # "none", "se", "gc", "nl"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
CINC2020TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `CINC2020TrainCfg.input_len`
CINC2020TrainCfg.input_len_tol = int(0.2 * CINC2020TrainCfg.input_len)
CINC2020TrainCfg.sig_slice_tol = 0.4  # None, do no slicing
CINC2020TrainCfg.siglen = CINC2020TrainCfg.input_len


# constants for model inference
CINC2020TrainCfg.bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
CINC2020TrainCfg.bin_pred_look_again_tol = 0.03
CINC2020TrainCfg.bin_pred_nsr_thr = 0.1
