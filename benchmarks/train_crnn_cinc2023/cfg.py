"""
Configurations for models, training, etc., as well as some constants.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from cfg_models import ModelArchCfg
from sklearn.model_selection import ParameterGrid

from torch_ecg.cfg import CFG
from torch_ecg.components.inputs import InputConfig
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

__all__ = [
    "BaseCfg",
    "TrainCfg",
    "ModelCfg",
    "MLCfg",
]


_BASE_DIR = pathlib.Path(__file__).absolute().parent


###############################################################################
# Base Configs,
# including path, data type, classes, etc.
###############################################################################

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.fs = 100
BaseCfg.recording_pattern = "(?P<sbj>[\\d]{4})\\_" "(?P<seg>[\\d]{3})\\_" "(?P<hour>[\\d]{3})\\_" "(?P<sig>EEG|ECG|REF|OTHER)"
# fmt: off
BaseCfg.common_eeg_channels = [
    "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]
BaseCfg.eeg_bipolar_channels = [  # from the unofficial phase
    "Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4",
    "T4-T6", "T6-O2", "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2", "Fz-Cz", "Cz-Pz",
]
# fmt: on
BaseCfg.hospitals = list("ABCDEFG")
BaseCfg.hour_limit = 72
BaseCfg.n_channels = len(BaseCfg.eeg_bipolar_channels)
BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32
BaseCfg.ignore_index = -100
BaseCfg.outcome = ["Good", "Poor"]
BaseCfg.outcome_map = {
    "Good": 0,
    "Poor": 1,
}
BaseCfg.cpc = [str(cpc_level) for cpc_level in range(1, 6)]
BaseCfg.cpc_map = {str(cpc_level): cpc_level - 1 for cpc_level in range(1, 6)}
BaseCfg.cpc2outcome_map = {
    "1": "Good",
    "2": "Good",
    "3": "Poor",
    "4": "Poor",
    "5": "Poor",
}
BaseCfg.output_target = "cpc"  # "cpc", "outcome"


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

###########################################
# common configurations for all tasks
###########################################

TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(exist_ok=True)
# TODO: add "contrastive_learning", "regression", "multi_task", etc.
TrainCfg.tasks = ["classification"]

TrainCfg.train_ratio = 0.8

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 80
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 24

# since the memory limit of the Challenge is 64GB,
# loading all data into memory is not feasible,
# each recording is randomly sampled a `input_len` segment
# one can reload data every `reload_data_every` epochs
# to sample different segments from the same recording
TrainCfg.reload_data_every = -1  # -1 for no reloading, positive integer for reloading

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 2.5e-3  # 5e-4, 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 8e-3  # for "one_cycle" scheduler, to adjust via expriments

# configs of callbacks, including early stopping, checkpoint, etc.
TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
TrainCfg.early_stopping.min_delta = 0.001  # should be non-negative
TrainCfg.early_stopping.patience = TrainCfg.n_epochs // 2
TrainCfg.keep_checkpoint_max = 10

# configs of loss function
# TrainCfg.loss = "AsymmetricLoss"  # "FocalLoss", "BCEWithLogitsLoss"
# TrainCfg.loss_kw = CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp")
TrainCfg.flooding_level = 0.0  # flooding performed if positive,

# configs of logging
TrainCfg.log_step = 20
# TrainCfg.eval_every = 20

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

###########################################
# classification configurations
###########################################

TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.n_channels = BaseCfg.n_channels
TrainCfg.classification.final_model_name = None
TrainCfg.classification.output_target = BaseCfg.output_target

# input format configurations
TrainCfg.classification.data_format = "channel_first"
TrainCfg.classification.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=TrainCfg.classification.n_channels,
    fs=TrainCfg.classification.fs,
)
TrainCfg.classification.num_channels = TrainCfg.classification.input_config.n_channels
TrainCfg.classification.input_len = int(180 * TrainCfg.classification.fs)  # units in seconds, to adjust
TrainCfg.classification.siglen = TrainCfg.classification.input_len  # alias
TrainCfg.classification.sig_slice_tol = None  # None, do no slicing

if TrainCfg.classification.output_target == "cpc":
    TrainCfg.classification.classes = deepcopy(BaseCfg.cpc)
    TrainCfg.classification.class_map = deepcopy(BaseCfg.cpc_map)
elif TrainCfg.classification.output_target == "outcome":
    TrainCfg.classification.classes = deepcopy(BaseCfg.outcome)
    TrainCfg.classification.class_map = deepcopy(BaseCfg.outcome_map)

# preprocess configurations
# NOTE: (only unofficial phase):
# all EEG data was pre-processed with bandpass filtering (0.5-20Hz, or 0.5-30Hz?)
# and resampled to 100 Hz.
TrainCfg.classification.resample = CFG(fs=TrainCfg.classification.fs)
TrainCfg.classification.bandpass = CFG(lowcut=0.5, highcut=30, filter_type="butter", filter_order=4)
TrainCfg.classification.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations
# TrainCfg.classification.label_smooth = False
# TrainCfg.classification.random_masking = False
# TrainCfg.classification.stretch_compress = False  # stretch or compress in time axis
# TrainCfg.classification.mixup = CFG(
#     prob=0.6,
#     alpha=0.3,
# )

# model choices
TrainCfg.classification.model_name = "crnn"  # "wav2vec", "crnn"
TrainCfg.classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.classification.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.classification.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.classification.loss = CFG(
    cpc="AsymmetricLoss",  # "FocalLoss", "BCEWithLogitsWithClassWeightLoss"
    outcome="AsymmetricLoss",  # "FocalLoss", "BCEWithLogitsWithClassWeightLoss"
)
TrainCfg.classification.loss_kw = CFG(
    cpc=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
    outcome=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
)

# monitor choices
# "outcome_score", "outcome_accuracy", "outcome_f_measure", "cpc_mae", "cpc_mse"
TrainCfg.classification.monitor = "outcome_score"

# TODO: consider a regression task for cpc


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################


_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype


ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

# adjust filter lengths, > 1 for enlarging, < 1 for shrinking
cnn_filter_length_ratio = 1.0

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t
    ModelCfg[t].output_target = TrainCfg[t].output_target
    ModelCfg[t].classes = TrainCfg[t].classes
    ModelCfg[t].fs = TrainCfg[t].fs

    ModelCfg[t].update(deepcopy(ModelArchCfg[t]))

    ModelCfg[t].num_channels = TrainCfg[t].num_channels
    ModelCfg[t].input_len = TrainCfg[t].input_len
    ModelCfg[t].model_name = TrainCfg[t].model_name
    ModelCfg[t].cnn_name = TrainCfg[t].cnn_name
    ModelCfg[t].rnn_name = TrainCfg[t].rnn_name
    ModelCfg[t].attn_name = TrainCfg[t].attn_name

    # adjust filter length; cnn, rnn, attn choices in model configs
    for mn in [
        "crnn",
        # "seq_lab",
        # "unet",
    ]:
        if mn not in ModelCfg[t]:
            continue
        ModelCfg[t][mn] = adjust_cnn_filter_lengths(ModelCfg[t][mn], int(ModelCfg[t].fs * cnn_filter_length_ratio))
        ModelCfg[t][mn].cnn.name = ModelCfg[t].cnn_name
        ModelCfg[t][mn].rnn.name = ModelCfg[t].rnn_name
        ModelCfg[t][mn].attn.name = ModelCfg[t].attn_name


# machine learning model configurations

MLCfg = CFG()
MLCfg.db_dir = None
MLCfg.log_dir = BaseCfg.log_dir
MLCfg.model_dir = BaseCfg.model_dir
MLCfg.log_step = 20
# MLCfg.task = "classification"  # "classification", "regression"
# MLCfg.output_target = None
MLCfg.output_target = BaseCfg.output_target
if MLCfg.output_target == "cpc":
    MLCfg.classes = deepcopy(BaseCfg.cpc)
    MLCfg.class_map = deepcopy(BaseCfg.cpc_map)
elif MLCfg.output_target == "outcome":
    MLCfg.classes = deepcopy(BaseCfg.outcome)
    MLCfg.class_map = deepcopy(BaseCfg.outcome_map)
# MLCfg.x_cols_cate = [  # categorical features
#     "Sex",
#     "OHCA",
#     "VFib",
#     "TTM",
# ]
# MLCfg.x_cols_cont = [  # continuous features
#     "Age",
#     "ROSC",
# ]
# MLCfg.x_cols = MLCfg.x_cols_cate + MLCfg.x_cols_cont
MLCfg.feature_list = [
    "age",  # continuous
    "sex_female",  # binarized
    "sex_male",  # binarized
    "sex_other",  # binarized
    "rosc",  # continuous
    "ohca",  # binary
    "vfib",  # binary, from "Shockable Rhythm" (official phase)
    "ttm",  # continuous (indeed, categorical)
]
MLCfg.cont_features = ["age", "rosc", "ttm"]
MLCfg.cont_scaler = "standard"  # "minmax", "standard"
MLCfg.grids = CFG()
MLCfg.grids.rf = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 3, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "bootstrap": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
)
MLCfg.grids.xgb = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        "max_depth": [3, 5, 8],
        "verbosity": [0],
    }
)
MLCfg.grids.gdbt = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "loss": ["deviance", "exponential"],
        "learning_rate": [0.01, 0.05, 0.1],
        "criterion": ["friedman_mse", "mse"],
        "min_samples_split": [2, 3, 4],
        "max_features": ["auto", "sqrt", "log2"],
        "warm_start": [True, False],
        "ccp_alpha": [0.0, 0.1, 0.5, 1.0],
    }
)
MLCfg.grids.svc = ParameterGrid(
    {
        "C": [0.1, 0.5, 1, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 5],  # for "poly" kernel
        "gamma": [
            "scale",
            "auto",
        ],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        "coef0": [0.0, 0.2, 0.5, 1.0],  # for 'poly' and 'sigmoid'
        "class_weight": ["balanced", None],
        "probability": [True],
        "shrinking": [True, False],
    }
)
MLCfg.grids.bagging = ParameterGrid(
    {
        "n_estimators": [10, 15, 20, 50, 100],
        "max_features": [0.1, 0.2, 0.5, 0.9, 1.0],
        "bootstrap": [True, False],
        "bootstrap_features": [True, False],
        "oob_score": [True, False],
        "warm_start": [True, False],
    }
)
MLCfg.monitor = "outcome_score"
