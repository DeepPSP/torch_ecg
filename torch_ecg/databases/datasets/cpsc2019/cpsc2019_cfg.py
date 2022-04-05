"""
"""

from ....cfg import CFG, DEFAULTS

__all__ = [
    "CPSC2019TrainCfg",
]


_NAME = "cinc2021"


CPSC2019TrainCfg = CFG()

CPSC2019TrainCfg.fs = 500  # Hz, CPSC2019 data fs
CPSC2019TrainCfg.classes = [
    "N",
]
CPSC2019TrainCfg.n_leads = 1

CPSC2019TrainCfg.db_dir = None
CPSC2019TrainCfg.log_dir = DEFAULTS.log_dir / _NAME
CPSC2019TrainCfg.checkpoints = DEFAULTS.checkpoints / _NAME
CPSC2019TrainCfg.model_dir = DEFAULTS.model_dir / _NAME
CPSC2019TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
CPSC2019TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
CPSC2019TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)

CPSC2019TrainCfg.bias_thr = (
    0.075 * CPSC2019TrainCfg.fs
)  # keep the same with `THR` in `cpsc2019_score.py`
# detected rpeaks that are within `skip_dist` from two ends of the signal will be ignored,
# as in the official entry function
CPSC2019TrainCfg.skip_dist = 0.5 * CPSC2019TrainCfg.fs
CPSC2019TrainCfg.torch_dtype = DEFAULTS.torch_dtype

CPSC2019TrainCfg.final_model_name = None
CPSC2019TrainCfg.keep_checkpoint_max = 20
CPSC2019TrainCfg.train_ratio = 0.8

CPSC2019TrainCfg.input_len = int(CPSC2019TrainCfg.fs * 10)  # 10 s

# configs of signal preprocessing
CPSC2019TrainCfg.normalize = False
# frequency band of the filter to apply, should be chosen very carefully
CPSC2019TrainCfg.bandpass = False
# CPSC2019TrainCfg.bandpass = CFG(
#     lowcut=0.5,
#     highcut=60,
# )

# configs of data aumentation
# NOTE: compared to data augmentation of CPSC2020,
# `stretch_compress` and `label_smoothing` are not used in CPSC2019
CPSC2019TrainCfg.label_smooth = False
CPSC2019TrainCfg.random_masking = False
CPSC2019TrainCfg.stretch_compress = False  # stretch or compress in time axis
CPSC2019TrainCfg.mixup = False
# CPSC2019TrainCfg.baseline_wander = CFG(  # too slow!
#     prob = 0.5,
#     bw_fs = np.array([0.33, 0.1, 0.05, 0.01]),
#     ampl_ratio = np.array([
#         [0.01, 0.01, 0.02, 0.03],  # low
#         [0.01, 0.02, 0.04, 0.05],  # low
#         [0.1, 0.06, 0.04, 0.02],  # low
#         [0.02, 0.04, 0.07, 0.1],  # low
#         [0.05, 0.1, 0.16, 0.25],  # medium
#         [0.1, 0.15, 0.25, 0.3],  # high
#         [0.25, 0.25, 0.3, 0.35],  # extremely high
#     ]),
#     gaussian = np.array([  # default gaussian, mean and std, in terms of ratio
#         [0.0, 0.001],
#         [0.0, 0.003],
#         [0.0, 0.01],
#     ]),
# )
CPSC2019TrainCfg.random_flip = CFG(
    prob=0.5,
)

# configs of training epochs, batch, etc.
CPSC2019TrainCfg.n_epochs = 150
CPSC2019TrainCfg.batch_size = 32

# configs of optimizers and lr_schedulers
CPSC2019TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
CPSC2019TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
CPSC2019TrainCfg.betas = (
    0.9,
    0.999,
)  # default values for corresponding PyTorch optimizers
CPSC2019TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

CPSC2019TrainCfg.learning_rate = 1e-3  # 1e-4
CPSC2019TrainCfg.lr = CPSC2019TrainCfg.learning_rate

CPSC2019TrainCfg.lr_scheduler = (
    "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
)
CPSC2019TrainCfg.lr_step_size = 50
CPSC2019TrainCfg.lr_gamma = 0.1
CPSC2019TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

CPSC2019TrainCfg.burn_in = 400
CPSC2019TrainCfg.steps = [5000, 10000]

CPSC2019TrainCfg.early_stopping = CFG()  # early stopping according to challenge metric
CPSC2019TrainCfg.early_stopping.min_delta = 0.0001  # should be non-negative
CPSC2019TrainCfg.early_stopping.patience = 15

# configs of loss function
CPSC2019TrainCfg.loss = "BCEWithLogitsLoss"
# CPSC2019TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
CPSC2019TrainCfg.flooding_level = 0.0  # flooding performed if positive

CPSC2019TrainCfg.log_step = 2

# model selection
CPSC2019TrainCfg.model_name = "seq_lab_crnn"  # "seq_lab_cnn", "unet", "subtract_unet"
CPSC2019TrainCfg.cnn_name = "multi_scopic"
CPSC2019TrainCfg.rnn_name = "lstm"
CPSC2019TrainCfg.attn_name = "se"

CPSC2019TrainCfg.reduction = 2**3  # TODO: automatic adjust via model config
CPSC2019TrainCfg.recover_length = True

CPSC2019TrainCfg.monitor = "qrs_score"
