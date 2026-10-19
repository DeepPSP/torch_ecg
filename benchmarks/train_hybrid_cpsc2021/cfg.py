"""
Configurations for the CPSC2021 benchmark (paroxysmal AF event detection).

The training configuration inherits from
:class:`torch_ecg.databases.datasets.cpsc2021.CPSC2021TrainCfg`; only the
artifact directories (logs, checkpoints, models) are made benchmark-local.
The task-specific model configurations are kept here.

References
----------
[1] http://2020.icbeb.org/CSPC2021
"""

from copy import deepcopy
from pathlib import Path

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG
from torch_ecg.databases.datasets.cpsc2021 import CPSC2021TrainCfg
from torch_ecg.model_configs import (
    ECG_SEQ_LAB_NET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
    RR_AF_CRF_CONFIG,
    RR_AF_VANILLA_CONFIG,
)

__all__ = [
    "TrainCfg",
    "ModelCfg",
]


_BASE_DIR = Path(__file__).absolute().parent


TrainCfg = deepcopy(CPSC2021TrainCfg)

TrainCfg.db_dir = None
TrainCfg.log_dir = _BASE_DIR / "log"
TrainCfg.model_dir = _BASE_DIR / "saved_models"
TrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
TrainCfg.model_dir.mkdir(parents=True, exist_ok=True)
TrainCfg.checkpoints = _BASE_DIR / "checkpoints"
TrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)


# Plan:
# R-peak detection using UNets, sequence labelling,
# main task via RR-LSTM using sequence of R peaks as input
# main task via UNets, sequence labelling using raw ECGs

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = TrainCfg.torch_dtype
_BASE_MODEL_CONFIG.fs = TrainCfg.fs
_BASE_MODEL_CONFIG.n_leads = TrainCfg.n_leads

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
ModelCfg.qrs_detection.seq_lab.fs = TrainCfg.fs
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
ModelCfg.qrs_detection.unet.fs = TrainCfg.fs


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
ModelCfg.main.seq_lab.fs = TrainCfg.fs
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
ModelCfg.main.unet.fs = TrainCfg.fs
ModelCfg.main.unet.reduction = 1
ModelCfg.main.unet.init_num_filters = 16  # keep the same with n_classes
ModelCfg.main.unet.down_num_filters = [
    ModelCfg.main.unet.init_num_filters * (2**idx) for idx in range(1, ModelCfg.main.unet.down_up_block_num + 1)
]
ModelCfg.main.unet.up_num_filters = [
    ModelCfg.main.unet.init_num_filters * (2**idx) for idx in range(ModelCfg.main.unet.down_up_block_num - 1, -1, -1)
]
ModelCfg.main.unet.up_mode = "interp"  # "deconv"
