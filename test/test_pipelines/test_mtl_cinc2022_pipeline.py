import json
import os
import re
import textwrap
import warnings
from abc import abstractmethod
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import shuffle, sample
from typing import Union, Optional, Any, List, Dict, Tuple, Sequence

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch_audiomentations as TA
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

try:
    import torchaudio
except Exception:
    torchaudio = None
try:
    import librosa
except Exception:
    librosa = None
try:
    import scipy.io.wavfile as sio_wav
except Exception:
    sio_wav = None
import IPython
from tqdm.auto import tqdm
from pcg_springer_features.schmidt_spike_removal import schmidt_spike_removal
from deprecated import deprecated

from torch_ecg.cfg import CFG
from torch_ecg.databases.base import PhysioNetDataBase, DataBaseInfo
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.utils.utils_data import ensure_siglen, stratified_train_test_split
from torch_ecg.utils.utils_nn import (
    adjust_cnn_filter_lengths,
    SizeMixin,
    default_collate_fn,
)
from torch_ecg.utils.utils_signal import butter_bandpass_filter
from torch_ecg.utils.misc import (
    get_record_list_recursive3,
    ReprMixin,
    list_sum,
    add_docstring,
)
from torch_ecg.utils.utils_metrics import _cls_to_bin
from torch_ecg.model_configs import (
    ECG_CRNN_CONFIG,
    ECG_SEQ_LAB_NET_CONFIG,
    ECG_UNET_VANILLA_CONFIG,
)
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.models import ECG_CRNN
from torch_ecg.models._nets import MLP
from torch_ecg.models.loss import (
    AsymmetricLoss,
    BCEWithLogitsWithClassWeightLoss,
    FocalLoss,
    MaskedBCEWithLogitsLoss,
)


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[1] / "tmp" / "test_mtl_cinc2022_pipeline"
_CWD.mkdir(parents=True, exist_ok=True)
_DB_DIR = _CWD.parents[2] / "sample-data" / "cinc2022"
_DB_DIR.mkdir(parents=True, exist_ok=True)
###############################################################################


###############################################################################
# set up configs

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.project_dir = _CWD
BaseCfg.log_dir = _CWD / "log"
BaseCfg.model_dir = _CWD / "saved_models"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.fs = 1000
BaseCfg.torch_dtype = torch.float32  # "double"
BaseCfg.np_dtype = np.float32
BaseCfg.ignore_index = -100
BaseCfg.ignore_unannotated = True

BaseCfg.outcomes = [
    "Abnormal",
    "Normal",
]
BaseCfg.classes = [
    "Present",
    "Unknown",
    "Absent",
]
BaseCfg.states = [
    "unannotated",
    "S1",
    "systolic",
    "S2",
    "diastolic",
]

# for example, can use scipy.signal.buttord(wp=[15, 250], ws=[5, 400], gpass=1, gstop=40, fs=1000)
BaseCfg.passband = [25, 400]  # Hz, candidates: [20, 500], [15, 250]
BaseCfg.filter_order = 3

# challenge specific configs, for merging results from multiple recordings into one
BaseCfg.merge_rule = "avg"  # "avg", "max"


class InputConfig(CFG):
    """ """

    __name__ = "InputConfig"

    def __init__(
        self,
        *args: Union[CFG, dict],
        input_type: str,
        n_channels: int,
        n_samples: int = -1,
        **kwargs: dict,
    ) -> None:
        """

        Parameters
        ----------
        input_type : str,
            the type of the input, can be
            - "waveform"
            - "spectrogram"
            - "mel_spectrogram" (with aliases `mel`, `melspectrogram`)
            - "mfcc"
            - "spectral" (concatenates the "spectrogram", the "mel_spectrogram" and the "mfcc")
        n_channels : int,
            the number of channels of the input
        n_samples : int,
            the number of samples of the input

        """
        super().__init__(
            *args,
            input_type=input_type,
            n_channels=n_channels,
            n_samples=n_samples,
            **kwargs,
        )
        assert "n_channels" in self and self.n_channels > 0
        assert "n_samples" in self and (self.n_samples > 0 or self.n_samples == -1)
        assert "input_type" in self and self.input_type.lower() in [
            "waveform",
            "spectrogram",
            "mel_spectrogram",
            "melspectrogram",
            "mel",
            "mfcc",
            "spectral",
        ]
        self.input_type = self.input_type.lower()
        if self.input_type in [
            "spectrogram",
            "mel_spectrogram",
            "melspectrogram",
            "mel",
            "mfcc",
            "spectral",
        ]:
            assert "n_bins" in self


###############################################################################
# training configurations for machine learning and deep learning
###############################################################################

TrainCfg = deepcopy(BaseCfg)

###########################################
# common configurations for all tasks
###########################################

TrainCfg.checkpoints = _CWD / "checkpoints"
TrainCfg.checkpoints.mkdir(exist_ok=True)

TrainCfg.train_ratio = 0.8

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 60
# TODO: automatic adjust batch size according to GPU capacity
# https://stackoverflow.com/questions/45132809/how-to-select-batch-size-automatically-to-fit-gpu
TrainCfg.batch_size = 24

# configs of optimizers and lr_schedulers
TrainCfg.optimizer = "adamw_amsgrad"  # "sgd", "adam", "adamw"
TrainCfg.momentum = 0.949  # default values for corresponding PyTorch optimizers
TrainCfg.betas = (0.9, 0.999)  # default values for corresponding PyTorch optimizers
TrainCfg.decay = 1e-2  # default values for corresponding PyTorch optimizers

TrainCfg.learning_rate = 5e-4  # 1e-3
TrainCfg.lr = TrainCfg.learning_rate

TrainCfg.lr_scheduler = "one_cycle"  # "one_cycle", "plateau", "burn_in", "step", None
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1
TrainCfg.max_lr = 2e-3  # for "one_cycle" scheduler, to adjust via expriments

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

###########################################
# task specific configurations
###########################################

# tasks of training
TrainCfg.tasks = [
    "classification",
    "segmentation",
    "multi_task",  # classification and segmentation with weight sharing
]

for t in TrainCfg.tasks:
    TrainCfg[t] = CFG()

###########################################
# classification configurations
###########################################

TrainCfg.classification.fs = BaseCfg.fs
TrainCfg.classification.final_model_name = None

# input format configurations
TrainCfg.classification.data_format = "channel_first"
TrainCfg.classification.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=1,
    fs=TrainCfg.classification.fs,
)
TrainCfg.classification.num_channels = TrainCfg.classification.input_config.n_channels
TrainCfg.classification.input_len = int(
    30 * TrainCfg.classification.fs
)  # 30 seconds, to adjust
TrainCfg.classification.siglen = TrainCfg.classification.input_len  # alias
TrainCfg.classification.sig_slice_tol = 0.2  # None, do no slicing
TrainCfg.classification.classes = deepcopy(BaseCfg.classes)
TrainCfg.classification.outcomes = deepcopy(BaseCfg.outcomes)
# TrainCfg.classification.outcomes = None
if TrainCfg.classification.outcomes is not None:
    TrainCfg.classification.outcome_map = {
        c: i for i, c in enumerate(TrainCfg.classification.outcomes)
    }
else:
    TrainCfg.classification.outcome_map = None
TrainCfg.classification.class_map = {
    c: i for i, c in enumerate(TrainCfg.classification.classes)
}

# preprocess configurations
TrainCfg.classification.resample = CFG(fs=TrainCfg.classification.fs)
TrainCfg.classification.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    filter_order=BaseCfg.filter_order,
)
TrainCfg.classification.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations via `from_dict` of `torch-audiomentations`
TrainCfg.classification.augmentations = [
    dict(
        transform="AddColoredNoise",
        params=dict(
            min_snr_in_db=1.0,
            max_snr_in_db=5.0,
            min_f_decay=-2.0,
            max_f_decay=2.0,
            mode="per_example",
            p=0.5,
            sample_rate=TrainCfg.classification.fs,
        ),
    ),
    dict(
        transform="PolarityInversion",
        params=dict(
            mode="per_example",
            p=0.6,
            sample_rate=TrainCfg.classification.fs,
        ),
    ),
]
TrainCfg.classification.augmentations_kw = CFG(
    p=0.7,
    p_mode="per_batch",
)

# model choices
TrainCfg.classification.model_name = "crnn"  # "wav2vec", "crnn"
TrainCfg.classification.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.classification.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.classification.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.classification.loss = CFG(
    # murmur="AsymmetricLoss",  # "FocalLoss"
    # outcome="CrossEntropyLoss",  # valid only if outcomes is not None
    murmur="BCEWithLogitsWithClassWeightLoss",
    outcome="BCEWithLogitsWithClassWeightLoss",
)
TrainCfg.classification.loss_kw = CFG(
    # murmur=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
    # outcome={},
    murmur=CFG(
        class_weight=torch.tensor([[5.0, 3.0, 1.0]])
    ),  # "Present", "Unknown", "Absent"
    outcome=CFG(class_weight=torch.tensor([[5.0, 1.0]])),  # "Abnormal", "Normal"
)

# monitor choices
# challenge metric is the **cost** of misclassification
# hence it is the lower the better
TrainCfg.classification.monitor = (
    "neg_weighted_cost"  # weighted_accuracy (not recommended)  # the higher the better
)
TrainCfg.classification.head_weights = CFG(
    # used to compute a numeric value to use the monitor
    murmur=0.5,
    outcome=0.5,
)

# freeze backbone configs, -1 for no freezing
TrainCfg.classification.freeze_backbone_at = int(0.6 * TrainCfg.n_epochs)

###########################################
# segmentation configurations
###########################################

TrainCfg.segmentation.fs = 1000
TrainCfg.segmentation.final_model_name = None

# input format configurations
TrainCfg.segmentation.data_format = "channel_first"
TrainCfg.segmentation.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=1,
    fs=TrainCfg.segmentation.fs,
)
TrainCfg.segmentation.num_channels = TrainCfg.segmentation.input_config.n_channels
TrainCfg.segmentation.input_len = int(
    30 * TrainCfg.segmentation.fs
)  # 30seconds, to adjust
TrainCfg.segmentation.siglen = TrainCfg.segmentation.input_len  # alias
TrainCfg.segmentation.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.segmentation.classes = deepcopy(BaseCfg.states)
if TrainCfg.ignore_unannotated:
    TrainCfg.segmentation.classes = [
        s for s in TrainCfg.segmentation.classes if s != "unannotated"
    ]
TrainCfg.segmentation.class_map = {
    c: i for i, c in enumerate(TrainCfg.segmentation.classes)
}

# preprocess configurations
TrainCfg.segmentation.resample = CFG(fs=TrainCfg.segmentation.fs)
TrainCfg.segmentation.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    filter_order=BaseCfg.filter_order,
)
TrainCfg.segmentation.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations via `from_dict` of `torch-audiomentations`
TrainCfg.segmentation.augmentations = [
    dict(
        transform="AddColoredNoise",
        params=dict(
            min_snr_in_db=1.0,
            max_snr_in_db=5.0,
            min_f_decay=-2.0,
            max_f_decay=2.0,
            mode="per_example",
            p=0.5,
            sample_rate=TrainCfg.segmentation.fs,
        ),
    ),
    dict(
        transform="PolarityInversion",
        params=dict(
            mode="per_example",
            p=0.6,
            sample_rate=TrainCfg.segmentation.fs,
        ),
    ),
]
TrainCfg.segmentation.augmentations_kw = CFG(
    p=0.7,
    p_mode="per_batch",
)

# model choices
TrainCfg.segmentation.model_name = "seq_lab"  # unet
TrainCfg.segmentation.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.segmentation.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.segmentation.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.segmentation.loss = CFG(
    segmentation="AsymmetricLoss",  # "FocalLoss"
)
TrainCfg.segmentation.loss_kw = CFG(
    segmentation=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
)

# monitor choices
TrainCfg.segmentation.monitor = "jaccard"

# freeze backbone configs, -1 for no freezing
TrainCfg.segmentation.freeze_backbone_at = -1


###########################################
# multi-task configurations
###########################################

TrainCfg.multi_task.fs = 1000
TrainCfg.multi_task.final_model_name = None

# input format configurations
TrainCfg.multi_task.data_format = "channel_first"
TrainCfg.multi_task.input_config = InputConfig(
    input_type="waveform",  # "waveform", "spectrogram", "mel", "mfcc", "spectral"
    n_channels=1,
    fs=TrainCfg.multi_task.fs,
)
TrainCfg.multi_task.num_channels = TrainCfg.multi_task.input_config.n_channels
TrainCfg.multi_task.input_len = int(30 * TrainCfg.multi_task.fs)  # 30seconds, to adjust
TrainCfg.multi_task.siglen = TrainCfg.multi_task.input_len  # alias
TrainCfg.multi_task.sig_slice_tol = 0.4  # None, do no slicing
TrainCfg.multi_task.classes = deepcopy(BaseCfg.classes)
TrainCfg.multi_task.class_map = {
    c: i for i, c in enumerate(TrainCfg.multi_task.classes)
}
TrainCfg.multi_task.outcomes = deepcopy(BaseCfg.outcomes)
TrainCfg.multi_task.outcome_map = {
    c: i for i, c in enumerate(TrainCfg.multi_task.outcomes)
}
TrainCfg.multi_task.states = deepcopy(BaseCfg.states)
if TrainCfg.ignore_unannotated:
    TrainCfg.multi_task.states = [
        s for s in TrainCfg.multi_task.states if s != "unannotated"
    ]
TrainCfg.multi_task.state_map = {s: i for i, s in enumerate(TrainCfg.multi_task.states)}

# preprocess configurations
TrainCfg.multi_task.resample = CFG(fs=TrainCfg.multi_task.fs)
TrainCfg.multi_task.bandpass = CFG(
    lowcut=BaseCfg.passband[0],
    highcut=BaseCfg.passband[1],
    filter_type="butter",
    filter_order=BaseCfg.filter_order,
)
TrainCfg.multi_task.normalize = CFG(  # None or False for no normalization
    method="z-score",
    mean=0.0,
    std=1.0,
)

# augmentations configurations via `from_dict` of `torch-audiomentations`
TrainCfg.multi_task.augmentations = [
    dict(
        transform="AddColoredNoise",
        params=dict(
            min_snr_in_db=1.0,
            max_snr_in_db=5.0,
            min_f_decay=-2.0,
            max_f_decay=2.0,
            mode="per_example",
            p=0.5,
            sample_rate=TrainCfg.multi_task.fs,
        ),
    ),
    dict(
        transform="PolarityInversion",
        params=dict(
            mode="per_example",
            p=0.6,
            sample_rate=TrainCfg.multi_task.fs,
        ),
    ),
]
TrainCfg.multi_task.augmentations_kw = CFG(
    p=0.7,
    p_mode="per_batch",
)

# model choices
TrainCfg.multi_task.model_name = "crnn"  # unet
TrainCfg.multi_task.cnn_name = "resnet_nature_comm_bottle_neck_se"
TrainCfg.multi_task.rnn_name = "lstm"  # "none", "lstm"
TrainCfg.multi_task.attn_name = "se"  # "none", "se", "gc", "nl"

# loss function choices
TrainCfg.multi_task.loss = CFG(
    # murmur="AsymmetricLoss",  # "FocalLoss"
    # outcome="CrossEntropyLoss",  # "FocalLoss", "AsymmetricLoss"
    murmur="BCEWithLogitsWithClassWeightLoss",
    outcome="BCEWithLogitsWithClassWeightLoss",
    segmentation="AsymmetricLoss",  # "FocalLoss", "CrossEntropyLoss"
)
TrainCfg.multi_task.loss_kw = CFG(
    # murmur=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
    # outcome={},
    murmur=CFG(
        class_weight=torch.tensor([[5.0 / 9.0, 3.0 / 9.0, 1.0 / 9.0]])
    ),  # "Present", "Unknown", "Absent"
    outcome=CFG(
        class_weight=torch.tensor([[5.0 / 6.0, 1.0 / 6.0]])
    ),  # "Abnormal", "Normal"
    segmentation=CFG(gamma_pos=0, gamma_neg=0.2, implementation="deep-psp"),
)

# monitor choices
TrainCfg.multi_task.monitor = "neg_weighted_cost"  # the higher the better
TrainCfg.multi_task.head_weights = CFG(
    # used to compute a numeric value to use the monitor
    murmur=0.5,
    outcome=0.5,
)
# freeze backbone configs, -1 for no freezing
TrainCfg.multi_task.freeze_backbone_at = int(0.6 * TrainCfg.n_epochs)


###############################################################################
# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
###############################################################################

ModelArchCfg = CFG()

# extra heads
OutcomeHeadCfg = CFG()
OutcomeHeadCfg.out_channels = [
    1024,
    # not including the last linear layer, whose out channels equals n_classes
]
OutcomeHeadCfg.activation = "mish"
OutcomeHeadCfg.bias = True
OutcomeHeadCfg.kernel_initializer = "he_normal"
OutcomeHeadCfg.dropouts = 0.2


SegmentationHeadCfg = CFG()
SegmentationHeadCfg.out_channels = [
    512,
    256,
]  # not including the last linear layer
SegmentationHeadCfg.activation = "mish"
SegmentationHeadCfg.bias = True
SegmentationHeadCfg.kernel_initializer = "he_normal"
SegmentationHeadCfg.dropouts = [0.2, 0.2, 0.0]
SegmentationHeadCfg.recover_length = True

ModelArchCfg.classification = CFG()
ModelArchCfg.classification.crnn = deepcopy(ECG_CRNN_CONFIG)

ModelArchCfg.classification.outcome_head = deepcopy(OutcomeHeadCfg)


ModelArchCfg.segmentation = CFG()
ModelArchCfg.segmentation.seq_lab = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
ModelArchCfg.segmentation.seq_lab.reduction = 1
ModelArchCfg.segmentation.seq_lab.recover_length = True
ModelArchCfg.segmentation.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)


ModelArchCfg.multi_task = CFG()
ModelArchCfg.multi_task.crnn = deepcopy(ECG_CRNN_CONFIG)

ModelArchCfg.multi_task.outcome_head = deepcopy(OutcomeHeadCfg)
ModelArchCfg.multi_task.segmentation_head = deepcopy(SegmentationHeadCfg)

_BASE_MODEL_CONFIG = CFG()
_BASE_MODEL_CONFIG.torch_dtype = BaseCfg.torch_dtype

ModelCfg = deepcopy(_BASE_MODEL_CONFIG)

for t in TrainCfg.tasks:
    ModelCfg[t] = deepcopy(_BASE_MODEL_CONFIG)
    ModelCfg[t].task = t
    ModelCfg[t].fs = TrainCfg[t].fs

    ModelCfg[t].update(deepcopy(ModelArchCfg[t]))

    ModelCfg[t].classes = TrainCfg[t].classes
    ModelCfg[t].num_channels = TrainCfg[t].num_channels
    ModelCfg[t].input_len = TrainCfg[t].input_len
    ModelCfg[t].model_name = TrainCfg[t].model_name
    ModelCfg[t].cnn_name = TrainCfg[t].cnn_name
    ModelCfg[t].rnn_name = TrainCfg[t].rnn_name
    ModelCfg[t].attn_name = TrainCfg[t].attn_name

    # adjust filter length; cnn, rnn, attn choices in model configs
    for mn in [
        "crnn",
        "seq_lab",
        # "unet",
    ]:
        if mn not in ModelCfg[t]:
            continue
        ModelCfg[t][mn] = adjust_cnn_filter_lengths(ModelCfg[t][mn], ModelCfg[t].fs)
        ModelCfg[t][mn].cnn.name = ModelCfg[t].cnn_name
        ModelCfg[t][mn].rnn.name = ModelCfg[t].rnn_name
        ModelCfg[t][mn].attn.name = ModelCfg[t].attn_name


# classification model outcome head
ModelCfg.classification.outcomes = deepcopy(TrainCfg.classification.outcomes)
if ModelCfg.classification.outcomes is None:
    ModelCfg.classification.outcome_head = None
else:
    ModelCfg.classification.outcome_head.loss = TrainCfg.classification.loss.outcome
    ModelCfg.classification.outcome_head.loss_kw = deepcopy(
        TrainCfg.classification.loss_kw.outcome
    )
ModelCfg.classification.states = None


# multi-task model outcome and segmentation head
ModelCfg.multi_task.outcomes = deepcopy(TrainCfg.multi_task.outcomes)
ModelCfg.multi_task.outcome_head.loss = TrainCfg.multi_task.loss.outcome
ModelCfg.multi_task.outcome_head.loss_kw = deepcopy(TrainCfg.multi_task.loss_kw.outcome)
ModelCfg.multi_task.states = deepcopy(TrainCfg.multi_task.states)
ModelCfg.multi_task.segmentation_head.loss = TrainCfg.multi_task.loss.segmentation
ModelCfg.multi_task.segmentation_head.loss_kw = deepcopy(
    TrainCfg.multi_task.loss_kw.segmentation
)


def remove_extra_heads(
    train_config: CFG, model_config: CFG, heads: Union[str, Sequence[str]]
) -> None:
    """
    remove extra heads from **task-specific** train config and model config,
    e.g. `TrainCfg.classification` and `ModelCfg.classification`

    Parameters
    ----------
    train_config : CFG
        train config
    model_config : CFG
        model config
    heads : str or sequence of str,
        names of heads to remove

    """
    if heads in ["", None, []]:
        return
    if isinstance(heads, str):
        heads = [heads]
    assert set(heads) <= set(["outcome", "outcomes", "segmentation"])
    for head in heads:
        if head.lower() in ["outcome", "outcomes"]:
            train_config.outcomes = None
            train_config.outcome_map = None
            train_config.loss.pop("outcome", None)
            train_config.loss_kw.pop("outcome", None)
            train_config.head_weights = {"murmur": 1.0}
            train_config.monitor = "murmur_weighted_accuracy"
            model_config.outcomes = None
            model_config.outcome_head = None
        if head.lower() in ["segmentation"]:
            train_config.states = None
            train_config.state_map = None
            train_config.loss.pop("segmentation", None)
            train_config.loss_kw.pop("segmentation", None)
            model_config.states = None
            model_config.segmentation_head = None


###############################################################################


###############################################################################
# data reader and data loader

_BACKEND_PRIORITY = [
    "torchaudio",
    "librosa",
    "scipy",
    "wfdb",
]


class PCGDataBase(PhysioNetDataBase):
    """ """

    __name__ = "PCGDataBase"

    def __init__(
        self,
        db_name: str,
        db_dir: str,
        fs: int = 1000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str, optional,
            storage path of the database
        fs: int, default 1000,
            (re-)sampling frequency of the audio
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(db_name, db_dir, working_dir, verbose, **kwargs)
        self.db_dir = (
            Path(self.db_dir).resolve().absolute()
        )  # will be fixed in `torch_ecg`
        self.fs = fs
        self.dtype = kwargs.get("dtype", BaseCfg.np_dtype)
        self.audio_backend = audio_backend.lower()
        if self.audio_backend not in self.available_backends():
            self.audio_backend = self.available_backends()[0]
            warnings.warn(
                f"audio backend {audio_backend.lower()} is not available, "
                f"using {self.audio_backend} instead",
                RuntimeWarning,
            )
        if self.audio_backend == "torchaudio":

            def torchaudio_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                try:
                    data, new_fs = torchaudio.load(file, normalize=True)
                except Exception:
                    data, new_fs = torchaudio.load(file, normalization=True)
                return data, new_fs

            self._audio_load_func = torchaudio_load
        elif self.audio_backend == "librosa":

            def librosa_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                data, _ = librosa.load(file, sr=fs, mono=False)
                return torch.from_numpy(data.reshape((-1, data.shape[-1]))), fs

            self._audio_load_func = librosa_load
        elif self.audio_backend == "scipy":

            def scipy_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                new_fs, data = sio_wav.read(file)
                data = (data / (2**15)).astype(self.dtype)[np.newaxis, :]
                return torch.from_numpy(data), new_fs

            self._audio_load_func = scipy_load
        elif self.audio_backend == "wfdb":
            warnings.warn(
                "loading result using wfdb is inconsistent with other backends",
                RuntimeWarning,
            )

            def wfdb_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                record = wfdb.rdrecord(file, physical=True)  # channel last
                sig = record.p_signal.T.astype(self.dtype)
                return torch.from_numpy(sig), record.fs[0]

            self._audio_load_func = wfdb_load
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self._all_records = None

    @staticmethod
    def available_backends() -> List[str]:
        """ """
        ab = ["wfdb"]
        if torchaudio is not None:
            ab.append("torchaudio")
        if librosa is not None:
            ab.append("librosa")
        if sio_wav is not None:
            ab.append("scipy")
        ab = sorted(ab, key=lambda x: _BACKEND_PRIORITY.index(x))
        return ab

    def _auto_infer_units(self) -> None:
        """
        disable this function implemented in the base class
        """
        print("DO NOT USE THIS FUNCTION for a PCG database!")

    @abstractmethod
    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        raise NotImplementedError

    def _reset_fs(self, new_fs: int) -> None:
        """ """
        self.fs = new_fs


_CINC2022_INFO = DataBaseInfo(
    title="""
    The CirCor DigiScope Phonocardiogram Dataset (main resource for CinC2022)
    """,
    about="""
    1. 5272 heart sound recordings (.wav format, sampling rate 4000 Hz) were collected from the main 4 auscultation locations of 1568 subjects, aged between 0 and 21 years (mean ± STD = 6.1 ± 4.3 years), with a duration between 4.8 to 80.4 seconds (mean ± STD = 22.9 ± 7.4 s)
    2. segmentation annotations (.tsv format) regarding the location of fundamental heart sounds (S1 and S2) in the recordings have been obtained using a semi-supervised scheme. The annotation files are composed of three distinct columns: the first column corresponds to the time instant (in seconds) where the wave was detected for the first time, the second column corresponds to the time instant (in seconds) where the wave was detected for the last time, and the third column corresponds to an identifier that uniquely identifies the detected wave. Here, we use the following convention:
        - The S1 wave is identified by the integer 1.
        - The systolic period is identified by the integer 2.
        - The S2 wave is identified by the integer 3.
        - The diastolic period is identified by the integer 4.
        - The unannotated segments of the signal are identified by the integer 0.
    """,
    usage=[
        "Heart murmur detection",
        "Heart sound segmentation",
    ],
    note="""
    1. the "Murmur" column (records whether heart murmur can be heard or not) and the "Outcome" column (the expert cardiologist's overall diagnosis using **clinical history, physical examination, analog auscultation, echocardiogram, etc.**) are **NOT RELATED**. All of the 6 combinations (["Present", "Absent", "Unknown"] x ["Abnormal", "Normal"]) occur in the dataset.
    2. the segmentation files do NOT in general (totally 132 such files) have the same length (namely the second column of the last row of these .tsv files) as the audio files.
    """,
    issues="""
    1. the segmentation file `50782_MV_1.tsv` (versions 1.0.2, 1.0.3) is broken.
    2. the challenge website states that the `Age` variable takes values in `Neonate`, `Infant`, `Child`, `Adolescent`, and `Young adult`. However, from the statistics csv file (training_data.csv), there's no subject whose `Age` column has value `Young adult`. Instead, there are 74 subject with null `Age` value, which only indicates that their ages were not recorded and may or may not belong to the “Young adult” age group.
    """,
    references=[
        "https://moody-challenge.physionet.org/2022/",
        "https://physionet.org/content/circor-heart-sound/1.0.3/",
    ],
    doi=[
        "10.1109/JBHI.2021.3137048",
        "10.13026/tshs-mw03",
    ],
)


@add_docstring(_CINC2022_INFO.format_database_docstring())
class CINC2022Reader(PCGDataBase):
    """ """

    __name__ = "CINC2022Reader"
    stats_fillna_val = "NA"

    def __init__(
        self,
        db_dir: str,
        fs: int = 4000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_dir: str,
            storage path of the database
        fs: int, default 4000,
            (re-)sampling frequency of the audio
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="circor-heart-sound",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        if "training_data" in os.listdir(self.db_dir):
            self.data_dir = self.db_dir / "training_data"
        else:
            self.data_dir = self.db_dir
        self.data_ext = "wav"
        self.ann_ext = "hea"
        self.segmentation_ext = "tsv"
        self.segmentation_states = deepcopy(BaseCfg.states)
        self.ignore_unannotated = kwargs.get("ignore_unannotated", True)
        if self.ignore_unannotated:
            self.segmentation_states = [
                s for s in self.segmentation_states if s != "unannotated"
            ]
        self.segmentation_map = {n: s for n, s in enumerate(self.segmentation_states)}
        if self.ignore_unannotated:
            self.segmentation_map[BaseCfg.ignore_index] = "unannotated"
        self.auscultation_locations = {
            "PV",
            "AV",
            "MV",
            "TV",
            "Phc",
        }

        self._rec_pattern = f"(?P<sid>[\\d]+)\\_(?P<loc>{'|'.join(self.auscultation_locations)})((?:\\_)(?P<num>\\d))?"

        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._exceptional_records = ["50782_MV_1"]
        self._ls_rec()

        self._df_stats = None
        self._stats_cols = [
            "Patient ID",
            "Locations",
            "Age",
            "Sex",
            "Height",
            "Weight",
            "Pregnancy status",
            "Outcome",  # added in version 1.0.2 in the official phase
            "Murmur",
            "Murmur locations",
            "Most audible location",
            "Systolic murmur timing",
            "Systolic murmur shape",
            "Systolic murmur grading",
            "Systolic murmur pitch",
            "Systolic murmur quality",
            "Diastolic murmur timing",
            "Diastolic murmur shape",
            "Diastolic murmur grading",
            "Diastolic murmur pitch",
            "Diastolic murmur quality",
            "Campaign",
            "Additional ID",
        ]
        self._df_stats_records = None
        self._stats_records_cols = [
            "Patient ID",
            "Location",
            "rec",
            "siglen",
            "siglen_sec",
            "Murmur",
        ]
        self._load_stats()

        # attributes for plot
        self.palette = {
            "systolic": "#d62728",
            "diastolic": "#2ca02c",
            "S1": "#17becf",
            "S2": "#bcbd22",
            "default": "#7f7f7f",
        }

    def _ls_rec(self) -> None:
        """
        list all records in the database
        """
        write_file = False
        self._df_records = pd.DataFrame(columns=["record", "path"])
        records_file = self.db_dir / "RECORDS"
        if records_file.exists():
            self._df_records["record"] = records_file.read_text().splitlines()
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
        else:
            write_file = True

        # self._all_records = wfdb.get_record_list(self.db_name)

        if len(self._df_records) == 0:
            write_file = True
            self._df_records["path"] = get_record_list_recursive3(
                self.db_dir, f"{self._rec_pattern}\\.{self.data_ext}", relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))

        data_dir = self._df_records["path"].apply(lambda x: x.parent).unique()
        assert len(data_dir) <= 1, "data_dir should be a single directory"
        if len(data_dir) == 1:  # in case no data found
            self.data_dir = data_dir[0]

        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records = self._df_records[
            ~self._df_records["record"].isin(self._exceptional_records)
        ]
        self._df_records.set_index("record", inplace=True)

        self._all_records = [
            item
            for item in self._df_records.index.tolist()
            if item not in self._exceptional_records
        ]
        self._all_subjects = sorted(
            set([item.split("_")[0] for item in self._all_records]),
            key=lambda x: int(x),
        )
        self._subject_records = defaultdict(list)
        for rec in self._all_records:
            self._subject_records[self.get_subject(rec)].append(rec)
        self._subject_records = dict(self._subject_records)

        if write_file:
            records_file.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

    def _load_stats(self) -> None:
        """
        collect statistics of the database
        """
        print("Reading the statistics from local file...")
        stats_file = self.db_dir / "training_data.csv"
        if stats_file.exists():
            self._df_stats = pd.read_csv(stats_file)
        elif self._all_records is not None and len(self._all_records) > 0:
            print("No cached statistics found, gathering from scratch...")
            self._df_stats = pd.DataFrame()
            with tqdm(
                self.all_subjects, total=len(self.all_subjects), desc="loading stats"
            ) as pbar:
                for s in pbar:
                    f = self.data_dir / f"{s}.txt"
                    content = f.read_text().splitlines()
                    new_row = {"Patient ID": s}
                    locations = set()
                    for line in content:
                        if not line.startswith("#"):
                            if line.split()[0] in self.auscultation_locations:
                                locations.add(line.split()[0])
                            continue
                        k, v = line.replace("#", "").split(":")
                        k, v = k.strip(), v.strip()
                        if v == "nan":
                            v = self.stats_fillna_val
                        new_row[k] = v
                    new_row["Recording locations:"] = "+".join(locations)
                    self._df_stats = self._df_stats.append(
                        new_row,
                        ignore_index=True,
                    )
            self._df_stats.to_csv(stats_file, index=False)
        else:
            print("No data found locally!")
            return
        self._df_stats = self._df_stats.fillna(self.stats_fillna_val)
        try:
            # the column "Locations" is changed to "Recording locations:" in version 1.0.2
            self._df_stats.Locations = self._df_stats.Locations.apply(
                lambda s: s.split("+")
            )
        except AttributeError:
            self._df_stats["Locations"] = self._df_stats["Recording locations:"].apply(
                lambda s: s.split("+")
            )
        self._df_stats["Murmur locations"] = self._df_stats["Murmur locations"].apply(
            lambda s: s.split("+")
        )
        self._df_stats["Patient ID"] = self._df_stats["Patient ID"].astype(str)
        self._df_stats = self._df_stats[self._stats_cols]
        for idx, row in self._df_stats.iterrows():
            for c in ["Height", "Weight"]:
                if row[c] == self.stats_fillna_val:
                    self._df_stats.at[idx, c] = np.nan

        # load stats of the records
        print("Reading the statistics of the records from local file...")
        stats_file = self.db_dir / "stats_records.csv"
        if stats_file.exists():
            self._df_stats_records = pd.read_csv(stats_file)
        else:
            self._df_stats_records = pd.DataFrame(columns=self._stats_records_cols)
            with tqdm(
                self._df_stats.iterrows(),
                total=len(self._df_stats),
                desc="loading record stats",
            ) as pbar:
                for _, row in pbar:
                    sid = row["Patient ID"]
                    for loc in row["Locations"]:
                        rec = f"{sid}_{loc}"
                        if rec not in self._all_records:
                            continue
                        header = wfdb.rdheader(str(self.data_dir / f"{rec}"))
                        if row["Murmur"] == "Unknown":
                            murmur = "Unknown"
                        if loc in row["Murmur locations"]:
                            murmur = "Present"
                        else:
                            murmur = "Absent"
                        new_row = {
                            "Patient ID": sid,
                            "Location": loc,
                            "rec": rec,
                            "siglen": header.sig_len,
                            "siglen_sec": header.sig_len / header.fs,
                            "Murmur": murmur,
                        }
                        self._df_stats_records = self._df_stats_records.append(
                            new_row,
                            ignore_index=True,
                        )
            self._df_stats_records.to_csv(stats_file, index=False)
        self._df_stats_records = self._df_stats_records.fillna(self.stats_fillna_val)

    def _decompose_rec(self, rec: Union[str, int]) -> Dict[str, str]:
        """
        decompose a record name into its components (subject, location, and number)

        Parameters
        ----------
        rec: str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        dict,
            the components (subject, location, and number) of the record,
            with keys "sid", "loc", and "num" respectively

        """
        if isinstance(rec, int):
            rec = self[rec]
        return list(re.finditer(self._rec_pattern, rec))[0].groupdict()

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """
        get the absolute path of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        extension: str, optional,
            extension of the file

        Returns
        -------
        Path,
            absolute path of the file

        """
        if isinstance(rec, int):
            rec = self[rec]
        path = self._df_records.loc[rec, "path"]
        if extension is not None and not extension.startswith("."):
            extension = f".{extension}"
        return path.with_suffix(extension or "").resolve()

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """
        load data from the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the record, defaults to `self.fs`,
            -1 for the sampling frequency from the audio file
        data_format : str, optional,
            the format of the returned data, defaults to `channel_first`
            can be `channel_last`, `channel_first`, `flat`,
            case insensitive
        data_type : str, default "np",
            the type of the returned data, can be one of "pt", "np",
            case insensitive

        Returns
        -------
        data : np.ndarray,
            the data of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.get_absolute_path(rec, self.data_ext)
        data, data_fs = self._audio_load_func(data_file, fs)
        # data of shape (n_channels, n_samples), of type torch.Tensor
        if fs is not None and data_fs != fs:
            data = torchaudio.transforms.Resample(data_fs, fs)(data)
        if data_format.lower() == "channel_last":
            data = data.T
        elif data_format.lower() == "flat":
            data = data.reshape(-1)
        if data_type.lower() == "np":
            data = data.numpy()
        elif data_type.lower() != "pt":
            raise ValueError(f"Unsupported data type: {data_type}")
        return data

    @add_docstring(load_data.__doc__)
    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """alias of `load_data`"""
        return self.load_data(rec, fs, data_format, data_type)

    def load_ann(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """
        load classification annotation of the record `rec` or the subject `sid`

        Parameters
        ----------
        rec_or_sid : str or int,
            the record name or the index of the record in `self.all_records`
            or the subject id
        class_map : dict, optional,
            the mapping of the annotation classes

        Returns
        -------
        ann : str or int,
            the class of the record,
            or the number of the class if `class_map` is provided

        """
        if isinstance(rec_or_sid, int):
            rec_or_sid = self[rec_or_sid]
        _class_map = class_map or {}
        if rec_or_sid in self.all_subjects:
            ann = self.df_stats[self.df_stats["Patient ID"] == rec_or_sid].iloc[0][
                "Murmur"
            ]
        elif rec_or_sid in self.all_records:
            decom = self._decompose_rec(rec_or_sid)
            sid, loc = decom["sid"], decom["loc"]
            row = self.df_stats[self.df_stats["Patient ID"] == sid].iloc[0]
            if row["Murmur"] == "Unknown":
                ann = "Unknown"
            if loc in row["Murmur locations"]:
                ann = "Present"
            else:
                ann = "Absent"
        else:
            raise ValueError(f"{rec_or_sid} is not a valid record or patient ID")
        ann = _class_map.get(ann, ann)
        return ann

    @add_docstring(load_ann.__doc__)
    def load_murmur(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """alias of `load_ann`"""
        return self.load_ann(rec_or_sid, class_map)

    def load_segmentation(
        self,
        rec: Union[str, int],
        seg_format: str = "df",
        ensure_same_len: bool = True,
        fs: Optional[int] = None,
    ) -> Union[pd.DataFrame, np.ndarray, dict]:
        """
        load the segmentation of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        seg_format : str, default `df`,
            the format of the returned segmentation,
            can be `df`, `dict`, `mask`, `binary`,
            case insensitive
        ensure_same_len : bool, default True,
            if True, the length of the segmentation will be
            the same as the length of the audio data
        fs : int, optional,
            the sampling frequency, defaults to `self.fs`,
            -1 for the sampling frequency from the audio file

        Returns
        -------
        pd.DataFrame or np.ndarray or dict,
            the segmentation of the record

        NOTE
        ----
        segmentation files do NOT have the same length (namely the second column of the last row of these .tsv files) as the audio files.

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = self.get_fs(rec)
        segmentation_file = self.get_absolute_path(rec, self.segmentation_ext)
        df_seg = pd.read_csv(segmentation_file, sep="\t", header=None)
        df_seg.columns = ["start_t", "end_t", "label"]
        if self.ignore_unannotated:
            df_seg["label"] = df_seg["label"].apply(
                lambda x: x - 1 if x > 0 else BaseCfg.ignore_index
            )
        df_seg["wave"] = df_seg["label"].apply(lambda s: self.segmentation_map[s])
        df_seg["start"] = (fs * df_seg["start_t"]).apply(round)
        df_seg["end"] = (fs * df_seg["end_t"]).apply(round)
        if ensure_same_len:
            sig_len = wfdb.rdheader(str(self.get_absolute_path(rec))).sig_len
            if sig_len != df_seg["end"].max():
                df_seg = df_seg.append(
                    dict(
                        start_t=df_seg["end"].max() / fs,
                        end_t=sig_len / fs,
                        start=df_seg["end"].max(),
                        end=sig_len,
                        wave="unannotated",
                        label=BaseCfg.ignore_index,
                    ),
                    ignore_index=True,
                )
        if seg_format.lower() in [
            "dataframe",
            "df",
        ]:
            return df_seg
        elif seg_format.lower() in [
            "dict",
            "dicts",
        ]:
            # dict of intervals
            return {
                k: [
                    [row["start"], row["end"]]
                    for _, row in df_seg[df_seg["wave"] == k].iterrows()
                ]
                for _, k in self.segmentation_map.items()
            }
        elif seg_format.lower() in [
            "mask",
        ]:
            # mask = np.zeros(df_seg.end.values[-1], dtype=int)
            mask = np.full(df_seg.end.values[-1], BaseCfg.ignore_index, dtype=int)
            for _, row in df_seg.iterrows():
                mask[row["start"] : row["end"]] = int(row["label"])
            return mask
        elif seg_format.lower() in [
            "binary",
        ]:
            bin_mask = np.zeros(
                (df_seg.end.values[-1], len(self.segmentation_states)), dtype=self.dtype
            )
            for _, row in df_seg.iterrows():
                if row["wave"] in self.segmentation_states:
                    bin_mask[
                        row["start"] : row["end"],
                        self.segmentation_states.index(row["wave"]),
                    ] = 1
            return bin_mask
        else:
            raise ValueError(f"{seg_format} is not a valid format")

    def load_meta_data(
        self,
        subject: str,
        keys: Optional[Union[Sequence[str], str]] = None,
    ) -> Union[dict, str, float, int]:
        """
        load meta data of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id
        keys : str or sequence of str, optional,
            the keys of the meta data to be returned,
            if None, return all meta data

        Returns
        -------
        meta_data : dict or str or float or int,
            the meta data of the subject

        """
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        meta_data = row.to_dict()
        if keys:
            if isinstance(keys, str):
                for k, v in meta_data.items():
                    if k.lower() == keys.lower():
                        return v
            else:
                _keys = [k.lower() for k in keys]
                return {k: v for k, v in meta_data.items() if k.lower() in _keys}
        return meta_data

    def load_outcome(self, rec_or_subject: Union[str, int]) -> str:
        """
        load the outcome of the subject or the subject related to the record

        Parameters
        ----------
        rec_or_subject : str or int,
            the record name or the index of the record in `self.all_records`,
            or the subject id (Patient ID)

        Returns
        -------
        outcome : str,
            the outcome of the record

        """
        if isinstance(rec_or_subject, int):
            rec_or_subject = self[rec_or_subject]
        if rec_or_subject in self.all_subjects:
            pass
        elif rec_or_subject in self.all_records:
            decom = self._decompose_rec(rec_or_subject)
            rec_or_subject = decom["sid"]
        else:
            raise ValueError(f"{rec_or_subject} is not a valid record or patient ID")
        outcome = self.load_outcome_(rec_or_subject)
        return outcome

    def load_outcome_(self, subject: str) -> str:
        """
        load the expert cardiologist's overall diagnosis of  of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id

        Returns
        -------
        str,
            the expert cardiologist's overall diagnosis,
            one of `Normal`, `Abnormal`

        """
        if isinstance(subject, int) or subject in self.all_records:
            raise ValueError("subject should be chosen from `self.all_subjects`")
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        return row.Outcome

    def _load_preprocessed_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        passband: Sequence[int] = BaseCfg.passband,
        order: int = BaseCfg.filter_order,
        spike_removal: bool = True,
    ) -> np.ndarray:
        """
        load preprocessed data of the record `rec`,
        with preprocessing procedure:
            - resample to `fs` (if `fs` is not None)
            - bandpass filter
            - spike removal

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the returned data
        data_format : str, default `channel_first`,
            the format of the returned data,
            can be `channel_first`, `channel_last` or `flat`,
            case insensitive
        passband : sequence of int, default `BaseCfg.passband`,
            the passband of the bandpass filter
        order : int, default `BaseCfg.filter_order`,
            the order of the bandpass filter
        spike_removal : bool, default True,
            whether to remove spikes using the Schmmidt algorithm

        Returns
        -------
        data : np.ndarray,
            the preprocessed data of the record

        """
        fs = fs or self.fs
        data = butter_bandpass_filter(
            self.load_data(rec, fs=fs, data_format="flat"),
            lowcut=passband[0],
            highcut=passband[1],
            fs=fs,
            order=order,
        ).astype(self.dtype)
        if spike_removal:
            data = schmidt_spike_removal(data, fs=fs)
        if data_format.lower() == "flat":
            return data
        data = np.atleast_2d(data)
        if data_format.lower() == "channel_last":
            data = data.T
        return data

    def get_fs(self, rec: Union[str, int]) -> int:
        """
        get the original sampling frequency of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        int,
            the original sampling frequency of the record

        """
        return wfdb.rdheader(str(self.get_absolute_path(rec))).fs

    def get_subject(self, rec: Union[str, int]) -> str:
        """
        get the subject id (Patient ID) of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        str,
            the subject id (Patient ID) of the record

        """
        return self._decompose_rec(rec)["sid"]

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    @property
    def df_stats(self) -> pd.DataFrame:
        if self._df_stats is None or self._df_stats.empty:
            self._load_stats()
        return self._df_stats

    @property
    def df_stats_records(self) -> pd.DataFrame:
        if self._df_stats_records is None or self._df_stats_records.empty:
            self._load_stats()
        return self._df_stats_records

    @property
    def murmur_feature_cols(self) -> List[str]:
        return [
            "Systolic murmur timing",
            "Systolic murmur shape",
            "Systolic murmur grading",
            "Systolic murmur pitch",
            "Systolic murmur quality",
            "Diastolic murmur timing",
            "Diastolic murmur shape",
            "Diastolic murmur grading",
            "Diastolic murmur pitch",
            "Diastolic murmur quality",
        ]

    def play(self, rec: Union[str, int], **kwargs) -> IPython.display.Audio:
        """
        play the record `rec` in a Juptyer Notebook

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        kwargs : dict,
            optional keyword arguments including `data`, `fs`,
            if specified, the data will be played instead of the record

        Returns
        -------
        IPython.display.Audio,
            the audio object of the record

        """
        if "data" in kwargs:
            return IPython.display.Audio(
                kwargs["data"], rate=kwargs.get("fs", self.get_fs(rec))
            )
        audio_file = self.get_absolute_path(rec)
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: Union[str, int], **kwargs) -> None:
        """
        plot the record `rec`, with metadata and segmentation

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        kwargs : dict,
            not used currently

        Returns
        -------
        fig: matplotlib.figure.Figure,
            the figure of the record
        ax: matplotlib.axes.Axes,
            the axes of the figure

        """
        import matplotlib.pyplot as plt

        waveforms = self.load_pcg(rec, data_format="flat")
        df_segmentation = self.load_segmentation(rec)
        meta_data = self.load_meta_data(self.get_subject(rec))
        labels = {
            "Outcome": meta_data["Outcome"],
            "Murmur": meta_data["Murmur"],
        }
        meta_data = {
            k: "NA" if meta_data[k] == self.stats_fillna_val else meta_data[k]
            for k in ["Age", "Sex", "Height", "Weight", "Pregnancy status"]
        }
        rec_dec = self._decompose_rec(rec)
        rec_dec = {
            "SubjectID": rec_dec["sid"],
            "Location": rec_dec["loc"],
            "Number": rec_dec["num"],
        }
        rec_dec = {k: v for k, v in rec_dec.items() if v is not None}
        figsize = (5 * len(waveforms) / self.fs, 5)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            np.arange(len(waveforms)) / self.fs,
            waveforms,
            color=self.palette["default"],
        )
        counter = {
            "systolic": 0,
            "diastolic": 0,
            "S1": 0,
            "S2": 0,
        }
        for _, row in df_segmentation.iterrows():
            if row.wave != "unannotated":
                # labels starting with "_" are ignored
                # ref. https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
                ax.axvspan(
                    row.start_t,
                    row.end_t,
                    color=self.palette[row.wave],
                    alpha=0.3,
                    label="_" * counter[row.wave] + row.wave,
                )
                counter[row.wave] += 1
        ax.legend(loc="upper right")
        bbox_prop = {
            "boxstyle": "round",
            "facecolor": "#EAEAF2",
            "edgecolor": "black",
        }
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in rec_dec.items()]),
            (0.01, 0.95),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in meta_data.items()]),
            (0.01, 0.80),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in labels.items()]),
            (0.01, 0.15),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )

        return fig, ax

    def plot_outcome_correlation(self, col: str = "Murmur", **kwargs: Any) -> object:
        """
        plot the correlation between the outcome and the feature `col`

        Parameters
        ----------
        col: str, default "Murmur",
            the feature to be used for the correlation, can be one of
            "Murmur", "Age", "Sex", "Pregnancy status"
        kwargs: dict,
            key word arguments,
            passed to the function `pd.DataFrame.plot`

        Returns
        -------
        ax: mpl.axes.Axes

        """
        # import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns

        # sns.set()
        sns.set_theme(style="white")  # darkgrid, whitegrid, dark, white, ticks
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["axes.labelsize"] = 24
        plt.rcParams["legend.fontsize"] = 18
        plt.rcParams["hatch.linewidth"] = 2.5

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        hatches = ["/", "\\", "|", ".", "x"]

        assert col in ["Murmur", "Age", "Sex", "Pregnancy status"]
        prefix_sep = " - "
        df_dummies = pd.get_dummies(
            self.df_stats[col], prefix=col, prefix_sep=prefix_sep
        )
        columns = df_dummies.columns.tolist()
        if f"{col}{prefix_sep}{self.stats_fillna_val}" in columns:
            idx = columns.index(f"{col}{prefix_sep}{self.stats_fillna_val}")
            columns[idx] = f"{col}{prefix_sep}{'NA'}"
            df_dummies.columns = columns
        df_stats = pd.concat((self.df_stats, df_dummies), axis=1)
        plot_kw = dict(
            kind="bar",
            figsize=(8, 8),
            ylabel="Number of Subjects (n.u.)",
            stacked=True,
            rot=0,
            ylim=(0, 620),
            yticks=np.arange(0, 700, 100),
            width=0.3,
            fill=True,
            # hatch=hatches[: len(columns)],
        )
        plot_kw.update(kwargs)
        ax = (
            df_stats.groupby("Outcome")
            .agg("sum")[df_dummies.columns.tolist()]
            .plot(**plot_kw)
        )
        for idx in range(len(columns)):
            ax.patches[2 * idx].set_hatch(hatches[idx])
            ax.patches[2 * idx + 1].set_hatch(hatches[idx])
        ax.legend(loc="upper left", ncol=int(np.ceil(len(columns) / 3)))
        plt.tight_layout()

        # mpl.rc_file_defaults()

        return ax


class CinC2022Dataset(Dataset, ReprMixin):
    """ """

    __name__ = "CinC2022Dataset"

    def __init__(
        self, config: CFG, task: str, training: bool = True, lazy: bool = True
    ) -> None:
        """ """
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()  # task will be set in self.__set_task
        self.training = training
        self.lazy = lazy

        self.reader = CINC2022Reader(
            self.config.db_dir,
            ignore_unannotated=self.config.get("ignore_unannotated", True),
        )

        self.subjects = self._train_test_split()
        df = self.reader.df_stats[
            self.reader.df_stats["Patient ID"].isin(self.subjects)
        ]
        self.records = list_sum(
            [self.reader.subject_records[row["Patient ID"]] for _, row in df.iterrows()]
        )
        if self.config.get("test_flag", True):
            self.records = sample(self.records, int(len(self.records) * 0.2))
        if self.training:
            shuffle(self.records)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config.classification))
        seg_ppm_config = CFG(random=False)
        seg_ppm_config.update(deepcopy(self.config.segmentation))
        self.ppm = PreprocManager.from_config(ppm_config)
        self.seg_ppm = PreprocManager.from_config(seg_ppm_config)

        self.__cache = None
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        """ """
        if self.cache is None:
            self._load_all_data()
        return self.cache["waveforms"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        if self.cache is None:
            self._load_all_data()
        return {k: v[index] for k, v in self.cache.items()}

    def __set_task(self, task: str, lazy: bool) -> None:
        """ """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if (
            hasattr(self, "task")
            and self.task == task.lower()
            and self.cache is not None
            and len(self.cache["waveforms"]) > 0
        ):
            return
        self.task = task.lower()
        self.siglen = int(self.config[self.task].fs * self.config[self.task].siglen)
        self.classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        if self.task in ["classification"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        elif self.task in ["segmentation"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.seg_ppm
            )
        elif self.task in ["multi_task"]:
            self.fdr = MutiTaskFastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        else:
            raise ValueError("Illegal task")

        if self.lazy:
            return

        tmp_cache = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="records") as pbar:
            for idx in pbar:
                tmp_cache.append(self.fdr[idx])
        keys = tmp_cache[0].keys()
        self.__cache = {k: np.concatenate([v[k] for v in tmp_cache]) for k in keys}
        for k in keys:
            if self.__cache[k].ndim == 1:
                self.__cache[k] = self.__cache[k]

    def _load_all_data(self) -> None:
        """ """
        self.__set_task(self.task, lazy=False)

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())

        df_train, df_test = stratified_train_test_split(
            self.reader.df_stats,
            [
                "Murmur",
                "Age",
                "Sex",
                "Pregnancy status",
                "Outcome",
            ],
            test_ratio=1 - train_ratio,
        )

        train_set = df_train["Patient ID"].tolist()
        test_set = df_test["Patient ID"].tolist()

        train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        shuffle(train_set)
        shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    @property
    def cache(self) -> List[Dict[str, np.ndarray]]:
        return self.__cache

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["task", "training"]


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: PCGDataBase,
        records: Sequence[str],
        config: CFG,
        task: str,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        rec = self.records[index]
        waveforms = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        waveforms = ensure_siglen(
            waveforms,
            siglen=self.config[self.task].input_len,
            fmt=self.config[self.task].data_format,
            tolerance=self.config[self.task].sig_slice_tol,
        ).astype(self.dtype)
        if waveforms.ndim == 2:
            waveforms = waveforms[np.newaxis, ...]

        n_segments = waveforms.shape[0]

        if self.task in ["classification"]:
            label = self.reader.load_ann(rec)
            if self.config[self.task].loss != "CrossEntropyLoss":
                label = (
                    np.isin(self.config[self.task].classes, label)
                    .astype(self.dtype)[np.newaxis, ...]
                    .repeat(n_segments, axis=0)
                )
            else:
                label = np.array(
                    [
                        self.config[self.task].class_map[label]
                        for _ in range(n_segments)
                    ],
                    dtype=int,
                )
            out = {"waveforms": waveforms, "murmur": label}
            if self.config[self.task].outcomes is not None:
                outcome = self.reader.load_outcome(rec)
                if self.config[self.task].loss["outcome"] != "CrossEntropyLoss":
                    outcome = (
                        np.isin(self.config[self.task].outcomes, outcome)
                        .astype(self.dtype)[np.newaxis, ...]
                        .repeat(n_segments, axis=0)
                    )
                else:
                    outcome = np.array(
                        [
                            self.config[self.task].outcome_map[outcome]
                            for _ in range(n_segments)
                        ],
                        dtype=int,
                    )
                out["outcome"] = outcome
            return out

        elif self.task in ["segmentation"]:
            label = self.reader.load_segmentation(
                rec,
                seg_format="binary",
                ensure_same_len=True,
                fs=self.config[self.task].fs,
            )
            label = ensure_siglen(
                label,
                siglen=self.config[self.task].input_len,
                fmt="channel_last",
                tolerance=self.config[self.task].sig_slice_tol,
            ).astype(self.dtype)
            return {"waveforms": waveforms, "segmentation": label}
        else:
            raise ValueError(f"Illegal task: {self.task}")

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]


class MutiTaskFastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: PCGDataBase,
        records: Sequence[str],
        config: CFG,
        task: str = "multi_task",
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        assert self.task == "multi_task"
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        rec = self.records[index]
        waveforms = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        waveforms = ensure_siglen(
            waveforms,
            siglen=self.config[self.task].input_len,
            fmt=self.config[self.task].data_format,
            tolerance=self.config[self.task].sig_slice_tol,
        ).astype(self.dtype)
        if waveforms.ndim == 2:
            waveforms = waveforms[np.newaxis, ...]

        n_segments = waveforms.shape[0]

        label = self.reader.load_ann(rec)
        if self.config[self.task].loss["murmur"] != "CrossEntropyLoss":
            label = (
                np.isin(self.config[self.task].classes, label)
                .astype(self.dtype)[np.newaxis, ...]
                .repeat(n_segments, axis=0)
            )
        else:
            label = np.array(
                [self.config[self.task].class_map[label] for _ in range(n_segments)],
                dtype=int,
            )
        out_tensors = {
            "waveforms": waveforms,
            "murmur": label,
        }

        if self.config[self.task].outcomes is not None:
            outcome = self.reader.load_outcome(rec)
            if self.config[self.task].loss["outcome"] != "CrossEntropyLoss":
                outcome = (
                    np.isin(self.config[self.task].outcomes, outcome)
                    .astype(self.dtype)[np.newaxis, ...]
                    .repeat(n_segments, axis=0)
                )
            else:
                outcome = np.array(
                    [
                        self.config[self.task].outcome_map[outcome]
                        for _ in range(n_segments)
                    ],
                    dtype=int,
                )
            out_tensors["outcome"] = outcome

        if self.config[self.task].states is not None:
            mask = self.reader.load_segmentation(
                rec,
                seg_format="binary",
                ensure_same_len=True,
                fs=self.config[self.task].fs,
            )
            mask = ensure_siglen(
                mask,
                siglen=self.config[self.task].input_len,
                fmt="channel_last",
                tolerance=self.config[self.task].sig_slice_tol,
            ).astype(self.dtype)
            out_tensors["segmentation"] = mask

        return out_tensors

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]


###############################################################################


###############################################################################
# models


@dataclass
class CINC2022Outputs:
    """ """

    murmur_output: ClassificationOutput
    outcome_output: ClassificationOutput
    segmentation_output: SequenceLabellingOutput
    murmur_loss: Optional[float] = None
    outcome_loss: Optional[float] = None
    segmentation_loss: Optional[float] = None


class MultiTaskHead(nn.Module, SizeMixin):
    """ """

    __name__ = "MultiTaskHead"

    def __init__(self, in_channels: int, config: CFG) -> None:
        """

        Parameters
        ----------
        in_channels: int,
            the number of input channels
        config: dict,
            configurations, ref. `cfg.ModelCfg`

        """
        super().__init__()
        self.in_channels = in_channels
        self.config = config

        self.heads = nn.ModuleDict()
        self.criteria = nn.ModuleDict()

        if self.config.get("outcome_head", None) is not None:
            self.outcomes = self.config.get("outcomes")
            self.config.outcome_head.out_channels.append(len(self.outcomes))
            self.heads["outcome"] = MLP(
                in_channels=self.in_channels,
                skip_last_activation=True,
                **self.config.outcome_head,
            )
            self.criteria["outcome"] = self._setup_criterion(
                loss=self.config.outcome_head.loss,
                loss_kw=self.config.outcome_head.loss_kw,
            )
        else:
            self.outcomes = None
        if self.config.get("segmentation_head", None) is not None:
            self.states = self.config.get("states")
            self.config.segmentation_head.out_channels.append(len(self.states))
            self.heads["segmentation"] = MLP(
                in_channels=self.in_channels,
                skip_last_activation=True,
                **self.config.segmentation_head,
            )
            self.criteria["segmentation"] = self._setup_criterion(
                loss=self.config.segmentation_head.loss,
                loss_kw=self.config.segmentation_head.loss_kw,
            )
        else:
            self.states = None

    def forward(
        self,
        features: torch.Tensor,
        pooled_features: torch.Tensor,
        original_len: int,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        Parameters
        ----------
        features: torch.Tensor,
            the feature tensor,
            of shape (batch_size, n_channels, seq_len)
        pooled_features: torch.Tensor,
            the pooled features of the input data,
            of shape (batch_size, n_channels)
        original_len: int,
            the original length of the input data,
            used when segmentation head's `recover_length` config is set `True`
        labels: dict of torch.Tensor, optional,
            the labels of the input data, including:
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes) or (batch_size,)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of torch.Tensor,
            the output of the model, including (some are optional):
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        if self.empty:
            warnings.warn("Empty model, DO NOT call forward function!", RuntimeWarning)
        out = dict(total_extra_loss=torch.scalar_tensor(0.0))
        if "segmentation" in self.heads:
            out["segmentation"] = self.heads["segmentation"](features.permute(0, 2, 1))
            if self.config.segmentation_head.get("recover_length", True):
                out["segmentation"] = F.interpolate(
                    out["segmentation"].permute(0, 2, 1),
                    size=original_len,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            if labels is not None and labels.get("segmentation", None) is not None:
                out["segmentation_loss"] = self.criteria["segmentation"](
                    out["segmentation"].reshape(-1, out["segmentation"].shape[0]),
                    labels["segmentation"].reshape(-1, labels["segmentation"].shape[0]),
                )
                out["total_extra_loss"] = (
                    out["total_extra_loss"].to(dtype=out["segmentation_loss"].dtype)
                    + out["segmentation_loss"]
                )
        if "outcome" in self.heads:
            out["outcome"] = self.heads["outcome"](pooled_features)
            if labels is not None and labels.get("outcome", None) is not None:
                out["outcome_loss"] = self.criteria["outcome"](
                    out["outcome"], labels["outcome"]
                )
                out["total_extra_loss"] = (
                    out["total_extra_loss"].to(dtype=out["outcome_loss"].dtype)
                    + out["outcome_loss"]
                )
        return out

    def _setup_criterion(self, loss: str, loss_kw: Optional[dict] = None) -> None:
        """ """
        if loss_kw is None:
            loss_kw = {}
        if loss == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif loss == "BCEWithLogitsWithClassWeightLoss":
            criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif loss == "BCELoss":
            criterion = nn.BCELoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "FocalLoss":
            criterion = FocalLoss(**loss_kw)
        elif loss == "AsymmetricLoss":
            criterion = AsymmetricLoss(**loss_kw)
        elif loss == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(
                f"loss `{loss}` not implemented! "
                "Please use one of the following: `BCEWithLogitsLoss`, `BCEWithLogitsWithClassWeightLoss`, "
                "`BCELoss`, `MaskedBCEWithLogitsLoss`, `MaskedBCEWithLogitsLoss`, `FocalLoss`, "
                "`AsymmetricLoss`, `CrossEntropyLoss`, or override this method to setup your own criterion."
            )
        return criterion

    @property
    def empty(self) -> bool:
        return len(self.heads) == 0


class CRNN_CINC2022(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "classification"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_CRNN_CINC2022(model_cfg)
        ````

        """
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )
        self.outcomes = _config.outcomes
        self.states = _config.states
        self.extra_heads = MultiTaskHead(
            in_channels=self.clf.in_channels,
            config=_config,
        )
        if self.extra_heads.empty:
            self.extra_heads = None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        freeze the backbone (CRNN part, excluding the heads) of the model

        Parameters
        ----------
        freeze: bool, default True,
            whether to freeze the backbone

        """
        for params in self.cnn.parameters():
            params.requires_grad = not freeze
        if getattr(self, "rnn") is not None:
            for params in self.rnn.parameters():
                params.requires_grad = not freeze
        if getattr(self, "attn") is not None:
            for params in self.attn.parameters():
                params.requires_grad = not freeze

    def forward(
        self,
        waveforms: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """

        Parameters
        ----------
        waveforms: torch.Tensor,
            of shape (batch_size, channels, seq_len)
        labels: dict of torch.Tensor, optional,
            the labels of the waveforms data, including:
            - "murmur": the murmur labels, of shape (batch_size, n_classes) or (batch_size,)
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes) or (batch_size,)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of torch.Tensor, with items (some are optional):
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        batch_size, channels, seq_len = waveforms.shape

        features = self.extract_features(waveforms)

        if self.pool:
            pooled_features = self.pool(features)  # (batch_size, channels, pool_size)
            # features = features.squeeze(dim=-1)
            pooled_features = rearrange(
                pooled_features,
                "batch_size channels pool_size -> batch_size (channels pool_size)",
            )
        else:
            # pooled_features of shape (batch_size, channels) or (batch_size, seq_len, channels)
            pooled_features = features

        # print(f"clf in shape = {x.shape}")
        pred = self.clf(pooled_features)  # batch_size, n_classes

        if self.extra_heads is not None:
            out = self.extra_heads(features, pooled_features, seq_len, labels)
            out["murmur"] = pred
        else:
            out = {"murmur": pred}

        return out

    @torch.no_grad()
    def inference(
        self,
        waveforms: Union[np.ndarray, torch.Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        waveforms: ndarray or torch.Tensor,
            waveforms tensor, of shape (batch_size, channels, seq_len)
        seg_thr: float, default 0.5,
            threshold for making binary predictions for
            the optional segmentaion head

        Returns
        -------
        CINC2022Outputs, with attributes:
            - murmur_output: ClassificationOutput, with items:
                - classes: list of str,
                    list of the class names
                - prob: ndarray or DataFrame,
                    scalar (probability) predictions,
                    (and binary predictions if `class_names` is True)
                - pred: ndarray,
                    the array of class number predictions
                - bin_pred: ndarray,
                    the array of binary predictions
                - forward_output: ndarray,
                    the array of output of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
            outcome_output: ClassificationOutput, optional, with items:
                - classes: list of str,
                    list of the outcome class names
                - prob: ndarray,
                    scalar (probability) predictions,
                - pred: ndarray,
                    the array of outcome class number predictions
                - forward_output: ndarray,
                    the array of output of the outcome head of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
            segmentation_output: SequenceLabellingOutput, optional, with items:
                - classes: list of str,
                    list of the state class names
                - prob: ndarray,
                    scalar (probability) predictions,
                - pred: ndarray,
                    the array of binarized prediction

        """
        self.eval()
        _input = torch.as_tensor(waveforms, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        forward_output = self.forward(_input)

        prob = self.softmax(forward_output["murmur"])
        pred = torch.argmax(prob, dim=-1)
        bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()

        murmur_output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output["murmur"].cpu().detach().numpy(),
        )

        if forward_output.get("outcome", None) is not None:
            prob = self.softmax(forward_output["outcome"])
            pred = torch.argmax(prob, dim=-1)
            bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            outcome_output = ClassificationOutput(
                classes=self.outcomes,
                prob=prob,
                pred=pred,
                bin_pred=bin_pred,
                forward_output=forward_output["outcome"].cpu().detach().numpy(),
            )
        else:
            outcome_output = None

        if forward_output.get("segmentation", None) is not None:
            # if "unannotated" in self.states, use softmax
            # else use sigmoid
            if "unannotated" in self.states:
                prob = self.softmax(forward_output["segmentation"])
                pred = torch.argmax(prob, dim=-1)
            else:
                prob = self.sigmoid(forward_output["segmentation"])
                pred = (prob > seg_thr).int() * (
                    prob == prob.max(dim=-1, keepdim=True).values
                ).int()
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            segmentation_output = SequenceLabellingOutput(
                classes=self.states,
                prob=prob,
                pred=pred,
                forward_output=forward_output["segmentation"].cpu().detach().numpy(),
            )
        else:
            segmentation_output = None

        return CINC2022Outputs(murmur_output, outcome_output, segmentation_output)

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        waveforms: Union[np.ndarray, torch.Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(waveforms, seg_thr)


###############################################################################


###############################################################################
# trainer
###############################################################################


class AugmenterManager(TA.SomeOf):
    """Audio data augmenters"""

    def __init__(
        self,
        transforms: Sequence[BaseWaveformTransform],
        p: float = 1.0,
        p_mode="per_batch",
    ) -> None:
        """ """
        super().__init__((1, None), transforms, p=p, p_mode=p_mode)

    @classmethod
    def from_config(cls, config: dict) -> "AugmenterManager":
        """ """
        transforms = [TA.from_dict(item) for item in config["augmentations"]]
        return cls(transforms, **config["augmentations_kw"])

    def __len__(self) -> int:
        """ """
        return len(self.transforms)


######################################
# custom metrics computation functions
######################################


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2022Outputs],
    require_both: bool = False,
) -> Dict[str, float]:
    """
    Compute the challenge metrics

    Parameters
    ----------
    labels: sequence of dict of ndarray,
        labels containing at least one of the following items:
            - "murmur":
                binary labels, of shape: (n_samples, n_classes);
                or categorical labels, of shape: (n_samples,)
            - "outcome":
                binary labels, of shape: (n_samples, n_classes),
                or categorical labels, of shape: (n_samples,)
    outputs: sequence of CINC2022Outputs,
        outputs containing at least one non-null attributes:
            - murmur_output: ClassificationOutput, with items:
                - classes: list of str,
                    list of the class names
                - prob: ndarray or DataFrame,
                    scalar (probability) predictions,
                    (and binary predictions if `class_names` is True)
                - pred: ndarray,
                    the array of class number predictions
                - bin_pred: ndarray,
                    the array of binary predictions
                - forward_output: ndarray,
                    the array of output of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
            - outcome_output: ClassificationOutput, optional, with items:
                - classes: list of str,
                    list of the outcome class names
                - prob: ndarray,
                    scalar (probability) predictions,
                - pred: ndarray,
                    the array of outcome class number predictions
                - forward_output: ndarray,
                    the array of output of the outcome head of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
    require_both: bool,
        whether to require both murmur and outcome labels and outputs to be provided

    Returns
    -------
    dict, a dict of the following metrics:
        - murmur_auroc: float,
            the macro-averaged area under the receiver operating characteristic curve for the murmur predictions
        - murmur_auprc: float,
            the macro-averaged area under the precision-recall curve for the murmur predictions
        - murmur_f_measure: float,
            the macro-averaged F-measure for the murmur predictions
        - murmur_accuracy: float,
            the accuracy for the murmur predictions
        - murmur_weighted_accuracy: float,
            the weighted accuracy for the murmur predictions
        - murmur_cost: float,
            the challenge cost for the murmur predictions
        - outcome_auroc: float,
            the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
        - outcome_auprc: float,
            the macro-averaged area under the precision-recall curve for the outcome predictions
        - outcome_f_measure: float,
            the macro-averaged F-measure for the outcome predictions
        - outcome_accuracy: float,
            the accuracy for the outcome predictions
        - outcome_weighted_accuracy: float,
            the weighted accuracy for the outcome predictions
        - outcome_cost: float,
            the challenge cost for the outcome predictions

    NOTE
    ----
    1. the "murmur_xxx" metrics are contained in the returned dict iff corr. labels and outputs are provided;
        the same applies to the "outcome_xxx" metrics.
    2. all labels should have a batch dimension, except for categorical labels

    """
    metrics = {}
    if require_both:
        assert all([set(lb.keys()) >= set(["murmur", "outcome"]) for lb in labels])
        assert all(
            [
                item.murmur_output is not None and item.outcome_output is not None
                for item in outputs
            ]
        )
    # metrics for murmurs
    # NOTE: labels all have a batch dimension, except for categorical labels
    if outputs[0].murmur_output is not None:
        murmur_labels = np.concatenate(
            [lb["murmur"] for lb in labels]  # categorical or binarized labels
        )
        murmur_scalar_outputs = np.concatenate(
            [np.atleast_2d(item.murmur_output.prob) for item in outputs]
        )
        murmur_binary_outputs = np.concatenate(
            [np.atleast_2d(item.murmur_output.bin_pred) for item in outputs]
        )
        murmur_classes = outputs[0].murmur_output.classes
        if murmur_labels.ndim == 1:
            murmur_labels = _cls_to_bin(
                murmur_labels, shape=(len(murmur_labels), len(murmur_classes))
            )
        metrics.update(
            _compute_challenge_metrics(
                murmur_labels,
                murmur_scalar_outputs,
                murmur_binary_outputs,
                murmur_classes,
            )
        )
    # metrics for outcomes
    if outputs[0].outcome_output is not None:
        outcome_labels = np.concatenate(
            [lb["outcome"] for lb in labels]  # categorical or binarized labels
        )
        outcome_scalar_outputs = np.concatenate(
            [np.atleast_2d(item.outcome_output.prob) for item in outputs]
        )
        outcome_binary_outputs = np.concatenate(
            [np.atleast_2d(item.outcome_output.bin_pred) for item in outputs]
        )
        outcome_classes = outputs[0].outcome_output.classes
        if outcome_labels.ndim == 1:
            outcome_labels = _cls_to_bin(
                outcome_labels, shape=(len(outcome_labels), len(outcome_classes))
            )
        metrics.update(
            _compute_challenge_metrics(
                outcome_labels,
                outcome_scalar_outputs,
                outcome_binary_outputs,
                outcome_classes,
            )
        )

    return metrics


def _compute_challenge_metrics(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, float]:
    """
    Compute macro-averaged metrics,
    modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names for murmurs or outcomes,
        e.g. `BaseCfg.classes` or `BaseCfg.outcomes`

    Returns
    -------
    dict, a dict of the following metrics:
        auroc: float,
            the macro-averaged area under the receiver operating characteristic curve
        auprc: float,
            the macro-averaged area under the precision-recall curve
        f_measure: float,
            the macro-averaged F-measure
        accuracy: float,
            the accuracy
        weighted_accuracy: float,
            the weighted accuracy
        cost: float,
            the challenge cost

    """
    detailed_metrics = _compute_challenge_metrics_detailed(
        labels, scalar_outputs, binary_outputs, classes
    )
    metrics = {
        f"""{detailed_metrics["prefix"]}_{k}""": v
        for k, v in detailed_metrics.items()
        if k
        in [
            "auroc",
            "auprc",
            "f_measure",
            "accuracy",
            "weighted_accuracy",
            "cost",
        ]
    }
    return metrics


def _compute_challenge_metrics_detailed(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Compute detailed metrics, modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names for murmurs or outcomes,
        e.g. `BaseCfg.classes` or `BaseCfg.outcomes`

    Returns
    -------
    dict, a dict of the following metrics:
        auroc: float,
            the macro-averaged area under the receiver operating characteristic curve
        auprc: float,
            the macro-averaged area under the precision-recall curve
        auroc_classes: np.ndarray,
            the area under the receiver operating characteristic curve for each class
        auprc_classes: np.ndarray,
            the area under the precision-recall curve for each class
        f_measure: float,
            the macro-averaged F-measure
        f_measure_classes: np.ndarray,
            the F-measure for each class
        accuracy: float,
            the accuracy
        accuracy_classes: np.ndarray,
            the accuracy for each class
        weighted_accuracy: float,
            the weighted accuracy
        cost: float,
            the challenge cost
        prefix: str,
            the prefix of the metrics, one of `"murmur"` or `"outcome"`

    """
    # For each patient, set the 'Unknown' class to positive if no class is positive or if multiple classes are positive.
    if list(classes) == BaseCfg.classes:
        positive_class = "Present"
        prefix = "murmur"
    elif list(classes) == BaseCfg.outcomes:
        positive_class = "Abnormal"
        prefix = "outcome"
    else:
        raise ValueError(f"Illegal sequence of classes: {classes}")
    labels = enforce_positives(labels, classes, positive_class)
    binary_outputs = enforce_positives(binary_outputs, classes, positive_class)

    # Evaluate the model by comparing the labels and outputs.
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    accuracy, accuracy_classes = compute_accuracy(labels, binary_outputs)
    weighted_accuracy = compute_weighted_accuracy(labels, binary_outputs, list(classes))
    # challenge_score = compute_challenge_score(labels, binary_outputs, classes)
    cost = compute_cost(labels, binary_outputs, BaseCfg.outcomes, classes)

    return dict(
        auroc=auroc,
        auprc=auprc,
        auroc_classes=auroc_classes,
        auprc_classes=auprc_classes,
        f_measure=f_measure,
        f_measure_classes=f_measure_classes,
        accuracy=accuracy,
        accuracy_classes=accuracy_classes,
        weighted_accuracy=weighted_accuracy,
        cost=cost,
        prefix=prefix,
    )


###########################################
# methods from the file evaluation_model.py
# of the official repository
###########################################


def enforce_positives(
    outputs: np.ndarray, classes: Sequence[str], positive_class: str
) -> np.ndarray:
    """
    For each patient, set a specific class to positive if no class is positive or multiple classes are positive.

    Parameters
    ----------
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: Sequence[str],
        class names
    positive_class: str,
        class name to be set to positive

    Returns
    -------
    outputs: np.ndarray,
        enforced binary outputs, of shape: (n_samples, n_classes)

    """
    num_patients, num_classes = np.shape(outputs)
    j = classes.index(positive_class)

    for i in range(num_patients):
        if np.sum(outputs[i, :]) != 1:
            outputs[i, :] = 0
            outputs[i, j] = 1
    return outputs


def compute_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        confusion matrix, of shape: (n_classes, n_classes)

    """
    assert np.shape(labels)[0] == np.shape(outputs)[0]
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A


def compute_one_vs_rest_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """
    Compute binary one-vs-rest confusion matrices, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        one-vs-rest confusion matrix, of shape: (n_classes, 2, 2)

    """
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


# alias
compute_ovr_confusion_matrix = compute_one_vs_rest_confusion_matrix


def compute_auc(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute macro AUROC and macro AUPRC, and AUPRCs, AUPRCs for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    macro_auroc: float,
        macro AUROC
    macro_auprc: float,
        macro AUPRC
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)

    """
    print("Computing AUROC and AUPRC...")
    num_patients, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_patients and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float("nan")
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float("nan")

    return macro_auroc, macro_auprc, auroc, auprc


def compute_accuracy(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute accuracy.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    accuracy: float,
        the accuracy
    accuracy_classes: np.ndarray,
        the accuracy for each class, of shape: (n_classes,)
    """
    print("Computing accuracy...")
    assert np.shape(labels) == np.shape(outputs)
    num_patients, num_classes = np.shape(labels)
    A = compute_confusion_matrix(labels, outputs)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float("nan")

    # Compute per-class accuracy.
    accuracy_classes = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(A[:, i]) > 0:
            accuracy_classes[i] = A[i, i] / np.sum(A[:, i])
        else:
            accuracy_classes[i] = float("nan")

    return accuracy, accuracy_classes


def compute_f_measure(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute macro F-measure, and F-measures for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    macro_f_measure: float,
        macro F-measure
    f_measure: np.ndarray,
        F-measures for each class, of shape: (n_classes,)

    """
    print("Computing F-measure...")
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure


def compute_weighted_accuracy(
    labels: np.ndarray, outputs: np.ndarray, classes: List[str]
) -> float:
    """
    compute weighted accuracy

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: List[str],
        list of class names, can be one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match

    Returns
    -------
    weighted_accuracy: float,
        weighted accuracy

    """
    print("Computing weighted accuracy...")
    # Define constants.
    if classes == ["Present", "Unknown", "Absent"]:
        weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    elif classes == ["Abnormal", "Normal"]:
        weights = np.array([[5, 1], [5, 1]])
    else:
        raise NotImplementedError(
            "Weighted accuracy undefined for classes {}".format(", ".join(classes))
        )

    # Compute confusion matrix.
    assert np.shape(labels) == np.shape(outputs)
    A = compute_confusion_matrix(labels, outputs)

    # Multiply the confusion matrix by the weight matrix.
    assert np.shape(A) == np.shape(weights)
    B = weights * A

    # Compute weighted_accuracy.
    if np.sum(B) > 0:
        weighted_accuracy = np.trace(B) / np.sum(B)
    else:
        weighted_accuracy = float("nan")

    return weighted_accuracy


def cost_algorithm(m: int) -> int:
    """total cost for algorithmic prescreening of m patients."""
    return 10 * m


def cost_expert(m: int, n: int) -> float:
    """total cost for expert screening of m patients out of a total of n total patients."""
    return (25 + 397 * (m / n) - 1718 * (m / n) ** 2 + 11296 * (m / n) ** 4) * n


def cost_treatment(m: int) -> int:
    """total cost for treatment of m patients."""
    return 10000 * m


def cost_error(m: int) -> int:
    """total cost for missed/late treatement of m patients."""
    return 50000 * m


def compute_cost(
    labels: np.ndarray,
    outputs: np.ndarray,
    label_classes: Sequence[str],
    output_classes: Sequence[str],
) -> float:
    """
    Compute Challenge cost metric.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    label_classes: Sequence[str],
        list of label class names, can one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match
        case sensitive
    output_classes: Sequence[str],
        list of predicted class names, can one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match
        case sensitive

    """
    print("Computing challenge cost...")
    # Define positive and negative classes for referral and treatment.
    positive_classes = ["Present", "Unknown", "Abnormal"]
    negative_classes = ["Absent", "Normal"]

    # Compute confusion matrix.
    A = compute_confusion_matrix(labels, outputs)

    # Identify positive and negative classes for referral.
    idx_label_positive = [
        i for i, x in enumerate(label_classes) if x in positive_classes
    ]
    idx_label_negative = [
        i for i, x in enumerate(label_classes) if x in negative_classes
    ]
    idx_output_positive = [
        i for i, x in enumerate(output_classes) if x in positive_classes
    ]
    idx_output_negative = [
        i for i, x in enumerate(output_classes) if x in negative_classes
    ]

    # Identify true positives, false positives, false negatives, and true negatives.
    tp = np.sum(A[np.ix_(idx_output_positive, idx_label_positive)])
    fp = np.sum(A[np.ix_(idx_output_positive, idx_label_negative)])
    fn = np.sum(A[np.ix_(idx_output_negative, idx_label_positive)])
    tn = np.sum(A[np.ix_(idx_output_negative, idx_label_negative)])
    total_patients = tp + fp + fn + tn

    # Compute total cost for all patients.
    total_cost = (
        cost_algorithm(total_patients)
        + cost_expert(tp + fp, total_patients)
        + cost_treatment(tp)
        + cost_error(fn)
    )

    # Compute mean cost per patient.
    if total_patients > 0:
        mean_cost = total_cost / total_patients
    else:
        mean_cost = float("nan")

    return mean_cost


@deprecated(reason="only used in the unofficial phase of the Challenge")
def compute_challenge_score(
    labels: np.ndarray, outputs: np.ndarray, classes: Sequence[str]
) -> float:
    """
    Compute Challenge score.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names

    Returns
    -------
    mean_score: float,
        mean Challenge score
    """
    # Define costs. Better to load these costs from an external file instead of defining them here.
    c_algorithm = 1  # Cost for algorithmic prescreening.
    c_gp = 250  # Cost for screening from a general practitioner (GP).
    c_specialist = 500  # Cost for screening from a specialist.
    c_treatment = 1000  # Cost for treatment.
    c_error = 10000  # Cost for diagnostic error.
    alpha = 0.5  # Fraction of murmur unknown cases that are positive.

    num_patients, num_classes = np.shape(labels)

    A = compute_confusion_matrix(labels, outputs)

    idx_positive = classes.index("Present")
    idx_unknown = classes.index("Unknown")
    idx_negative = classes.index("Absent")

    n_pp = A[idx_positive, idx_positive]
    n_pu = A[idx_positive, idx_unknown]
    n_pn = A[idx_positive, idx_negative]
    n_up = A[idx_unknown, idx_positive]
    n_uu = A[idx_unknown, idx_unknown]
    n_un = A[idx_unknown, idx_negative]
    n_np = A[idx_negative, idx_positive]
    n_nu = A[idx_negative, idx_unknown]
    n_nn = A[idx_negative, idx_negative]

    n_total = n_pp + n_pu + n_pn + n_up + n_uu + n_un + n_np + n_nu + n_nn

    total_score = (
        c_algorithm * n_total
        + c_gp * (n_pp + n_pu + n_pn)
        + c_specialist * (n_pu + n_up + n_uu + n_un)
        + c_treatment * (n_pp + alpha * n_pu + n_up + alpha * n_uu)
        + c_error * (n_np + alpha * n_nu)
    )
    if n_total > 0:
        mean_score = total_score / n_total
    else:
        mean_score = float("nan")

    return mean_score


class CINC2022Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CINC2022Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        """

        Parameters
        ----------
        model: Module,
            the model to be trained
        model_config: dict,
            the configuration of the model,
            used to keep a record in the checkpoints
        train_config: dict,
            the configuration of the training,
            including configurations for the data loader, for the optimization, etc.
            will also be recorded in the checkpoints.
            `train_config` should at least contain the following keys:
                "monitor": str,
                "loss": str,
                "n_epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "lr_scheduler": str,
                    "lr_step_size": int, optional, depending on the scheduler
                    "lr_gamma": float, optional, depending on the scheduler
                    "max_lr": float, optional, depending on the scheduler
                "optimizer": str,
                    "decay": float, optional, depending on the optimizer
                    "momentum": float, optional, depending on the optimizer
        device: torch.device, optional,
            the device to be used for training
        lazy: bool, default True,
            whether to initialize the data loader lazily

        """
        super().__init__(
            model=model,
            dataset_cls=CinC2022Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=True,
                lazy=False,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if self.device == torch.device("cpu"):
            num_workers = 1
        else:
            num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _setup_augmenter_manager(self) -> None:
        """ """
        self.augmenter_manager = AugmenterManager.from_config(
            config=self.train_config[self.train_config.task]
        )

    def _setup_criterion(self) -> None:
        """ """
        loss_kw = (
            self.train_config[self.train_config.task]
            .get("loss_kw", {})
            .get(self._criterion_key, {})
        )
        if self.train_config.loss[self._criterion_key] == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif (
            self.train_config.loss[self._criterion_key]
            == "BCEWithLogitsWithClassWeightLoss"
        ):
            self.criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "BCELoss":
            self.criterion = nn.BCELoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "FocalLoss":
            self.criterion = FocalLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "AsymmetricLoss":
            self.criterion = AsymmetricLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(
                f"loss `{self.train_config.loss}` not implemented! "
                "Please use one of the following: `BCEWithLogitsLoss`, `BCEWithLogitsWithClassWeightLoss`, "
                "`BCELoss`, `MaskedBCEWithLogitsLoss`, `MaskedBCEWithLogitsLoss`, `FocalLoss`, "
                "`AsymmetricLoss`, `CrossEntropyLoss`, or override this method to setup your own criterion."
            )
        self.criterion.to(device=self.device, dtype=self.dtype)

    def train_one_epoch(self, pbar: tqdm) -> None:
        """
        train one epoch, and update the progress bar

        Parameters
        ----------
        pbar: tqdm,
            the progress bar for training

        """
        if (
            self.epoch
            >= self.train_config[self.train_config.task].freeze_backbone_at
            > 0
        ):
            self._model.freeze_backbone(True)
        else:
            self._model.freeze_backbone(False)
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            n_samples = input_tensors["waveforms"].shape[self.batch_dim]
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            input_tensors["waveforms"] = self.augmenter_manager(
                input_tensors["waveforms"]
            )

            # out_tensors is a dict of tensors, with the following items (some are optional):
            # - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            # - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            # - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            # - "outcome_loss": loss of the outcome predictions
            # - "segmentation_loss": loss of the segmentation predictions
            # - "total_extra_loss": total loss of the extra heads
            out_tensors = self.run_one_step(input_tensors)

            # WARNING:
            # When `module` (self._model) returns a scalar (i.e., 0-dimensional tensor) in forward(),
            # `DataParallel` will return a vector of length equal to number of devices used in data parallelism,
            # containing the result from each device.
            # ref. https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
            loss = self.criterion(
                out_tensors[self._criterion_key],
                input_tensors[self._criterion_key].to(
                    dtype=self.dtype, device=self.device
                ),
            ).to(dtype=self.dtype, device=self.device) + out_tensors.get(
                "total_extra_loss",
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
            ).mean().to(
                dtype=self.dtype, device=self.device
            )

            if self.train_config.flooding_level > 0:
                flood = (
                    loss - self.train_config.flooding_level
                ).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)

    def run_one_step(
        self, input_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """

        Parameters
        ----------
        input_tensors: dict of Tensors,
            the tensors to be processed for training one step (batch), with the following items:
                - "waveforms" (required): the input waveforms
                - "murmur" (optional): the murmur labels, for classification task and multi task
                - "outcome" (optional): the outcome labels, for classification task and multi task
                - "segmentation" (optional): the segmentation labels, for segmentation task and multi task

        Returns
        -------
        out_tensors: dict of Tensors, with the following items (some are optional):
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        waveforms = input_tensors.pop("waveforms").to(self.device)
        input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        out_tensors = self.model(waveforms, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """

        self.model.eval()

        all_outputs = []
        all_labels = []

        for input_tensors in data_loader:
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            waveforms = input_tensors.pop("waveforms")
            waveforms = waveforms.to(device=self.device, dtype=self.dtype)
            labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}

            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            all_outputs.append(self._model.inference(waveforms))

        if self.val_train_loader is not None and self.train_config.task not in [
            "segmentation"
        ]:
            log_head_num = 5
            head_scalar_preds = all_outputs[0].murmur_output.prob[:log_head_num]
            head_bin_preds = all_outputs[0].murmur_output.bin_pred[:log_head_num]
            head_preds_classes = [
                np.array(all_outputs[0].murmur_output.classes)[np.where(row)[0]]
                for row in head_bin_preds
            ]
            head_labels = all_labels[0]["murmur"][:log_head_num]
            head_labels_classes = [
                np.array(all_outputs[0].murmur_output.classes)[np.where(row)]
                if head_labels.ndim == 2
                else np.array(all_outputs[0].murmur_output.classes)[row]
                for row in head_labels
            ]
            log_head_num = min(log_head_num, len(head_scalar_preds))
            for n in range(log_head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                murmur scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                murmur binary prediction:    {head_bin_preds[n].tolist()}
                murmur labels:               {head_labels[n].astype(int).tolist()}
                murmur predicted classes:    {head_preds_classes[n].tolist()}
                murmur label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)
            if "outcome" in input_tensors:
                head_scalar_preds = all_outputs[0].outcome_output.prob[:log_head_num]
                head_bin_preds = all_outputs[0].outcome_output.bin_pred[:log_head_num]
                head_preds_classes = [
                    np.array(all_outputs[0].outcome_output.classes)[np.where(row)[0]]
                    for row in head_bin_preds
                ]
                head_labels = all_labels[0]["outcome"][:log_head_num]
                head_labels_classes = [
                    np.array(all_outputs[0].outcome_output.classes)[np.where(row)[0]]
                    if head_labels.ndim == 2
                    else np.array(all_outputs[0].outcome_output.classes)[row]
                    for row in head_labels
                ]
                log_head_num = min(log_head_num, len(head_scalar_preds))
                for n in range(log_head_num):
                    msg = textwrap.dedent(
                        f"""
                    ----------------------------------------------
                    outcome scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                    outcome binary prediction:    {head_bin_preds[n].tolist()}
                    outcome labels:               {head_labels[n].astype(int).tolist()}
                    outcome predicted classes:    {head_preds_classes[n].tolist()}
                    outcome label classes:        {head_labels_classes[n].tolist()}
                    ----------------------------------------------
                    """
                    )
                    self.log_manager.log_message(msg)

        eval_res = compute_challenge_metrics(
            labels=all_labels,
            outputs=all_outputs,
            require_both=False,
        )
        # eval_res contains the following items:
        # murmur_auroc: float,
        #     the macro-averaged area under the receiver operating characteristic curve for the murmur predictions
        # murmur_auprc: float,
        #     the macro-averaged area under the precision-recall curve for the murmur predictions
        # murmur_f_measure: float,
        #     the macro-averaged F-measure for the murmur predictions
        # murmur_accuracy: float,
        #     the accuracy for the murmur predictions
        # murmur_weighted_accuracy: float,
        #     the weighted accuracy for the murmur predictions
        # murmur_cost: float,
        #     the challenge cost for the murmur predictions
        # outcome_auroc: float,
        #     the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
        # outcome_auprc: float,
        #     the macro-averaged area under the precision-recall curve for the outcome predictions
        # outcome_f_measure: float,
        #     the macro-averaged F-measure for the outcome predictions
        # outcome_accuracy: float,
        #     the accuracy for the outcome predictions
        # outcome_weighted_accuracy: float,
        #     the weighted accuracy for the outcome predictions
        # outcome_cost: float,
        #     the challenge cost for the outcome predictions

        weighted_cost = 0
        if eval_res.get("murmur_cost", None) is not None:
            weighted_cost += (
                eval_res["murmur_cost"]
                * self.train_config[self.train_config.task].head_weights.murmur
            )
        if eval_res.get("outcome_cost", None) is not None:
            weighted_cost += (
                eval_res["outcome_cost"]
                * self.train_config[self.train_config.task].head_weights.outcome
            )
        eval_res["neg_weighted_cost"] = -weighted_cost

        # in case possible memeory leakage?
        del all_labels
        del all_outputs

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        prefix = f"task-{self.train_config.task}_{self._model.__name__}"
        if hasattr(self.model_config, "cnn"):
            prefix = f"{prefix}_{self.model_config.cnn.name}_epoch"
        else:
            prefix = f"{prefix}_epoch"
        return prefix

    def extra_log_suffix(self) -> str:
        suffix = f"task-{self.train_config.task}_{super().extra_log_suffix()}"
        if hasattr(self.model_config, "cnn"):
            suffix = f"{suffix}_{self.model_config.cnn.name}"
        return suffix

    @property
    def _criterion_key(self) -> str:
        return {
            "multi_task": "murmur",
            "classification": "murmur",
            "segmentation": "segmentation",
        }[self.train_config.task]


def collate_fn(
    batch: Sequence[Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]]
) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """ """
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        collated = default_collate_fn([tuple(b[k] for k in keys) for b in batch])
        return {k: collated[i] for i, k in enumerate(keys)}
    else:
        return default_collate_fn(batch)


def _set_task(task: str, config: CFG) -> None:
    """"""
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]


TASK = "classification"  # "multi_task"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_MAP = {
    "crnn": CRNN_CINC2022,
}


dr = CINC2022Reader(_DB_DIR)
dr.download(compressed=True)
dr._ls_rec()
del dr


def test_dataset():
    """ """
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = _DB_DIR

    ds_train = CinC2022Dataset(ds_config, TASK, training=True, lazy=True)
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)

    ds_train._load_all_data()
    ds_val._load_all_data()

    print("dataset test passed")


def test_models():
    """ """
    model = CRNN_CINC2022(ModelCfg[TASK])
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = _DB_DIR
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)
    ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, input_tensors in enumerate(dl):
        waveforms = input_tensors.pop("waveforms").to(DEVICE)
        # input_tensors = {k: v.to(DEVICE) for k, v in input_tensors.items()}
        # out_tensors = model(waveforms, input_tensors)
        print(model.inference(waveforms))
        if idx > 10:
            break

    print("models test passed")


def test_challenge_metrics():
    """ """
    outputs = [
        CINC2022Outputs(
            murmur_output=ClassificationOutput(
                classes=["Present", "Unknown", "Absent"],
                prob=np.array([[0.75, 0.15, 0.1]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0, 0]]),
            ),
            outcome_output=ClassificationOutput(
                classes=["Abnormal", "Normal"],
                prob=np.array([[0.6, 0.4]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0]]),
            ),
            segmentation_output=None,
        ),
        CINC2022Outputs(
            murmur_output=ClassificationOutput(
                classes=["Present", "Unknown", "Absent"],
                prob=np.array([[0.3443752, 0.32366553, 0.33195925]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0, 0]]),
            ),
            outcome_output=ClassificationOutput(
                classes=["Abnormal", "Normal"],
                prob=np.array([[0.5230, 0.0202]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0]]),
            ),
            segmentation_output=None,
        ),
    ]
    labels = [
        {
            "murmur": np.array([[0.0, 0.0, 1.0]]),
            "outcome": np.array([0]),
        },
        {
            "murmur": np.array([[0.0, 1.0, 0.0]]),
            "outcome": np.array([1]),
        },
    ]

    compute_challenge_metrics(labels, outputs)

    print("challenge metrics test passed")


def test_trainer():
    """ """
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = _DB_DIR
    train_config.debug = True

    train_config.n_epochs = 1
    train_config.batch_size = 4  # 16G (Tesla T4)

    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    assert train_config[TASK].model_name == "crnn"
    model_name = model_config.model_name = train_config[TASK].model_name
    model_config[model_name].cnn_name = train_config[TASK].cnn_name
    model_config[model_name].rnn_name = train_config[TASK].rnn_name
    model_config[model_name].attn_name = train_config[TASK].attn_name

    model_cls = _MODEL_MAP[model_config.model_name]
    model_cls.__DEBUG__ = False

    model = model_cls(config=model_config)
    model.to(device=DEVICE)

    trainer = CINC2022Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )

    best_state_dict = trainer.train()
