"""
Resources:
----------
1. ECG CRNN
2. special detectors
3. to add more

Rules:
------
to write
"""

from .ecg_crnn import ECG_CRNN
from .ecg_seq_lab_net import ECG_SEQ_LAB_NET
from .ecg_subtract_unet import ECG_SUBTRACT_UNET
from .ecg_unet import ECG_UNET
from .grad_cam import GradCam

from .nets import (
    Mish, Swish,
    Initializers, Activations,
    Bn_Activation, Conv_Bn_Activation,
    MultiConv, BranchedConv,
    DownSample,
    BidirectionalLSTM, StackedLSTM,
    # "AML_Attention", "AML_GatedAttention",
    AttentionWithContext,
    MultiHeadAttention, SelfAttention,
    AttentivePooling,
    ZeroPadding,
    SeqLin,
    WeightedBCELoss, BCEWithLogitsWithClassWeightLoss,
)


# __all__ = [s for s in dir() if not s.startswith('_')]
__all__ = [
    "ECG_CRNN",
    "ECG_SEQ_LAB_NET",
    "ECG_SUBTRACT_UNET",
    "ECG_UNET",
    "GradCam",
    "Mish", "Swish",
    "Initializers", "Activations",
    "Bn_Activation", "Conv_Bn_Activation",
    "MultiConv", "BranchedConv",
    "DownSample",
    "BidirectionalLSTM", "StackedLSTM",
    # "AML_Attention", "AML_GatedAttention",
    "GlobalContextBlock",
    "AttentionWithContext",
    "MultiHeadAttention", "SelfAttention",
    "AttentivePooling",
    "ZeroPadding",
    "SeqLin",
    "NonLocalBlock", "SEBlock", "GlobalContextBlock",
    "WeightedBCELoss", "BCEWithLogitsWithClassWeightLoss",
]
