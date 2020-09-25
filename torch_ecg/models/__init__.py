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
    default_collate_fn,
)


__all__ = [s for s in dir() if not s.startswith('_')]
