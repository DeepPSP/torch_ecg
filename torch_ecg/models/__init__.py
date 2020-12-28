"""
Resources:
----------
1. ECG CRNN
2. ECG sequence labeling models
3. ECG UNets
4. LSTM models using RR intervals as inputs
5. ECG ``Object Detection'' models
6. to add more...
"""

from .ecg_crnn import ECG_CRNN
from .ecg_seq_lab_net import ECG_SEQ_LAB_NET
from .ecg_subtract_unet import ECG_SUBTRACT_UNET
from .ecg_unet import ECG_UNET
from .rr_lstm import RR_LSTM
from .grad_cam import GradCam

from .nets import (
    Mish, Swish,
    Initializers, Activations,
    Bn_Activation, Conv_Bn_Activation,
    MultiConv, BranchedConv,
    DownSample,
    BidirectionalLSTM, StackedLSTM,
    AttentionWithContext,
    MultiHeadAttention, SelfAttention,
    AttentivePooling,
    ZeroPadding,
    SeqLin,
    NonLocalBlock, SEBlock, GlobalContextBlock,
    CRF, ExtendedCRF,
    WeightedBCELoss, BCEWithLogitsWithClassWeightLoss,
)

from .cnn import (
    VGGBlock, VGG16,
    ResNetBasicBlock, ResNet,
    MultiScopicBasicBlock, MultiScopicBranch, MultiScopicCNN,
    DenseBasicBlock, DenseBottleNeck, DenseMacroBlock, DenseTransition, DenseNet,
)


# __all__ = [s for s in dir() if not s.startswith('_')]
__all__ = [
    # large models
    "ECG_CRNN",
    "ECG_SEQ_LAB_NET",
    "ECG_SUBTRACT_UNET",
    "ECG_UNET",
    "RR_LSTM",

    # grad cam
    "GradCam",

    # building blocks
    "Mish", "Swish",
    "Initializers", "Activations",
    "Bn_Activation", "Conv_Bn_Activation",
    "MultiConv", "BranchedConv",
    "DownSample",
    "BidirectionalLSTM", "StackedLSTM",
    "AttentionWithContext",
    "MultiHeadAttention", "SelfAttention",
    "AttentivePooling",
    "ZeroPadding",
    "SeqLin",
    "NonLocalBlock", "SEBlock", "GlobalContextBlock",
    "CRF", "ExtendedCRF",
    "WeightedBCELoss", "BCEWithLogitsWithClassWeightLoss",

    # named CNNs
    "VGGBlock", "VGG16",
    "ResNetBasicBlock", "ResNetBottleNeck", "ResNet",
    "MultiScopicBasicBlock", "MultiScopicBranch", "MultiScopicCNN",
    "DenseBasicBlock", "DenseBottleNeck", "DenseMacroBlock", "DenseTransition", "DenseNet",
]
