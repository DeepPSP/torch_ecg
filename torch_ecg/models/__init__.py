"""
torch_ecg.models
================

This module contains the model architectures for ECG various tasks.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.models

Convolutional neural backbones
------------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    ResNet
    RegNet
    VGG16
    Xception
    DenseNet
    MobileNetV1
    MobileNetV2
    MobileNetV3
    MultiScopicCNN

Downstream task models
----------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    ECG_UNET
    ECG_SUBTRACT_UNET
    ECG_CRNN
    ECG_SEQ_LAB_NET
    RR_LSTM

Saliency analysis
-----------------
.. autosummary::
    :toctree: generated/
    :recursive:

    GradCam

"""

from . import _nets, loss
from .cnn import VGG16, DenseNet, MobileNetV1, MobileNetV2, MobileNetV3, MultiScopicCNN, RegNet, ResNet, Xception
from .ecg_crnn import ECG_CRNN
from .ecg_seq_lab_net import ECG_SEQ_LAB_NET
from .grad_cam import GradCam
from .rr_lstm import RR_LSTM
from .transformers import Transformer
from .unets import ECG_SUBTRACT_UNET, ECG_UNET

__all__ = [
    # CNN backbone
    "ResNet",
    "RegNet",
    "VGG16",
    "Xception",
    "DenseNet",
    "MobileNetV1",
    "MobileNetV2",
    "MobileNetV3",
    "MultiScopicCNN",
    # downstream task models
    "ECG_UNET",
    "ECG_SUBTRACT_UNET",
    "ECG_CRNN",
    "ECG_SEQ_LAB_NET",
    "GradCam",
    "RR_LSTM",
    # etc
    "Transformer",
    "loss",
    "_nets",
]
