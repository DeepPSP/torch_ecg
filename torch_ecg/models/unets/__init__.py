"""
Various UNets for ECG waveform segmentation
"""

from .ecg_subtract_unet import ECG_SUBTRACT_UNET  # CPSC2019 unet
from .ecg_unet import ECG_UNET  # vanilla unet

__all__ = [
    "ECG_SUBTRACT_UNET",
    "ECG_UNET",
]
