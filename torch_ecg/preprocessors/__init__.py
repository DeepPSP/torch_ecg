"""
PreProcessors as torch.nn.Module
"""

from .bandpass import BandPass
from .baseline_remove import BaselineRemove
from .normalize import MinMaxNormalize, NaiveNormalize, Normalize, ZScoreNormalize
from .preproc_manager import PreprocManager
from .resample import Resample

__all__ = [
    "PreprocManager",
    "BandPass",
    "BaselineRemove",
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
    "Resample",
]
