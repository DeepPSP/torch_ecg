"""
PreProcessors as torch.nn.Module
"""

from .normalize import (
    Normalize,
    MinMaxNormalize,
    NaiveNormalize,
    ZScoreNormalize,
)
from .resample import Resample


__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
    "Resample",
]
