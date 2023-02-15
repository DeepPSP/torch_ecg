"""
Preprocessors for signals of torch Tensor format
================================================

This module contains the preprocessors for signals of torch Tensor format.

.. currentmodule:: torch_ecg.preprocessors

.. autosummary::
    :toctree: generated/

    PreprocManager
    BandPass
    BaselineRemove
    Normalize
    MinMaxNormalize
    NaiveNormalize
    ZScoreNormalize
    Resample

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
