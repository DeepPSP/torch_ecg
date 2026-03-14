"""
torch_ecg.preprocessors
=======================

This module contains the preprocessors for signals of torch Tensor format.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.preprocessors

.. autosummary::
    :toctree: generated/
    :recursive:

    PREPROCESSORS
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
from .registry import PREPROCESSORS
from .resample import Resample

__all__ = [
    "PREPROCESSORS",
    "PreprocManager",
    "BandPass",
    "BaselineRemove",
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
    "Resample",
]
