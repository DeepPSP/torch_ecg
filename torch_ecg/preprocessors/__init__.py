"""
torch_ecg.preprocessors
=======================

This module contains the preprocessors for signals of torch Tensor format.

.. contents:: torch_ecg.preprocessors
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.preprocessors

.. autosummary::
    :toctree: generated/
    :recursive:

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
