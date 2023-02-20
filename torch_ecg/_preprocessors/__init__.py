"""
Preprocessors for signals of numpy array format
===============================================

This module contains a set of preprocessors for signals of numpy array format.

.. currentmodule:: torch_ecg._preprocessors

.. autosummary::
    :toctree: generated/
    :recursive:

    PreprocManager
    PreProcessor
    BandPass
    BaselineRemove
    Normalize
    MinMaxNormalize
    NaiveNormalize
    ZScoreNormalize
    Resample
    preprocess_multi_lead_signal
    preprocess_single_lead_signal

"""

from .bandpass import BandPass
from .base import (
    PreProcessor,
    preprocess_multi_lead_signal,
    preprocess_single_lead_signal,
)
from .baseline_remove import BaselineRemove
from .normalize import MinMaxNormalize, NaiveNormalize, Normalize, ZScoreNormalize
from .preproc_manager import PreprocManager
from .resample import Resample

__all__ = [
    "PreprocManager",
    "PreProcessor",
    "BandPass",
    "BaselineRemove",
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
    "Resample",
    "preprocess_multi_lead_signal",
    "preprocess_single_lead_signal",
]
