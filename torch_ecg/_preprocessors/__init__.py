"""
torch_ecg._preprocessors
========================

This module contains a set of preprocessors for signals of numpy array format.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

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
