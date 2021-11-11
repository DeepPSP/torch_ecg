"""
"""

from pytest import approx

from torch_ecg.preprocessors import (
    BandPass,
    BaselineRemove,
    NaiveNormalize,
    MinMaxNormalize,
    ZScoreNormalize,
    Resample,
)

from .test_data import load_test_clf_data
