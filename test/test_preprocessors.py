"""
"""

from pytest import approx
import torch

try:
    import torch_ecg
except:
    import sys
    from pathlib import Path

    sys.path.append(Path(__file__).absolute().parent.parent)
    import torch_ecg

from torch_ecg._preprocessors import (
    BandPass,
    BaselineRemove,
    Normalize,
    MinMaxNormalize,
    NaiveNormalize,
    ZScoreNormalize,
    Resample,
)

from .test_data import load_test_clf_data
