"""
"""

import pytest
import numpy as np

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.cfg import DEFAULTS
from torch_ecg.components.metrics import (
    ClassificationMetrics,
    RPeaksDetectionMetrics,
    WaveDelineationMetrics,
)


def test_classification_metrics():
    cm = ClassificationMetrics()

    # binary labels (100 samples, 10 classes, multi-label)
    labels = DEFAULTS.RNG_randint(0, 1, (100, 10))
    # probability outputs (100 samples, 10 classes, multi-label)
    outputs_prob = DEFAULTS.RNG.random((100, 10))
    # binarize outputs (100 samples, 10 classes, multi-label)
    outputs_bin = DEFAULTS.RNG_randint(0, 1, (100, 10))
    # categorical outputs (100 samples, 10 classes)
    outputs_cate = DEFAULTS.RNG_randint(0, 9, (100,))

    cm(labels, outputs_prob)
    with pytest.warns(
        RuntimeWarning, match="`outputs` is probably binary, AUC may be incorrect"
    ):
        cm(labels, outputs_bin)
    with pytest.warns(
        RuntimeWarning, match="`outputs` is probably binary, AUC may be incorrect"
    ):
        cm(labels, outputs_cate)


def test_rpeaks_detection_metrics():
    rdm = RPeaksDetectionMetrics()

    labels = [np.array([500, 1000])]
    outputs = [np.array([500, 700, 1000])]  # a false positive at 700
    rdm(labels, outputs, fs=500)
    assert rdm.qrs_score == pytest.approx(0.7)


def test_wave_delineation_metrics():
    wdm = WaveDelineationMetrics()

    truth_masks = DEFAULTS.RNG_randint(0, 3, (1, 1, 5000))
    pred_masks = DEFAULTS.RNG_randint(0, 3, (1, 1, 5000))
    class_map = {
        "pwave": 1,
        "qrs": 2,
        "twave": 3,
    }
    wdm(truth_masks, pred_masks, class_map, fs=500)
