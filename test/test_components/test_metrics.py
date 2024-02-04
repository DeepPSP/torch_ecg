"""
"""

import numpy as np
import pytest

from torch_ecg.cfg import DEFAULTS
from torch_ecg.components.metrics import ClassificationMetrics, Metrics, RPeaksDetectionMetrics, WaveDelineationMetrics
from torch_ecg.utils.utils_metrics import accuracy, auc, f_measure, precision, sensitivity, specificity, top_n_accuracy


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
    assert isinstance(cm.accuracy, float)
    assert isinstance(cm.precision, float)
    assert isinstance(cm.recall, float)
    assert isinstance(cm.sensitivity, float)
    assert isinstance(cm.hit_rate, float)
    assert isinstance(cm.true_positive_rate, float)
    assert isinstance(cm.specificity, float)
    assert isinstance(cm.selectivity, float)
    assert isinstance(cm.true_negative_rate, float)
    assert isinstance(cm.positive_predictive_value, float)
    assert isinstance(cm.negative_predictive_value, float)
    assert isinstance(cm.jaccard_index, float)
    assert isinstance(cm.threat_score, float)
    assert isinstance(cm.critical_success_index, float)
    assert isinstance(cm.phi_coefficient, float)
    assert isinstance(cm.matthews_correlation_coefficient, float)
    assert isinstance(cm.false_negative_rate, float)
    assert isinstance(cm.miss_rate, float)
    assert isinstance(cm.false_positive_rate, float)
    assert isinstance(cm.fall_out, float)
    assert isinstance(cm.false_discovery_rate, float)
    assert isinstance(cm.false_omission_rate, float)
    assert isinstance(cm.positive_likelihood_ratio, float)
    assert isinstance(cm.negative_likelihood_ratio, float)
    assert isinstance(cm.prevalence_threshold, float)
    assert isinstance(cm.balanced_accuracy, float)
    assert isinstance(cm.f1_measure, float)
    assert isinstance(cm.fowlkes_mallows_index, float)
    assert isinstance(cm.bookmaker_informedness, float)
    assert isinstance(cm.markedness, float)
    assert isinstance(cm.diagnostic_odds_ratio, float)
    assert isinstance(cm.area_under_the_receiver_operater_characteristic_curve, float)
    assert isinstance(cm.auroc, float)
    assert isinstance(cm.area_under_the_precision_recall_curve, float)
    assert isinstance(cm.auprc, float)

    assert isinstance(cm.classification_report, dict)
    assert cm.extra_metrics == {}

    cm.set_macro(False)
    assert isinstance(cm.accuracy, np.ndarray)
    assert isinstance(cm.precision, np.ndarray)
    assert isinstance(cm.recall, np.ndarray)
    assert isinstance(cm.sensitivity, np.ndarray)
    assert isinstance(cm.hit_rate, np.ndarray)
    assert isinstance(cm.true_positive_rate, np.ndarray)
    assert isinstance(cm.specificity, np.ndarray)
    assert isinstance(cm.selectivity, np.ndarray)
    assert isinstance(cm.true_negative_rate, np.ndarray)
    assert isinstance(cm.positive_predictive_value, np.ndarray)
    assert isinstance(cm.negative_predictive_value, np.ndarray)
    assert isinstance(cm.jaccard_index, np.ndarray)
    assert isinstance(cm.threat_score, np.ndarray)
    assert isinstance(cm.critical_success_index, np.ndarray)
    assert isinstance(cm.phi_coefficient, np.ndarray)
    assert isinstance(cm.matthews_correlation_coefficient, np.ndarray)
    assert isinstance(cm.false_negative_rate, np.ndarray)
    assert isinstance(cm.miss_rate, np.ndarray)
    assert isinstance(cm.false_positive_rate, np.ndarray)
    assert isinstance(cm.fall_out, np.ndarray)
    assert isinstance(cm.false_discovery_rate, np.ndarray)
    assert isinstance(cm.false_omission_rate, np.ndarray)
    assert isinstance(cm.positive_likelihood_ratio, np.ndarray)
    assert isinstance(cm.negative_likelihood_ratio, np.ndarray)
    assert isinstance(cm.prevalence_threshold, np.ndarray)
    assert isinstance(cm.balanced_accuracy, np.ndarray)
    assert isinstance(cm.f1_measure, np.ndarray)
    assert isinstance(cm.fowlkes_mallows_index, np.ndarray)
    assert isinstance(cm.bookmaker_informedness, np.ndarray)
    assert isinstance(cm.markedness, np.ndarray)
    assert isinstance(cm.diagnostic_odds_ratio, np.ndarray)
    assert isinstance(cm.area_under_the_receiver_operater_characteristic_curve, np.ndarray)
    assert isinstance(cm.auroc, np.ndarray)
    assert isinstance(cm.area_under_the_precision_recall_curve, np.ndarray)
    assert isinstance(cm.auprc, np.ndarray)

    with pytest.warns(RuntimeWarning, match="`outputs` is probably binary, AUC may be incorrect"):
        cm(labels, outputs_bin)
    with pytest.warns(RuntimeWarning, match="`outputs` is probably binary, AUC may be incorrect"):
        cm(labels, outputs_cate)

    assert str(cm) == repr(cm)


def test_rpeaks_detection_metrics():
    rdm = RPeaksDetectionMetrics()

    labels = [np.array([500, 1000])]
    outputs = [np.array([500, 700, 1000])]  # a false positive at 700
    rdm(labels, outputs, fs=500)
    assert rdm.qrs_score == pytest.approx(0.7)
    assert rdm.extra_metrics == {}
    assert str(rdm) == repr(rdm)


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
    assert isinstance(wdm.sensitivity, float)
    assert isinstance(wdm.precision, float)
    assert isinstance(wdm.f1_score, float)
    assert isinstance(wdm.mean_error, float)
    assert isinstance(wdm.standard_deviation, float)
    assert isinstance(wdm.jaccard_index, float)

    wdm.set_macro(False)
    assert isinstance(wdm.sensitivity, dict)
    assert isinstance(wdm.precision, dict)
    assert isinstance(wdm.f1_score, dict)
    assert isinstance(wdm.mean_error, dict)
    assert isinstance(wdm.standard_deviation, dict)
    assert isinstance(wdm.jaccard_index, dict)

    assert wdm.extra_metrics == {}

    assert str(wdm) == repr(wdm)


def test_base_metrics():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Metrics()


def test_metric_functions():
    # 100 samples, 10 classes
    labels = DEFAULTS.RNG_randint(0, 9, (100))
    outputs = DEFAULTS.RNG.uniform(0, 1, (100, 10))
    acc = top_n_accuracy(labels, outputs, 3)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1
    acc = top_n_accuracy(labels, outputs, [1, 3, 5])
    assert isinstance(acc, dict)
    assert acc.keys() == {"top_1_acc", "top_3_acc", "top_5_acc"}
    assert all([0 <= v <= 1 for v in acc.values()]), acc.values()

    macro_score, scores = f_measure(labels, outputs)
    assert isinstance(macro_score, float)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert 0 <= macro_score <= 1
    assert all([0 <= v <= 1 for v in scores]), scores

    macro_score, scores = precision(labels, outputs)
    assert isinstance(macro_score, float)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert 0 <= macro_score <= 1
    assert all([0 <= v <= 1 for v in scores]), scores

    macro_score, scores = sensitivity(labels, outputs)
    assert isinstance(macro_score, float)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert 0 <= macro_score <= 1
    assert all([0 <= v <= 1 for v in scores]), scores

    macro_score, scores = specificity(labels, outputs)
    assert isinstance(macro_score, float)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert 0 <= macro_score <= 1
    assert all([0 <= v <= 1 for v in scores]), scores

    macro_score, scores = accuracy(labels, outputs)
    assert isinstance(macro_score, float)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert 0 <= macro_score <= 1
    assert all([0 <= v <= 1 for v in scores]), scores

    macro_auroc, macro_auprc, auroc, auprc = auc(labels, outputs)
    assert isinstance(macro_auroc, float)
    assert isinstance(macro_auprc, float)
    assert isinstance(auroc, np.ndarray)
    assert isinstance(auprc, np.ndarray)
    assert auroc.shape == (10,)
    assert auprc.shape == (10,)
    assert 0 <= macro_auroc <= 1
    assert 0 <= macro_auprc <= 1
    assert all([0 <= v <= 1 for v in auroc]), auroc
    assert all([0 <= v <= 1 for v in auprc]), auprc

    with pytest.raises(
        ValueError,
        match="outputs must be of shape \\(n_samples, n_classes\\) to compute AUC",
    ):
        auc(labels, outputs[:, 0])
