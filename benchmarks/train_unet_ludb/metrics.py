"""

Reference
---------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.

Section 3.2 of ref. [1] describes the metrics

KEY points
----------
1. an onset or an offset is detected correctly, if their deviation from the doctor annotations does not exceed in absolute value the tolerance of 150 ms
2. if there is no corresponding critical point (onsets and offset of ECG waveforms P, QRS, T) in the test sample in the neighborhood of ±tolerance of the detected critical point, then the I type error is counted (false positive, FP)
3. if the algorithm does not detect a critical point, then the II type error is counted (false negative, FN)
"""
from numbers import Real
from typing import Union, Optional, Sequence, Dict, Tuple

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from easydict import EasyDict as ED

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.utils.misc import (
    ECGWaveForm,
    masks_to_waveforms,
)
from torch_ecg.cfg import DEFAULTS


__all__ = [
    "compute_metrics",
    "compute_metrics_waveform",
]


__TOLERANCE = 150  # ms
__WaveNames = ["pwave", "qrs", "twave"]


def compute_metrics(truth_masks:Sequence[np.ndarray],
                    pred_masks:Sequence[np.ndarray],
                    class_map:Dict[str, int],
                    fs:Real,
                    mask_format:str="channel_first") -> Dict[str, Dict[str, float]]:
    """ finished, checked,

    compute metrics
    (sensitivity, precision, f1_score, mean error and standard deviation of the mean errors)
    for multiple evaluations

    Parameters
    ----------
    truth_masks: sequence of ndarray,
        a sequence of ground truth masks,
        each of which can also hold multiple masks from different samples (differ by record or by lead)
    pred_masks: sequence of ndarray,
        predictions corresponding to `truth_masks`
    class_map: dict,
        class map, mapping names to waves to numbers from 0 to n_classes-1,
        the keys should contain 'pwave', 'qrs', 'twave'
    fs: real number,
        sampling frequency of the signal corresponding to the masks,
        used to compute the duration of each waveform,
        hence the error and standard deviations of errors
    mask_format: str, default "channel_first",
        format of the mask, one of the following:
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first')

    Returns
    -------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation
    """
    assert len(truth_masks) == len(pred_masks)
    truth_waveforms, pred_waveforms = [], []
    # compute for each element
    for tm, pm in zip(truth_masks, pred_masks):
        n_masks = tm.shape[0] if mask_format.lower() in ['channel_first', 'lead_first'] \
            else tm.shape[1]

        new_t = masks_to_waveforms(tm, class_map, fs, mask_format)
        new_t = [new_t[f"lead_{idx+1}"] for idx in range(n_masks)]  # list of list of `ECGWaveForm`s
        truth_waveforms += new_t

        new_p = masks_to_waveforms(pm, class_map, fs, mask_format)
        new_p = [new_p[f"lead_{idx+1}"] for idx in range(n_masks)]  # list of list of `ECGWaveForm`s
        pred_waveforms += new_p

    scorings = compute_metrics_waveform(truth_waveforms, pred_waveforms, fs)

    return scorings


def compute_metrics_waveform(truth_waveforms:Sequence[Sequence[ECGWaveForm]],
                             pred_waveforms:Sequence[Sequence[ECGWaveForm]],
                             fs:Real) -> Dict[str, Dict[str, float]]:
    """ finished, checked,

    compute the sensitivity, precision, f1_score, mean error and standard deviation of the mean errors,
    of evaluations on a multiple samples (differ by records, or leads)

    Parameters
    ----------
    truth_waveforms: sequence of sequence of `ECGWaveForm`s,
        the ground truth,
        each element is a sequence of `ECGWaveForm`s from the same sample
    pred_waveforms: sequence of sequence of `ECGWaveForm`s,
        the predictions corresponding to `truth_waveforms`,
        each element is a sequence of `ECGWaveForm`s from the same sample
    fs: real number,
        sampling frequency of the signal corresponding to the waveforms,
        used to compute the duration of each waveform,
        hence the error and standard deviations of errors

    Returns
    -------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation
    """
    truth_positive = ED({
        f"{wave}_{term}": 0 \
            for wave in ["pwave", "qrs", "twave",] for term in ["onset", "offset"]
    })
    false_positive = ED({
        f"{wave}_{term}": 0 \
            for wave in ["pwave", "qrs", "twave",] for term in ["onset", "offset"]
    })
    false_negative = ED({
        f"{wave}_{term}": 0 \
            for wave in ["pwave", "qrs", "twave",] for term in ["onset", "offset"]
    })
    errors = ED({
        f"{wave}_{term}": [] \
            for wave in ["pwave", "qrs", "twave",] for term in ["onset", "offset"]
    })
    # accumulating results
    for tw, pw in zip(truth_waveforms, pred_waveforms):
        s = _compute_metrics_waveform(tw, pw, fs)
        for wave in ["pwave", "qrs", "twave",]:
            for term in ["onset", "offset"]:
                truth_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"]["truth_positive"]
                false_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"]["false_positive"]
                false_negative[f"{wave}_{term}"] += s[f"{wave}_{term}"]["false_negative"]
                errors[f"{wave}_{term}"] += s[f"{wave}_{term}"]["errors"]
    scorings = ED()
    for wave in ["pwave", "qrs", "twave",]:
        for term in ["onset", "offset"]:
            tp = truth_positive[f"{wave}_{term}"]
            fp = false_positive[f"{wave}_{term}"]
            fn = false_negative[f"{wave}_{term}"]
            err = errors[f"{wave}_{term}"]
            sensitivity = tp / (tp + fn + DEFAULTS.eps)
            precision = tp / (tp + fp + DEFAULTS.eps)
            f1_score = 2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
            mean_error = np.mean(err) * 1000 / fs
            standard_deviation = np.std(err) * 1000 / fs
            scorings[f"{wave}_{term}"] = ED(
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )

    return scorings


def _compute_metrics_waveform(truths:Sequence[ECGWaveForm],
                              preds:Sequence[ECGWaveForm],
                              fs:Real) -> Dict[str, Dict[str, float]]:
    """ finished, checked,

    compute the sensitivity, precision, f1_score, mean error and standard deviation of the mean errors,
    of evaluations on a single sample (the same record, the same lead)

    Parameters
    ----------
    truths: sequence of `ECGWaveForm`s,
        the ground truth
    preds: sequence of `ECGWaveForm`s,
        the predictions corresponding to `truths`,
    fs: real number,
        sampling frequency of the signal corresponding to the waveforms,
        used to compute the duration of each waveform,
        hence the error and standard deviations of errors

    Returns
    -------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation
    """
    pwave_onset_truths, pwave_offset_truths, pwave_onset_preds, pwave_offset_preds = \
        [], [], [], []
    qrs_onset_truths, qrs_offset_truths, qrs_onset_preds, qrs_offset_preds = \
        [], [], [], []
    twave_onset_truths, twave_offset_truths, twave_onset_preds, twave_offset_preds = \
        [], [], [], []

    for item in ["truths", "preds"]:
        for w in eval(item):
            for term in ["onset", "offset"]:
                eval(f"{w.name}_{term}_{item}.append(w.{term})")

    scorings = ED()
    for wave in ["pwave", "qrs", "twave",]:
        for term in ["onset", "offset"]:
            truth_positive, false_negative, false_positive, errors, \
            sensitivity, precision, f1_score, mean_error, standard_deviation = \
                _compute_metrics_base(
                    eval(f"{wave}_{term}_truths"),
                    eval(f"{wave}_{term}_preds"),
                    fs
                )
            scorings[f"{wave}_{term}"] = ED(
                truth_positive=truth_positive,
                false_negative=false_negative,
                false_positive=false_positive,
                errors=errors,
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )
    return scorings


def _compute_metrics_base(truths:Sequence[Real],
                          preds:Sequence[Real],
                          fs:Real) -> Dict[str,float]:
    """ finished, checked,

    Parameters
    ----------
    truths: sequence of real numbers,
        ground truth of indices of corresponding critical points
    preds: sequence of real numbers,
        predicted indices of corresponding critical points
    fs: real number,
        sampling frequency of the signal corresponding to the critical points,
        used to compute the duration of each waveform,
        hence the error and standard deviations of errors

    Returns
    -------
    tuple of metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation
        see ref. [1]
    """
    _tolerance = __TOLERANCE * fs / 1000
    _truths = np.array(truths)
    _preds = np.array(preds)
    truth_positive, false_positive, false_negative = 0, 0, 0
    errors = []
    n_included = 0
    for point in truths:
        _pred = _preds[np.where(np.abs(_preds-point)<=_tolerance)[0].tolist()]
        if len(_pred) > 0:
            truth_positive += 1
            errors.append(_pred[0]-point)
        else:
            false_negative += 1
        n_included += len(_pred)
    
    # false_positive = len(_preds) - n_included
    false_positive = len(_preds) - truth_positive

    # print(f"""
    # truth_positive = {truth_positive}
    # false_positive = {false_positive}
    # false_negative = {false_negative}
    # """)

    # print(f"len(truths) = {len(truths)}, truth_positive + false_negative = {truth_positive + false_negative}")
    # print(f"len(preds) = {len(preds)}, truth_positive + false_positive = {truth_positive + false_positive}")

    sensitivity = truth_positive / (truth_positive + false_negative + DEFAULTS.eps)
    precision = truth_positive / (truth_positive + false_positive + DEFAULTS.eps)
    f1_score = 2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
    mean_error = np.mean(errors) * 1000 / fs
    standard_deviation = np.std(errors) * 1000 / fs

    return truth_positive, false_negative, false_positive, errors, sensitivity, precision, f1_score, mean_error, standard_deviation
