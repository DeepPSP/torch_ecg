"""
utilities for computing metrics.

NOTE that only the widely used metrics are implemented here,
challenge (e.g. CinC, CPSC series) specific metrics are not included.

"""

from numbers import Real
from typing import Dict, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
from torch import Tensor

from .misc import add_docstring
from .utils_data import ECGWaveForm, ECGWaveFormNames, masks_to_waveforms
from ..cfg import DEFAULTS


__all__ = [
    "top_n_accuracy",
    "confusion_matrix",
    "ovr_confusion_matrix",
    "QRS_score",
    "metrics_from_confusion_matrix",
    "compute_wave_delineation_metrics",
]


def top_n_accuracy(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    n: Union[int, Sequence[int]] = 1,
) -> Union[float, Dict[str, float]]:
    """

    Parameters
    ----------
    labels: np.ndarray or Tensor,
        labels of class indices,
        of shape (batch_size,) or (batch_size, d_1, ..., d_m)
    outputs: np.ndarray or Tensor,
        predicted probabilities, of shape (batch_size, num_classes) or (batch_size, d_1, ..., d_m, num_classes)
        of shape (batch_size, num_classes) or (batch_size, num_classes, d_1, ..., d_m)
    n: int or list of int,
        top n to be considered

    Returns
    -------
    acc: float or dict of float,
        top n accuracy

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> labels, outputs = DEFAULTS.RNG_randint(0, 10, (100)), DEFAULTS.RNG.uniform(0, 1, (100, 10))  # 100 samples, 10 classes
    >>> top_n_accuracy(labels, outputs, 3)
    0.32
    >>> top_n_accuracy(labels, outputs, [1,3,5])
    {'top_1_acc': 0.12, 'top_3_acc': 0.32, 'top_5_acc': 0.52}

    """
    assert outputs.shape[0] == labels.shape[0]
    labels, outputs = torch.as_tensor(labels), torch.as_tensor(outputs)
    batch_size, n_classes, *extra_dims = outputs.shape
    if isinstance(n, int):
        ln = [n]
    else:
        ln = n
    acc = {}
    for _n in ln:
        key = f"top_{_n}_acc"
        _, indices = torch.topk(
            outputs, _n, dim=1
        )  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
        pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
        pattern = f"batch_size {pattern} -> batch_size n {pattern}"
        correct = torch.sum(indices == einops.repeat(labels, pattern, n=_n))
        acc[key] = correct.item() / outputs.shape[0]
        for d in extra_dims:
            acc[key] = acc[key] / d
    if len(ln) == 1:
        return acc[key]
    return acc


def confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray or Tensor,
        binary labels, of shape: (n_samples, n_classes)
        or indices of each label class, of shape: (n_samples,)
    outputs: np.ndarray or Tensor,
        binary outputs, of shape: (n_samples, n_classes)
        or indices of each class predicted, of shape: (n_samples,)
    num_classes: int, optional,
        number of classes,
        if `labels` and `outputs` are both of shape (n_samples,),
        then `num_classes` must be specified.

    Returns
    -------
    cm: np.ndarray,
        confusion matrix, of shape: (n_classes, n_classes)

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    cm = np.zeros((num_classes, num_classes))
    for k in range(num_samples):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        cm[i, j] += 1

    return cm


def one_vs_rest_confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute binary one-vs-rest confusion matrices,
    where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray or Tensor,
        binary labels, of shape: (n_samples, n_classes)
        or indices of each label class, of shape: (n_samples,)
    outputs: np.ndarray or Tensor,
        binary outputs, of shape: (n_samples, n_classes)
        or indices of each class predicted, of shape: (n_samples,)
    num_classes: int, optional,
        number of classes,
        if `labels` and `outputs` are both of shape (n_samples,),
        then `num_classes` must be specified.

    Returns
    -------
    ovr_cm: np.ndarray,
        one-vs-rest confusion matrix, of shape: (n_classes, 2, 2)

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    ovr_cm = np.zeros((num_classes, 2, 2))
    for i in range(num_samples):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                ovr_cm[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                ovr_cm[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                ovr_cm[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                ovr_cm[j, 1, 1] += 1

    return ovr_cm


# alias
ovr_confusion_matrix = one_vs_rest_confusion_matrix


_METRICS_FROM_CONFUSION_MATRIX_PARAMS = """
    Compute macro {metric}, and {metrics} for each class.

    Parameters
    ----------
    labels: np.ndarray or Tensor,
        binary labels, of shape: (n_samples, n_classes)
        or indices of each label class, of shape: (n_samples,)
    outputs: np.ndarray or Tensor,
        binary outputs, of shape: (n_samples, n_classes)
        or indices of each class predicted, of shape: (n_samples,)
    num_classes: int, optional,
        number of classes,
        if `labels` and `outputs` are both of shape (n_samples,),
        then `num_classes` must be specified.
    weights: np.ndarray or Tensor, optional,
        weights for each class, of shape: (n_classes,),
        used to compute macro {metric},
"""


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(metric="metrics", metrics="metrics"),
    "prepend",
)
def metrics_from_confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Returns
    -------
    metrics: dict,
        metrics computed from the one-vs-rest confusion matrix

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> labels, outputs = DEFAULTS.RNG_randint(0,2,(100,10)), DEFAULTS.RNG_randint(0,2,(100,10))
    >>> metrics = metrics_from_confusion_matrix(labels, outputs)

    References
    ----------
    1. https://en.wikipedia.org/wiki/Precision_and_recall

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    num_samples, num_classes = np.shape(labels)

    ovr_cm = ovr_confusion_matrix(labels, outputs)

    # sens: sensitivity, recall, hit rate, or true positive rate
    # spec: specificity, selectivity or true negative rate
    # prec: precision or positive predictive value
    # npv: negative predictive value
    # jac: jaccard index, threat score, or critical success index
    # acc: accuracy
    # phi: phi coefficient, or matthews correlation coefficient
    # NOTE: never use repeat here, because it will cause bugs
    # sens, spec, prec, npv, jac, acc, phi = list(repeat(np.zeros(num_classes), 7))
    sens, spec, prec, npv, jac, acc, phi = [np.zeros(num_classes) for _ in range(7)]
    auroc = np.zeros(
        num_classes
    )  # area under the receiver-operater characteristic curve (ROC AUC)
    auprc = np.zeros(num_classes)  # area under the precision-recall curve
    for k in range(num_classes):
        tp, fp, fn, tn = (
            ovr_cm[k, 0, 0],
            ovr_cm[k, 0, 1],
            ovr_cm[k, 1, 0],
            ovr_cm[k, 1, 1],
        )
        if tp + fn > 0:
            sens[k] = tp / (tp + fn)
        else:
            sens[k] = float("nan")
        if tp + fp > 0:
            prec[k] = tp / (tp + fp)
        else:
            prec[k] = float("nan")
        if tn + fp > 0:
            spec[k] = tn / (tn + fp)
        else:
            spec[k] = float("nan")
        if tn + fn > 0:
            npv[k] = tn / (tn + fn)
        else:
            npv[k] = float("nan")
        if tp + fn + fp > 0:
            jac[k] = tp / (tp + fn + fp)
        else:
            jac[k] = float("nan")
        acc[k] = (tp + tn) / num_samples
        phi[k] = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )

        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_samples and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr_ = np.zeros(num_thresholds)
        tnr_ = np.zeros(num_thresholds)
        ppv_ = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr_[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr_[j] = float("nan")
            if fp[j] + tn[j]:
                tnr_[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr_[j] = float("nan")
            if tp[j] + fp[j]:
                ppv_[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv_[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr_[j + 1] - tpr_[j]) * (tnr_[j + 1] + tnr_[j])
            auprc[k] += (tpr_[j + 1] - tpr_[j]) * ppv_[j + 1]

    fnr = 1 - sens  # false negative rate, miss rate
    fpr = 1 - spec  # false positive rate, fall-out
    fdr = 1 - prec  # false discovery rate
    for_ = 1 - npv  # false omission rate
    plr = sens / fpr  # positive likelihood ratio
    nlr = fnr / spec  # negative likelihood ratio
    pt = np.sqrt(fpr) / (np.sqrt(sens) + np.sqrt(fpr))  # prevalence threshold
    ba = (sens + spec) / 2  # balanced accuracy
    f1 = 2 * sens * prec / (sens + prec)  # f1-measure
    fm = np.sqrt(prec * sens)  # fowlkes-mallows index
    bm = sens + spec - 1  # informedness, bookmaker informedness
    mk = prec + npv - 1  # markedness
    dor = plr / nlr  # diagnostic odds ratio

    if weights is None:
        _weights = np.ones(num_classes)
    else:
        _weights = weights / np.mean(weights)
    metrics = {}
    for m in [
        "sens",  # sensitivity, recall, hit rate, or true positive rate
        "spec",  # specificity, selectivity or true negative rate
        "prec",  # precision or positive predictive value
        "npv",  # negative predictive value
        "jac",  # jaccard index, threat score, or critical success index
        "acc",  # accuracy
        "phi",  # phi coefficient, or matthews correlation coefficient
        "fnr",  # false negative rate, miss rate
        "fpr",  # false positive rate, fall-out
        "fdr",  # false discovery rate
        "for_",  # false omission rate
        "plr",  # positive likelihood ratio
        "nlr",  # negative likelihood ratio
        "pt",  # prevalence threshold
        "ba",  # balanced accuracy
        "f1",  # f1-measure
        "fm",  # fowlkes-mallows index
        "bm",  # bookmaker informedness
        "mk",  # markedness
        "dor",  # diagnostic odds ratio
        "auroc",  # area under the receiver-operater characteristic curve (ROC AUC)
        "auprc",  # area under the precision-recall curve
    ]:
        metrics[m.strip("_")] = eval(m)
        metrics[f"macro_{m}".strip("_")] = (
            np.nanmean(eval(m) * _weights)
            if np.any(np.isfinite(eval(m)))
            else float("nan")
        )
    return metrics


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="F1-measure", metrics="F1-measures"
    ),
    "prepend",
)
def f_measure(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_f1: float,
        macro F1-measure
    f1: np.ndarray,
        F1-measures for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_f1"], m["f1"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="sensitivity", metrics="sensitivities"
    ),
    "prepend",
)
def sensitivity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_sens: float,
        macro sensitivity
    sens: np.ndarray,
        sensitivities for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_sens"], m["sens"]


# aliases
recall = sensitivity
true_positive_rate = sensitivity
hit_rate = sensitivity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="precision", metrics="precisions"
    ),
    "prepend",
)
def precision(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_prec: float,
        macro precision
    prec: np.ndarray,
        precisions for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_prec"], m["prec"]


# aliases
positive_predictive_value = precision


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="specificity", metrics="specificities"
    ),
    "prepend",
)
def specificity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_spec: float,
        macro specificity
    spec: np.ndarray,
        specificities for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_spec"], m["spec"]


# aliases
selectivity = specificity
true_negative_rate = specificity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="AUROC and macro AUPRC", metrics="AUPRCs, AUPRCs"
    ),
    "prepend",
)
def auc(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    macro_auroc: float,
        macro AUROC
    macro_auprc: float,
        macro AUPRC
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)
    auprc: np.ndarray,
        AUPRCs for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_auroc"], m["macro_auprc"], m["auroc"], m["auprc"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format(
        metric="accuracy", metrics="accuracies"
    ),
    "prepend",
)
def accuracy(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
    weights: Optional[Union[np.ndarray, Tensor]] = None,
) -> float:
    """
    Returns
    -------
    macro_acc: float,
        the macro accuracy
    acc: np.ndarray,
        accuracies for each class, of shape: (n_classes,)

    """
    m = metrics_from_confusion_matrix(labels, outputs, num_classes, weights)

    return m["macro_acc"], m["acc"]


def QRS_score(
    rpeaks_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rpeaks_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    thr: float = 0.075,
) -> float:
    """

    QRS accuracy score, proposed in CPSC2019.

    Parameters
    ----------
    rpeaks_truths: sequence,
        sequence of ground truths of rpeaks locations (indices) from multiple records
    rpeaks_preds: sequence,
        predictions of ground truths of rpeaks locations (indices) for multiple records
    fs: real number,
        sampling frequency of ECG signal
    thr: float, default 0.075,
        threshold for a prediction to be truth positive,
        with units in seconds,

    Returns
    -------
    rec_acc: float,
        accuracy of predictions

    """
    assert len(rpeaks_truths) == len(
        rpeaks_preds
    ), f"number of records does not match, truth indicates {len(rpeaks_truths)}, while pred indicates {len(rpeaks_preds)}"
    n_records = len(rpeaks_truths)
    record_flags = np.ones((len(rpeaks_truths),), dtype=float)
    thr_ = thr * fs

    for idx, (truth_arr, pred_arr) in enumerate(zip(rpeaks_truths, rpeaks_preds)):
        false_negative = 0
        false_positive = 0
        true_positive = 0
        extended_truth_arr = np.concatenate((truth_arr.astype(int), [int(9.5 * fs)]))
        for j, t_ind in enumerate(extended_truth_arr[:-1]):
            next_t_ind = extended_truth_arr[j + 1]
            loc = np.where(np.abs(pred_arr - t_ind) <= thr_)[0]
            if j == 0:
                err = np.where(
                    (pred_arr >= 0.5 * fs + thr_) & (pred_arr <= t_ind - thr_)
                )[0]
            else:
                err = np.array([], dtype=int)
            err = np.append(
                err,
                np.where((pred_arr >= t_ind + thr_) & (pred_arr <= next_t_ind - thr_))[
                    0
                ],
            )

            false_positive += len(err)
            if len(loc) >= 1:
                true_positive += 1
                false_positive += len(loc) - 1
            elif len(loc) == 0:
                false_negative += 1

        if false_negative + false_positive > 1:
            record_flags[idx] = 0
        elif false_negative == 1 and false_positive == 0:
            record_flags[idx] = 0.3
        elif false_negative == 0 and false_positive == 1:
            record_flags[idx] = 0.7

    rec_acc = round(np.sum(record_flags) / n_records, 4)

    return rec_acc


def cls_to_bin(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    if isinstance(labels, Tensor):
        labels = labels.cpu().numpy()
    if isinstance(outputs, Tensor):
        outputs = outputs.cpu().numpy()
    if labels.ndim == outputs.ndim == 1:
        assert num_classes is not None
        shape = (labels.shape[0], num_classes)
        labels = _cls_to_bin(labels, shape)
        outputs = _cls_to_bin(outputs, shape)
    elif labels.ndim == 1:
        shape = outputs.shape
        labels = _cls_to_bin(labels, shape)
    elif outputs.ndim == 1:
        shape = labels.shape
        outputs = _cls_to_bin(outputs, shape)
    return labels, outputs


def _cls_to_bin(cls: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    """ """
    bin_ = np.zeros(shape)
    for i in range(shape[0]):
        bin_[i, cls[i]] = 1
    return bin_


def compute_wave_delineation_metrics(
    truth_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    class_map: Dict[str, int],
    fs: Real,
    mask_format: str = "channel_first",
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    f"""

    compute metrics for the task of ECG wave delineation
    (sensitivity, precision, f1_score, mean error and standard deviation of the mean errors)
    for multiple evaluations

    Parameters
    ----------
    truth_masks: sequence of ndarray,
        a sequence of ground truth masks,
        each of which can also hold multiple masks from different samples (differ by record or by lead).
        Each mask is of shape (n_channels, n_timesteps) or (n_timesteps, n_channels)
    pred_masks: sequence of ndarray,
        predictions corresponding to `truth_masks`,
        of the same shapes.
    class_map: dict,
        class map, mapping names to waves to numbers from 0 to n_classes-1,
        the keys should contain {", ".join([f'"{item}"' for item in ECGWaveFormNames])}.
    fs: real number,
        sampling frequency of the signal corresponding to the masks,
        used to compute the duration of each waveform,
        hence the error and standard deviations of errors
    mask_format: str, default "channel_first",
        format of the mask, one of the following:
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first')
    tol: float, default 0.15,
        tolerance for the duration of the waveform,
        with units in seconds

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
        n_masks = (
            tm.shape[0]
            if mask_format.lower() in ["channel_first", "lead_first"]
            else tm.shape[1]
        )

        new_t = masks_to_waveforms(tm, class_map, fs, mask_format)
        new_t = [
            new_t[f"lead_{idx+1}"] for idx in range(n_masks)
        ]  # list of list of `ECGWaveForm`s
        truth_waveforms += new_t

        new_p = masks_to_waveforms(pm, class_map, fs, mask_format)
        new_p = [
            new_p[f"lead_{idx+1}"] for idx in range(n_masks)
        ]  # list of list of `ECGWaveForm`s
        pred_waveforms += new_p

    scorings = compute_metrics_waveform(truth_waveforms, pred_waveforms, fs, tol)

    return scorings


def compute_metrics_waveform(
    truth_waveforms: Sequence[Sequence[ECGWaveForm]],
    pred_waveforms: Sequence[Sequence[ECGWaveForm]],
    fs: Real,
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    """

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
    tol: float, default 0.15,
        tolerance for the duration of the waveform,
        with units in seconds

    Returns
    -------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation

    """
    truth_positive = dict(
        {
            f"{wave}_{term}": 0
            for wave in ECGWaveFormNames
            for term in ["onset", "offset"]
        }
    )
    false_positive = dict(
        {
            f"{wave}_{term}": 0
            for wave in ECGWaveFormNames
            for term in ["onset", "offset"]
        }
    )
    false_negative = dict(
        {
            f"{wave}_{term}": 0
            for wave in ECGWaveFormNames
            for term in ["onset", "offset"]
        }
    )
    errors = dict(
        {
            f"{wave}_{term}": []
            for wave in ECGWaveFormNames
            for term in ["onset", "offset"]
        }
    )
    # accumulating results
    for tw, pw in zip(truth_waveforms, pred_waveforms):
        s = _compute_metrics_waveform(tw, pw, fs, tol)
        for wave in [
            "pwave",
            "qrs",
            "twave",
        ]:
            for term in ["onset", "offset"]:
                truth_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"][
                    "truth_positive"
                ]
                false_positive[f"{wave}_{term}"] += s[f"{wave}_{term}"][
                    "false_positive"
                ]
                false_negative[f"{wave}_{term}"] += s[f"{wave}_{term}"][
                    "false_negative"
                ]
                errors[f"{wave}_{term}"] += s[f"{wave}_{term}"]["errors"]
    scorings = dict()
    for wave in ECGWaveFormNames:
        for term in ["onset", "offset"]:
            tp = truth_positive[f"{wave}_{term}"]
            fp = false_positive[f"{wave}_{term}"]
            fn = false_negative[f"{wave}_{term}"]
            err = errors[f"{wave}_{term}"]
            sensitivity = tp / (tp + fn + DEFAULTS.eps)
            precision = tp / (tp + fp + DEFAULTS.eps)
            f1_score = (
                2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
            )
            mean_error = np.mean(err) * 1000 / fs if len(err) > 0 else np.nan
            standard_deviation = np.std(err) * 1000 / fs if len(err) > 0 else np.nan
            scorings[f"{wave}_{term}"] = dict(
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )

    return scorings


def _compute_metrics_waveform(
    truths: Sequence[ECGWaveForm],
    preds: Sequence[ECGWaveForm],
    fs: Real,
    tol: Real = 0.15,
) -> Dict[str, Dict[str, float]]:
    """

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
    tol: float, default 0.15,
        tolerance for the duration of the waveform,
        with units in seconds

    Returns
    -------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation

    """
    pwave_onset_truths, pwave_offset_truths, pwave_onset_preds, pwave_offset_preds = (
        [],
        [],
        [],
        [],
    )
    qrs_onset_truths, qrs_offset_truths, qrs_onset_preds, qrs_offset_preds = (
        [],
        [],
        [],
        [],
    )
    twave_onset_truths, twave_offset_truths, twave_onset_preds, twave_offset_preds = (
        [],
        [],
        [],
        [],
    )

    for item in ["truths", "preds"]:
        for w in eval(item):
            for term in ["onset", "offset"]:
                eval(f"{w.name}_{term}_{item}.append(w.{term})")

    scorings = dict()
    for wave in ECGWaveFormNames:
        for term in ["onset", "offset"]:
            (
                truth_positive,
                false_negative,
                false_positive,
                errors,
                sensitivity,
                precision,
                f1_score,
                mean_error,
                standard_deviation,
            ) = _compute_metrics_base(
                eval(f"{wave}_{term}_truths"), eval(f"{wave}_{term}_preds"), fs, tol
            )
            scorings[f"{wave}_{term}"] = dict(
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


def _compute_metrics_base(
    truths: Sequence[Real], preds: Sequence[Real], fs: Real, tol: Real = 0.15
) -> Dict[str, float]:
    r"""

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
    tol: float, default 0.15,
        tolerance for the duration of the waveform,
        with units in seconds

    Returns
    -------
    tuple of metrics:
        truth_positive, false_negative, false_positive, errors,
        sensitivity, precision, f1_score, mean_error, standard_deviation
        see ref. \[[1](#ref1)\]

    References
    ----------
    1. <a name="ref1"></a> Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.

    """
    _tolerance = round(tol * fs)
    _truths = np.array(truths)
    _preds = np.array(preds)
    truth_positive, false_positive, false_negative = 0, 0, 0
    errors = []
    n_included = 0
    for point in truths:
        _pred = _preds[np.where(np.abs(_preds - point) <= _tolerance)[0].tolist()]
        if len(_pred) > 0:
            truth_positive += 1
            idx = np.argmin(np.abs(_pred - point))
            errors.append(_pred[idx] - point)
        else:
            false_negative += 1
        n_included += len(_pred)

    false_positive = len(_preds) - truth_positive

    sensitivity = truth_positive / (truth_positive + false_negative + DEFAULTS.eps)
    precision = truth_positive / (truth_positive + false_positive + DEFAULTS.eps)
    f1_score = 2 * sensitivity * precision / (sensitivity + precision + DEFAULTS.eps)
    mean_error = np.mean(errors) * 1000 / fs if len(errors) > 0 else np.nan
    standard_deviation = np.std(errors) * 1000 / fs if len(errors) > 0 else np.nan

    return (
        truth_positive,
        false_negative,
        false_positive,
        errors,
        sensitivity,
        precision,
        f1_score,
        mean_error,
        standard_deviation,
    )
