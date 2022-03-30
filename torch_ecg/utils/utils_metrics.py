"""
utilities for computing metrics.

NOTE that only the widely used metrics are implemented here,
challenge (e.g. CinC, CPSC series) specific metrics are not included.

"""

from itertools import repeat
from typing import Union, Optional, Dict, Tuple, Sequence
from numbers import Number, Real

import numpy as np
import einops
import torch
from torch import Tensor
from torch import nn

from .misc import add_docstring


__all__ = [
    "top_n_accuracy",
    "confusion_matrix",
    "ovr_confusion_matrix",
    "auc",
    "accuracy",
    "f_measure",
    "QRS_score",
]


def top_n_accuracy(preds: Tensor, labels: Tensor, n: int = 1) -> float:
    """

    Parameters
    ----------
    preds: Tensor,
        of shape (batch_size, num_classes) or (batch_size, num_classes, d_1, ..., d_m)
    labels: Tensor,
        of shape (batch_size,) or (batch_size, d_1, ..., d_m)
    n: int,
        top n to be considered

    Returns
    -------
    acc: float,
        top n accuracy

    """
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    _, indices = torch.topk(
        preds, n, dim=1
    )  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc = correct.item() / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
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
    A: np.ndarray,
        confusion matrix, of shape: (n_classes, n_classes)

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_samples):
        i = np.argmax(outputs[k, :])
        j = np.argmax(labels[k, :])
        A[i, j] += 1

    return A


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
    A: np.ndarray,
        one-vs-rest confusion matrix, of shape: (n_classes, 2, 2)

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1) for value in np.unique(labels))
    assert all(value in (0, 1) for value in np.unique(outputs))

    num_samples, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_samples):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


# alias
ovr_confusion_matrix = one_vs_rest_confusion_matrix


_METRICS_FROM_CONFUSION_MATRIX_PARAMS = """
    Compute macro {}, and {} for each class.
    
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
"""


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("metrics", "metrics"), "prepend"
)
def _metrics_from_confusion_matrix(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Returns
    -------
    metrics: dict,
        metrics computed from the one-vs-rest confusion matrix

    Examples
    --------
    >>> labels, outputs = np.random.randint(0,2,(100,10)), np.random.randint(0,2,(100,10))
    >>> metrics = _metrics_from_confusion_matrix(labels, outputs)

    References
    ----------
    1. https://en.wikipedia.org/wiki/Precision_and_recall

    """
    labels, outputs = cls_to_bin(labels, outputs, num_classes)
    num_samples, num_classes = np.shape(labels)

    A = ovr_confusion_matrix(labels, outputs)

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
    auroc = np.zeros(num_classes)  # area under the receiver-operater characteristic curve (ROC AUC)
    auprc = np.zeros(num_classes)  # area under the precision-recall curve
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if tp + fn > 0:
            sens[k] = tp / (tp + fn)
        else:
            sens[k] = float("nan")
        # continue
        if tp + fp > 0:
            prec[k] = tp / (tp + fp)
        else:
            prec[k] = float("nan")
        if tn + fn > 0:
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
            np.nanmean(eval(m)) if np.any(np.isfinite(eval(m))) else float("nan")
        )
    return metrics


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("F1-measure", "F1-measures"), "prepend"
)
def f_measure(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_f1: float,
        macro F1-measure
    f1: np.ndarray,
        F1-measures for each class, of shape: (n_classes,)

    """
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

    return m["macro_f1"], m["f1"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("sensitivity", "sensitivities"),
    "prepend",
)
def sensitivity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_sens: float,
        macro sensitivity
    sens: np.ndarray,
        sensitivities for each class, of shape: (n_classes,)

    """
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

    return m["macro_sens"], m["sens"]


# aliases
recall = sensitivity
true_positive_rate = sensitivity
hit_rate = sensitivity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("precision", "precisions"), "prepend"
)
def precision(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_prec: float,
        macro precision
    prec: np.ndarray,
        precisions for each class, of shape: (n_classes,)

    """
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

    return m["macro_prec"], m["prec"]


# aliases
positive_predictive_value = precision


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("specificity", "specificities"),
    "prepend",
)
def specificity(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns
    -------
    macro_spec: float,
        macro specificity
    spec: np.ndarray,
        specificities for each class, of shape: (n_classes,)

    """
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

    return m["macro_spec"], m["spec"]


# aliases
selectivity = specificity
true_negative_rate = specificity


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("AUROC and macro AUPRC", "AUPRCs, AUPRCs"),
    "prepend",
)
def auc(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
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
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

    return m["macro_auroc"], m["macro_auprc"], m["auroc"], m["auprc"]


@add_docstring(
    _METRICS_FROM_CONFUSION_MATRIX_PARAMS.format("accuracy", "accuracies"),
    "prepend",
)
def accuracy(
    labels: Union[np.ndarray, Tensor],
    outputs: Union[np.ndarray, Tensor],
    num_classes: Optional[int] = None,
) -> float:
    """
    Returns
    -------
    macro_acc: float,
        the macro accuracy
    acc: np.ndarray,
        accuracies for each class, of shape: (n_classes,)

    """
    m = _metrics_from_confusion_matrix(labels, outputs, num_classes)

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
