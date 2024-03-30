from typing import Dict, Sequence, Tuple

import numpy as np
from cfg import BaseCfg
from helper_code import is_nan
from outputs import CINC2023Outputs

__all__ = [
    "compute_challenge_metrics",
]


######################################
# custom metrics computation functions
######################################


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2023Outputs],
    hospitals: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Compute the challenge metrics.

    Parameters
    ----------
    labels : Sequence[Dict[str, np.ndarray]]
        labels containing at least one of the following items:
            - "cpc":
              binary labels, of shape: ``(n_samples, n_classes)``;
              or categorical labels, of shape: ``(n_samples,)``
            - "outcome":
              binary labels, of shape: ``(n_samples, n_classes)``,
              or categorical labels, of shape: ``(n_samples,)``
    outputs : Sequence[CINC2023Outputs]
        outputs containing at least one non-null attributes:
            - cpc_output: ClassificationOutput, with items:
                - classes: list of str,
                  list of the class names
                - prob: ndarray or DataFrame,
                  scalar (probability) predictions,
                  (and binary predictions if `class_names` is True)
                - pred: ndarray,
                  the array of class number predictions
                - bin_pred: ndarray,
                  the array of binary predictions
                - forward_output: ndarray,
                  the array of output of the model's forward function,
                  useful for producing challenge result using
                  multiple recordings
            - cpc_value: Sequence[float],
                the array of cpc value predictions
            - outcome_output: ClassificationOutput, with items:
                - classes: list of str,
                  list of the outcome class names
                - prob: ndarray,
                  scalar (probability) predictions,
                - pred: ndarray,
                  the array of outcome class number predictions
                - forward_output: ndarray,
                  the array of output of the outcome head of the model's forward function,
                  useful for producing challenge result using
                  multiple recordings
            - outcome: Sequence[str],
                the array of outcome predictions (class names)
    hospitals : Sequence[Sequence[str]]
        The hospital names for each patient,
        each of shape ``(n_samples,)``.

    Returns
    -------
    dict
        A dict of the following metrics:
            - outcome_score: float,
            the Challenge score for the outcome predictions
            - outcome_auroc: float,
            the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
            - outcome_auprc: float,
            the macro-averaged area under the precision-recall curve for the outcome predictions
            - outcome_f_measure: float,
            the macro-averaged F-measure for the outcome predictions
            - outcome_accuracy: float,
            the accuracy for the outcome predictions
            - cpc_mse: float,
            the mean squared error for the cpc predictions
            - cpc_mae: float,
            the mean absolute error for the cpc predictions

    NOTE
    ----
    1. the "cpc_xxx" metrics are contained in the returned dict iff corr. labels and outputs are provided;
       the same applies to the "outcome_xxx" metrics.
    2. all labels should have a batch dimension, except for categorical labels

    """
    metrics = {}

    # compute the outcome metrics
    if outputs[0].outcome_output is not None:
        outcome_labels = np.concatenate([label["outcome"] for label in labels])  # categorical or binarized labels
        if outcome_labels.ndim == 2 and outcome_labels.shape[1] == len(BaseCfg.outcome):
            outcome_labels = outcome_labels.argmax(axis=1)
        # outcome_prob_outputs <- probabilities of the "Poor" class
        outcome_prob_outputs = np.concatenate(
            [item.outcome_output.prob[:, item.outcome_output.classes.index("Poor")] for item in outputs]  # probability outputs
        )
        # outcome_prob_outputs = outcome_prob_outputs.max(axis=1)
        outcome_pred_outputs = np.concatenate([item.outcome_output.pred for item in outputs])  # categorical outputs
        hospitals = np.concatenate(hospitals)
        metrics.update(compute_outcome_metrics(outcome_labels, outcome_prob_outputs, outcome_pred_outputs, hospitals))

    # compute the cpc metrics
    if outputs[0].cpc_output is not None:
        cpc_labels = np.concatenate([label["cpc"] for label in labels])  # categorical or binarized labels
        if cpc_labels.ndim == 2 and cpc_labels.shape[1] == len(BaseCfg.cpc):
            cpc_labels = cpc_labels.argmax(axis=1) + 1
        cpc_pred_outputs = np.concatenate([item.cpc_value for item in outputs])  # categorical or regression outputs
        metrics.update(compute_cpc_metrics(cpc_labels, cpc_pred_outputs))

    return metrics


def compute_outcome_metrics(
    outcome_labels: np.ndarray,
    outcome_prob_outputs: np.ndarray,
    outcome_pred_outputs: np.ndarray,
    hospitals: Sequence[str],
) -> Dict[str, float]:
    """Compute the outcome metrics.

    Parameters
    ----------
    outcome_labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outcome_prob_outputs : np.ndarray
        The probability outputs for `outcome`,
        of shape ``(num_patients, num_classes)``.
    outcome_pred_outputs : np.ndarray
        The categorical (class number) outputs for `outcome`,
        of shape ``(num_patients,)``.
    hospitals : Sequence[str]
        The hospital names for each patient,
        of shape ``(num_patients,)``.

    Returns
    -------
    dict
        A dict of the following metrics:
            - outcome_auroc: float,
              the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
            - outcome_auprc: float,
              the macro-averaged area under the precision-recall curve for the outcome predictions
            - outcome_f_measure: float,
              the macro-averaged F-measure for the outcome predictions
            - outcome_accuracy: float,
              the accuracy for the outcome predictions
            - outcome_score: float,
              the Challenge score for the outcome predictions

    """
    metrics = {}
    metrics["outcome_score"] = compute_challenge_score(outcome_labels, outcome_prob_outputs, hospitals)
    auroc, auprc = compute_auc(outcome_labels, outcome_prob_outputs)
    metrics["outcome_auroc"] = auroc
    metrics["outcome_auprc"] = auprc
    metrics["outcome_f_measure"] = compute_f_measure(outcome_labels, outcome_pred_outputs)[0]
    metrics["outcome_accuracy"] = compute_accuracy(outcome_labels, outcome_pred_outputs)[0]
    return metrics


def compute_cpc_metrics(cpc_labels: np.ndarray, cpc_pred_outputs: np.ndarray) -> Dict[str, float]:
    """Compute the CPC metrics.

    Parameters
    ----------
    cpc_labels : np.ndarray
        The categorical ground truth labels for `cpc`,
        of shape ``(num_patients,)``.
    cpc_pred_outputs : np.ndarray
        The categorical (class number) outputs for `cpc`,
        or the regression outputs for `cpc`,
        of shape ``(num_patients,)``.

    Returns
    -------
    dict
        A dict of the following metrics:
            - cpc_mse: float,
              the mean squared error for the cpc predictions
            - cpc_mae: float,
              the mean absolute error for the cpc predictions

    """
    metrics = {}
    metrics["cpc_mse"] = compute_mse(cpc_labels, cpc_pred_outputs)
    metrics["cpc_mae"] = compute_mae(cpc_labels, cpc_pred_outputs)
    return metrics


###########################################
# methods from the file evaluation_model.py
# of the official repository
###########################################


def compute_challenge_score(labels: np.ndarray, outputs: np.ndarray, hospitals: Sequence[str]) -> float:
    """Compute the Challenge score.

    The Challenge score is the largest TPR such that FPR <= 0.05
    for `outcome`.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The probability outputs (probabilities of the "Poor" class) for `outcome`,
        of shape ``(num_patients)``.
    hospitals : Sequence[str]
        The hospital names for each patient,
        of shape ``(num_patients,)``.

    Returns
    -------
    max_tpr : float
        The Challenge score, the largest TPR such that FPR <= 0.05.

    """
    # Check the data.
    assert len(labels) == len(outputs)

    # Convert the data to NumPy arrays for easier indexing.
    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)

    # Identify the unique hospitals.
    unique_hospitals = sorted(set(hospitals))
    num_hospitals = len(unique_hospitals)

    # Initialize a confusion matrix for each hospital.
    tps = np.zeros(num_hospitals)
    fps = np.zeros(num_hospitals)
    fns = np.zeros(num_hospitals)
    tns = np.zeros(num_hospitals)

    # Compute the confusion matrix at each output threshold separately for each hospital.
    for i, hospital in enumerate(unique_hospitals):
        idx = [j for j, x in enumerate(hospitals) if x == hospital]
        current_labels = labels[idx]
        current_outputs = outputs[idx]
        num_instances = len(current_labels)

        # Collect the unique output values as the thresholds for the positive and negative classes.
        thresholds = np.unique(current_outputs)
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        idx = np.argsort(current_outputs)[::-1]

        # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)

        tp[0] = 0
        fp[0] = 0
        fn[0] = np.sum(current_labels == 1)
        tn[0] = np.sum(current_labels == 0)

        # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
        k = 0
        for thr_ in range(1, num_thresholds):
            tp[thr_] = tp[thr_ - 1]
            fp[thr_] = fp[thr_ - 1]
            fn[thr_] = fn[thr_ - 1]
            tn[thr_] = tn[thr_ - 1]

            while k < num_instances and current_outputs[idx[k]] >= thresholds[thr_]:
                if current_labels[idx[k]] == 1:
                    tp[thr_] += 1
                    fn[thr_] -= 1
                else:
                    fp[thr_] += 1
                    tn[thr_] -= 1
                k += 1

        # Compute the FPRs.
        fpr = np.zeros(num_thresholds)
        for thr_ in range(num_thresholds):
            if tp[thr_] + fn[thr_] > 0:
                fpr[thr_] = float(fp[thr_]) / float(tp[thr_] + fn[thr_])
            else:
                fpr[thr_] = float("nan")

        # Find the threshold such that FPR <= 0.05.
        max_fpr = 0.05
        if np.any(fpr <= max_fpr):
            thr_ = max(thr_ for thr_, x in enumerate(fpr) if x <= max_fpr)
            tps[i] = tp[thr_]
            fps[i] = fp[thr_]
            fns[i] = fn[thr_]
            tns[i] = tn[thr_]
        else:
            tps[i] = tp[0]
            fps[i] = fp[0]
            fns[i] = fn[0]
            tns[i] = tn[0]

    # Compute the TPR at FPR <= 0.05 for each hospital.
    tp = np.sum(tps)
    fp = np.sum(fps)
    fn = np.sum(fns)
    tn = np.sum(tns)

    if tp + fn > 0:
        max_tpr = tp / (tp + fn)
    else:
        max_tpr = float("nan")

    return max_tpr


def compute_auc(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float]:
    """Compute area under the receiver operating characteristic curve (AUROC)
    and area under the precision recall curve (AUPRC).

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The probability outputs (probabilities of the "Poor" class) for `outcome`,
        of shape ``(num_patients)``.

    Returns
    -------
    auroc: float
        The AUROC.
    auprc : float
        The AUPRC.

    """
    assert len(labels) == len(outputs)
    num_patients = len(labels)

    # Use the unique output values as the thresholds for the positive and negative classes.
    thresholds = np.unique(outputs)
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    thresholds = thresholds[::-1]
    num_thresholds = len(thresholds)

    idx = np.argsort(outputs)[::-1]

    # Initialize the TPs, FPs, FNs, and TNs with no positive outputs.
    tp = np.zeros(num_thresholds)
    fp = np.zeros(num_thresholds)
    fn = np.zeros(num_thresholds)
    tn = np.zeros(num_thresholds)

    tp[0] = 0
    fp[0] = 0
    fn[0] = np.sum(labels == 1)
    tn[0] = np.sum(labels == 0)

    # Update the TPs, FPs, FNs, and TNs using the values at the previous threshold.
    i = 0
    for j in range(1, num_thresholds):
        tp[j] = tp[j - 1]
        fp[j] = fp[j - 1]
        fn[j] = fn[j - 1]
        tn[j] = tn[j - 1]

        while i < num_patients and outputs[idx[i]] >= thresholds[j]:
            if labels[idx[i]]:
                tp[j] += 1
                fn[j] -= 1
            else:
                fp[j] += 1
                tn[j] -= 1
            i += 1

    # Compute the TPRs, TNRs, and PPVs at each threshold.
    tpr = np.zeros(num_thresholds)
    tnr = np.zeros(num_thresholds)
    ppv = np.zeros(num_thresholds)
    for j in range(num_thresholds):
        if tp[j] + fn[j]:
            tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
        else:
            tpr[j] = float("nan")
        if fp[j] + tn[j]:
            tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
        else:
            tnr[j] = float("nan")
        if tp[j] + fp[j]:
            ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
        else:
            ppv[j] = float("nan")

    # Compute AUROC as the area under a piecewise linear function
    # with TPR/sensitivity (x-axis) and TNR/specificity (y-axis) and
    # AUPRC as the area under a piecewise constant
    # with TPR/recall (x-axis) and PPV/precision (y-axis).
    auroc = 0.0
    auprc = 0.0
    for j in range(num_thresholds - 1):
        auroc += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
        auprc += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    return auroc, auprc


def compute_one_hot_encoding(data: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Construct the one-hot encoding of data for the given classes.

    Parameters
    ----------
    data : np.ndarray
        The (categorical) data to encode,
        of shape ``(num_patients,)``.
    classes : np.ndarray
        The classes to use for the encoding,
        of shape ``(num_classes,)``.

    Returns
    -------
    np.ndarray
        The one-hot encoding of the data,
        of shape ``(num_patients, num_classes)``.

    """
    num_patients = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_patients, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for j, y in enumerate(classes):
            if (x == y) or (is_nan(x) and is_nan(y)):
                one_hot_encoding[i, j] = 1

    return one_hot_encoding


def compute_confusion_matrix(labels: np.ndarray, outputs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Compute the binary confusion matrix.

    The columns are the expert labels and
    the rows are the classifier labels for the given classes.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients, num_classes)``.
    outputs : np.ndarray
        The binarized (one-hot encoded) classifier outputs for `outcome`,
        of shape ``(num_patients, num_classes)``.
    classes : np.ndarray
        The classes to use for the confusion matrix,
        of shape ``(num_classes,)``.

    Returns
    -------
    np.ndarray
        The confusion matrix,
        of shape ``(num_classes, num_classes)``.

    """
    assert np.shape(labels) == np.shape(outputs)

    num_patients = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, num_classes))
    for k in range(num_patients):
        for i in range(num_classes):
            for j in range(num_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A


def compute_one_vs_rest_confusion_matrix(labels: np.ndarray, outputs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Construct the binary one-vs-rest (OVR) confusion matrices.

    The columns are the expert labels and
    the rows are the classifier for the given classes.

    Parameters
    ----------
    labels : np.ndarray
        The binarized (one-hot encoded) ground truth labels for `outcome`,
        of shape ``(num_patients, num_classes)``.
    outputs : np.ndarray
        The binarized (one-hot encoded) classifier outputs for `outcome`,
    classes : np.ndarray
        The classes to use for the confusion matrices.

    Returns
    -------
    np.ndarray
        The one-vs-rest confusion matrices,
        of shape ``(num_classes, 2, 2)``.

    """
    assert np.shape(labels) == np.shape(outputs)

    num_patients = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
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


def compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the accuracy and per-class accuracy.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The categorical classifier outputs for `outcome`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The macro-averaged accuracy.
    np.ndarray
        The per-class accuracy,
        of shape ``(num_classes,)``.
    np.ndarray
        The array of classes,
        of shape ``(num_classes,)``.

    """
    # Compute the confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_confusion_matrix(labels, outputs, classes)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float("nan")

    # Compute per-class accuracy.
    num_classes = len(classes)
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(labels[:, i]) > 0:
            per_class_accuracy[i] = A[i, i] / np.sum(A[:, i])
        else:
            per_class_accuracy[i] = float("nan")

    return accuracy, per_class_accuracy, classes


def compute_f_measure(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the F-measure and per-class F-measure.

    Parameters
    ----------
    labels : np.ndarray
        The categorical ground truth labels for `outcome`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The categorical classifier outputs for `outcome`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The macro-averaged F-measure.
    np.ndarray
        The per-class F-measure,
        of shape ``(num_classes,)``.
    np.ndarray
        The array of classes,
        of shape ``(num_classes,)``.

    """
    # Compute confusion matrix.
    classes = np.unique(np.concatenate((labels, outputs)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float("nan")

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, per_class_f_measure, classes


def compute_mse(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute the mean-squared error (MSE).

    Parameters
    ----------
    labels : np.ndarray
        The continuous (actually categorical) ground truth labels for `cpc`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The continuous (actually categorical) classifier outputs for `cpc`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The MSE.

    """
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mse = np.mean((labels - outputs) ** 2)

    return mse


def compute_mae(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute the mean-absolute error (MAE).

    Parameters
    ----------
    labels : np.ndarray
        The continuous (actually categorical) ground truth labels for `cpc`,
        of shape ``(num_patients,)``.
    outputs : np.ndarray
        The continuous (actually categorical) classifier outputs for `cpc`,
        of shape ``(num_patients,)``.

    Returns
    -------
    float
        The MAE.

    """
    assert len(labels) == len(outputs)

    labels = np.asarray(labels, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    mae = np.mean(np.abs(labels - outputs))

    return mae
