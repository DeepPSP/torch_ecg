"""
metrics from the official scoring repository
"""

from typing import Tuple, Sequence, Dict, List, Union

import numpy as np
from deprecated import deprecated
from torch_ecg.utils.utils_metrics import _cls_to_bin

from cfg import BaseCfg
from outputs import CINC2022Outputs


__all__ = [
    "compute_challenge_metrics",
]


######################################
# custom metrics computation functions
######################################


def compute_challenge_metrics(
    labels: Sequence[Dict[str, np.ndarray]],
    outputs: Sequence[CINC2022Outputs],
    require_both: bool = False,
) -> Dict[str, float]:
    """
    Compute the challenge metrics

    Parameters
    ----------
    labels: sequence of dict of ndarray,
        labels containing at least one of the following items:
            - "murmur":
                binary labels, of shape: (n_samples, n_classes);
                or categorical labels, of shape: (n_samples,)
            - "outcome":
                binary labels, of shape: (n_samples, n_classes),
                or categorical labels, of shape: (n_samples,)
    outputs: sequence of CINC2022Outputs,
        outputs containing at least one non-null attributes:
            - murmur_output: ClassificationOutput, with items:
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
            - outcome_output: ClassificationOutput, optional, with items:
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
    require_both: bool,
        whether to require both murmur and outcome labels and outputs to be provided

    Returns
    -------
    dict, a dict of the following metrics:
        - murmur_auroc: float,
            the macro-averaged area under the receiver operating characteristic curve for the murmur predictions
        - murmur_auprc: float,
            the macro-averaged area under the precision-recall curve for the murmur predictions
        - murmur_f_measure: float,
            the macro-averaged F-measure for the murmur predictions
        - murmur_accuracy: float,
            the accuracy for the murmur predictions
        - murmur_weighted_accuracy: float,
            the weighted accuracy for the murmur predictions
        - murmur_cost: float,
            the challenge cost for the murmur predictions
        - outcome_auroc: float,
            the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
        - outcome_auprc: float,
            the macro-averaged area under the precision-recall curve for the outcome predictions
        - outcome_f_measure: float,
            the macro-averaged F-measure for the outcome predictions
        - outcome_accuracy: float,
            the accuracy for the outcome predictions
        - outcome_weighted_accuracy: float,
            the weighted accuracy for the outcome predictions
        - outcome_cost: float,
            the challenge cost for the outcome predictions

    NOTE
    ----
    1. the "murmur_xxx" metrics are contained in the returned dict iff corr. labels and outputs are provided;
        the same applies to the "outcome_xxx" metrics.
    2. all labels should have a batch dimension, except for categorical labels

    """
    metrics = {}
    if require_both:
        assert all([set(lb.keys()) >= set(["murmur", "outcome"]) for lb in labels])
        assert all(
            [
                item.murmur_output is not None and item.outcome_output is not None
                for item in outputs
            ]
        )
    # metrics for murmurs
    # NOTE: labels all have a batch dimension, except for categorical labels
    if outputs[0].murmur_output is not None:
        murmur_labels = np.concatenate(
            [lb["murmur"] for lb in labels]  # categorical or binarized labels
        )
        murmur_scalar_outputs = np.concatenate(
            [np.atleast_2d(item.murmur_output.prob) for item in outputs]
        )
        murmur_binary_outputs = np.concatenate(
            [np.atleast_2d(item.murmur_output.bin_pred) for item in outputs]
        )
        murmur_classes = outputs[0].murmur_output.classes
        if murmur_labels.ndim == 1:
            murmur_labels = _cls_to_bin(
                murmur_labels, shape=(len(murmur_labels), len(murmur_classes))
            )
        metrics.update(
            _compute_challenge_metrics(
                murmur_labels,
                murmur_scalar_outputs,
                murmur_binary_outputs,
                murmur_classes,
            )
        )
    # metrics for outcomes
    if outputs[0].outcome_output is not None:
        outcome_labels = np.concatenate(
            [lb["outcome"] for lb in labels]  # categorical or binarized labels
        )
        outcome_scalar_outputs = np.concatenate(
            [np.atleast_2d(item.outcome_output.prob) for item in outputs]
        )
        outcome_binary_outputs = np.concatenate(
            [np.atleast_2d(item.outcome_output.bin_pred) for item in outputs]
        )
        outcome_classes = outputs[0].outcome_output.classes
        if outcome_labels.ndim == 1:
            outcome_labels = _cls_to_bin(
                outcome_labels, shape=(len(outcome_labels), len(outcome_classes))
            )
        metrics.update(
            _compute_challenge_metrics(
                outcome_labels,
                outcome_scalar_outputs,
                outcome_binary_outputs,
                outcome_classes,
            )
        )

    return metrics


def _compute_challenge_metrics(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, float]:
    """
    Compute macro-averaged metrics,
    modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names for murmurs or outcomes,
        e.g. `BaseCfg.classes` or `BaseCfg.outcomes`

    Returns
    -------
    dict, a dict of the following metrics:
        auroc: float,
            the macro-averaged area under the receiver operating characteristic curve
        auprc: float,
            the macro-averaged area under the precision-recall curve
        f_measure: float,
            the macro-averaged F-measure
        accuracy: float,
            the accuracy
        weighted_accuracy: float,
            the weighted accuracy
        cost: float,
            the challenge cost

    """
    detailed_metrics = _compute_challenge_metrics_detailed(
        labels, scalar_outputs, binary_outputs, classes
    )
    metrics = {
        f"""{detailed_metrics["prefix"]}_{k}""": v
        for k, v in detailed_metrics.items()
        if k
        in [
            "auroc",
            "auprc",
            "f_measure",
            "accuracy",
            "weighted_accuracy",
            "cost",
        ]
    }
    # metrics["task_score"] = (
    #     metrics["weighted_accuracy"]
    #     if list(classes) == BaseCfg.classes
    #     else metrics["cost"]
    # )
    return metrics


def _compute_challenge_metrics_detailed(
    labels: np.ndarray,
    scalar_outputs: np.ndarray,
    binary_outputs: np.ndarray,
    classes: Sequence[str],
) -> Dict[str, Union[float, np.ndarray, str]]:
    """
    Compute detailed metrics, modified from the function `evaluate_model`
    in `evaluate_model.py` in the official scoring repository.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    scalar_outputs: np.ndarray,
        scalar outputs (probabilities), of shape: (n_samples, n_classes)
    binary_outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names for murmurs or outcomes,
        e.g. `BaseCfg.classes` or `BaseCfg.outcomes`

    Returns
    -------
    dict, a dict of the following metrics:
        auroc: float,
            the macro-averaged area under the receiver operating characteristic curve
        auprc: float,
            the macro-averaged area under the precision-recall curve
        auroc_classes: np.ndarray,
            the area under the receiver operating characteristic curve for each class
        auprc_classes: np.ndarray,
            the area under the precision-recall curve for each class
        f_measure: float,
            the macro-averaged F-measure
        f_measure_classes: np.ndarray,
            the F-measure for each class
        accuracy: float,
            the accuracy
        accuracy_classes: np.ndarray,
            the accuracy for each class
        weighted_accuracy: float,
            the weighted accuracy
        cost: float,
            the challenge cost
        prefix: str,
            the prefix of the metrics, one of `"murmur"` or `"outcome"`

    """
    # For each patient, set the 'Unknown' class to positive if no class is positive or if multiple classes are positive.
    if list(classes) == BaseCfg.classes:
        positive_class = "Present"
        prefix = "murmur"
    elif list(classes) == BaseCfg.outcomes:
        positive_class = "Abnormal"
        prefix = "outcome"
    else:
        raise ValueError(f"Illegal sequence of classes: {classes}")
    labels = enforce_positives(labels, classes, positive_class)
    binary_outputs = enforce_positives(binary_outputs, classes, positive_class)

    # Evaluate the model by comparing the labels and outputs.
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    accuracy, accuracy_classes = compute_accuracy(labels, binary_outputs)
    weighted_accuracy = compute_weighted_accuracy(labels, binary_outputs, list(classes))
    # challenge_score = compute_challenge_score(labels, binary_outputs, classes)
    cost = compute_cost(labels, binary_outputs, BaseCfg.outcomes, classes)

    return dict(
        auroc=auroc,
        auprc=auprc,
        auroc_classes=auroc_classes,
        auprc_classes=auprc_classes,
        f_measure=f_measure,
        f_measure_classes=f_measure_classes,
        accuracy=accuracy,
        accuracy_classes=accuracy_classes,
        weighted_accuracy=weighted_accuracy,
        cost=cost,
        prefix=prefix,
    )


###########################################
# methods from the file evaluation_model.py
# of the official repository
###########################################


def enforce_positives(
    outputs: np.ndarray, classes: Sequence[str], positive_class: str
) -> np.ndarray:
    """
    For each patient, set a specific class to positive if no class is positive or multiple classes are positive.

    Parameters
    ----------
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: Sequence[str],
        class names
    positive_class: str,
        class name to be set to positive

    Returns
    -------
    outputs: np.ndarray,
        enforced binary outputs, of shape: (n_samples, n_classes)

    """
    num_patients, num_classes = np.shape(outputs)
    j = classes.index(positive_class)

    for i in range(num_patients):
        if np.sum(outputs[i, :]) != 1:
            outputs[i, :] = 0
            outputs[i, j] = 1
    return outputs


def compute_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Compute a binary confusion matrix, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        confusion matrix, of shape: (n_classes, n_classes)

    """
    assert np.shape(labels)[0] == np.shape(outputs)[0]
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A


def compute_one_vs_rest_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """
    Compute binary one-vs-rest confusion matrices, where the columns are expert labels and rows are classifier labels.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    A: np.ndarray,
        one-vs-rest confusion matrix, of shape: (n_classes, 2, 2)

    """
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

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


# alias
compute_ovr_confusion_matrix = compute_one_vs_rest_confusion_matrix


def compute_auc(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute macro AUROC and macro AUPRC, and AUPRCs, AUPRCs for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

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
    print("Computing AUROC and AUPRC...")
    num_patients, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
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
            while i < num_patients and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
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

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float("nan")
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float("nan")

    return macro_auroc, macro_auprc, auroc, auprc


def compute_accuracy(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute accuracy.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    accuracy: float,
        the accuracy
    accuracy_classes: np.ndarray,
        the accuracy for each class, of shape: (n_classes,)
    """
    print("Computing accuracy...")
    assert np.shape(labels) == np.shape(outputs)
    num_patients, num_classes = np.shape(labels)
    A = compute_confusion_matrix(labels, outputs)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float("nan")

    # Compute per-class accuracy.
    accuracy_classes = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(A[:, i]) > 0:
            accuracy_classes[i] = A[i, i] / np.sum(A[:, i])
        else:
            accuracy_classes[i] = float("nan")

    return accuracy, accuracy_classes


def compute_f_measure(
    labels: np.ndarray, outputs: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute macro F-measure, and F-measures for each class.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)

    Returns
    -------
    macro_f_measure: float,
        macro F-measure
    f_measure: np.ndarray,
        F-measures for each class, of shape: (n_classes,)

    """
    print("Computing F-measure...")
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure


def compute_weighted_accuracy(
    labels: np.ndarray, outputs: np.ndarray, classes: List[str]
) -> float:
    """
    compute weighted accuracy

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: List[str],
        list of class names, can be one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match

    Returns
    -------
    weighted_accuracy: float,
        weighted accuracy

    """
    print("Computing weighted accuracy...")
    # Define constants.
    if classes == ["Present", "Unknown", "Absent"]:
        weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    elif classes == ["Abnormal", "Normal"]:
        weights = np.array([[5, 1], [5, 1]])
    else:
        raise NotImplementedError(
            "Weighted accuracy undefined for classes {}".format(", ".join(classes))
        )

    # Compute confusion matrix.
    assert np.shape(labels) == np.shape(outputs)
    A = compute_confusion_matrix(labels, outputs)

    # Multiply the confusion matrix by the weight matrix.
    assert np.shape(A) == np.shape(weights)
    B = weights * A

    # Compute weighted_accuracy.
    if np.sum(B) > 0:
        weighted_accuracy = np.trace(B) / np.sum(B)
    else:
        weighted_accuracy = float("nan")

    return weighted_accuracy


def cost_algorithm(m: int) -> int:
    """total cost for algorithmic prescreening of m patients."""
    return 10 * m


def cost_expert(m: int, n: int) -> float:
    """total cost for expert screening of m patients out of a total of n total patients."""
    return (25 + 397 * (m / n) - 1718 * (m / n) ** 2 + 11296 * (m / n) ** 4) * n


def cost_treatment(m: int) -> int:
    """total cost for treatment of m patients."""
    return 10000 * m


def cost_error(m: int) -> int:
    """total cost for missed/late treatement of m patients."""
    return 50000 * m


def compute_cost(
    labels: np.ndarray,
    outputs: np.ndarray,
    label_classes: Sequence[str],
    output_classes: Sequence[str],
) -> float:
    """
    Compute Challenge cost metric.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    label_classes: Sequence[str],
        list of label class names, can one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match
        case sensitive
    output_classes: Sequence[str],
        list of predicted class names, can one of the following:
        ['Present', 'Unknown', 'Absent'],
        ['Abnormal', 'Normal'],
        cases and ordering must match
        case sensitive

    """
    print("Computing challenge cost...")
    # Define positive and negative classes for referral and treatment.
    positive_classes = ["Present", "Unknown", "Abnormal"]
    negative_classes = ["Absent", "Normal"]

    # Compute confusion matrix.
    A = compute_confusion_matrix(labels, outputs)

    # Identify positive and negative classes for referral.
    idx_label_positive = [
        i for i, x in enumerate(label_classes) if x in positive_classes
    ]
    idx_label_negative = [
        i for i, x in enumerate(label_classes) if x in negative_classes
    ]
    idx_output_positive = [
        i for i, x in enumerate(output_classes) if x in positive_classes
    ]
    idx_output_negative = [
        i for i, x in enumerate(output_classes) if x in negative_classes
    ]

    # Identify true positives, false positives, false negatives, and true negatives.
    tp = np.sum(A[np.ix_(idx_output_positive, idx_label_positive)])
    fp = np.sum(A[np.ix_(idx_output_positive, idx_label_negative)])
    fn = np.sum(A[np.ix_(idx_output_negative, idx_label_positive)])
    tn = np.sum(A[np.ix_(idx_output_negative, idx_label_negative)])
    total_patients = tp + fp + fn + tn

    # Compute total cost for all patients.
    total_cost = (
        cost_algorithm(total_patients)
        + cost_expert(tp + fp, total_patients)
        + cost_treatment(tp)
        + cost_error(fn)
    )

    # Compute mean cost per patient.
    if total_patients > 0:
        mean_cost = total_cost / total_patients
    else:
        mean_cost = float("nan")

    return mean_cost


@deprecated(reason="only used in the unofficial phase of the Challenge")
def compute_challenge_score(
    labels: np.ndarray, outputs: np.ndarray, classes: Sequence[str]
) -> float:
    """
    Compute Challenge score.

    Parameters
    ----------
    labels: np.ndarray,
        binary labels, of shape: (n_samples, n_classes)
    outputs: np.ndarray,
        binary outputs, of shape: (n_samples, n_classes)
    classes: sequence of str,
        class names

    Returns
    -------
    mean_score: float,
        mean Challenge score
    """
    # Define costs. Better to load these costs from an external file instead of defining them here.
    c_algorithm = 1  # Cost for algorithmic prescreening.
    c_gp = 250  # Cost for screening from a general practitioner (GP).
    c_specialist = 500  # Cost for screening from a specialist.
    c_treatment = 1000  # Cost for treatment.
    c_error = 10000  # Cost for diagnostic error.
    alpha = 0.5  # Fraction of murmur unknown cases that are positive.

    num_patients, num_classes = np.shape(labels)

    A = compute_confusion_matrix(labels, outputs)

    idx_positive = classes.index("Present")
    idx_unknown = classes.index("Unknown")
    idx_negative = classes.index("Absent")

    n_pp = A[idx_positive, idx_positive]
    n_pu = A[idx_positive, idx_unknown]
    n_pn = A[idx_positive, idx_negative]
    n_up = A[idx_unknown, idx_positive]
    n_uu = A[idx_unknown, idx_unknown]
    n_un = A[idx_unknown, idx_negative]
    n_np = A[idx_negative, idx_positive]
    n_nu = A[idx_negative, idx_unknown]
    n_nn = A[idx_negative, idx_negative]

    n_total = n_pp + n_pu + n_pn + n_up + n_uu + n_un + n_np + n_nu + n_nn

    total_score = (
        c_algorithm * n_total
        + c_gp * (n_pp + n_pu + n_pn)
        + c_specialist * (n_pu + n_up + n_uu + n_un)
        + c_treatment * (n_pp + alpha * n_pu + n_up + alpha * n_uu)
        + c_error * (n_np + alpha * n_nu)
    )
    if n_total > 0:
        mean_score = total_score / n_total
    else:
        mean_score = float("nan")

    return mean_score
