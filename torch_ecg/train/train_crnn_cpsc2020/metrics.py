"""
"""
from numbers import Real
from typing import Union, Optional, Any, List, Tuple, Sequence

import numpy as np
from easydict import EasyDict as ED

from .utils import dict_to_str
from .cfg import BaseCfg


__all__ = [
    "CPSC2020_loss",
    "CPSC2020_score",
    "eval_score",
]


def CPSC2020_loss(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str, verbose:int=0) -> int:
    """ finished, updated with the latest (updated on 2020.8.31) official function

    Parameters:
    -----------
    y_true: ndarray,
        array of ground truth of beat types
    y_true: ndarray,
        array of predictions of beat types
    y_indices: ndarray,
        indices of beat (rpeak) in the original ecg signal
    dtype: type, default str,
        dtype of `y_true` and `y_pred`

    Returns:
    --------
    total_loss: int,
        the total loss of all ectopic beat types (SBP, PVC)
    """
    classes = ['S', 'V']

    truth_arr = {}
    pred_arr = {}
    if dtype == str:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==c)[0]]
            pred_arr[c] = y_indices[np.where(y_pred==c)[0]]
    elif dtype == int:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==BaseCfg.class_map[c])[0]]
            pred_arr[c] = y_indices[np.where(y_pred==BaseCfg.class_map[c])[0]]

    true_positive = {c: 0 for c in classes}

    for c in classes:
        for tc in truth_arr[c]:
            pc = np.where(abs(pred_arr[c]-tc) <= BaseCfg.bias_thr)[0]
            if pc.size > 0:
                true_positive[c] += 1

    false_positive = {
        c: len(pred_arr[c]) - true_positive[c] for c in classes
    }
    false_negative = {
        c: len(truth_arr[c]) - true_positive[c] for c in classes
    }

    false_positive_loss = {c: 1 for c in classes}
    false_negative_loss = {c: 5 for c in classes}

    if verbose >= 1:
        print(f"true_positive = {dict_to_str(true_positive)}")
        print(f"false_positive = {dict_to_str(false_positive)}")
        print(f"false_negative = {dict_to_str(false_negative)}")

    total_loss = sum([
        false_positive[c] * false_positive_loss[c] + false_negative[c] * false_negative_loss[c] \
            for c in classes
    ])
    
    return total_loss


def CPSC2020_score(sbp_true:List[np.ndarray], pvc_true:List[np.ndarray], sbp_pred:List[np.ndarray], pvc_pred:List[np.ndarray], verbose:int=0) -> Union[Tuple[int],dict]:
    """ finished, checked,

    Score Function for all (test) records

    Parameters:
    -----------
    sbp_true, pvc_true, sbp_pred, pvc_pred: list of ndarray,
    verbose: int

    Returns:
    --------
    retval: tuple or dict,
        tuple of (negative) scores for each ectopic beat type (SBP, PVC), or
        dict of more scoring details, including
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type
    """
    s_score = np.zeros([len(sbp_true), ], dtype=int)
    v_score = np.zeros([len(sbp_true), ], dtype=int)
    ## Scoring ##
    for i, (s_ref, v_ref, s_pos, v_pos) in enumerate(zip(sbp_true, pvc_true, sbp_pred, pvc_pred)):
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        # SBP
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= BaseCfg.bias_thr)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
                    s_fp += len(s_pos_cand) - 1
        # PVC
        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= BaseCfg.bias_thr)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
                    v_fp += len(v_pos_cand) - 1
        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)

        if verbose >= 1:
            print(f"for the {i}-th record")
            print(f"s_tp = {s_tp}, s_fp = {s_fp}, s_fn = {s_fn}")
            print(f"v_tp = {v_tp}, v_fp = {v_fp}, s_fn = {v_fn}")
            print(f"s_score[{i}] = {s_score[i]}, v_score[{i}] = {v_score[i]}")

    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    if verbose >= 1:
        retval = ED(
            total_loss=-(Score1+Score2),
            class_loss={'S':-Score1, 'V':-Score2},
            true_positive={'S':s_tp, 'V':v_tp},
            false_positive={'S':s_fp, 'V':v_fp},
            false_negative={'S':s_fn, 'V':v_fn},
        )
    else:
        retval = Score1, Score2

    return retval



# -------------------------------------------------------
# the following are borrowed from CINC2020
# for classification of segments of ECGs

def eval_score(classes:List[str], truth:Sequence, binary_pred:Sequence, scalar_pred:Sequence) -> Tuple[float]:
    """ finished, checked,
    
    for classification of segments of ECGs

    Parameters:
    -----------
    classes: list of str,
        list of all the classes, in the format of abbrevations
    truth: sequence,
        ground truth array, of shape (n_records, n_classes), with values 0 or 1
    binary_pred: sequence,
        binary predictions, of shape (n_records, n_classes), with values 0 or 1
    scalar_pred: sequence,
        probability predictions, of shape (n_records, n_classes), with values within [0,1]

    Returns:
    --------
    auroc: float,
    auprc: float,
    accuracy: float,
    f_measure: float,
    f_beta_measure: float,
    g_beta_measure: float,
    """
    _truth = np.array(truth)
    _binary_pred = np.array(binary_pred)
    _scalar_pred = np.array(scalar_pred)

    print('- AUROC and AUPRC...')
    auroc, auprc = compute_auc(_truth, _scalar_pred)

    print('- Accuracy...')
    accuracy = compute_accuracy(_truth, _binary_pred)

    print('- F-measure...')
    f_measure = compute_f_measure(_truth, _binary_pred)

    print('- F-beta and G-beta measures...')
    f_beta_measure, g_beta_measure = compute_beta_measures(_truth, _binary_pred, beta=2)

    print('Done.')

    # Return the results.
    return auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure


# Compute recording-wise accuracy.
def compute_accuracy(labels:np.ndarray, outputs:np.ndarray) -> float:
    """ checked,
    """
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(labels:np.ndarray, outputs:np.ndarray, normalize:bool=False) -> np.ndarray:
    """ checked,
    """
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A


# Compute macro F-measure.
def compute_f_measure(labels:np.ndarray, outputs:np.ndarray) -> float:
    """ checked,
    """
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure


# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(labels:np.ndarray, outputs:np.ndarray, beta:Real) -> Tuple[float, float]:
    """ checked,
    """
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1+beta**2)*tp + fp + beta**2*fn:
            f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
        else:
            f_beta_measure[k] = float('nan')
        if tp + fp + beta*fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
        else:
            g_beta_measure[k] = float('nan')

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels:np.ndarray, outputs:np.ndarray) -> Tuple[float, float]:
    """ checked,
    """
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
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
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc
