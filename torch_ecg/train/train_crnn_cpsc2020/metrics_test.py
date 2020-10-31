"""
"""
from typing import Union, Optional, Any, List, Tuple

import numpy as np
from easydict import EasyDict as ED

from .utils import dict_to_str, in_generalized_interval
from .cfg import BaseCfg
from .metrics import CPSC2020_loss, CPSC2020_score


__all__ = [
    "CPSC2020_loss_test",
    "CPSC2020_score_test",
]


def CPSC2020_loss_test(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str) -> int:
    """

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
    retval: dict, including the following items
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type
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

    # true_positive = {
    #     c: np.array([in_generalized_interval(idx, pred_intervals[c]) for idx in truth_arr[c]]).astype(int).sum() \
    #         for c in classes
    # }
    false_positive = {
        c: len(pred_arr[c]) - true_positive[c] for c in classes
    }
    false_negative = {
        c: len(truth_arr[c]) - true_positive[c] for c in classes
    }

    false_positive_loss = {c: 1 for c in classes}
    false_negative_loss = {c: 5 for c in classes}

    print(f"true_positive = {dict_to_str(true_positive)}")
    print(f"false_positive = {dict_to_str(false_positive)}")
    print(f"false_negative = {dict_to_str(false_negative)}")

    class_loss = {
        c: false_positive[c] * false_positive_loss[c] + false_negative[c] * false_negative_loss[c] \
            for c in classes
    }

    total_loss = sum(class_loss.values())

    retval = ED(
        total_loss=total_loss,
        class_loss=class_loss,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    return retval


def CPSC2020_score_test(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str) -> int:
    """

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
    retval: dict, including the following items
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type
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

    retval = CPSC2020_score(
        [truth_arr['S']],[truth_arr['V']],[pred_arr['S']],[pred_arr['V']],
        verbose=1,
    )

    return retval


@DeprecationWarning
def CPSC2020_loss_v0(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str) -> int:
    """ finished, too slow!

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
    retval: dict, including the following items
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - class_loss: loss of each ectopic beat type
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type
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

    pred_intervals = {
        c: [[idx-BaseCfg.bias_thr, idx+BaseCfg.bias_thr] for idx in pred_arr[c]] \
            for c in classes
    }

    true_positive = {
        c: np.array([in_generalized_interval(idx, pred_intervals[c]) for idx in truth_arr[c]]).astype(int).sum() \
            for c in classes
    }
    false_positive = {
        c: len(pred_arr[c]) - true_positive[c] for c in classes
    }
    false_negative = {
        c: len(truth_arr[c]) - true_positive[c] for c in classes
    }

    false_positive_loss = {c: 1 for c in classes}
    false_negative_loss = {c: 5 for c in classes}

    class_loss = {
        false_positive[c] * false_positive_loss[c] + false_negative[c] * false_negative_loss[c] \
            for c in classes
    }

    total_loss = sum(class_loss.values())

    retval = ED(
        total_loss=total_loss,
        class_loss=class_loss,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )

    return retval
