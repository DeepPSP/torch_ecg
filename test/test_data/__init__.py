"""
"""

import os
from typing import List, Tuple, Dict

import wfdb
import numpy as np


__all__ = [
    "load_test_data",
    "load_test_seg_data",
]


_test_clf_records = [
    "E00666", "JS00666",  # from CinC2021
]

_test_seg_records = [
    "data_66_16",  # from CPSC2021
]

_CWD = os.path.dirname(os.path.abspath(__file__))


def load_test_clf_data() -> List[Tuple[np.ndarray, List[str]]]:
    """
    load test data for classification
    """
    examples = []
    for rec in _test_clf_records:
        path = os.path.join(_CWD, rec)
        sig = wfdb.rdrecord(path).p_signal.T
        header = wfdb.rdheader(path)
        label = [l.replace("Dx:", "").strip() for l in header.comments if "Dx:" in l][0].split(",")
        examples.append((sig, label))
    return examples


def load_test_seg_data() -> List[Dict[str, np.ndarray]]:
    """
    load test data for segmentation
    """
    qrs_radius = int(0.06*200)
    examples = []
    for rec in _test_seg_records:
        path = os.path.join(_CWD, rec)
        sig = wfdb.rdrecord(path).p_signal.T
        ann = wfdb.rdann(path, extension="atr")
        qrs_mask = np.zeros((sig.shape[1],1))
        af_mask = np.zeros((sig.shape[1],1))
        af_start, af_end = None, None
        for idx, beat_type, af_identifier in zip(ann.sample, ann.symbol, ann.aux_note):
            if beat_type == "+":
                if af_identifier in ["(AFIB", "(AFL",]:
                    af_start = idx
                    af_end = None
                else:  # af_identifier == "N"
                    af_end = idx
                    af_mask[af_start:af_end] = 1
                    af_start = None
            else:
                qrs_mask[max(0,idx-qrs_radius):min(sig.shape[1],idx+qrs_radius)] = 1
        if af_start is not None and af_end is None:
            af_mask[af_start, sig.shape[1]] = 1
        examples.append({"sig": sig, "qrs_mask": qrs_mask, "af_mask": af_mask})
    return examples
