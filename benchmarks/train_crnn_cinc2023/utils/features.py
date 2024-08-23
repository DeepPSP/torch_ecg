"""
features from metadata
"""

from typing import Literal, Union

import numpy as np
import pandas as pd
from helper_code import get_age, get_cpc, get_ohca, get_outcome, get_rosc, get_sex, get_shockable_rhythm, get_ttm  # get_vfib,

__all__ = [
    "get_features",
    "get_labels",
]


def get_features(patient_metadata: str, ret_type: Literal["np", "pd", "dict"] = "np") -> Union[np.ndarray, pd.DataFrame, dict]:
    """Extract features from the patient metadata.

    Adapted from the official repo.

    Parameters
    ----------
    patient_metadata : str
        The patient metadata.
    ret_type : {"np", "pd", "dict"}, default "np"
        The return value type.

    Returns
    -------
    np.ndarray or pd.DataFrame or dict
        The patient features.

    """
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_shockable_rhythm(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    sex_features = np.zeros(2, dtype=int)
    if sex == "Female":
        female = 1
        male = 0
        other = 0
    elif sex == "Male":
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    # Combine the patient features.
    if ret_type == "np":
        patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])
    elif ret_type == "pd":
        patient_features = pd.DataFrame(
            {
                "age": age,
                "sex_female": female,
                "sex_male": male,
                "sex_other": other,
                "rosc": rosc,
                "ohca": ohca,
                "vfib": vfib,
                "ttm": ttm,
            },
            index=[0],
        )
    elif ret_type == "dict":
        patient_features = {
            "age": age,
            "sex_female": female,
            "sex_male": male,
            "sex_other": other,
            "rosc": rosc,
            "ohca": ohca,
            "vfib": vfib,
            "ttm": ttm,
        }

    return patient_features


def get_labels(patient_metadata: str, ret_type: Literal["np", "pd", "dict"] = "dict") -> Union[np.ndarray, pd.DataFrame, dict]:
    """Extract labels from the patient metadata.

    Adapted from the official repo.

    Parameters
    ----------
    patient_metadata : str
        The patient metadata.
    ret_type : {"np", "pd", "dict"}, default "dict"
        The return value type.

    Returns
    -------
    dict or np.ndarray or pd.DataFrame
        The patient labels, including
        - "outcome" (int)
        - "cpc" (float)

    """
    labels = {}
    labels["outcome"] = get_outcome(patient_metadata)
    labels["cpc"] = get_cpc(patient_metadata)

    if ret_type == "dict":
        pass
    elif ret_type == "np":
        labels = np.array([labels["outcome"], labels["cpc"]])
    elif ret_type == "pd":
        labels = pd.DataFrame(labels, index=[0])

    return labels
