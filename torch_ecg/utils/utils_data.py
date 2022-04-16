"""
utilities for convertions of data, labels, masks, etc.

"""

import os
import warnings
from collections import namedtuple, Counter
from copy import deepcopy
from numbers import Real
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from torch import Tensor
from sklearn.utils import compute_class_weight
from wfdb import MultiRecord, Record
from wfdb.io import _header

from ..cfg import CFG, DEFAULTS


__all__ = [
    "get_mask",
    "class_weight_to_sample_weight",
    "rdheader",
    "ensure_lead_fmt",
    "ensure_siglen",
    "ECGWaveForm",
    "ECGWaveFormNames",
    "masks_to_waveforms",
    "mask_to_intervals",
    "uniform",
    "stratified_train_test_split",
    "cls_to_bin",
    "generate_weight_mask",
]


def get_mask(
    shape: Union[int, Sequence[int]],
    critical_points: np.ndarray,
    left_bias: int,
    right_bias: int,
    return_fmt: str = "mask",
) -> Union[np.ndarray, list]:
    """

    get the mask around the `critical_points`

    Parameters
    ----------
    shape: int, or sequence of int,
        shape of the mask (and the original data)
    critical_points: ndarray,
        indices (of the last dimension) of the points around which to be masked (value 1)
    left_bias: int, non-negative
        bias to the left of the critical points for the mask
    right_bias: int, non-negative
        bias to the right of the critical points for the mask
    return_fmt: str, default "mask",
        format of the return values,
        "mask" for the usual mask,
        can also be "intervals", which consists of a list of intervals

    Returns
    -------
    mask: ndarray or list,
        the mask array

    """
    if isinstance(shape, int):
        shape = (shape,)
    l_itv = [
        [max(0, cp - left_bias), min(shape[-1], cp + right_bias)]
        for cp in critical_points
    ]
    if return_fmt.lower() == "mask":
        mask = np.zeros(shape=shape, dtype=int)
        for itv in l_itv:
            mask[..., itv[0] : itv[1]] = 1
    elif return_fmt.lower() == "intervals":
        mask = l_itv
    return mask


def class_weight_to_sample_weight(
    y: np.ndarray, class_weight: Union[str, List[float], np.ndarray, dict] = "balanced"
) -> np.ndarray:
    """

    transform class weight to sample weight

    Parameters
    ----------
    y: ndarray,
        the label (class) of each sample
    class_weight: str, or list, or ndarray, or dict, default "balanced",
        the weight for each sample class,
        if is "balanced", the class weight will automatically be given by
        if `y` is of string type, then `class_weight` should be a dict,
        if `y` is of numeric type, and `class_weight` is array_like,
        then the labels (`y`) should be continuous and start from 0

    Returns
    -------
    sample_weight: ndarray,
        the array of sample weight

    """
    if not class_weight:
        sample_weight = np.ones_like(y, dtype=float)
        return sample_weight

    try:
        sample_weight = y.copy().astype(int)
    except Exception:
        sample_weight = y.copy()
        assert (
            isinstance(class_weight, dict) or class_weight.lower() == "balanced"
        ), "if `y` are of type str, then class_weight should be \042balanced\042 or a dict"

    if isinstance(class_weight, str) and class_weight.lower() == "balanced":
        classes = np.unique(y).tolist()
        cw = compute_class_weight("balanced", classes=classes, y=y)
        sample_weight = np.vectorize(lambda s: cw[classes.index(s)])(sample_weight)
    else:
        sample_weight = np.vectorize(lambda s: class_weight[s])(sample_weight)
    sample_weight = sample_weight / np.max(sample_weight)
    return sample_weight


def rdheader(header_data: Union[str, Sequence[str]]) -> Union[Record, MultiRecord]:
    """

    modified from `wfdb.rdheader`

    Parameters
    ----------
    head_data: str, or sequence of str,
        path of the .hea header file, or lines of the .hea header file

    """
    if isinstance(header_data, str):
        if not header_data.endswith(".hea"):
            _header_data = header_data + ".hea"
        else:
            _header_data = header_data
        if not os.path.isfile(_header_data):
            raise FileNotFoundError
        with open(_header_data, "r") as f:
            _header_data = f.read().splitlines()
    elif isinstance(header_data, Sequence):
        _header_data = header_data
    else:
        raise TypeError(
            f"header_data must be str or sequence of str, but got {type(header_data)}"
        )
    # Read the header file. Separate comment and non-comment lines
    header_lines, comment_lines = [], []
    for line in _header_data:
        striped_line = line.strip()
        # Comment line
        if striped_line.startswith("#"):
            comment_lines.append(striped_line)
        # Non-empty non-comment line = header line.
        elif striped_line:
            # Look for a comment in the line
            ci = striped_line.find("#")
            if ci > 0:
                header_lines.append(striped_line[:ci])
                # comment on same line as header line
                comment_lines.append(striped_line[ci:])
            else:
                header_lines.append(striped_line)

    # Get fields from record line
    record_fields = _header._parse_record_line(header_lines[0])

    # Single segment header - Process signal specification lines
    if record_fields["n_seg"] is None:
        # Create a single-segment WFDB record object
        record = Record()

        # There are signals
        if len(header_lines) > 1:
            # Read the fields from the signal lines
            signal_fields = _header._parse_signal_lines(header_lines[1:])
            # Set the object's signal fields
            for field in signal_fields:
                setattr(record, field, signal_fields[field])

        # Set the object's record line fields
        for field in record_fields:
            if field == "n_seg":
                continue
            setattr(record, field, record_fields[field])
    # Multi segment header - Process segment specification lines
    else:
        # Create a multi-segment WFDB record object
        record = MultiRecord()
        # Read the fields from the segment lines
        segment_fields = _header._read_segment_lines(header_lines[1:])
        # Set the object's segment fields
        for field in segment_fields:
            setattr(record, field, segment_fields[field])
        # Set the objects' record fields
        for field in record_fields:
            setattr(record, field, record_fields[field])

        # Determine whether the record is fixed or variable
        if record.seg_len[0] == 0:
            record.layout = "variable"
        else:
            record.layout = "fixed"

    # Set the comments field
    record.comments = [line.strip(" \t#") for line in comment_lines]

    return record


def ensure_lead_fmt(
    values: Sequence[Real], n_leads: int = 12, fmt: str = "lead_first"
) -> np.ndarray:
    """

    ensure the `n_leads`-lead (ECG) signal to be of the format of `fmt`

    Parameters
    ----------
    values: sequence,
        values of the `n_leads`-lead (ECG) signal
    n_leads: int, default 12,
        number of leads
    fmt: str, default "lead_first", case insensitive,
        format of the output values, can be one of
        "lead_first" (alias "channel_first"), "lead_last" (alias "channel_last")

    Returns
    -------
    out_values: ndarray,
        ECG signal in the format of `fmt`

    """
    out_values = np.array(values)
    lead_dim = np.where(np.array(out_values.shape) == n_leads)[0]
    if not any([[0] == lead_dim or [1] == lead_dim]):
        raise ValueError(f"not valid {n_leads}-lead signal")
    lead_dim = lead_dim[0]
    if (lead_dim == 1 and fmt.lower() in ["lead_first", "channel_first"]) or (
        lead_dim == 0 and fmt.lower() in ["lead_last", "channel_last"]
    ):
        out_values = out_values.T
        return out_values
    return out_values


def ensure_siglen(
    values: Sequence[Real],
    siglen: int,
    fmt: str = "lead_first",
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """

    ensure the (ECG) signal to be of length `siglen`,
    strategy:
        If `values` has length greater than `siglen`,
        the central `siglen` samples will be adopted;
        otherwise, zero padding will be added to both sides.
        If `tolerance` is given,
        then if the length of `values` is longer than `siglen` by more than `tolerance`,
        the `values` will be sliced to have multiple of `siglen` samples.

    Parameters
    ----------
    values: sequence,
        values of the `n_leads`-lead (ECG) signal
    siglen: int,
        length of the signal supposed to have
    fmt: str, default "lead_first", case insensitive,
        format of the input and output values, can be one of
        "lead_first" (alias "channel_first"), "lead_last" (alias "channel_last")

    Returns
    -------
    out_values: ndarray,
        ECG signal in the format of `fmt` and of fixed length `siglen`,
        of ndim=3 if `tolerence` is given, otherwise ndim=2

    """
    if fmt.lower() in ["channel_last", "lead_last"]:
        _values = np.array(values).T
    else:
        _values = np.array(values).copy()
    original_siglen = _values.shape[1]
    n_leads = _values.shape[0]

    if tolerance is None or original_siglen <= siglen * (1 + tolerance):
        if original_siglen >= siglen:
            start = (original_siglen - siglen) // 2
            end = start + siglen
            out_values = _values[..., start:end]
        else:
            pad_len = siglen - original_siglen
            pad_left = pad_len // 2
            pad_right = pad_len - pad_left
            out_values = np.concatenate(
                [
                    np.zeros((n_leads, pad_left)),
                    _values,
                    np.zeros((n_leads, pad_right)),
                ],
                axis=1,
            )

        if fmt.lower() in ["channel_last", "lead_last"]:
            out_values = out_values.T
        if tolerance is not None:
            out_values = out_values[np.newaxis, ...]

        return out_values

    forward_len = int(round(siglen * tolerance))
    out_values = np.array(
        [
            _values[..., idx * forward_len : idx * forward_len + siglen]
            for idx in range((original_siglen - siglen) // forward_len + 1)
        ]
    )
    if fmt.lower() in ["channel_last", "lead_last"]:
        out_values = np.moveaxis(out_values, 1, -1)
    return out_values


ECGWaveForm = namedtuple(
    typename="ECGWaveForm",
    field_names=["name", "onset", "offset", "peak", "duration"],
)
ECGWaveFormNames = [
    "pwave",
    "qrs",
    "twave",
]


def masks_to_waveforms(
    masks: np.ndarray,
    class_map: Dict[str, int],
    fs: Real,
    mask_format: str = "channel_first",
    leads: Optional[Sequence[str]] = None,
) -> Dict[str, List[ECGWaveForm]]:
    """

    convert masks into lists of waveforms

    Parameters
    ----------
    masks: ndarray,
        wave delineation in the form of masks,
        of shape (n_leads, seq_len), or (seq_len,)
    class_map: dict,
        class map, mapping names to waves to numbers from 0 to n_classes-1,
        the keys should contain "pwave", "qrs", "twave"
    fs: real number,
        sampling frequency of the signal corresponding to the `masks`,
        used to compute the duration of each waveform
    mask_format: str, default "channel_first",
        format of the mask, used only when `masks.ndim = 2`
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first")
    leads: str or list of str, optional,
        the names of leads corresponding to the channels of the `masks`

    Returns
    -------
    waves: dict,
        each item value is a list containing the `ECGWaveForm`s corr. to the lead;
        each item key is from `leads` if `leads` is set,
        otherwise would be "lead_1", "lead_2", ..., "lead_n"

    """
    if masks.ndim == 1:
        _masks = masks[np.newaxis, ...]
    elif masks.ndim == 2:
        if mask_format.lower() not in [
            "channel_first",
            "lead_first",
        ]:
            _masks = masks.T
        else:
            _masks = masks.copy()
    else:
        raise ValueError(
            f"masks should be of dim 1 or 2, but got a {masks.ndim}d array"
        )

    _leads = (
        [f"lead_{idx+1}" for idx in range(_masks.shape[0])] if leads is None else leads
    )
    assert len(_leads) == _masks.shape[0]

    _class_map = CFG(deepcopy(class_map))

    waves = CFG({lead_name: [] for lead_name in _leads})
    for channel_idx, lead_name in enumerate(_leads):
        current_mask = _masks[channel_idx, ...]
        for wave_name, wave_number in _class_map.items():
            if wave_name.lower() not in ECGWaveFormNames:
                continue
            current_wave_inds = np.where(current_mask == wave_number)[0]
            if len(current_wave_inds) == 0:
                continue
            np.where(np.diff(current_wave_inds) > 1)
            split_inds = np.where(np.diff(current_wave_inds) > 1)[0].tolist()
            split_inds = sorted(split_inds + [i + 1 for i in split_inds])
            split_inds = [0] + split_inds + [len(current_wave_inds) - 1]
            for i in range(len(split_inds) // 2):
                itv_start = current_wave_inds[split_inds[2 * i]]
                itv_end = current_wave_inds[split_inds[2 * i + 1]] + 1
                w = ECGWaveForm(
                    name=wave_name.lower(),
                    onset=itv_start,
                    offset=itv_end,
                    peak=np.nan,
                    duration=1000 * (itv_end - itv_start) / fs,  # ms
                )
                waves[lead_name].append(w)
        waves[lead_name].sort(key=lambda w: w.onset)
    return waves


def mask_to_intervals(
    mask: np.ndarray,
    vals: Optional[Union[int, Sequence[int]]] = None,
    right_inclusive: bool = False,
) -> Union[list, dict]:
    """

    Parameters
    ----------
    mask: ndarray,
        1d mask
    vals: int or sequence of int, optional,
        values in `mask` to obtain intervals
    right_inclusive: bool, default False,
        if True, the intervals will be right inclusive
        otherwise, right exclusive

    Returns
    -------
    intervals: dict or list,
        the intervals corr. to each value in `vals` if `vals` is `None` or `Sequence`;
        or the intervals corr. to `vals` if `vals` is int.
        each interval is of the form `[a,b]`

    """
    if vals is None:
        _vals = list(set(mask))
    elif isinstance(vals, int):
        _vals = [vals]
    else:
        _vals = vals
    # assert set(_vals) & set(mask) == set(_vals)
    bias = 0 if right_inclusive else 1

    intervals = {v: [] for v in _vals}
    for v in _vals:
        valid_inds = np.where(np.array(mask) == v)[0]
        if len(valid_inds) == 0:
            continue
        split_indices = np.where(np.diff(valid_inds) > 1)[0]
        split_indices = split_indices.tolist() + (split_indices + 1).tolist()
        split_indices = sorted([0] + split_indices + [len(valid_inds) - 1])
        for idx in range(len(split_indices) // 2):
            intervals[v].append(
                [
                    valid_inds[split_indices[2 * idx]],
                    valid_inds[split_indices[2 * idx + 1]] + bias,
                ]
            )

    if isinstance(vals, int):
        intervals = intervals[vals]

    return intervals


def uniform(low: Real, high: Real, num: int) -> List[float]:
    """

    Parameters
    ----------
    low: real number,
        lower bound of the interval of the uniform distribution
    high: real number,
        upper bound of the interval of the uniform distribution
    num: int,
        number of random numbers to generate

    Returns
    -------
    arr: list of float,
        array of randomly generated numbers with uniform distribution

    """
    arr = [DEFAULTS.RNG.uniform(low, high) for _ in range(num)]
    return arr


def stratified_train_test_split(
    df: pd.DataFrame, stratified_cols: Sequence[str], test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Do stratified train-test split on the dataframe,

    Parameters
    ----------
    df: pd.DataFrame,
        dataframe to be split
    stratified_cols: sequence of str,
        columns to be stratified, assuming each column is a categorical variable
        each class in any of the columns will be
        split into train and test sets with an approximate ratio of `test_ratio`
    test_ratio: float, default 0.2,
        ratio of test set to the whole dataframe

    Returns
    -------
    df_train: pd.DataFrame,
        the dataframe of the train set
    df_test: pd.DataFrame,
        the dataframe of the test set

    For example,
    if one has a dataframe with columns `sex`, `nationality`, etc.,
    assuming `sex` includes `male`, `female`; `nationality` includes `Chinese`, `American`,
    and sets `stratified_cols = ["sex", "nationality"]` with `test_ratio = 0.2`,
    then approximately 20% of the male and 20% of the female subjects
    will be put into the test set,
    and **at the same time**, approximately 20% of the Chinese and 20% of the Americans
    lie in the test set as well.

    """
    invalid_cols = [
        col
        for col in stratified_cols
        if not all([v > 1 for v in Counter(df[col]).values()])
    ]
    if len(invalid_cols) > 0:
        warnings.warn(
            f"invalid columns: {invalid_cols}, "
            "each of which has classes with only one member (row), "
        )
    stratified_cols = [col for col in stratified_cols if col not in invalid_cols]
    df_inspection = df[stratified_cols].copy()
    for item in stratified_cols:
        all_entities = df_inspection[item].unique().tolist()
        entities_dict = {e: str(i) for i, e in enumerate(all_entities)}
        df_inspection[item] = df_inspection[item].apply(lambda e: entities_dict[e])

    inspection_col_name = "Inspection" * (
        max([len(c) for c in stratified_cols]) // 10 + 1
    )
    df_inspection[inspection_col_name] = ""
    for idx, row in df_inspection.iterrows():
        cn = "-".join([row[sc] for sc in stratified_cols])
        df_inspection.loc[idx, inspection_col_name] = cn
    item_names = df_inspection[inspection_col_name].unique().tolist()
    item_indices = {
        n: df_inspection.index[df_inspection[inspection_col_name] == n].tolist()
        for n in item_names
    }
    for n in item_names:
        DEFAULTS.RNG.shuffle(item_indices[n])

    test_indices = []
    for n in item_names:
        item_test_indices = item_indices[n][: round(test_ratio * len(item_indices[n]))]
        test_indices += item_test_indices
    df_test = df.loc[df.index.isin(test_indices)].reset_index(drop=True)
    df_train = df.loc[~df.index.isin(test_indices)].reset_index(drop=True)
    return df_train, df_test


def cls_to_bin(
    cls_array: Union[np.ndarray, Tensor], num_classes: Optional[int] = None
) -> np.ndarray:
    """

    converting a categorical (class indices) array of shape (n,)
    to a one-hot (binary) array of shape (n, num_classes)

    Parameters
    ----------
    cls_array: ndarray,
        class indices array of shape (n,)
    num_classes: int, optional,
        number of classes,
        if not specified, it will be inferred from the values of `cls_array`

    Returns
    -------
    bin_array: ndarray,
        binary array of shape (n, num_classes)

    """
    if isinstance(cls_array, Tensor):
        cls_array = cls_array.cpu().numpy()
    if num_classes is None:
        num_classes = cls_array.max() + 1
    assert (
        num_classes > 0 and num_classes == cls_array.max() + 1
    ), "num_classes must be greater than 0 and equal to the max value of cls_array"
    if cls_array.ndim == 2 and cls_array.shape[1] == num_classes:
        bin_array = cls_array
    else:
        shape = (cls_array.shape[0], num_classes)
        bin_array = np.zeros(shape)
        for i in range(shape[0]):
            bin_array[i, cls_array[i]] = 1
    return bin_array


def generate_weight_mask(
    target_mask: np.ndarray,
    fg_weight: float,
    fs: int,
    reduction: int,
    radius: float,
    boundary_weight: float,
    plot: bool = False,
) -> np.ndarray:
    """

    generate weight mask for a binary target mask,
    accounting the foreground weight and boundary weight

    Parameters
    ----------
    target_mask: ndarray,
        the target mask, assumed to be 1d
    fg_weight: float,
        foreground weight, usually > 1
    fs: int,
        sampling frequency of the signal
    reduction: int,
        reduction ratio of the mask w.r.t. the signal
    boundary_weight: float,
        weight for the boundaries (positions where values change) of the target map
    plot: bool, default False,
        if True, target_mask and the result weight_mask will be plotted

    Returns
    -------
    weight_mask: ndarray,
        the weight mask

    """
    weight_mask = np.ones_like(target_mask, dtype=float)
    sigma = int((radius * fs) / reduction)
    weight = np.full_like(target_mask, fg_weight) - 1
    weight_mask += (target_mask > 0.5) * weight
    border = np.where(np.diff(target_mask) != 0)[0]
    for idx in border:
        # weight = np.zeros_like(target_mask, dtype=float)
        # weight[max(0, idx-sigma): (idx+sigma)] = boundary_weight
        weight = np.full_like(target_mask, boundary_weight, dtype=float)
        weight = weight * np.exp(
            -np.power(np.arange(len(target_mask)) - idx, 2) / sigma**2
        )
        weight_mask += weight
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(target_mask, label="target mask")
        ax.plot(weight_mask, label="weight mask")
        ax.set_xlabel("samples")
        ax.set_ylabel("weight")
        ax.legend(loc="best")
        plt.show()
    return weight_mask
