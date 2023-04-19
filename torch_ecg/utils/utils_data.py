"""
Utilities for convertions of data, labels, masks, etc.
"""

import os
import warnings
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
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
from torch import Tensor, from_numpy
from torch.nn.functional import interpolate
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
    """Get the mask around the given critical points.

    Parameters
    ----------
    shape : int or Sequence[int]
        Shape of the mask (and the original data).
    critical_points : numpy.ndarray
        Indices (of the last dimension) of the points
        around which to be masked (value 1).
    left_bias : int
        Bias to the left of the critical points for the mask,
        non-negative.
    right_bias : int
        Bias to the right of the critical points for the mask,
        non-negative.
    return_fmt : {"mask", "intervals"}
        Format of the return values, by default "mask".
        "mask" stands for the usual mask, while
        "intervals" means a list of intervals of the
        form ``[start, end]``.

    Returns
    -------
    numpy.ndarray or list
        The mask array or the list of intervals.

    Examples
    --------
    >>> mask = get_mask((12, 5000), np.arange(250, 5000 - 250, 400), 50, 50)
    >>> mask.shape
    (12, 5000)
    >>> mask.sum(axis=1)
    array([1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200])
    >>> intervals = get_mask((12, 5000), np.arange(250, 5000 - 250, 400), 50, 50, return_fmt="intervals")
    >>> intervals
    [[200, 300], [600, 700], [1000, 1100], [1400, 1500], [1800, 1900], [2200, 2300], [2600, 2700], [3000, 3100], [3400, 3500], [3800, 3900], [4200, 4300], [4600, 4700]]

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
    y: np.ndarray, class_weight: Union[str, dict, List[float], np.ndarray] = "balanced"
) -> np.ndarray:
    """Transform class weight to sample weight.

    Parameters
    ----------
    y : numpy.ndarray
        The label (class) of each sample.
    class_weight : str or dict or List[float] or numpy.ndarray, default "balanced"
        The weight for each sample class.
        If is "balanced", the class weight will automatically be given by
        the inverse of the class frequency.
        If `y` is of string `dtype`, then `class_weight` should be a :class:`dict`.
        if `y` is of numeric `dtype`, and `class_weight` is array_like,
        then the labels (`y`) should be continuous and start from 0.

    Returns
    -------
    sample_weight : numpy.ndarray
        The array of sample weight.

    Examples
    --------
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 2])
    >>> class_weight_to_sample_weight(y, class_weight="balanced").tolist()
    [0.25, 0.25, 0.25, 0.25, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1.0]
    >>> class_weight_to_sample_weight(y, class_weight=[1, 1, 3]).tolist()
    [0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1.0]
    >>> class_weight_to_sample_weight(y, class_weight={0: 2, 1: 1, 2: 3}).tolist()
    [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 1.0]
    >>> y = ["dog", "dog", "cat", "dog", "cattle", "cat", "dog", "cat"]
    >>> class_weight_to_sample_weight(y, class_weight={"dog": 1, "cat": 2, "cattle": 3}).tolist()
    [0.3333333333333333, 0.3333333333333333, 0.6666666666666666, 0.3333333333333333, 1.0, 0.6666666666666666, 0.3333333333333333, 0.6666666666666666]
    >>> class_weight_to_sample_weight(y, class_weight=[1, 2, 3])
    AssertionError: if `y` are of type str, then class_weight should be "balanced" or a dict

    """
    if not class_weight:
        sample_weight = np.ones_like(y, dtype=DEFAULTS.np_dtype)
        return sample_weight

    try:
        sample_weight = np.array(y.copy()).astype(int)
    except ValueError:
        sample_weight = np.array(y.copy())
        assert isinstance(class_weight, dict) or (
            isinstance(class_weight, str) and class_weight.lower() == "balanced"
        ), "if `y` are of type str, then class_weight should be \042balanced\042 or a dict"

    if isinstance(class_weight, str) and class_weight.lower() == "balanced":
        classes = np.unique(y).tolist()
        cw = compute_class_weight("balanced", classes=classes, y=y)
        sample_weight = np.vectorize(lambda s: cw[classes.index(s)])(sample_weight)
    else:
        sample_weight = np.vectorize(lambda s: class_weight[s])(sample_weight)
    sample_weight = sample_weight / np.max(sample_weight)
    return sample_weight


def rdheader(
    header_data: Union[Path, str, Sequence[str]]
) -> Union[Record, MultiRecord]:
    """Modified from `wfdb.rdheader`.

    Parameters
    ----------
    head_data : pathlib.Path or str or Sequence[str]
        Path of the .hea header file,
        or lines of the .hea header file.

    Returns
    -------
    wfdb.Record or wfdb.MultiRecord
        The record object.

    """
    if isinstance(header_data, (str, Path)):
        header_data = str(header_data)
        if not header_data.endswith(".hea"):
            _header_data = header_data + ".hea"
        else:
            _header_data = header_data
        if not os.path.isfile(_header_data):
            raise FileNotFoundError(f"file `{_header_data}` not found")
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
    values: np.ndarray, n_leads: int = 12, fmt: str = "lead_first"
) -> np.ndarray:
    """Ensure the multi-lead (ECG) signal to be of specified format.

    Parameters
    ----------
    values : numpy.ndarray
        Values of the `n_leads`-lead (ECG) signal.
    n_leads : int, default 12
        Number of leads of the multi-lead (ECG) signal.
    fmt : str, default "lead_first"
        Format of the output values, can be one of
        "lead_first" (alias "channel_first"),
        "lead_last" (alias "channel_last"), case insensitive.

    Returns
    -------
    numpy.ndarray
        ECG signal in the specified format.

    Examples
    --------
    >>> values = np.random.randn(5000, 12)
    >>> new_values = ensure_lead_fmt(values, fmt="lead_first")
    >>> new_values.shape
    (5000, 12)
    >>> np.allclose(values, new_values.T)
    True

    """
    dtype = values.dtype
    out_values = np.array(values, dtype=dtype)
    lead_dim = np.where(np.array(out_values.shape) == n_leads)[0].tolist()
    if not any([[0] == lead_dim or [1] == lead_dim]):
        raise ValueError(f"not valid {n_leads}-lead signal")
    lead_dim = lead_dim[0]
    if (lead_dim == 1 and fmt.lower() in ["lead_first", "channel_first"]) or (
        lead_dim == 0 and fmt.lower() in ["lead_last", "channel_last"]
    ):
        out_values = out_values.T
    elif fmt.lower() not in [
        "lead_first",
        "channel_first",
        "lead_last",
        "channel_last",
    ]:
        raise ValueError(f"not valid fmt: `{fmt}`")
    return out_values


def ensure_siglen(
    values: np.ndarray,
    siglen: int,
    fmt: str = "lead_first",
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """Ensure the (ECG) signal to be of specified length.

    Strategy:

        1. If `values` has length greater than `siglen`,
           the central `siglen` samples will be adopted;
           otherwise, zero padding will be added to both sides.
        2. If `tolerance` is given, then if the length of `values` is
           longer than `siglen` by more than `tolerance` in percentage,
           the `values` will be sliced to have multiple of `siglen` samples,
           each with ``(1 - tolerance) * siglen`` overlap.

    Parameters
    ----------
    values : numpy.ndarray
        Values of the `n_leads`-lead (ECG) signal.
    siglen : int
        Length of the signal supposed to have.
    fmt : str, default "lead_first"
        Format of the input and output values, can be one of
        "lead_first" (alias "channel_first"),
        "lead_last" (alias "channel_last"), case insensitive.
    tolerance : float, optional
        Tolerance of the length of `values` to be
        longer than `siglen` in percentage.

    Returns
    -------
    numpy.ndarray
        ECG signal in the format of `fmt` and of fixed length `siglen`,
        of ``ndim=3`` if `tolerence` is given, otherwise ``ndim=2``.

    Examples
    --------
    >>> values = np.random.randn(12, 4629)
    >>> new_values = ensure_siglen(values, 5000, fmt="lead_first")
    >>> new_values.shape
    (12, 5000)
    >>> new_values = ensure_siglen(values, 4000, tolerance=0.1, fmt="lead_first")
    >>> new_values.shape
    (2, 12, 4000)
    >>> new_values = ensure_siglen(values, 4000, tolerance=0.2, fmt="lead_first")
    >>> new_values.shape
    (1, 12, 4000)

    """
    dtype = values.dtype
    if fmt.lower() in ["channel_last", "lead_last"]:
        # to lead_first
        _values = np.array(values, dtype=dtype).T
    else:
        _values = np.array(values, dtype=dtype).copy()
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
            out_values = np.pad(
                _values, ((0, 0), (pad_left, pad_right)), "constant", constant_values=0
            ).astype(dtype)

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
        ],
        dtype=dtype,
    )
    if fmt.lower() in ["channel_last", "lead_last"]:
        out_values = np.moveaxis(out_values, 1, -1)
    return out_values.astype(dtype)


@dataclass
class ECGWaveForm:
    """Dataclass for ECG waveform information.

    Attributes
    ----------
    name : str
        Name of the wave, e.g. "N", "p", "t", etc.
    onset : numbers.Real
        Onset index of the wave,
        :class:`~numpy.nan` for unknown/unannotated onset.
    offset : numbers.Real
        Offset index of the wave,
        :class:`~numpy.nan` for unknown/unannotated offset.
    peak : numbers.Real
        Peak index of the wave,
        :class:`~numpy.nan` for unknown/unannotated peak.
    duration : numbers.Real
        Suration of the wave, with units in milliseconds,
        :class:`~numpy.nan` for unknown/unannotated duration.

    TODO
    ----
    Add `fs` field to indicate the sampling rate of the waveform,
    and make `duration` a property computed from `fs`, `onset`, and `offset`.
    """

    name: str
    onset: Real
    offset: Real
    peak: Real
    duration: Real

    @property
    def duration_(self) -> Real:
        """Duration of the wave, with units in number of samples."""
        try:
            return self.offset - self.onset
        except TypeError:
            return np.nan


# ECGWaveForm = namedtuple(
#     typename="ECGWaveForm",
#     field_names=["name", "onset", "offset", "peak", "duration"],
# )

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
    """Convert masks into lists of :class:`ECGWaveForm` for each lead.

    Parameters
    ----------
    masks : numpy.ndarray
        Wave delineation in the form of masks,
        of shape ``(n_leads, seq_len)``, or ``(seq_len,)``.
    class_map : dict
        Class map, mapping names to waves to numbers from 0 to n_classes-1,
        the keys should contain "pwave", "qrs", "twave".
    fs : numbers.Real
        Sampling frequency of the signal corresponding to the `masks`,
        used to compute the duration of each waveform.
    mask_format : str, default "channel_first"
        Format of the mask, used only when ``masks.ndim = 2``, can be
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first"), case insensitive.
    leads : str or List[str], optional
        Names of leads corresponding to the channels of the `masks`.

    Returns
    -------
    dict
        Each item value is a list containing the :class:`ECGWaveForm`
        corr. to the lead.
        Each item key is from `leads` if `leads` is set,
        otherwise would be ``"lead_1", "lead_2", ..., "lead_n"``.

    Examples
    --------
    >>> class_map = {
    ...     "pwave": 1,
    ...     "qrs": 2,
    ...     "twave": 3,
    ... }
    >>> masks = np.zeros((2, 500), dtype=int)  # 2 leads, 5000 samples
    >>> masks[:, 100:150] = 1
    >>> masks[:, 160:205] = 2
    >>> masks[:, 250:340] = 3
    >>> waveforms = masks_to_waveforms(masks, class_map=class_map, fs=500, leads=["III", "aVR"])
    >>> waveforms["III"][0]
    ECGWaveForm(name='pwave', onset=100, offset=150, peak=nan, duration=100.0)

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
    """Convert a mask into a list of intervals,
    or a dict of lists of intervals.

    Parameters
    ----------
    mask : numpy.ndarray
        The (1D) mask to convert.
    vals : int or Sequence[int], optional
        Values in `mask` to obtain the intervals.
    right_inclusive : bool, default False
        If True, the intervals will be right inclusive,
        otherwise, right exclusive.

    Returns
    -------
    intervals : dict or list
        The intervals corr. to each value in `vals`
        if `vals` is `None` or of sequence type;
        or the intervals corr. to `vals` if `vals` is int.
        each interval is of the form ``[a, b]``.

    Examples
    --------
    >>> mask = np.zeros(100, dtype=int)
    >>> mask[10: 20] = 1
    >>> mask[80: 95] = 1
    >>> mask[50: 60] = 2
    >>> mask_to_intervals(mask, vals=1)
    [[10, 20], [80, 95]]
    >>> mask_to_intervals(mask, vals=[0, 2])
    {0: [[0, 10], [20, 50], [60, 80], [95, 100]], 2: [[50, 60]]}
    >>> mask_to_intervals(mask)
    {0: [[0, 10], [20, 50], [60, 80], [95, 100]], 1: [[10, 20], [80, 95]], 2: [[50, 60]]}
    >>> mask_to_intervals(mask, vals=[1, 2], right_inclusive=True)
    {1: [[10, 19], [80, 94]], 2: [[50, 59]]}

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
    """Generate a list of numbers uniformly distributed.

    Parameters
    ----------
    low : numbers.Real
        Lower bound of the interval of the uniform distribution.
    high : numbers.Real
        Upper bound of the interval of the uniform distribution.
    num : int
        Number of random numbers to generate.

    Returns
    -------
    List[float]
        Array of randomly generated numbers with uniform distribution.

    Examples
    --------
    >>> arr = uniform(0, 1, 10)
    >>> all([0 <= x <= 1 for x in arr])
    True

    """
    arr = [DEFAULTS.RNG.uniform(low, high) for _ in range(num)]
    return arr


def stratified_train_test_split(
    df: pd.DataFrame,
    stratified_cols: Sequence[str],
    test_ratio: float = 0.2,
    reset_index: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratified train-test split on the dataframe.

    For example,
    if one has a dataframe with columns `sex`, `nationality`, etc.,
    assuming `sex` includes `male`, `female`; `nationality` includes `Chinese`, `American`,
    and sets `stratified_cols = ["sex", "nationality"]` with `test_ratio = 0.2`,
    then approximately 20% of the male and 20% of the female subjects
    will be put into the test set,
    and **at the same time**, approximately 20% of the Chinese and 20% of the Americans
    lie in the test set as well.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be split.
    stratified_cols : Sequence[str]
        Columns to be stratified,
        assuming each column is a categorical variable.
        Each class in any of the columns will be split into
        train and test sets with an approximate ratio of `test_ratio`.
    test_ratio : float, default 0.2
        Ratio of test set to the whole dataframe.
    reset_index : bool, default False
        Whether to reset the index of the dataframes.

    Returns
    -------
    df_train : pandas.DataFrame
        The dataframe of the train set.
    df_test : pandas.DataFrame
        The dataframe of the test set.

    """
    invalid_cols = [
        col
        for col in stratified_cols
        if not all([v > 1 for v in Counter(df[col].apply(str)).values()])
    ]
    if len(invalid_cols) > 0:
        warnings.warn(
            f"invalid columns: {invalid_cols}, "
            "each of which has classes with only one member (row).",
            RuntimeWarning,
        )
    stratified_cols = [col for col in stratified_cols if col not in invalid_cols]
    # map to str to avoid incorrect comparison of nan values
    df_inspection = df[stratified_cols].copy().applymap(str)
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
    df_test = df.loc[df.index.isin(test_indices)]
    df_train = df.loc[~df.index.isin(test_indices)]
    if reset_index:
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
    return df_train, df_test


def cls_to_bin(
    cls_array: Union[np.ndarray, Tensor], num_classes: Optional[int] = None
) -> np.ndarray:
    """Convert a categorical array to a one-hot array.

    Convert a categorical (class indices) array of shape ``(n,)``
    to a one-hot (binary) array of shape ``(n, num_classes)``.

    Parameters
    ----------
    cls_array : numpy.ndarray or torch.Tensor
        Class indices array (tensor) of shape ``(num_samples,)``;
        or of shape ``(num_samples, num_samples)`` if `num_classes` is not None,
        in which case `cls_array` should be consistant with `num_classes`,
        and the function will return `cls_array` directly.
    num_classes : int, optional
        Number of classes. If not specified,
        it will be inferred from the values of `cls_array`.

    Returns
    -------
    numpy.ndarray
        Binary array of shape ``(num_samples, num_classes)``.

    Examples
    --------
    >>> cls_array = torch.randint(0, 26, size=(1000,))
    >>> bin_array = cls_to_bin(cls_array)
    >>> cls_array = np.random.randint(0, 26, size=(1000,))
    >>> bin_array = cls_to_bin(cls_array)

    """
    if isinstance(cls_array, Tensor):
        cls_array = cls_array.cpu().numpy()
    if num_classes is None:
        assert (
            cls_array.ndim == 1
        ), "`cls_array` should be 1D if num_classes is not specified"
        num_classes = cls_array.max() + 1
    if cls_array.ndim == 1:
        assert num_classes > 0 and num_classes >= cls_array.max() + 1, (
            "num_classes must be greater than 0 and greater than or equal to "
            "the max value of `cls_array` if `cls_array` is 1D and `num_classes` is specified"
        )
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
    fg_weight: Real,
    fs: Real,
    reduction: Real,
    radius: Real,
    boundary_weight: Real,
    plot: bool = False,
) -> np.ndarray:
    """Generate weight mask for a binary target mask,
    accounting the foreground weight and boundary weight.

    Parameters
    ----------
    target_mask : numpy.ndarray
        The target mask, assumed to be 1D and binary.
    fg_weight: numbers.Real
        Foreground (value 1) weight, usually > 1.
    fs : numbers.Real
        Sampling frequency of the signal.
    reduction : numbers.Real
        Reduction ratio of the mask w.r.t. the signal.
    radius : numbers.Real
        Radius of the boundary, with units in seconds.
    boundary_weight : numbers.Real
        Weight for the boundaries (positions where values change)
        of the target map.
    plot : bool, default False
        If True, `target_mask` and the result `weight_mask` will be plotted.

    Returns
    -------
    weight_mask : numpy.ndarray
        Weight mask generated from `target_mask`.

    Examples
    --------
    >>> target_mask = np.zeros(50000, dtype=int)
    >>> target_mask[500:14000] = 1
    >>> target_mask[35800:44600] = 1
    >>> fg_weight = 2.0
    >>> fs = 500
    >>> reduction = 1
    >>> radius = 0.8
    >>> boundary_weight = 5.0
    >>> weight_mask = generate_weight_mask(
    ...     target_mask, fg_weight, fs, reduction, radius, boundary_weight
    ... )
    >>> weight_mask.shape
    (50000,)
    >>> reduction = 10
    >>> weight_mask = generate_weight_mask(
    ...     target_mask, fg_weight, fs, reduction, radius, boundary_weight
    ... )
    >>> weight_mask.shape
    (5000,)

    """
    assert target_mask.ndim == 1, "`target_mask` should be 1D"
    assert set(np.unique(target_mask)).issubset(
        {0, 1}
    ), "`target_mask` should be binary"
    assert (
        isinstance(reduction, Real) and reduction >= 1
    ), "`reduction` should be a real number greater than 1"
    if reduction > 1:
        # downsample the target mask
        target_mask = (
            interpolate(
                from_numpy(target_mask).unsqueeze(0).unsqueeze(0).float(),
                mode="nearest",
                scale_factor=1 / reduction,
                recompute_scale_factor=True,
            )
            .squeeze(0)
            .squeeze(0)
            .numpy()
            .astype(DEFAULTS.DTYPE.NP)
        )
    weight_mask = np.ones_like(target_mask, dtype=DEFAULTS.DTYPE.NP)
    sigma = int((radius * fs) / reduction)
    weight = np.full_like(target_mask, fg_weight) - 1
    weight_mask += (target_mask > 0.5) * weight
    border = np.where(np.diff(target_mask) != 0)[0]
    for idx in border:
        # weight = np.zeros_like(target_mask, dtype=DEFAULTS.DTYPE.NP)
        # weight[max(0, idx-sigma): (idx+sigma)] = boundary_weight
        weight = np.full_like(target_mask, boundary_weight, dtype=DEFAULTS.DTYPE.NP)
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
