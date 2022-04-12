"""
denoise, mainly concerning the motion artefacts

some of the CPSC2020 records have segments of severe motion artefacts,
such segments should be eliminated from feature computation

Process
-------
1. detect segments of nearly constant values, and slice these segments out
2. detect motion artefact (large variation of values in a short time), and further slice the record into segments without motion artefact
3. more?

References
----------
to add
"""
from numbers import Real
from typing import List

import numpy as np

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[3]))

from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import mask_to_intervals


__all__ = [
    "ecg_denoise",
]


def ecg_denoise(filtered_sig: np.ndarray, fs: Real, config: CFG) -> List[List[int]]:
    """

    a naive function removing non-ECG segments (flat and motion artefact)

    Parameters
    ----------
    filtered_sig: ndarray,
        1d filtered (typically bandpassed) ECG signal,
    fs: real number,
        sampling frequency of `filtered_sig`
    config: dict,
        configs of relavant parameters, like window, step, etc.

    Returns
    -------
    intervals: list of (length 2) list of int,
        list of intervals of non-noise segment of `filtered_sig`
    """
    _LABEL_VALID, _LABEL_INVALID = 1, 0
    # constants
    siglen = len(filtered_sig)
    window = int(config.get("window", 2000) * fs / 1000)  # 2000 ms
    step = int(config.get("step", window / 5))
    ampl_min = config.get("ampl_min", 0.2)  # 0.2 mV
    ampl_max = config.get("ampl_max", 6.0)  # 6 mV

    mask = np.zeros_like(filtered_sig, dtype=int)

    if siglen < window:
        result = []
        return result

    # detect and remove flat part
    n_seg, residue = divmod(siglen - window + step, step)
    start_inds = [idx * step for idx in range(n_seg)]
    if residue != 0:
        start_inds.append(siglen - window)
        n_seg += 1

    for idx in start_inds:
        window_vals = filtered_sig[idx : idx + window]
        ampl = np.max(window_vals) - np.min(window_vals)
        if ampl > ampl_min:
            mask[idx : idx + window] = _LABEL_VALID

    # detect and remove motion artefact
    window = window // 2  # 1000 ms
    step = window // 5
    n_seg, residue = divmod(siglen - window + step, step)
    start_inds = [idx * step for idx in range(n_seg)]
    if residue != 0:
        start_inds.append(siglen - window)
        n_seg += 1

    for idx in start_inds:
        window_vals = filtered_sig[idx : idx + window]
        ampl = np.max(window_vals) - np.min(window_vals)
        if ampl > ampl_max:
            mask[idx : idx + window] = _LABEL_INVALID

    # mask to intervals
    interval_threshold = int(config.get("len_threshold", 5) * fs)  # 5s
    intervals = mask_to_intervals(mask, _LABEL_VALID)
    intervals = [item for item in intervals if item[1] - item[0] > interval_threshold]

    return intervals
