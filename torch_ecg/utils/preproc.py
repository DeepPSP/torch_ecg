"""
preprocess of (single lead) ecg signal:
    band pass (dep. on purpose?) --> remove baseline (?) --> find rpeaks
    --> wave delineation (?, put into several stand alone files)

NOTE:
-----
1. ALL signals are assumed to have units in mV

References:
-----------
[1] https://github.com/PIA-Group/BioSPPy
[2] to add

"""

import multiprocessing as mp
from collections import Counter
from numbers import Real
from typing import Dict, List, Optional

import numpy as np

# from scipy.signal import medfilt
# https://github.com/scipy/scipy/issues/9680
from biosppy.signals.tools import filter_signal
from scipy.ndimage.filters import median_filter

from ..cfg import CFG
from .misc import list_sum, ms2samples
from .utils_data import get_mask
from .rpeaks import (
    christov_detect,
    engzee_detect,
    gamboa_detect,
    gqrs_detect,
    hamilton_detect,
    pantompkins_detect,
    ssf_detect,
    xqrs_detect,
)

__all__ = [
    "preprocess_multi_lead_signal",
    "preprocess_single_lead_signal",
    "rpeaks_detect_multi_leads",
    "merge_rpeaks",
]


QRS_DETECTORS = {
    "xqrs": xqrs_detect,
    "gqrs": gqrs_detect,
    "pantompkins": pantompkins_detect,
    "hamilton": hamilton_detect,
    "ssf": ssf_detect,
    "christov": christov_detect,
    "engzee": engzee_detect,
    "gamboa": gamboa_detect,
    # "seq_lab": seq_lab_net_detect,
}
DL_QRS_DETECTORS = [
    # "seq_lab",  # currently set empty
]
# ecg signal preprocessing configurations
PreprocCfg = CFG()
# PreprocCfg.fs = 500
PreprocCfg.rpeak_mask_radius = 50  # ms
PreprocCfg.rpeak_lead_num_thr = (
    8 / 12
)  # ratio of leads, used for merging rpeaks detected from multiple leads
PreprocCfg.beat_winL = 250
PreprocCfg.beat_winR = 250


def preprocess_multi_lead_signal(
    raw_sig: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    bl_win: Optional[List[Real]] = None,
    band_fs: Optional[List[Real]] = None,
    rpeak_fn: Optional[str] = None,
    verbose: int = 0,
) -> Dict[str, np.ndarray]:
    """

    perform preprocessing for multi-lead ecg signal (with units in mV),
    preprocessing may include median filter, bandpass filter, and rpeaks detection, etc.

    Parameters
    ----------
    raw_sig: ndarray,
        the raw ecg signal, with units in mV
    fs: real number,
        sampling frequency of `raw_sig`
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    bl_win: list (of 2 real numbers), optional,
        window (units in second) of baseline removal using `median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is [0.2, 0.6],
        if is None or empty, baseline removal will not be performed
    band_fs: list (of 2 real numbers), optional,
        frequency band of the bandpass filter,
        a typical pair is [0.5, 45],
        be careful when detecting paced rhythm,
        if is None or empty, bandpass filtering will not be performed
    rpeak_fn: str, optional,
        name of the function detecting rpeaks,
        can be one of keys of `QRS_DETECTORS`, case insensitive
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    retval: dict,
        with items
        - "filtered_ecg": the array of the processed ecg signal
        - "rpeaks": the array of indices of rpeaks; empty if `rpeak_fn` is not given

    NOTE: currently NEVER set verbose >= 3

    """
    assert sig_fmt.lower() in [
        "channel_first",
        "lead_first",
        "channel_last",
        "lead_last",
    ]
    if sig_fmt.lower() in ["channel_last", "lead_last"]:
        filtered_ecg = raw_sig.T
    else:
        filtered_ecg = raw_sig.copy()
    rpeaks_candidates = []
    cpu_num = max(1, mp.cpu_count() - 3)
    with mp.Pool(processes=cpu_num) as pool:
        results = pool.starmap(
            func=preprocess_single_lead_signal,
            iterable=[
                (filtered_ecg[lead, ...], fs, bl_win, band_fs, rpeak_fn, verbose)
                for lead in range(filtered_ecg.shape[0])
            ],
        )
    for lead in range(filtered_ecg.shape[0]):
        # filtered_metadata = preprocess_single_lead_signal(
        #     raw_sig=filtered_ecg[lead,...],
        #     fs=fs,
        #     bl_win=bl_win,
        #     band_fs=band_fs,
        #     rpeak_fn=rpeak_fn
        # )
        filtered_metadata = results[lead]
        filtered_ecg[lead, ...] = filtered_metadata["filtered_ecg"]
        rpeaks_candidates.append(filtered_metadata["rpeaks"])
        if verbose >= 1:
            print(
                f"for the {lead}-th lead, rpeaks_candidates = {filtered_metadata['rpeaks']}"
            )

    if rpeak_fn and rpeak_fn in DL_QRS_DETECTORS:
        rpeaks = rpeaks_detect_multi_leads(
            sig=filtered_ecg,
            fs=fs,
            sig_fmt=sig_fmt,
            rpeak_fn=rpeak_fn,
            verbose=verbose,
        )
    rpeaks = merge_rpeaks(rpeaks_candidates, raw_sig, fs, verbose)
    retval = CFG(
        {
            "filtered_ecg": filtered_ecg,
            "rpeaks": rpeaks,
        }
    )
    return retval


def preprocess_single_lead_signal(
    raw_sig: np.ndarray,
    fs: Real,
    bl_win: Optional[List[Real]] = None,
    band_fs: Optional[List[Real]] = None,
    rpeak_fn: Optional[str] = None,
    verbose: int = 0,
) -> Dict[str, np.ndarray]:
    """

    perform preprocessing for single lead ecg signal (with units in mV),
    preprocessing may include median filter, bandpass filter, and rpeaks detection, etc.

    Parameters
    ----------
    raw_sig: ndarray,
        the raw ecg signal, with units in mV
    fs: real number,
        sampling frequency of `raw_sig`
    bl_win: list (of 2 real numbers), optional,
        window (units in second) of baseline removal using `median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is [0.2, 0.6],
        if is None or empty, baseline removal will not be performed
    band_fs: list (of 2 real numbers), optional,
        frequency band of the bandpass filter,
        a typical pair is [0.5, 45],
        be careful when detecting paced rhythm,
        if is None or empty, bandpass filtering will not be performed
    rpeak_fn: str, optional,
        name of the function detecting rpeaks,
        can be one of keys of `QRS_DETECTORS`, case insensitive
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    retval: dict,
        with items
        - "filtered_ecg": the array of the processed ecg signal
        - "rpeaks": the array of indices of rpeaks; empty if `rpeak_fn` is not given

    """
    filtered_ecg = raw_sig.copy()

    # remove baseline
    if bl_win:
        window1 = 2 * (int(bl_win[0] * fs) // 2) + 1  # window size must be odd
        window2 = 2 * (int(bl_win[1] * fs) // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode="nearest")
        baseline = median_filter(baseline, size=window2, mode="nearest")
        filtered_ecg = filtered_ecg - baseline

    # filter signal
    if band_fs:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype="FIR",
            # ftype="butter",
            band="bandpass",
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=band_fs,
        )["signal"]

    if rpeak_fn and rpeak_fn not in DL_QRS_DETECTORS:
        rpeaks = QRS_DETECTORS[rpeak_fn.lower()](filtered_ecg, fs).astype(int)
    else:
        rpeaks = np.array([], dtype=int)

    retval = CFG(
        {
            "filtered_ecg": filtered_ecg,
            "rpeaks": rpeaks,
        }
    )

    if verbose >= 3:
        from cfg import PlotCfg
        from utils.misc import plot_single_lead

        t = np.arange(len(filtered_ecg)) / fs
        waves = {
            "qrs": get_mask(
                shape=len(filtered_ecg),
                critical_points=rpeaks,
                left_bias=ms2samples(PlotCfg.qrs_radius, fs),
                right_bias=ms2samples(PlotCfg.qrs_radius, fs),
                return_fmt="intervals",
            )
        }
        plot_single_lead(t=t, sig=filtered_ecg, ticks_granularity=2, waves=waves)

    return retval


def rpeaks_detect_multi_leads(
    sig: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    rpeak_fn: str = "xqrs",
    verbose: int = 0,
) -> np.ndarray:
    """finished, NOT checked,

    detect rpeaks from the filtered multi-lead ecg signal (with units in mV)

    Parameters
    ----------
    sig: ndarray,
        the (better be filtered) ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    rpeak_fn: str,
        name of the function detecting rpeaks,
        can be one of keys of `QRS_DETECTORS`, case insensitive
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    rpeaks: np.ndarray,
        array of indices of the detected rpeaks of the multi-lead ecg signal

    """
    assert sig_fmt.lower() in [
        "channel_first",
        "lead_first",
        "channel_last",
        "lead_last",
    ]
    if sig_fmt.lower() in ["channel_last", "lead_last"]:
        s = sig.T
    else:
        s = sig.copy()
    rpeaks = []
    for lead in range(s.shape[0]):
        rpeaks.append((QRS_DETECTORS[rpeak_fn.lower()](s[lead], fs)).astype(int))
    rpeaks = merge_rpeaks(rpeaks, sig, fs, verbose)
    return rpeaks


def merge_rpeaks(
    rpeaks_candidates: List[np.ndarray], sig: np.ndarray, fs: Real, verbose: int = 0
) -> np.ndarray:
    """

    merge rpeaks that are detected from each lead of multi-lead signals (with units in mV),
    using certain criterion merging qrs masks from each lead

    Parameters
    ----------
    rpeaks_candidates: list of ndarray,
        each element (ndarray) is the array of indices of rpeaks of corr. lead
    sig: ndarray,
        the multi-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    final_rpeaks: np.ndarray
        the final rpeaks obtained by merging the rpeaks from all the leads

    """
    rpeak_masks = np.zeros_like(sig, dtype=int)
    sig_len = sig.shape[1]
    radius = ms2samples(PreprocCfg.rpeak_mask_radius, fs)
    if verbose >= 1:
        print(f"sig_len = {sig_len}, radius = {radius}")
    for lead in range(sig.shape[0]):
        for r in rpeaks_candidates[lead]:
            rpeak_masks[lead, max(0, r - radius) : min(sig_len - 1, r + radius)] = 1
    rpeak_masks = (
        rpeak_masks.sum(axis=0) >= int(PreprocCfg.rpeak_lead_num_thr * sig.shape[0])
    ).astype(int)
    rpeak_masks[0], rpeak_masks[-1] = 0, 0
    split_indices = np.where(np.diff(rpeak_masks) != 0)[0]
    if verbose >= 1:
        print(
            f"split_indices = {split_indices}, with total number = {len(split_indices)}"
        )
        if verbose >= 2:
            print(
                f"the corresponding intervals are {[[split_indices[2*idx], split_indices[2*idx+1]] for idx in range(len(split_indices)//2)]}"
            )

    final_rpeaks = []
    for idx in range(len(split_indices) // 2):
        start_idx = split_indices[2 * idx]
        end_idx = split_indices[2 * idx + 1]
        rc = list_sum(  # `lr`: list of rpeaks
            [
                lr[np.where((lr >= start_idx) & (lr <= end_idx))].tolist()
                for lr in rpeaks_candidates
            ]
        )
        if verbose >= 2:
            print(
                f"at the {idx}-th interval, start_idx = {start_idx}, end_idx = {end_idx}"
            )
            print(f"rpeak candidates = {rc}")
        counter = Counter(rc).most_common()
        if len(counter) > 0 and counter[0][1] >= len(rc) // 2 + 1:
            # might have the case where
            # rc is empty
            final_rpeaks.append(counter[0][0])
        elif len(counter) > 0:
            final_rpeaks.append(int(np.mean(rc)))
    final_rpeaks = np.array(final_rpeaks)
    return final_rpeaks
