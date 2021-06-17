"""

References:
-----------
[1] https://github.com/mondejar/ecg-classification
[2] to add

mainly RR intervals + morphological features

--> (lstm using RR intervals) + (cnn using raw (or processed?) signal)
"""
import os
from copy import deepcopy
from typing import Optional
import operator
import multiprocessing as mp
from functools import reduce

import pywt
import numpy as np
from scipy.io import savemat
from easydict import EasyDict as ED

from train.train_crnn_cpsc2020.cfg import FeatureCfg
from train.train_crnn_cpsc2020.utils import list_sum, compute_local_average


__all__ = [
    "compute_ecg_features",
    "compute_wavelet_descriptor",
    "compute_rr_descriptor",
    "compute_morph_descriptor",
]


def compute_ecg_features(sig:np.ndarray, rpeaks:np.ndarray, config:Optional[ED]=None, save_dir:Optional[str]=None, save_fmt:str="npy") -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    sig: ndarray,
        the filtered ecg signal
    rpeaks: ndarray,
        indices of R peaks
    config: dict, optional,
        extra process configurations,
        `FeatureCfg` will `update` this `config`

    Returns
    -------
    features: ndarray,
        the computed features, of shape (m,n), where
            m = the number of beats (the number of `rpeaks`)
            n = the dimension of the features
    """
    cfg = deepcopy(FeatureCfg)
    cfg.update(deepcopy(config) or {})

    filtered_rpeaks = rpeaks[np.where( (rpeaks>=cfg.beat_winL) & (rpeaks<len(sig)-cfg.beat_winR) )[0]]

    beats = []
    for r in filtered_rpeaks:
        beats.append(sig[r-cfg.beat_winL:r+cfg.beat_winR])
    features = np.empty((len(beats), 0))

    # NOTE: ordering keep in accordance with FeatureCfg.features
    if 'wavelet' in cfg.features:
        tmp = []
        for beat in beats:
            tmp.append(np.array(compute_wavelet_descriptor(beat, cfg)))
        features = np.concatenate((features, np.array(tmp)), axis=1)
    if 'rr' in cfg.features:
        tmp = compute_rr_descriptor(filtered_rpeaks, cfg)
        features = np.concatenate((features, tmp), axis=1)
    if 'morph' in cfg.features:
        tmp = []
        for beat in beats:
            tmp.append(np.array(compute_morph_descriptor(beat, cfg)))
        features = np.concatenate((features, np.array(tmp)), axis=1)

    if save_dir:
        save_suffix = "-".join(cfg.features)
        save_path = os.path.join(save_dir, f"ecg-features-{save_suffix}.{save_fmt.lower()}")
        if save_fmt.lower() == "npy":
            np.save(save_path, features)
        elif save_fmt.lower() == "mat":
            savemat(save_path, {"features": features}, format='5')

    return features


def compute_wavelet_descriptor(beat:np.ndarray, config:ED) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    beat: ndarray,
        a window properly covers the qrs complex, perhaps even the q, t waves
    config: dict,
        process configurations,
    
    Returns
    -------
    coeffs: ndarray,
        the `level`-th level decomposition coefficients

    References
    ----------
    [1] https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html?highlight=wavedec#multilevel-decomposition-using-wavedec
    [2] https://en.wikipedia.org/wiki/Wavelet
    """
    wave_family = pywt.Wavelet(config.wt_family)
    coeffs = pywt.wavedec(beat, wave_family, level=config.wt_level)[0]
    return coeffs


def compute_rr_descriptor(rpeaks:np.ndarray, config:Optional[ED]=None) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rpeaks: ndarray,
        indices of r peaks
    config: dict, optional,
        extra process configurations,
        `FeatureCfg` will `update` this `config`

    Returns
    -------
    features_rr: ndarray,
        rr features, including (in the following ordering)
        pre_rr: rr intervals to the previous r peak
        post_rr: rr intervals to the next r peak
        local_rr: mean rr interval of the previous n (c.f. `FeatureCfg`) beats
        global_rr: mean rr interval of the previous n (c.f. `FeatureCfg`) minutes
    """
    cfg = deepcopy(FeatureCfg)
    cfg.update(deepcopy(config) or {})
    
    # NOTE that for np.diff:
    # The first difference is given by ``out[n] = a[n+1] - a[n]``
    rr_intervals = np.diff(rpeaks)

    pre_rr = _compute_pre_rr(rr_intervals)
    post_rr = _compute_post_rr(rr_intervals)
    local_rr = _compute_local_rr(pre_rr, cfg)
    global_rr = _compute_global_rr(rpeaks, pre_rr, cfg)

    if cfg.rr_normalize_radius is not None:
        # normalized RR features: use local average
        pre_rr = pre_rr / compute_local_average(pre_rr, cfg.rr_normalize_radius)
        post_rr = post_rr / compute_local_average(post_rr, cfg.rr_normalize_radius)
        local_rr = local_rr / compute_local_average(local_rr, cfg.rr_normalize_radius)
        global_rr = global_rr / compute_local_average(global_rr, cfg.rr_normalize_radius)
        features_rr = np.column_stack((pre_rr, post_rr, local_rr, global_rr))
    else:
        features_rr = np.column_stack((pre_rr, post_rr, local_rr, global_rr))
        features_rr = features_rr / cfg.fs  #  units to sec
            
    return features_rr

def _compute_pre_rr(rr_intervals:np.ndarray) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rr_intervals: ndarray,
        array of rr intervals (to the next r peak)

    Returns
    -------
    pre_rr: ndarray,
        array of rr intervals to the previous r peak
    """
    try:
        pre_rr = np.append(rr_intervals[0], rr_intervals)
    except:  # in case empty rr_intervals
        pre_rr = np.array([], dtype=int)
    return pre_rr

def _compute_post_rr(rr_intervals:np.ndarray) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rr_intervals: ndarray,
        array of rr intervals (to the next r peak)

    Returns
    -------
    post_rr: ndarray,
        array of rr intervals to the next r peak,
        with the last element of `rr_intervals` duplicated
    """
    try:
        post_rr = np.append(rr_intervals, rr_intervals[-1])
    except:  # in case empty rr_intervals
        post_rr = np.array([], dtype=int)
    return post_rr

def _compute_local_rr(prev_rr:np.ndarray, config:ED) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rr_intervals: ndarray,
        array of rr intervals (to the next r peak)
    config: dict,
        configurations (local range) for computing local rr intervals

    Returns
    -------
    local_rr: ndarray,
        array of the local mean rr intervals
    """
    local_rr = np.array([], dtype=int)
    for i in range(config.rr_local_range-1):  # head
        local_rr = np.append(local_rr, np.mean(prev_rr[:i+1]))
    local_rr = np.append(
        local_rr,
        np.mean(np.array([prev_rr[i:len(prev_rr)-(config.rr_local_range-i-1)] for i in range(config.rr_local_range)]), axis=0)
    )
    return local_rr

def _compute_global_rr_epoch(rpeaks:np.ndarray, prev_rr:np.ndarray, epoch_start:int, epoch_end:int, global_range:int) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rpeaks: ndarray,
        indices of r peaks
    prev_rr: ndarray,
        array of rr intervals to the previous r peak
    epoch_start: int,
        index in `rpeaks` of the epoch start
    epoch_end: int,
        index in `rpeaks` of the epoch end
    global_range: int,
        range in number of samples for computing the 'global' mean rr intervals

    Returns
    -------
    global_rr: ndarray,
        array of the global mean rr intervals
    """
    global_rr = []
    for idx in range(epoch_start,epoch_end):
        nb_samples = len(np.where(rpeaks[idx]-rpeaks[:idx]<global_range)[0])
        global_rr.append(np.mean(prev_rr[idx-nb_samples:idx+1]))
    return global_rr

def _compute_global_rr(rpeaks:np.ndarray, prev_rr:np.ndarray, config:ED) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    rpeaks: ndarray,
        indices of r peaks
    prev_rr: ndarray,
        array of rr intervals to the previous r peak
    config: dict,
        configurations (global range) for computing global rr intervals

    Returns
    -------
    global_rr: ndarray,
        array of the global mean rr intervals
    """
    split_indices = [0]
    one_hour = config.fs * 3600
    for i in range(1, int(rpeaks[-1]) // one_hour):
        split_indices.append(len(np.where(rpeaks < i*one_hour)[0])+1)
    if len(split_indices) == 1 or split_indices[-1] < len(rpeaks): # tail
        split_indices.append(len(rpeaks))
    
    cpu_num = max(1, mp.cpu_count()-3)
    with mp.Pool(processes=cpu_num) as pool:
        result = pool.starmap(
            func=_compute_global_rr_epoch,
            iterable=[
                (
                    rpeaks,
                    prev_rr,
                    split_indices[idx],
                    split_indices[idx+1],
                    config.rr_global_range
                )\
                    for idx in range(len(split_indices)-1)
            ],
        )
    # list_addition = lambda a,b: a+b
    # global_rr = np.array(reduce(list_addition, result))
    global_rr = np.array(list_sum(result))
    return global_rr


def compute_morph_descriptor(beat:np.ndarray, config:ED) -> np.ndarray:
    """ finished, checked,

    Parameters
    ----------
    beat: ndarray,
        a window properly covers the qrs complex, perhaps even the q, t waves
    config: dict,
        process configurations,

    Returns
    -------
    morph: ndarray
    """
    R_pos = int((config.beat_winL + config.beat_winR) / 2)

    itv = config.morph_intervals
    itv_num = len(itv)

    R_value = beat[R_pos]
    morph = np.zeros((itv_num,))
    y_values = np.zeros(itv_num)
    x_values = np.zeros(itv_num)
    # Obtain (max/min) values and index from the intervals
    for n in range(itv_num):
        [x_values[n], y_values[n]] = \
            max(enumerate(beat[itv[n][0]:itv[n][1]]), key=operator.itemgetter(1))
    
    for n in range(1, itv_num):
        x_values[n] = x_values[n] + itv[n][0]
    
    # Norm data before compute distance
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))
    
    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)
                
    for n in range(itv_num):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n]) 
        y_diff = R_value - y_values[n]
        morph[n] =  np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))
    
    if np.isnan(morph[n]):
        morph[n] = 0.0

    return morph
