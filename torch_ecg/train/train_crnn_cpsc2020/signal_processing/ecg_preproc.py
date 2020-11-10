"""
preprocess of (single lead) ecg signal:
    band pass --> remove baseline --> find rpeaks --> denoise (mainly deal with motion artefact)

TODO:
    1. motion artefact detection,
       and slice the signal into continuous (no motion artefact within) segments
    2. to add

References:
-----------
[1] https://github.com/PIA-Group/BioSPPy
[2] to add
"""
import os, time
import multiprocessing as mp
from copy import deepcopy
from numbers import Real
from typing import Union, Optional, Any, List, Dict

import numpy as np
from easydict import EasyDict as ED
from scipy.ndimage.filters import median_filter
from scipy.signal.signaltools import resample
from scipy.io import savemat
# from scipy.signal import medfilt
# https://github.com/scipy/scipy/issues/9680
from biosppy.signals.tools import filter_signal

from torch_ecg.train.train_crnn_cpsc2020.cfg import PreprocCfg
from .ecg_rpeaks import (
    xqrs_detect, gqrs_detect, pantompkins,
    hamilton_detect, ssf_detect, christov_detect, engzee_detect, gamboa_detect,
)
from .ecg_rpeaks_dl import seq_lab_net_detect


__all__ = [
    "preprocess_signal",
    "parallel_preprocess_signal",
]


QRS_DETECTORS = {
    "xqrs": xqrs_detect,
    "gqrs": gqrs_detect,
    "pantompkins": pantompkins,
    "hamilton": hamilton_detect,
    "ssf": ssf_detect,
    "christov": christov_detect,
    "engzee": engzee_detect,
    "gamboa": gamboa_detect,
    "seq_lab": seq_lab_net_detect,
}
DL_QRS_DETECTORS = [
    "seq_lab",
]


def preprocess_signal(raw_sig:np.ndarray, fs:Real, config:Optional[ED]=None) -> Dict[str, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_sig`
    config: dict, optional,
        extra process configuration,
        `PreprocCfg` will be updated by this `config`

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if 'rpeaks' in `config` is not set
    """
    filtered_ecg = raw_sig.copy()

    cfg = deepcopy(PreprocCfg)
    cfg.update(config or {})

    if fs != cfg.fs:
        filtered_ecg = resample(filtered_ecg, int(round(len(filtered_ecg)*cfg.fs/fs)))

    # remove baseline
    if 'baseline' in cfg.preproc:
        window1 = 2 * (cfg.baseline_window1 // 2) + 1  # window size must be odd
        window2 = 2 * (cfg.baseline_window2 // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode='nearest')
        baseline = median_filter(baseline, size=window2, mode='nearest')
        filtered_ecg = filtered_ecg - baseline
    
    # filter signal
    if 'bandpass' in cfg.preproc:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=cfg.filter_band,
        )['signal']

    if cfg.rpeaks and cfg.rpeaks.lower() not in DL_QRS_DETECTORS:
        # dl detectors not for parallel computing using `mp`
        detector = QRS_DETECTORS[cfg.rpeaks.lower()]
        rpeaks = detector(sig=filtered_ecg, fs=fs).astype(int)
    else:
        rpeaks = np.array([], dtype=int)

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })
    
    return retval
    

def parallel_preprocess_signal(raw_sig:np.ndarray, fs:Real, config:Optional[ED]=None, save_dir:Optional[str]=None, save_fmt:str='npy', verbose:int=0) -> Dict[str, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_sig`
    config: dict, optional,
        extra process configuration,
        `PreprocCfg` will `update` this `config`
    save_dir: str, optional,
        directory for saving the outcome ('filtered_ecg' and 'rpeaks')
    save_fmt: str, default 'npy',
        format of the save files, 'npy' or 'mat'

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if 'rpeaks' in `config` is not set
    """
    start_time = time.time()
    cfg = deepcopy(PreprocCfg)
    cfg.update(config or {})

    epoch_len = int(cfg.parallel_epoch_len * fs)
    epoch_overlap_half = int(cfg.parallel_epoch_overlap * fs) // 2
    epoch_overlap = 2 * epoch_overlap_half
    epoch_forward = epoch_len - epoch_overlap

    if len(raw_sig) <= 3 * epoch_len:  # too short, no need for parallel computing
        retval = preprocess_signal(raw_sig, fs, cfg)
        if cfg.rpeaks and cfg.rpeaks.lower() in DL_QRS_DETECTORS:
            rpeaks = QRS_DETECTORS[cfg.rpeaks.lower()](sig=raw_sig, fs=fs, verbose=verbose).astype(int)
            retval.rpeaks = rpeaks
        return retval
    
    l_epoch = [
        raw_sig[idx*epoch_forward: idx*epoch_forward + epoch_len] \
            for idx in range((len(raw_sig)-epoch_overlap)//epoch_forward)
    ]

    if cfg.parallel_keep_tail:
        tail_start_idx = epoch_forward * len(l_epoch) + epoch_overlap
        if len(raw_sig) - tail_start_idx < 30 * fs:  # less than 30s, make configurable?
            # append to the last epoch
            l_epoch[-1] = np.append(l_epoch[-1], raw_sig[tail_start_idx:])
        else:  # long enough
            tail_epoch = raw_sig[tail_start_idx-epoch_overlap:]
            l_epoch.append(tail_epoch)

    cpu_num = max(1, mp.cpu_count()-3)
    with mp.Pool(processes=cpu_num) as pool:
        result = pool.starmap(
            func=preprocess_signal,
            iterable=[(e, fs, cfg) for e in l_epoch],
        )

    if cfg.parallel_keep_tail:
        tail_result = result[-1]
        result = result[:-1]
    
    filtered_ecg = result[0]['filtered_ecg'][:epoch_len-epoch_overlap_half]
    rpeaks = result[0]['rpeaks'][np.where(result[0]['rpeaks']<epoch_len-epoch_overlap_half)[0]]
    for idx, e in enumerate(result[1:]):
        filtered_ecg = np.append(
            filtered_ecg, e['filtered_ecg'][epoch_overlap_half: -epoch_overlap_half]
        )
        epoch_rpeaks = e['rpeaks'][np.where( (e['rpeaks'] >= epoch_overlap_half) & (e['rpeaks'] < epoch_len-epoch_overlap_half) )[0]]
        rpeaks = np.append(rpeaks, (idx+1)*epoch_forward + epoch_rpeaks)

    if cfg.parallel_keep_tail:
        filtered_ecg = np.append(filtered_ecg, tail_result['filtered_ecg'][epoch_overlap_half:])
        tail_rpeaks = tail_result['rpeaks'][np.where(tail_result['rpeaks'] >= epoch_overlap_half)[0]]
        rpeaks = np.append(rpeaks, len(result)*epoch_forward + tail_rpeaks)

    if verbose >= 1:
        if cfg.rpeaks.lower() in DL_QRS_DETECTORS:
            print(f"signal processing took {round(time.time()-start_time, 3)} seconds")
        else:
            print(f"signal processing and R peaks detection took {round(time.time()-start_time, 3)} seconds")
        start_time = time.time()

    if cfg.rpeaks and cfg.rpeaks.lower() in DL_QRS_DETECTORS:
        rpeaks = QRS_DETECTORS[cfg.rpeaks.lower()](sig=raw_sig, fs=fs, verbose=verbose).astype(int)
        if verbose >= 1:
            print(f"R peaks detection using {cfg.rpeaks} took {round(time.time()-start_time, 3)} seconds")

    if save_dir:
        # NOTE: this part is not tested
        os.makedirs(save_dir, exist_ok=True)
        if save_fmt.lower() == 'npy':
            np.save(os.path.join(save_dir, "filtered_ecg.npy"), filtered_ecg)
            np.save(os.path.join(save_dir, "rpeaks.npy"), rpeaks)
        elif save_fmt.lower() == 'mat':
            # save into 2 files, keep in accordance
            savemat(os.path.join(save_dir, "filtered_ecg.mat"), {"filtered_ecg": filtered_ecg}, format='5')
            savemat(os.path.join(save_dir, "rpeaks.mat"), {"rpeaks": rpeaks}, format='5')

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })

    return retval

"""
to check correctness of the function `parallel_preprocess_signal`,
say for record A01, one can call
>>> raw_sig = loadmat("./data/A01.mat")['ecg'].flatten()
>>> processed = parallel_preprocess_signal(raw_sig, 400)
>>> print(len(processed['filtered_ecg']) - len(raw_sig))
>>> start_t = int(3600*24.7811)
>>> len_t = 10
>>> fig, ax = plt.subplots(figsize=(20,6))
>>> ax.plot(hehe['filtered_ecg'][start_t*400:(start_t+len_t)*400])
>>> for r in [p for p in hehe['rpeaks'] if start_t*400 <= p < (start_t+len_t)*400]:
>>>    ax.axvline(r-start_t*400,c='red',linestyle='dashed')
>>> plt.show()

or one can use the 'dataset.py'
"""
