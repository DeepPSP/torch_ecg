#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import detrend

mne.set_log_level(verbose="WARNING")
from mne.filter import filter_data, notch_filter

sys.path.append(str(Path(__file__).resolve().parent))

from peakdetect import peakdetect

# from read_delirium_data import datenum


seg_mask_explanation = [
    "normal",
    "NaN in EEG",  # _[1,3] (append channel ids)
    "overly high/low amplitude",
    "flat signal",
    "NaN in feature",
    "NaN in spectrum",
    "overly high/low total power",
    "muscle artifact",
    "multiple assessment scores",
    "spurious spectrum",
    "fast rising decreasing",
    "1Hz artifact",
]


def peak_detect(signal, max_change_points, min_change_amp, lookahead=200, delta=0):
    # signal: #channel x #points
    res = []
    for cid in range(signal.shape[0]):
        local_max, local_min = peakdetect(signal[cid], lookahead=lookahead, delta=delta)
        if len(local_min) <= 0 and len(local_max) <= 0:
            res.append(False)
        else:
            if len(local_min) <= 0:
                local_extremes = np.array(local_max)
            elif len(local_max) <= 0:
                local_extremes = np.array(local_min)
            else:
                local_extremes = np.r_[local_max, local_min]
            local_extremes = local_extremes[np.argsort(local_extremes[:, 0])]
            res.append(
                np.logical_and(
                    np.diff(local_extremes[:, 0]) <= max_change_points, np.abs(np.diff(local_extremes[:, 1])) >= min_change_amp
                ).sum()
            )
    return res


def smooth(x, window_len=12, window="hanning"):
    #'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    s = np.r_[x[window_len // 2 - 1 : 0 : -1], x, x[-2 : -window_len // 2 - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    return np.convolve(w / w.sum(), s, mode="valid")


def autocorrelate_noncentral_max_abs(x):
    ress = []
    EPS = 1e-15  # avoid divide by 0
    for ii in range(x.shape[1]):
        # --- modification by wenh06 starts ---
        res = np.correlate(x[:, ii], x[:, ii], mode="full") / (np.correlate(x[:, ii], x[:, ii], mode="valid")[0] + EPS)
        # --- modification by wenh06 ends ---
        ress.append(np.max(res[len(res) // 2 + 7 : len(res) // 2 + 20]))  # ECG range: 40/1min(0.7Hz) -- 120/1min(2Hz)
    return ress


def segment_EEG(
    EEG,
    Ch_names,
    window_time,
    step_time,
    Fs,
    notch_freq=None,
    bandpass_freq=None,
    start_end_remove_window_num=0,
    amplitude_thres=500,
    n_jobs=1,
    to_remove_mean=False,
):  #
    """Segment EEG signals.

    Arguments:
    EEG -- np.ndarray, size=(channel_num, sample_num)
    labels -- np.ndarray, size=(sample_num,2), col1: time, col2: score
    times -- array of datetime.datetime, size=(sample_num,)
    window_time -- in seconds
    Fz -- in Hz

    Keyword arguments:
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the EEG signal
    amplitude_thres -- default 1000, mark all segments with np.any(EEG_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of EEG signal from each channel

    Outputs:
    EEG segments -- a list of np.ndarray, each has size=(window_size, channel_num)
    labels --  a list of labels of each window
    segment start ids -- a list of starting ids of each window in (sample_num,)
    segment masks --
    """
    std_thres1 = 0.2
    std_thres2 = 0.5
    flat_seconds = 2

    ## KEEP AN EYE ON IT
    if to_remove_mean:
        EEG = EEG - np.mean(EEG, axis=1, keepdims=True)
    window_size = int(round(window_time * Fs))
    step_size = int(round(step_time * Fs))
    flat_length = int(round(flat_seconds * Fs))
    ## start_ids

    start_ids = np.arange(0, EEG.shape[1] - window_size + 1, step_size)
    if start_end_remove_window_num > 0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    if len(start_ids) <= 0:
        raise ValueError("No EEG segments")

    seg_masks = [seg_mask_explanation[0]] * len(start_ids)
    """
    for i in range(len(start_ids)):
        ll = labels[start_ids[i]:start_ids[i]+window_size]
        nll = np.isnan(ll)
        if not np.all(nll) and (np.any(nll) or len(set(ll))!=1):
            seg_masks[i] = seg_mask_explanation[8]
    """

    ## filter signal
    # IF high end of bandpass (aka max) is more than notch filter
    if np.max(bandpass_freq) >= notch_freq:
        # JUST DO NOTCH FILTERING
        EEG = notch_filter(EEG, Fs, notch_freq, n_jobs=n_jobs, verbose="ERROR")  # (#window, #ch, window_size+2padding)
    EEG = filter_data(
        EEG, Fs, bandpass_freq[0], bandpass_freq[1], n_jobs=n_jobs, verbose="ERROR"
    )  # take the value starting from *padding*, (#window, #ch, window_size+2padding)

    ## detect burst suppression

    # import pdb;pdb.set_trace()
    EEG_tmp = np.zeros_like(EEG)
    for i in range(EEG.shape[0]):
        eeg_smooth = smooth(EEG[i, :], window_len=10, window="flat")
        EEG_tmp[i, : eeg_smooth.shape[0]] = eeg_smooth
    EEG_mne = mne.io.RawArray(
        np.array(EEG_tmp, copy=True), mne.create_info(Ch_names, Fs, ch_types="eeg", verbose="ERROR"), verbose="ERROR"
    )
    EEG_mne.apply_hilbert(envelope=True, n_jobs=-1, verbose="ERROR")
    BS = EEG_mne.get_data()

    bs_window_size = int(round(120 * Fs))  # BSR estimation window 2min
    bs_start_ids = np.arange(0, EEG.shape[1] - bs_window_size + 1, bs_window_size)
    if len(bs_start_ids) <= 0:
        bs_start_ids = np.array([0], dtype=int)
    if EEG.shape[1] > bs_start_ids[-1] + bs_window_size:  # if incomplete divide
        bs_start_ids = np.r_[bs_start_ids, EEG.shape[1] - bs_window_size]
    BS_segs = BS[:, list(map(lambda x: np.arange(x, min(BS.shape[1], x + bs_window_size)), bs_start_ids))]
    BSR_segs = np.sum(BS_segs <= 5, axis=2).T * 1.0 / bs_window_size
    BSR = np.zeros_like(EEG)
    for ii, bsi in enumerate(bs_start_ids):
        BSR[:, bsi : min(BSR.shape[1], bsi + bs_window_size)] = BSR_segs[ii].reshape(-1, 1)

    ## segment signal
    BSR_segs = BSR[:, list(map(lambda x: np.arange(x, x + window_size), start_ids))].transpose(1, 0, 2).mean(axis=2)
    EEG_segs = EEG[:, list(map(lambda x: np.arange(x, x + window_size), start_ids))].transpose(
        1, 0, 2
    )  # (#window, #ch, window_size+2padding)

    ## find nan in signal

    nan2d = np.any(np.isnan(EEG_segs), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = "%s_%s" % (seg_mask_explanation[1], np.where(nan2d[i])[0])

    ## calculate spectrogram

    # TODO detrend(EEG_segs)
    # TODO remove_mean(EEG_segs) to remove frequency at 0Hz

    # mne_epochs = mne.EpochsArray(EEG_segs, mne.create_info(ch_names=list(map(str, range(EEG_segs.shape[1]))), sfreq=Fs, ch_types='eeg'), verbose=False)
    BW = 2.0
    specs, freq = mne.time_frequency.psd_array_multitaper(
        EEG_segs,
        Fs,
        fmin=bandpass_freq[0],
        fmax=bandpass_freq[1],
        adaptive=False,
        low_bias=False,
        n_jobs=n_jobs,
        verbose="ERROR",
        bandwidth=BW,
        normalization="full",
    )
    df = freq[1] - freq[0]
    # --- modification by wenh06 starts ---
    EPS = 1e-15  # avoid log(0)
    specs = 10 * np.log10(specs.transpose(0, 2, 1) + EPS)
    # --- modification by wenh06 ends ---

    ## find nan in spectrum
    specs[np.isinf(specs)] = np.nan
    nan2d = np.any(np.isnan(specs), axis=1)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    nonan_spec_id = np.where(np.all(np.logical_not(np.isnan(specs)), axis=(1, 2)))[0]
    for i in nan1d:
        seg_masks[i] = "%s_%s" % (seg_mask_explanation[5], np.where(nan2d[i])[0])

    ## find staircase-like spectrum
    # | \      +-+
    # |  \     | |
    # |   -----+ +--\
    # +--------------=====
    spec_smooth_window = int(round(1.0 / df))  # 1 Hz
    specs2 = specs[nonan_spec_id][:, np.logical_and(freq >= 5, freq <= 20)]
    freq2 = freq[np.logical_and(freq >= 5, freq <= 20)][spec_smooth_window:-spec_smooth_window]
    ww = np.hanning(spec_smooth_window * 2 + 1)
    ww = ww / ww.sum()
    # print(f"specs.shape : {specs.shape}, specs2.shape : {specs2.shape}")
    smooth_specs = np.apply_along_axis(lambda m: np.convolve(m, ww, mode="valid"), axis=1, arr=specs2)
    dspecs = specs2[:, spec_smooth_window:-spec_smooth_window] - smooth_specs
    # dspecs_std = np.std(dspecs, axis=1, keepdims=True)
    # dspecs_std[dspecs_std<1e-3] = 1.
    dspecs = dspecs - dspecs.mean(axis=1, keepdims=True)  # ()/dspecs_std
    aa = np.apply_along_axis(
        lambda m: np.convolve(m, np.array([-1.0, -1.0, 0, 1.0, 1.0, 1.0, 1.0]), mode="same"), axis=1, arr=dspecs
    )  # increasing staircase-like pattern
    bb = np.apply_along_axis(
        lambda m: np.convolve(m, np.array([1.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0]), mode="same"), axis=1, arr=dspecs
    )  # decreasing staircase-like pattern
    stsp2d = np.logical_or(np.maximum(aa, bb).max(axis=1) >= 10, np.any(np.abs(np.diff(specs2, axis=1)) >= 11, axis=1))
    # stsp2d = np.logical_and(np.maximum(aa,bb)>=14, np.abs(np.concatenate([dspecs[:,2:]-dspecs[:,:-2], np.zeros((dspecs.shape[0],2,dspecs.shape[2]))))>=1.8, axis=1))
    # stsp2d = np.any(np.logical_and(np.concatenate([np.zeros((dspecs.shape[0],1,dspecs.shape[2])),np.abs(np.diff(np.arctan(np.diff(dspecs,axis=1)/df),axis=1))/np.pi*180],axis=1)[:,:-1]>80, np.logical_or(np.abs(np.diff(dspecs,axis=1))[:,:-1]>2,np.abs(dspecs[:,2:]-dspecs[:,:-2])>3)),axis=1)
    # stsp2d = np.logical_or(stsp2d, np.any(np.abs(np.diff(specs2,axis=1))>=6, axis=1))
    stsp1d = nonan_spec_id[np.any(stsp2d, axis=1)]
    """
    stsp2d = Parallel(n_jobs=n_jobs,verbose=True)(delayed(peak_detect_num_amp)(dspecs[sid].T, lookahead=5, delta=1.5) for sid in range(dspecs.shape[0]))
    stsp2d = (np.array(map(lambda x:x[0],stsp2d))>=8) & (np.array(map(lambda x:x[1],stsp2d))>=6)
    stsp1d = nonan_spec_id[np.where(np.any(stsp2d, axis=1))[0]]
    """
    for i in stsp1d:
        seg_masks[i] = seg_mask_explanation[9]

    ## check ECG in spectrum (~1Hz and harmonics)
    # dspecs2 = dspecs[:,np.logical_and(freq2>=6,freq2<=10)]
    autocorrelation = Parallel(n_jobs=n_jobs, prefer="threads", verbose=True)(
        delayed(autocorrelate_noncentral_max_abs)(spec) for spec in dspecs
    )
    autocorrelation = np.array(autocorrelation)
    ecg2d = autocorrelation > 0.7
    ecg1d = nonan_spec_id[np.any(ecg2d, axis=1)]
    for i in ecg1d:
        seg_masks[i] = seg_mask_explanation[11]

    ## find overly fast rising/decreasing signal

    max_change_points = 0.1 * Fs
    min_change_amp = 1.8 * amplitude_thres
    fast_rising2d = Parallel(n_jobs=n_jobs, prefer="threads", verbose=True)(
        delayed(peak_detect)(EEG_segs[sid], max_change_points, min_change_amp, lookahead=50, delta=0)
        for sid in range(EEG_segs.shape[0])
    )
    fast_rising2d = np.array(fast_rising2d) > 0
    fast_rising1d = np.where(np.any(fast_rising2d, axis=1))[0]
    for i in fast_rising1d:
        seg_masks[i] = seg_mask_explanation[10]

    ## find large amplitude in signal

    amplitude_large2d = np.max(EEG_segs, axis=2) - np.min(EEG_segs, axis=2) > 2 * amplitude_thres
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
        seg_masks[i] = "%s_%s" % (seg_mask_explanation[2], np.where(amplitude_large2d[i])[0])

    ## find flat signal
    # careful about burst suppression
    EEG_segs_temp = EEG_segs[:, :, : (EEG_segs.shape[2] // flat_length) * flat_length]
    short_segs = EEG_segs_temp.reshape(
        EEG_segs_temp.shape[0], EEG_segs_temp.shape[1], EEG_segs_temp.shape[2] // flat_length, flat_length
    )
    # print(f"short_seg contain infs : {np.any(np.isinf(short_segs))} or NaNs : {np.any(np.isnan(short_segs))}")
    flat2d = np.any(detrend(short_segs, axis=3).std(axis=3) <= std_thres1, axis=2)
    flat2d = np.logical_or(flat2d, np.std(EEG_segs, axis=2) <= std_thres2)
    flat1d = np.where(np.any(flat2d, axis=1))[0]
    for i in flat1d:
        seg_masks[i] = "%s_%s" % (seg_mask_explanation[3], np.where(flat2d[i])[0])

    """
    ## local z-transform
    eeg = np.array(EEG_segs[:,:,:100], copy=True)
    eeg[np.isinf(eeg)] = np.nan
    mu = np.nanmean(eeg, axis=2)
    sigma2 = np.nanstd(eeg, axis=2)**2
    EEG_segs_ = np.array(EEG_segs, copy=True)
    for t in range(EEG_segs.shape[2]):
        eeg = np.array(EEG_segs[:,:,t], copy=True)
        eeg[np.logical_or(np.isnan(eeg),np.isinf(eeg))] = 0.

        sigma2_no0 = np.array(sigma2, copy=True)
        sigma2_no0[np.abs(sigma2_no0)<=1e-3] = 1.

        eeg = (eeg-mu)/np.sqrt(sigma2_no0)
        mu = 0.001*eeg+0.999*mu
        sigma2 = 0.001*(eeg-mu)**2+0.999*sigma2

        EEG_segs_[:,:,t] = eeg
    EEG_segs = detrend(EEG_segs_, axis=2)
    """
    BW = 1.0  # frequency resolution 1Hz
    specs, freq = mne.time_frequency.psd_array_multitaper(
        EEG_segs,
        Fs,
        fmin=bandpass_freq[0],
        fmax=bandpass_freq[1],
        adaptive=False,
        low_bias=False,
        n_jobs=n_jobs,
        verbose="ERROR",
        bandwidth=BW,
        normalization="full",
    )
    # --- modification by wenh06 starts ---
    EPS = 1e-15  # avoid log(0)
    specs = 10 * np.log10(specs.transpose(0, 2, 1) + EPS)
    # --- modification by wenh06 ends ---

    return EEG_segs, BSR_segs, start_ids, seg_masks, specs, freq
