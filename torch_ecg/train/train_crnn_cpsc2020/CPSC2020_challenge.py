"""
"""
import math
import time
from numbers import Real
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly

from .signal_processing.ecg_preproc import parallel_preprocess_signal
from .signal_processing.ecg_denoise import ecg_denoise
from .saved_models import load_model
from .cfg import ModelCfg


CRNN_MODEL, SEQ_LAB_MODEL = load_model(which="both")
CRNN_CFG, SEQ_LAB_CFG = ModelCfg.crnn, ModelCfg.seq_lab

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = np.float64
else:
    _DTYPE = np.float32


def CPSC2020_challenge(ECG, fs):
    """
    % This function can be used for events 1 and 2. Participants are free to modify any
    % components of the code. However the function prototype must stay the same
    % [S_pos,V_pos] = CPSC2020_challenge(ECG,fs) where the inputs and outputs are specified
    % below.
    %
    %% Inputs
    %       ECG : raw ecg vector signal 1-D signal
    %       fs  : sampling rate
    %
    %% Outputs
    %       S_pos : the position where SPBs detected
    %       V_pos : the position where PVCs detected
    %
    %
    %
    % Copyright (C) 2020 Dr. Chengyu Liu
    % Southeast university
    % chengyu@seu.edu.cn
    %
    % Last updated : 02-23-2020

    """

    #   ====== arrhythmias detection =======
    # finished, checked,

    print("\n" + "*"*80)
    msg = "   CPSC2020_challenge starts ...  "
    print("*"*((80-len(msg))//2) + msg + "*"*((80-len(msg))//2))
    print("*"*80)
    start_time = time.time()
    timer = time.time()

    FS = 400

    if int(fs) != FS:
        sig = resample_poly(np.array(ECG).flatten(), up=FS, down=int(fs))
    else:
        sig = np.array(ECG).flatten()
    pps = parallel_preprocess_signal(sig, fs)  # use default config in `cfg`
    filtered_ecg = pps['filtered_ecg']
    rpeaks = pps['rpeaks']
    valid_intervals = ecg_denoise(filtered_ecg, fs=FS, config={"ampl_min":0.15})
    rpeaks = [r for r in rpeaks if any([itv[0]<=r<=itv[1] for itv in valid_intervals])]
    rpeaks = np.array(rpeaks, dtype=int)

    print(f"signal preprocessing used {time.time()-timer:.3f} seconds")
    timer = time.time()

    # classify and sequence labeling models

    seq_lab_granularity = 8
    model_input_len = 10 * FS  # 10s
    half_overlap_len = 512  # should be divisible by `model_granularity`
    overlap_len = 2 * half_overlap_len
    forward_len = model_input_len - overlap_len

    n_segs, residue = divmod(len(filtered_ecg)-overlap_len, forward_len)
    if residue != 0:
        filtered_ecg = np.append(filtered_ecg, np.zeros((forward_len-residue,)))
        n_segs += 1
    batch_size = 64
    n_batches = math.ceil(n_segs / batch_size)

    print(f"number of batches = {n_batches}")

    S_pos_rsmp, V_pos_rsmp = np.array([], dtype=int), np.array([], dtype=int)

    MEAN, STD = 0.01, 0.25  # rescale to this mean and standard deviation
    segs = list(range(n_segs))
    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_segs = segs[b_start: b_start + batch_size]
        b_input = []
        b_rpeaks = []
        for idx in b_segs:
            start = idx*forward_len
            end = idx*forward_len+model_input_len
            seg = filtered_ecg[start: end]
            if np.std(seg) > 0:
                seg = (seg - np.mean(seg) + MEAN) / np.std(seg) * STD
            b_input.append(seg)
            seg_rpeaks = rpeaks[np.where((rpeaks>=start) & (rpeaks<end))[0]] - start
            b_rpeaks.append(seg_rpeaks)
        b_input = np.vstack(b_input).reshape((-1, 1, model_input_len))
        b_input = b_input.astype(_DTYPE)
        
        _, crnn_out = \
            CRNN_MODEL.inference(b_input, bin_pred_thr=0.5)  # (batch_size, 3)
        _, SPB_indices, PVC_indices = \
            SEQ_LAB_MODEL.inference(b_input, bin_pred_thr=0.5, rpeak_inds=b_rpeaks)

        for i, idx in enumerate(b_segs):
            if crnn_out[i, CRNN_CFG.classes.index("N")] == 1:
                # the classifier predicts non-premature segment
                continue
            if crnn_out[i, CRNN_CFG.classes.index("S")] == 1:
                seg_spb = np.array(SPB_indices[i])
                seg_spb = seg_spb[np.where((seg_spb>=half_overlap_len) & (seg_spb<model_input_len-half_overlap_len))[0]] + idx * forward_len
                S_pos_rsmp = np.append(S_pos_rsmp, seg_spb)
            if crnn_out[i, CRNN_CFG.classes.index("V")] == 1:
                seg_pvc = np.array(PVC_indices[i])
                seg_pvc = seg_pvc[np.where((seg_pvc>=half_overlap_len) & (seg_pvc<model_input_len-half_overlap_len))[0]] + idx * forward_len
                V_pos_rsmp = np.append(V_pos_rsmp, seg_pvc)

        print(f"{b_idx+1}/{n_batches} batches", end="\r")

    print(f"\nprediction used {time.time()-timer:.3f} seconds")
    print(f"\ntotal time cost is {time.time()-start_time:.3f} seconds")

    S_pos_rsmp = S_pos_rsmp[np.where(S_pos_rsmp<len(filtered_ecg))[0]]
    V_pos_rsmp = V_pos_rsmp[np.where(V_pos_rsmp<len(filtered_ecg))[0]]

    if int(fs) != FS:
        S_pos = np.round(S_pos_rsmp * fs / FS).astype(int)
        V_pos = np.round(V_pos_rsmp * fs / FS).astype(int)
    else:
        S_pos, V_pos = S_pos_rsmp.astype(int), V_pos_rsmp.astype(int)

    print("*"*80)
    msg = "   CPSC2020_challenge ends ...  "
    print("*"*((80-len(msg))//2) + msg + "*"*((80-len(msg))//2))
    print("*"*80 + "\n")

    return S_pos, V_pos


if __name__ == "__main__":
    from ..database_reader.database_reader.other_databases import CPSC2020 as CR
    from .cfg import TrainCfg

    dr = CR(TrainCfg.db_dir)
    for rec in dr.all_records:
        print(f"rec = {rec}")
        input_ecg = dr.load_data(rec, keep_dim=False)
        print(f"input_ecg.shape = {input_ecg.shape}")
        S_pos, V_pos = CPSC2020_challenge(input_ecg, 400)
        print(f"S_pos = {S_pos}")
        print(f"V_pos = {V_pos}")
        print("\n" + "*"*80 + "\n")
