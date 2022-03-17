#!/usr/bin/env python3

import os, sys, time, json
from pathlib import Path
from itertools import repeat

import wfdb
import numpy as np
import torch
import scipy.signal as SS

try:
    import torch_ecg
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from torch_ecg.cfg import CFG
from torch_ecg.utils.preproc import (
    preprocess_multi_lead_signal,
    remove_spikes_naive,
)
from torch_ecg.utils.utils_signal import normalize
from torch_ecg.utils.utils_interval import (
    generalized_intervals_intersection,
    generalized_intervals_union,
)

from model import (
    ECG_SEQ_LAB_NET_CPSC2021,
    ECG_UNET_CPSC2021,
    RR_LSTM_CPSC2021,
    _qrs_detection_post_process,
    _main_task_post_process,
)

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""


ECG_SEQ_LAB_NET_CPSC2021.__DEBUG__ = False
ECG_UNET_CPSC2021.__DEBUG__ = False
RR_LSTM_CPSC2021.__DEBUG__ = False
_BASE_DIR = Path(__file__).absolute().parent
_CUDA = torch.device("cuda")
_CPU = torch.device("cpu")
_BATCH_SIZE = 32

_VERBOSE = 1


_ENTRY_CONFIG = CFG()
_ENTRY_CONFIG.use_rr_lstm_model = True
_ENTRY_CONFIG.use_main_seq_lab_model = True
_ENTRY_CONFIG.use_main_unet_model = False
_ENTRY_CONFIG.merge_rule = "union"

_MODEL_FILENAME = CFG(
    qrs_detection="BestModel_qrs_detection.pth.tar",
    rr_lstm="BestModel_rr_lstm.pth.tar",
    main_seq_lab="BestModel_main_seq_lab.pth.tar",
    main_unet="BestModel_main_unet.pth.tar",  # BestModel_main_unet_deconv.pth.tar
    # it seems that the unet_deconv model is completely useless
)


def save_dict(filename, dic):
    """save dict into json file"""
    with open(filename, "w") as json_file:
        json.dump(dic, json_file, ensure_ascii=False)


@torch.no_grad()
def challenge_entry(sample_path):
    """
    This is a baseline method.
    """
    assert any([
        _ENTRY_CONFIG.use_rr_lstm_model,
        _ENTRY_CONFIG.use_main_seq_lab_model,
        _ENTRY_CONFIG.use_main_unet_model
    ]), "NO model is used, please check `_ENTRY_CONFIG`"

    print("\n" + "*"*100)
    msg = "   CPSC2021 challenge entry starts   ".center(100, "#")
    print(msg)
    print("*"*100 + "\n")
    print(f"processing {sample_path} under config\n{_ENTRY_CONFIG}")
    start_time = time.time()
    timer = time.time()

    # all models are loaded into cpu
    # when using, move to gpu
    rpeak_model, rpeak_cfg = ECG_SEQ_LAB_NET_CPSC2021.from_checkpoint(
        str(_BASE_DIR / "saved_models" / _MODEL_FILENAME.qrs_detection),
        device=_CPU,
    )
    rpeak_model.eval()
    rpeak_cfg = CFG(rpeak_cfg)
    if _VERBOSE >= 1:
        print("QRS detection model is loaded")
    rr_lstm_model, rr_cfg = RR_LSTM_CPSC2021.from_checkpoint(
        str(_BASE_DIR / "saved_models" / _MODEL_FILENAME.rr_lstm),
        device=_CPU,
    )
    rr_lstm_model.eval()
    rr_cfg = CFG(rr_cfg)
    if _VERBOSE >= 1:
        print("RR LSTM model is loaded")
    if _ENTRY_CONFIG.use_main_seq_lab_model:
        # SeqLab (SeqTag) model for the main task
        main_task_model, main_task_cfg = ECG_SEQ_LAB_NET_CPSC2021.from_checkpoint(
            str(_BASE_DIR / "saved_models" / _MODEL_FILENAME.main_seq_lab),
            device=_CPU,
        )
        if _VERBOSE >= 1:
            print("Main task SeqLab model is loaded")
    else:
        # UNet model for the main task
        main_task_model, main_task_cfg = ECG_UNET_CPSC2021.from_checkpoint(
            str(_BASE_DIR / "saved_models" / _MODEL_FILENAME.main_unet),
            device=_CPU,
        )
        if _VERBOSE >= 1:
            print("Main task UNet model is loaded")
    main_task_model.eval()
    main_task_cfg = CFG(main_task_cfg)

    if _VERBOSE >= 1:
        print(f"models loaded in {time.time()-timer:.2f} seconds...")
        timer = time.time()

    _sample_path = Path(sample_path).with_suffix("")
    try:
        wfdb_rec = wfdb.rdrecord(sample_path, physical=True)
    except:
        wfdb_rec = wfdb.rdrecord(_sample_path, physical=True)
    sig = np.asarray(wfdb_rec.p_signal.T)
    for idx in range(sig.shape[0]):
        sig[idx, ...] = remove_spikes_naive(sig[idx, ...])

    # preprocessing, e.g. resample, bandpass, normalization, etc.
    # finished, checked,
    if main_task_cfg.fs != wfdb_rec.fs:
        sig = SS.resample_poly(sig, main_task_cfg.fs, wfdb_rec.fs, axis=1)
    if "baseline" in main_task_cfg:
        bl_win = [main_task_cfg.baseline_window1, main_task_cfg.baseline_window2]
    else:
        bl_win = None
    if "bandpass" in main_task_cfg:
        band_fs = main_task_cfg.filter_band
    else:
        band_fs = None
    sig = preprocess_multi_lead_signal(
        sig,
        fs=main_task_cfg.fs,
        bl_win=bl_win,
        band_fs=band_fs,
        verbose=_VERBOSE,
    )["filtered_ecg"]
    original_siglen = sig.shape[1]

    if _VERBOSE >= 1:
        print(f"data preprocessed in {time.time()-timer:.2f} seconds...")
        timer = time.time()

    # slice data into segments for rpeak detection and main task
    # finished, checked,

    seglen = main_task_cfg[main_task_cfg.task].input_len
    overlap_len = 8 * main_task_cfg.fs
    forward_len = seglen - overlap_len
    dl_input = np.array([]).reshape((0, main_task_cfg.n_leads, seglen))

    # the last few sample points are dropped
    if sig.shape[1] > seglen:
        sig = sig[..., :sig.shape[1] // main_task_cfg[main_task_cfg.task].reduction * main_task_cfg[main_task_cfg.task].reduction]

    if _VERBOSE >= 2:
        print(f"seglen = {seglen}, overlap_len = {overlap_len}, forward_len = {forward_len}")

    for idx in range((sig.shape[1]-seglen) // forward_len + 1):
        seg_data = sig[..., forward_len*idx: forward_len*idx+seglen]
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                # std=list(repeat(1*main_task_cfg.random_normalize_std[0], main_task_cfg.n_leads)),
                per_channel=True,
            )
        dl_input = np.concatenate((dl_input, seg_data[np.newaxis, ...]))
    # add tail
    if sig.shape[1] > seglen:
        seg_data = sig[..., max(0,sig.shape[1]-seglen):sig.shape[1]]
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                per_channel=True,
            )
        dl_input = np.concatenate((dl_input, seg_data[np.newaxis, ...]))
    else:  # too short to form one slice
        seg_data = sig.copy()
        if main_task_cfg.random_normalize:  # to keep consistency of data distribution
            seg_data = normalize(
                sig=seg_data,
                mean=list(repeat(np.mean(main_task_cfg.random_normalize_mean), main_task_cfg.n_leads)),
                std=list(repeat(np.mean(main_task_cfg.random_normalize_std), main_task_cfg.n_leads)),
                per_channel=True,
            )
        dl_input = seg_data[np.newaxis,...]

    if _VERBOSE >= 1:
        print(f"data sliced in {time.time()-timer:.2f} seconds...")
        print(f"sig.shape = {sig.shape}, dl_input.shape = {dl_input.shape}")
        timer = time.time()

    # detect rpeaks
    # finished, checked,
    rpeaks = _detect_rpeaks(
        model=rpeak_model,
        sig=dl_input,
        siglen=sig.shape[1],
        overlap_len=overlap_len,
        config=rpeak_cfg,
    )
    # return rpeaks

    # rr_lstm
    # finished, checked,
    if _ENTRY_CONFIG.use_rr_lstm_model:
        rr_pred = _rr_lstm(
            model=rr_lstm_model,
            rpeaks=rpeaks,
            siglen=original_siglen,
            config=rr_cfg,
        )
        if len(rr_pred) == 0:
            rr_pred_cls = "N"
        elif len(rr_pred) == 1 and np.diff(rr_pred[0])[0] == original_siglen-1:
            rr_pred_cls = "AFf"
        else:
            rr_pred_cls = "AFp"
    else:
        rr_pred = []  # turn off rr_lstm_model, for inspecting the main_task_model
        rr_pred_cls = None
    if _VERBOSE >= 1:
        print(f"\nprediction of rr_lstm_model = {rr_pred}")
    # return rr_pred

    # main_task
    # finished, checked,
    if any([_ENTRY_CONFIG.use_main_seq_lab_model, _ENTRY_CONFIG.use_main_seq_lab_model]):
        main_pred = _main_task(
            model=main_task_model,
            sig=dl_input,
            siglen=original_siglen,
            overlap_len=overlap_len,
            rpeaks=rpeaks,
            config=main_task_cfg,
        )
        if len(main_pred) == 0:
            main_pred_cls = "N"
        elif len(main_pred) == 1 and np.diff(main_pred[0])[0] == original_siglen-1:
            main_pred_cls = "AFf"
        else:
            main_pred_cls = "AFp"
    else:
        main_pred = []  # turn off main_task_model, for inspecting the lstm model
        main_pred_cls = None
    if _VERBOSE >= 1:
        print(f"\nprediction of main_task_model = {main_pred}")
    # return main_pred

    # merge results from rr_lstm and main_task
    # finished, checked,
    # TODO: more sophisticated merge methods?
    if _ENTRY_CONFIG.merge_rule == "union":
        # final_pred = generalized_intervals_union(
        #     [rr_pred, main_pred,]
        # )
        final_pred = _merge_rule_union(rr_pred, rr_pred_cls, main_pred, main_pred_cls)
    else:  # intersection
        final_pred = generalized_intervals_intersection(
            rr_pred, main_pred,
        )

    # TODO: need further filtering to filter out normal episodes shorter than 5 beats?

    if _VERBOSE >= 1:
        print(f"\nfinal prediction = {final_pred}")

    # numpy dtypes to native python dtypes
    # to make json serilizable
    for idx in range(len(final_pred)):
        try:
            final_pred[idx][0] = final_pred[idx][0].item()
        except:
            pass
        try:
            final_pred[idx][1] = final_pred[idx][1].item()
        except:
            pass

    pred_dict = {
        "predict_endpoints": final_pred
    }

    if _VERBOSE >= 1:
        print(f"processing of {sample_path} totally cost {time.time()-start_time:.2f} seconds")

    del rpeak_model
    del rr_lstm_model
    del main_task_model

    print("\n" + "*"*100)
    msg = "   CPSC2021 challenge entry ends   ".center(100, "#")
    print(msg)
    print("*"*100 + "\n\n")

    return pred_dict


def _detect_rpeaks(model, sig, siglen, overlap_len, config):
    """ finished, checked,

    NOTE: sig are sliced data with overlap,
    hence DO NOT directly use model's inference method
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    _device = next(model.parameters()).device
    _dtype = next(model.parameters()).dtype
    sig = torch.as_tensor(sig, device=_device, dtype=_dtype)
    if sig.ndim == 2:
        sig = sig.unsqueeze(0)  # add a batch dimension
    batch_size, channels, seq_len = sig.shape

    l_pred = []
    for idx in range(batch_size//_BATCH_SIZE):
        pred = model.forward(sig[_BATCH_SIZE*idx:_BATCH_SIZE*(idx+1), ...])
        pred = model.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)
        l_pred.append(pred)
    if batch_size % _BATCH_SIZE != 0:
        pred = model.forward(sig[batch_size//_BATCH_SIZE * _BATCH_SIZE:, ...])
        pred = model.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)
        l_pred.append(pred)
    pred = np.concatenate(l_pred)

    # merge the prob array
    seglen = config[config.task].input_len // config[config.task].reduction
    qua_overlap_len = overlap_len // 4 // config[config.task].reduction
    forward_len = seglen - overlap_len // config[config.task].reduction
    _siglen = siglen // config[config.task].reduction

    if _VERBOSE >= 2:
        print("\nin function _detect_rpeaks...")
        print(f"pred.shape = {pred.shape}")
        print(f"seglen = {seglen}, qua_overlap_len = {qua_overlap_len}, forward_len = {forward_len}")

    merged_pred = np.zeros((_siglen,))
    if pred.shape[0] > 1:
        merged_pred[:seglen-qua_overlap_len] = pred[0, :seglen-qua_overlap_len]
        merged_pred[_siglen-(seglen-qua_overlap_len):] = pred[-1,qua_overlap_len:]
        for idx in range(1,pred.shape[0]-1):
            to_compare = np.zeros((_siglen,))
            start_idx = forward_len*idx + qua_overlap_len
            end_idx = forward_len*idx + seglen - qua_overlap_len
            to_compare[start_idx: end_idx] = pred[idx,qua_overlap_len: seglen-qua_overlap_len]
            merged_pred = np.maximum(merged_pred, to_compare)
        # tail
        to_compare = np.zeros((_siglen,))
        to_compare[_siglen-seglen+qua_overlap_len:] = pred[-1, qua_overlap_len:]
        merged_pred = np.maximum(merged_pred, to_compare,)
    else:  # too short to form one slice
        merged_pred = pred[0, ...]
    merged_pred = merged_pred[np.newaxis, ...]
    
    rpeaks = _qrs_detection_post_process(
        pred=merged_pred,
        fs=config.fs, 
        reduction=config[config.task].reduction,
        bin_pred_thr=0.5,
    )[0]

    return rpeaks


def _rr_lstm(model, rpeaks, siglen, config):
    """ finished, checked,
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    rr = np.diff(rpeaks) / config.fs
    # just use the model's inference method
    pred, af_episodes = model.inference(
        input=rr,
        bin_pred_thr=0.5,
        rpeaks=rpeaks,
        episode_len_thr=5,
    )
    af_episodes = af_episodes[0]
    # move to the first and (or) the last sample point of the record if necessary
    if len(af_episodes) > 0:
        # print(af_episodes)
        # print(rpeaks[0], rpeaks[-1])
        if af_episodes[0][0] == rpeaks[0]:
            af_episodes[0][0] = 0
        if af_episodes[-1][-1] == rpeaks[-1]:
            af_episodes[-1][-1] = siglen-1
    return af_episodes


def _main_task(model, sig, siglen, overlap_len, rpeaks, config):
    """ finished, checked,
    """
    try:
        model = model.to(_CUDA)
    except:
        pass
    _device = next(model.parameters()).device
    _dtype = next(model.parameters()).dtype
    sig = torch.as_tensor(sig, device=_device, dtype=_dtype)
    if sig.ndim == 2:
        sig = sig.unsqueeze(0)  # add a batch dimension
    batch_size, channels, seq_len = sig.shape
    
    l_pred = []
    for idx in range(batch_size//_BATCH_SIZE):
        pred = model.forward(sig[_BATCH_SIZE*idx:_BATCH_SIZE*(idx+1), ...])
        pred = model.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)
        l_pred.append(pred)
    if batch_size % _BATCH_SIZE != 0:
        pred = model.forward(sig[batch_size//_BATCH_SIZE * _BATCH_SIZE:, ...])
        pred = model.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)
        l_pred.append(pred)
    pred = np.concatenate(l_pred)

    # merge the prob array
    seglen = config[config.task].input_len // config[config.task].reduction
    qua_overlap_len = overlap_len // 4 // config[config.task].reduction
    forward_len = seglen - overlap_len // config[config.task].reduction
    _siglen = siglen // config[config.task].reduction

    if _VERBOSE >= 2:
        print("\nin function _main_task...")
        print(f"pred.shape = {pred.shape}")
        print(f"seglen = {seglen}, qua_overlap_len = {qua_overlap_len}, forward_len = {forward_len}")

    merged_pred = np.zeros((_siglen,))
    if pred.shape[0] > 1:
        merged_pred[:seglen-qua_overlap_len] = pred[0, :seglen-qua_overlap_len]
        merged_pred[_siglen-(seglen-qua_overlap_len):] = pred[-1,qua_overlap_len:]
        for idx in range(1,pred.shape[0]-1):
            to_compare = np.zeros((_siglen,))
            start_idx = forward_len*idx + qua_overlap_len
            end_idx = forward_len*idx + seglen - qua_overlap_len
            to_compare[start_idx: end_idx] = pred[idx,qua_overlap_len: seglen-qua_overlap_len]
            merged_pred = np.maximum(merged_pred, to_compare)
        # tail
        to_compare = np.zeros((_siglen,))
        to_compare[_siglen-seglen+qua_overlap_len:] = pred[-1, qua_overlap_len:]
        merged_pred = np.maximum(merged_pred, to_compare,)
    else:  # too short to form one slice
        merged_pred = pred[0, ...]
    merged_pred = merged_pred[np.newaxis, ...]

    af_episodes = _main_task_post_process(
        pred=merged_pred,
        fs=config.fs, 
        reduction=config[config.task].reduction,
        bin_pred_thr=0.5,
        rpeaks=[rpeaks],
        siglens=[siglen],
    )[0]

    return af_episodes


def _merge_rule_union(rr_pred, rr_pred_cls, main_pred, main_pred_cls):
    """
    By studying the results (the confusion matrices) on the validation set,
    RR_LSTM model and SeqLab model both seldom have false positives on the classes "N" and "AFf".
    Anlyzing the false positives on the class "AFp", we find that
    the RR_LSTM tends to mistake "N" for "AFp",
    while the SeqLab model tends to mistake "AFf" for "AFp". 
    """
    if rr_pred_cls is None:
        return main_pred
    if main_pred_cls is None:
        return rr_pred
    if (rr_pred_cls == "N" and main_pred_cls != "AFf") or (main_pred_cls == "N" and rr_pred_cls != "AFf"):
        return []
    final_pred = generalized_intervals_union(
            [rr_pred, main_pred,]
        )
    return final_pred


if __name__ == '__main__':
    DATA_PATH = Path(sys.argv[1])
    RESULT_PATH = Path(sys.argv[2])
    if not RESULT_PATH.exists():
        RESULT_PATH.mkdir(parents=True)
        
    test_set = (DATA_PATH / 'RECORDS').read_text().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = DATA_PATH / sample
        pred_dict = challenge_entry(str(sample_path))

        save_dict(str(RESULT_PATH / sample+'.json'), pred_dict)
