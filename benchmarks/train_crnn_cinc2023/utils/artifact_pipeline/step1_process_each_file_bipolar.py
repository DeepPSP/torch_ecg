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

import datetime
import math
import os
import os.path
import pdb
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import mne as mne
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio

sys.path.append(str(Path(__file__).resolve().parent))

from segment_EEG import *
from segment_EEG_without_detection import *

Fs = 100.0
# assess_time_before = 1800  # [s]
# assess_time_after = 1800  # [s]
window_time = 5  # [s]
window_step = 5  # [s]
# sub_window_time = 5  # [s] for calculating features
# sub_window_step = 1  # [s]
start_end_remove_window_num = 0
amplitude_thres = 500  # 500  # [uV]
line_freq = 60.0  # [Hz]
bandpass_freq = [0.5, 30.0]  # [Hz]
tostudy_freq = [0.5, 30.0]  # [Hz]
# available_channels = ['C3', 'C4', 'O1', 'O2', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FZ', 'FP1', 'FP2', 'FPZ', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']#BWH
# available_channels = ['EEG Fp1-Ref1', 'EEG F3-Ref1', 'EEG C3-Ref1', 'EEG P3-Ref1', 'EEG F7-Ref1', 'EEG T3-Ref1', 'EEG T5-Ref1', 'EEG O1-Ref1', 'EEG Fz-Ref1', 'EEG Cz-Ref1', 'EEG Pz-Ref1', 'EEG Fp2-Ref1',  'EEG F4-Ref1', 'EEG C4-Ref1', 'EEG P4-Ref1', 'EEG F8-Ref1', 'EEG T4-Ref1', 'EEG T6-Ref1', 'EEG O2-Ref1']  # UTW
# available_channels = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2',  'F4', 'C4', 'P4', 'F8', 'T8', 'P8', 'O2']
available_channels = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
]  # MGH BWH ULB
bipolar_channels = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fz-Cz",
    "Cz-Pz",
]
# available_channels = ['EEGFP1_', 'EEGFP2_', 'EEGFPZ_', 'EEGF7__', 'EEGF8__']
# eeg_channels = ['C3', 'C4', 'O1', 'O2', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FZ', 'FP1', 'FP2', 'FPZ', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']#['Fp1-F7','Fp2-F8','Fp1-Fp2','F7-F8']#'Fp1','Fp2','F7','F8',
# algorithms = ['cnn_lstm_ae', 'lstm', 'dnn_clf', 'dnn_ord', 'moe_dnn']#'RandomForest','SVM','ELM']'blr', 'dnn_reg', 'logreg',
random_state = 1
# normal_only = False
# labeled_only = False

seg_mask_explanation = np.array(
    [
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
)

if __name__ == "__main__":
    # """
    ##################
    # use data_list_paths to specify what data to use
    # data_list.txt:
    # data_path    spec_path    feature_path    state
    # eeg1.mat     specs1.mat   Features1.mat   good
    # ...
    # note: eeg segments are contained in features
    ##################
    # file = "D:\\205011.edf"

    file_path = "Z:\\"
    save_path = "Z:\\"
    file_list = [f for f in os.listdir(file_path) if f.endswith(".edf")]

    # file_list = sorted(file_list)
    # import pdb;pdb.set_trace()
    # file_list = os.listdir(file_path)
    file_list = file_list[155:156]

    for ifile in file_list:
        file = file_path + ifile
        print(file)
        # import pdb;pdb.set_trace()
        #        if os.path.isfile(save_path+ifile+'.mat'):
        #            continue
        #        else:
        #            try:
        # data = mne.io.read_raw_edf(file,preload=True)
        data = mne.io.read_raw_edf(file, stim_channel=None, exclude="EDF Annotations", preload=True)
        raw_data = data.get_data(picks=range(23))
        info = data.info
        fs = info["sfreq"]
        # raw_data = scipy.signal.resample(raw_data, int(math.floor(raw_data.shape[1]*Fs/fs)),axis=1)
        raw_data = scipy.signal.resample_poly(raw_data, Fs, fs, axis=1)
        # raw_data = mne.filter.resample(raw_data, down=fs/Fs, npad='auto')

        # import pdb;pdb.set_trace()
        raw_data = raw_data * 10e5  # V->uV

        channels = data.ch_names
        channels = [x.upper() for x in channels]
        chan_index = list()
        for chNo in available_channels:
            chan_index.append(channels.index(chNo.upper()))
        raw_data = raw_data[chan_index, :]

        ## Bipolar reference
        bipolar_data = np.zeros((18, raw_data.shape[1]))
        bipolar_data[8, :] = raw_data[0, :] - raw_data[1, :]
        # Fp1-F3
        bipolar_data[9, :] = raw_data[1, :] - raw_data[2, :]
        # F3-C3
        bipolar_data[10, :] = raw_data[2, :] - raw_data[3, :]
        # C3-P3
        bipolar_data[11, :] = raw_data[3, :] - raw_data[7, :]
        # P3-O1

        bipolar_data[12, :] = raw_data[11, :] - raw_data[12, :]
        # Fp2-F4
        bipolar_data[13, :] = raw_data[12, :] - raw_data[13, :]
        # F4-C4
        bipolar_data[14, :] = raw_data[13, :] - raw_data[14, :]
        # C4-P4
        bipolar_data[15, :] = raw_data[14, :] - raw_data[18, :]
        # P4-O2

        bipolar_data[0, :] = raw_data[0, :] - raw_data[4, :]
        # Fp1-F7
        bipolar_data[1, :] = raw_data[4, :] - raw_data[5, :]
        # F7-T3
        bipolar_data[2, :] = raw_data[5, :] - raw_data[6, :]
        # T3-T5
        bipolar_data[3, :] = raw_data[6, :] - raw_data[7, :]
        # T5-O1

        bipolar_data[4, :] = raw_data[11, :] - raw_data[15, :]
        # Fp2-F8
        bipolar_data[5, :] = raw_data[15, :] - raw_data[16, :]
        # F8-T4
        bipolar_data[6, :] = raw_data[16, :] - raw_data[17, :]
        # T4-T6
        bipolar_data[7, :] = raw_data[17, :] - raw_data[18, :]
        # T6-O2

        bipolar_data[16, :] = raw_data[8, :] - raw_data[9, :]
        # Fz-Cz
        bipolar_data[17, :] = raw_data[9, :] - raw_data[10, :]
        # Cz-Pz

        # ## save 5s monopolar/bipolar epoches using notch/band pass/artifact detection/resampling ### DEBUG
        # segs_monpolar = segment_EEG_without_detection(raw_data,available_channels,window_time, window_step, Fs,
        #                     notch_freq=line_freq, bandpass_freq=bandpass_freq,
        #                     to_remove_mean=False, amplitude_thres=amplitude_thres, n_jobs=-1, start_end_remove_window_num=start_end_remove_window_num)
        del raw_data
        segs_, bs_, seg_start_ids_, seg_mask, specs_, freqs_ = segment_EEG(
            bipolar_data,
            bipolar_channels,
            window_time,
            window_step,
            Fs,
            notch_freq=line_freq,
            bandpass_freq=bandpass_freq,
            to_remove_mean=False,
            amplitude_thres=amplitude_thres,
            n_jobs=-1,
            start_end_remove_window_num=start_end_remove_window_num,
        )

        if len(segs_) <= 0:
            raise ValueError("No segments")

        seg_mask2 = map(lambda x: x.split("_")[0], seg_mask)
        sm = Counter(seg_mask2)
        for ex in seg_mask_explanation:
            if ex in sm:
                print("%s: %d/%d, %g%%" % (ex, sm[ex], len(seg_mask), sm[ex] * 100.0 / len(seg_mask)))

        if segs_.shape[0] <= 0:
            raise ValueError("No EEG signal")
            if segs_.shape[1] != len(bipolar_channels):
                raise ValueError("Incorrect #chanels")

        fd = os.path.split(save_path)[0]
        if not os.path.exists(fd):
            os.mkdir(fd)
        res = {
            "EEG_segs_bipolar": segs_.astype("float16"),
            "EEG_segs_monopolar": segs_monpolar.astype("float16"),
            "EEG_specs": specs_.astype("float16"),
            "burst_suppression": bs_.astype("float16"),
            "EEG_frequency": freqs_,
            "seg_start_ids": seg_start_ids_,
            "Fs": Fs,
            "seg_masks": seg_mask,
            "channel_names": bipolar_channels,
        }
        sio.savemat(save_path + ifile, res, do_compression=True)
