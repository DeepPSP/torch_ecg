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


import nitime.algorithms as tsa
import numpy as np
from scipy.signal import detrend


def multitaper_spectrogram(EEG, Fs, NW, window_length, window_step):
    """Compute spectrogram using multitaper estimation.

    Arguments:
    EEG -- EEG signal, size=(sample_point_num, channel_num)
    Fs -- sampling frequency in Hz
    NW -- the time-halfbandwidth product, usually 2, 3, or 4
    window_length -- length of windows in seconds
    window_step -- step of windows in seconds

    Outputs:
    psd estimation, size=(window_num, freq_point_num, channel_num)
    frequencies, size=(freq_point_num,)
    """
    signal_length, channel_num = EEG.shape

    window_length = int(round(window_length * Fs))
    window_step = int(round(window_step * Fs))

    nfft = max(1 << (window_length - 1).bit_length(), window_length)

    window_start = np.arange(0, signal_length - window_length + 1, window_step)
    window_num = len(window_start)

    freqs = np.arange(0, Fs, Fs * 1.0 / nfft)[: nfft // 2 + 1]

    mt_pxx = np.zeros((window_num, len(freqs), channel_num))
    for wi in range(window_num):
        eeg_seg = detrend(EEG[window_start[wi] : window_start[wi] + window_length, :], axis=0)
        for chi in range(channel_num):
            _, pxx, _ = tsa.multi_taper_psd(
                eeg_seg[:, chi], Fs=Fs, NW=NW, adaptive=True, jackknife=False, low_bias=True, NFFT=nfft
            )
            mt_pxx[wi, :, chi] = pxx

    return mt_pxx, freqs


if __name__ == "__main__":
    import pdb

    from scipy import io as sio

    ff = sio.loadmat(r"C:\\multitaper_example.mat")
    EEG = ff["eeg"]
    EEG_after_detrend = ff["eeg_after_detrend"]
    ss = ff["ss"]

    Fs = 200
    NW = 2
    window_length = 2
    window_step = 0.2

    mt_pxx, freqs = multitaper_spectrogram(EEG, Fs, NW, window_length, window_step)
    print(mt_pxx)
    print(freqs)
