"""
"""

from typing import Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch


__all__ = [
    "plot_spectrogram",
]


def plot_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    title: str = "Spectrogram",
    xlim: Optional[Sequence[float]] = None,
) -> None:
    """
    modified from the function `plot_specgram`
    in https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

    Parameters
    ----------
    waveform: np.ndarray,
        raw audio signal
    sample_rate: int,
        sampling rate
    title: str,
        title of the plot
    xlim: 2-sequence of float,
        x-axis limits

    """
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)
