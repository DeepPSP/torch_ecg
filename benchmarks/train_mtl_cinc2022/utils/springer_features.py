"""
"""

from decimal import localcontext, Decimal, ROUND_HALF_UP
from typing import Optional, Sequence

import numpy as np
import scipy.signal as SS
from easydict import EasyDict as ED
from torch_ecg.utils.utils_signal import butter_bandpass_filter, normalize

from .schmidt_spike_removal import schmidt_spike_removal
from .springer_dwt import get_dwt_features


__all__ = [
    "get_springer_features",
]


def get_springer_features(
    signal: np.ndarray,
    fs: int,
    feature_fs: int,
    feature_format: str = "flat",
    config: Optional[dict] = None,
) -> np.ndarray:
    """

    This function **almost** re-implements the original matlab
    implementation of the Springer features.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    feature_fs: int,
        The sampling frequency of the features.
    feature_format: str, default "flat",
        The format of the features, can be one of
        "flat", "channel_first", "channel_last",
        case insensitive.
    config: dict, optional,
        The configuration for extraction methods of the features.

    Returns
    -------
    springer_features: np.ndarray,
        The extracted features, of shape
        (4 * feature_len,) if `feature_format` is "flat",
        (4, feature_len) if `feature_format` is "channel_first",
        (feature_len, 4) if `feature_format` is "channel_last".
        The features are in the following order:
        - homomorphic_envelope
        - hilbert envelope
        - PSD
        - DWT

    """
    assert feature_format.lower() in [
        "flat",
        "channel_first",
        "channel_last",
    ], f"`feature_format` must be one of 'flat', 'channel_first', 'channel_last', but got {feature_format}"
    cfg = ED(
        order=2,
        lowcut=25,
        highcut=400,
        lpf_freq=8,
        seg_tol=0.1,
        psd_freq_lim=(40, 60),
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    filtered_signal = butter_bandpass_filter(
        signal,
        fs=fs,
        lowcut=cfg.lowcut,
        highcut=cfg.highcut,
        order=cfg.order,
        btype="lohi",
    )
    filtered_signal = schmidt_spike_removal(filtered_signal, fs)

    homomorphic_envelope = homomorphic_envelope_with_hilbert(
        filtered_signal, fs, lpf_freq=cfg.lpf_freq
    )
    downsampled_homomorphic_envelope = SS.resample_poly(
        homomorphic_envelope, feature_fs, fs
    )
    downsampled_homomorphic_envelope = normalize(
        downsampled_homomorphic_envelope, method="z-score", mean=0.0, std=1.0
    )

    amplitude_envelope = hilbert_envelope(filtered_signal, fs)
    downsampled_hilbert_envelope = SS.resample_poly(amplitude_envelope, feature_fs, fs)
    downsampled_hilbert_envelope = normalize(
        downsampled_hilbert_envelope, method="z-score", mean=0.0, std=1.0
    )

    psd = get_PSD_feature(filtered_signal, fs, freq_lim=cfg.psd_freq_lim)
    psd = SS.resample_poly(psd, len(downsampled_homomorphic_envelope), len(psd))
    psd = normalize(psd, method="z-score", mean=0.0, std=1.0)

    wavelet_feature = np.abs(get_dwt_features(filtered_signal, fs, config=cfg))
    wavelet_feature = wavelet_feature[: len(homomorphic_envelope)]
    wavelet_feature = SS.resample_poly(wavelet_feature, feature_fs, fs)
    wavelet_feature = normalize(wavelet_feature, method="z-score", mean=0.0, std=1.0)

    func = dict(
        flat=np.concatenate,
        channel_first=np.row_stack,
        channel_last=np.column_stack,
    )

    springer_features = func[feature_format.lower()](
        [
            downsampled_homomorphic_envelope,
            downsampled_hilbert_envelope,
            psd,
            wavelet_feature,
        ]
    )
    return springer_features


def hilbert_envelope(signal: np.ndarray, fs: int) -> np.ndarray:
    """

    Compute the envelope of the signal using the Hilbert transform.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.

    Returns
    -------
    ndarray:
        The envelope of the signal.

    """
    return np.abs(SS.hilbert(signal))


def homomorphic_envelope_with_hilbert(
    signal: np.ndarray, fs: int, lpf_freq: int = 8, order: int = 1
) -> np.ndarray:
    """

    Compute the homomorphic envelope of the signal using the Hilbert transform.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    lpf_freq: int, default 8,
        The low-pass filter frequency (high cut frequency).
        The filter will be applied to log of the Hilbert envelope.
    order: int, default 1,
        The order of the butterworth low-pass filter.

    Returns
    -------
    homomorphic_envelope: ndarray,
        The homomorphic envelope of the signal.

    """
    amplitude_envelope = hilbert_envelope(signal, fs)
    homomorphic_envelope = np.exp(
        butter_bandpass_filter(np.log(amplitude_envelope), 0, lpf_freq, fs, order=order)
    )
    homomorphic_envelope[0] = homomorphic_envelope[1]
    return homomorphic_envelope


def get_PSD_feature(
    signal: np.ndarray,
    fs: int,
    freq_lim: Sequence[int] = (40, 60),
    window_size: float = 1 / 40,
    overlap_size: float = 1 / 80,
) -> np.ndarray:
    """

    Compute the PSD (power spectral density) of the signal.

    Parameters
    ----------
    signal: np.ndarray,
        The signal (1D) to extract features from.
    fs: int,
        The sampling frequency of the signal.
    freq_lim: sequence of int, default (40,60),
        The frequency range to compute the PSD.
    window_size: float, default 1/40,
        The size of the window to compute the PSD,
        with units in seconds.
    overlap_size: float, default 1/80,
        The size of the overlap between windows to compute the PSD,
        with units in seconds.

    Returns
    -------
    psd: ndarray,
        The PSD of the signal.

    NOTE:
    The `round` function in matlab is different from python's `round` function,
    ref. https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules.
    The rounding rule for matlab is `to nearest, ties away from zero`,
    while the rounding rule for python is `to nearest, ties to even`.

    """
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        nperseg = int(Decimal(fs * window_size).to_integral_value())
        noverlap = int(Decimal(fs * overlap_size).to_integral_value())
    f, t, Sxx = SS.spectrogram(
        signal,
        fs,
        "hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=fs,
        return_onesided=True,
        scaling="density",
        mode="psd",
    )
    inds = np.where((f >= freq_lim[0]) & (f <= freq_lim[1]))[0]
    psd = np.mean(Sxx[inds, :], axis=0)
    return psd
