"""Base class for preprocessors."""

from abc import ABC, abstractmethod
from itertools import repeat
from numbers import Real
from typing import List, Literal, Optional, Tuple

import numpy as np
from biosppy.signals.tools import filter_signal
from scipy.ndimage import median_filter

from ..cfg import DEFAULTS
from ..utils.misc import ReprMixin, add_docstring
from ..utils.utils_signal import butter_bandpass_filter

# from scipy.signal import medfilt
# https://github.com/scipy/scipy/issues/9680


__all__ = [
    "PreProcessor",
    "preprocess_multi_lead_signal",
    "preprocess_single_lead_signal",
]


class PreProcessor(ReprMixin, ABC):
    """Base class for preprocessors."""

    __name__ = "PreProcessor"

    @abstractmethod
    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        """Apply the preprocessor to `sig`.

        Parameters
        ----------
        sig : numpy.ndarray
            The ECG signal, can be
                - 1d array, which is a single-lead ECG;
                - 2d array, which is a multi-lead ECG of "lead_first" format;
                - 3d array, which is a tensor of several ECGs, of shape ``(batch, lead, siglen)``.
        fs : numbers.Real
            Sampling frequency of the ECG signal.

        """
        raise NotImplementedError

    @add_docstring(apply)
    def __call__(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        """alias of :meth:`self.apply`."""
        return self.apply(sig, fs)

    def _check_sig(self, sig: np.ndarray) -> None:
        """Check validity of the signal.

        Parameters
        ----------
        sig : numpy.ndarray
            The ECG signal, can be
                - 1d array, which is a single-lead ECG;
                - 2d array, which is a multi-lead ECG of "lead_first" format;
                - 3d array, which is a tensor of several ECGs, of shape ``(batch, lead, siglen)``.

        """
        if sig.ndim not in [1, 2, 3]:
            raise ValueError(
                "Invalid input ECG signal. Should be"
                "1d array, which is a single-lead ECG;"
                "or 2d array, which is a multi-lead ECG of `lead_first` format;"
                "or 3d array, which is a tensor of several ECGs, of shape (batch, lead, siglen)."
            )


def preprocess_multi_lead_signal(
    raw_sig: np.ndarray,
    fs: Real,
    sig_fmt: Literal["channel_first", "lead_first", "channel_last", "lead_last"] = "channel_first",
    bl_win: Optional[List[Real]] = None,
    band_fs: Optional[List[Real]] = None,
    filter_type: Literal["butter", "fir"] = "butter",
    filter_order: Optional[int] = None,
) -> np.ndarray:
    """Perform preprocessing for multi-lead ECG signal (with units in mV).

    preprocessing may include median filter, bandpass filter, and rpeaks detection, etc.
    Also works for single-lead ECG signal (setting ``sig_fmt="channel_first"``).

    Parameters
    ----------
    raw_sig : numpy.ndarray
        The raw ECG signal, with units in mV.
    fs : numbers.Real
        Sampling frequency of `raw_sig`.
    sig_fmt : str, default "channel_first"
        Format of the multi-lead ECG signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first").
    bl_win : List[numbers.Real], optional
        Window (units in second) of baseline removal
        using :meth:`~scipy.ndimage.median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is ``[0.2, 0.6]``.
        If is None or empty, baseline removal will not be performed.
    band_fs : List[numbers.Real], optional
        Frequency band of the bandpass filter,
        a typical pair is ``[0.5, 45]``.
        Be careful when detecting paced rhythm.
        If is None or empty, bandpass filtering will not be performed.
    filter_type : {"butter", "fir"}, default "butter"
        Type of the bandpass filter.
    filter_order : int, optional
        Order of the bandpass filter.

    Returns
    -------
    filtered_ecg : numpy.ndarray
        The array of the processed ECG signal.
        The format of the signal is kept the same with the original signal,
        i.e. `sig_fmt`.

    """
    raw_sig = np.asarray(raw_sig)
    assert raw_sig.ndim in [2, 3], "multi-lead signal should be 2d or 3d array"
    assert sig_fmt.lower() in [
        "channel_first",
        "lead_first",
        "channel_last",
        "lead_last",
    ], f"multi-lead signal format `{sig_fmt}` not supported"
    if sig_fmt.lower() in ["channel_last", "lead_last"]:
        # might have a batch dimension at the first axis
        filtered_ecg = np.moveaxis(raw_sig, -2, -1).astype(DEFAULTS.np_dtype)
    else:
        filtered_ecg = np.asarray(raw_sig, dtype=DEFAULTS.np_dtype)

    # remove baseline
    if bl_win:
        window1, window2 = list(repeat(1, filtered_ecg.ndim)), list(repeat(1, filtered_ecg.ndim))
        window1[-1] = 2 * (int(bl_win[0] * fs) // 2) + 1  # window size must be odd
        window2[-1] = 2 * (int(bl_win[1] * fs) // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode="nearest")
        baseline = median_filter(baseline, size=window2, mode="nearest")
        filtered_ecg = filtered_ecg - baseline

    # filter signal
    if band_fs:
        assert band_fs[0] < band_fs[1], "Invalid frequency band"
        nyq = 0.5 * fs
        if band_fs[0] <= 0 and band_fs[1] < nyq:
            band = "lowpass"
            frequency = band_fs[1]
        elif band_fs[1] >= nyq and band_fs[0] > 0:
            band = "highpass"
            frequency = band_fs[0]
        elif band_fs[0] > 0 and band_fs[1] < nyq:
            band = "bandpass"
            frequency = band_fs
        else:
            raise AssertionError("Invalid frequency band")
        if filter_type.lower() == "fir":
            filtered_ecg = filter_signal(
                signal=filtered_ecg,
                ftype="FIR",
                # ftype="butter",
                band=band,
                order=filter_order or int(0.2 * fs),
                sampling_rate=fs,
                frequency=frequency,
            )["signal"]
        elif filter_type.lower() == "butter":
            filtered_ecg = butter_bandpass_filter(
                data=filtered_ecg,
                lowcut=band_fs[0],
                highcut=band_fs[1],
                fs=fs,
                order=filter_order or round(0.01 * fs),  # better be determined by the `buttord`
            )
        else:
            raise ValueError(f"Unsupported filter type `{filter_type}`")

    if sig_fmt.lower() in ["channel_last", "lead_last"]:
        filtered_ecg = filtered_ecg.T

    return filtered_ecg


def preprocess_single_lead_signal(
    raw_sig: np.ndarray,
    fs: Real,
    bl_win: Optional[List[Real]] = None,
    band_fs: Optional[List[Real]] = None,
    filter_type: Literal["butter", "fir"] = "butter",
    filter_order: Optional[int] = None,
) -> np.ndarray:
    """Perform preprocessing for single lead ECG signal (with units in mV).

    Preprocessing may include median filter, bandpass filter, and rpeaks detection, etc.

    Parameters
    ----------
    raw_sig : numpy.ndarray
        Raw ECG signal, with units in mV.
    fs : numbers.Real
        Sampling frequency of `raw_sig`.
    bl_win : list (of 2 numbers.Real), optional
        Window (units in second) of baseline removal
        using :meth:`~scipy.ndimage.median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is ``[0.2, 0.6]``.
        If is None or empty, baseline removal will not be performed.
    band_fs : list of numbers.Real, optional
        Frequency band of the bandpass filter,
        a typical pair is ``[0.5, 45]``.
        Be careful when detecting paced rhythm.
        If is None or empty, bandpass filtering will not be performed.
    filter_type : {"butter", "fir"}, default "butter"
        Type of the bandpass filter.
    filter_order : int, optional
        Order of the bandpass filter.

    Returns
    -------
    filtered_ecg : numpy.ndarray
        The array of the processed ECG signal.

    NOTE
    ----
    Bandpass filter uses FIR filters, an alternative can be Butterworth filter,
    e.g. :meth:`~torch_ecg.utils.butter_bandpass_filter`.

    """
    filtered_ecg = np.asarray(raw_sig, dtype=DEFAULTS.np_dtype)
    assert filtered_ecg.ndim == 1, "single-lead signal should be 1d array"

    # remove baseline
    if bl_win:
        window1 = 2 * (int(bl_win[0] * fs) // 2) + 1  # window size must be odd
        window2 = 2 * (int(bl_win[1] * fs) // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode="nearest")
        baseline = median_filter(baseline, size=window2, mode="nearest")
        filtered_ecg = filtered_ecg - baseline

    # filter signal
    if band_fs:
        assert band_fs[0] < band_fs[1], "Invalid frequency band"
        nyq = 0.5 * fs
        if band_fs[0] <= 0 and band_fs[1] < nyq:
            band = "lowpass"
            frequency = band_fs[1]
        elif band_fs[1] >= nyq and band_fs[0] > 0:
            band = "highpass"
            frequency = band_fs[0]
        elif band_fs[0] > 0 and band_fs[1] < nyq:
            band = "bandpass"
            frequency = band_fs
        else:
            raise AssertionError("Invalid frequency band")
        if filter_type.lower() == "fir":
            filtered_ecg = filter_signal(
                signal=filtered_ecg,
                ftype="FIR",
                # ftype="butter",
                band=band,
                order=int(0.3 * fs),
                sampling_rate=fs,
                frequency=frequency,
            )["signal"]
        elif filter_type.lower() == "butter":
            filtered_ecg = butter_bandpass_filter(
                data=filtered_ecg,
                lowcut=band_fs[0],
                highcut=band_fs[1],
                fs=fs,
                order=filter_order or round(0.01 * fs),  # better be determined by the `buttord`
            )
        else:
            raise ValueError(f"Unsupported filter type `{filter_type}`")

    return filtered_ecg
