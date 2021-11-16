"""
use median filter to remove baseline,
note that highpass filters also have the effect of baseline removal
"""

from typing import NoReturn, Any, Tuple
from numbers import Real
import warnings

import numpy as np

from .base import (
    PreProcessor,
    preprocess_multi_lead_signal,
    preprocess_single_lead_signal,
)


__all__ = ["BaselineRemove",]


class BaselineRemove(PreProcessor):
    """
    """
    __name__ = "BaselineRemove"

    def __init__(self, window1:float=0.2, window2:float=0.6, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        window1: float, default 0.2,
            the smaller window size of the median filter, with units in seconds
        highcut: float, default 0.6,
            the larger window size of the median filter, with units in seconds
        """
        self.window1 = window1
        self.window2 = window2
        if self.window2 < self.window1:
            self.window1, self.window2 = self.window2, self.window1
            warnings.warn("values of window1 and window2 are switched")

    def apply(self, sig:np.ndarray, fs:Real) -> Tuple[np.ndarray, int]:
        """ finished, checked,

        apply the preprocessor to `sig`

        Parameters
        ----------
        sig: ndarray,
            the ECG signal, can be
            1d array, which is a single-lead ECG
            2d array, which is a multi-lead ECG of "lead_first" format
            3d array, which is a tensor of several ECGs, of shape (batch, lead, siglen)
        fs: real number,
            sampling frequency of the ECG signal

        Returns
        -------
        filtered_sig: ndarray,
            the median filtered (hence baseline removed) ECG signal
        fs: int,
            the sampling frequency of the filtered ECG signal
        """
        self._check_sig(sig)
        if sig.ndim == 1:
            filtered_sig = preprocess_single_lead_signal(
                raw_sig=sig,
                fs=fs,
                bl_win=[self.window1, self.window2],
            )
        elif sig.ndim == 2:
            filtered_sig = preprocess_multi_lead_signal(
                raw_sig=sig,
                fs=fs,
                bl_win=[self.window1, self.window2],
            )
        elif sig.ndim == 3:
            filtered_sig = np.zeros_like(sig)
            for b in range(filtered_sig.shape[0]):
                filtered_sig[b, ...] = preprocess_multi_lead_signal(
                    raw_sig=sig[b, ...],
                    fs=fs,
                    bl_win=[self.window1, self.window2],
                )
        return filtered_sig, fs
