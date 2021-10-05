"""
"""

from typing import NoReturn, Optional
from numbers import Real
import warnings

import numpy as np

from .base import (
    PreProcessor,
    preprocess_multi_lead_signal,
    preprocess_single_lead_signal,
)


__all__ = ["BandPass",]


class BandPass(PreProcessor):
    """
    """
    __name__ = "BandPass"

    def __init__(self, lowcut:Optional[Real]=None, highcut:Optional[Real]=None, **kwargs) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        lowcut: real number, optional,
            low cutoff frequency
        highcut: real number, optional,
            high cutoff frequency
        """
        self.lowcut = lowcut
        self.highcut = highcut
        assert any([self.lowcut is not None, self.highcut is not None]), \
            "At least one of lowcut and highcut should be set"
        if not self.lowcut:
            self.lowcut = 0
        if not self.highcut:
            self.highcut = float("inf")

    def apply(self, sig:np.ndarray, fs:Real) -> np.ndarray:
        """ finished, NOT checked,

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
            the bandpass filtered ECG signal
        """
        if sig.ndim == 1:
            filtered_sig = preprocess_single_lead_signal(
                raw_sig=sig,
                fs=fs,
                band_fs=[self.lowcut, self.highcut],
            )
        elif sig.ndim == 2:
            filtered_sig = preprocess_multi_lead_signal(
                raw_sig=sig,
                fs=fs,
                band_fs=[self.lowcut, self.highcut],
            )
        elif sig.ndim == 3:
            filtered_sig = np.zeros_like(sig)
            for b in range(filtered_sig.shape[0]):
                filtered_sig[b, ...] = preprocess_multi_lead_signal(
                    raw_sig=sig[b, ...],
                    fs=fs,
                    band_fs=[self.lowcut, self.highcut],
                )
        else:
            raise ValueError(
                "Invalid input ECG signal. Should be"
                "1d array, which is a single-lead ECG;"
                "or 2d array, which is a multi-lead ECG of `lead_first` format;"
                "or 3d array, which is a tensor of several ECGs, of shape (batch, lead, siglen)."
            )
        return filtered_sig
