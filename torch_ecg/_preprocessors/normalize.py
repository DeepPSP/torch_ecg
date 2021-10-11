"""
normalization of the signals
"""

from typing import NoReturn, Any
from numbers import Real

import numpy as np

from .base import PreProcessor
from ..utils.utils_signal import normalize


__all__ = ["Normalize",]


class Normalize(PreProcessor):
    """
    """
    __name__ = "Normalize"

    def __init__(self,
                 mean:Union[Real,np.ndarray]=0.0,
                 std:Union[Real,np.ndarray]=1.0,
                 per_channel:bool=True,
                 **kwargs:Any) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        mean: real number of ndarray, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal
        std: real number of ndarray, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal
        sig_fmt: str, default "channel_first",
            format of the signal, can be of one of
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        per_channel: bool, default False,
            if True, normalization will be done per channel
        """
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        if isinstance(std, Real):
            assert std > 0, "standard deviation should be positive"
        else:
            assert (std > 0).all(), "standard deviations should all be positive"
        if not per_channel:
            assert isinstance(mean, Real) and isinstance(std, Real), \
                f"mean and std should be real numbers in the non per-channel setting"

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
        normalized_sig: ndarray,
            the normalized ECG signal
        """
        self.__check_sig(sig)
        normalized_sig = normalize(
            sig=sig,
            mean=self.mean,
            std=self.std,
            sig_fmt="channel_first",
            per_channel=self.per_channel,
        )
        return normalized_sig
