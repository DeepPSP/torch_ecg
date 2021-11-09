"""
normalization of the signals
"""

from typing import NoReturn, Any, Union
from numbers import Real

import numpy as np

from .base import PreProcessor
from ..utils.utils_signal import normalize


__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
]


class Normalize(PreProcessor):
    """
    perform z-score normalization on `sig`,
    to make it has fixed mean and standard deviation,
    or perform min-max normalization on `sig`,
    or normalize `sig` using `mean` and `std` via (sig - mean) / std.
    More precisely,

        .. math::
            \begin{align*}
            \text{Min-Max normalization:} & \frac{sig - \min(sig)}{\max(sig) - \min(sig)} \\
            \text{Naive normalization:} & \frac{sig - m}{s} \\
            \text{Z-score normalization:} & \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m
            \end{align*}
    """
    __name__ = "Normalize"

    def __init__(self,
                 method:str,
                 mean:Union[Real,np.ndarray]=0.0,
                 std:Union[Real,np.ndarray]=1.0,
                 per_channel:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        method: str,
            normalization method, case insensitive, can be one of
            "naive", "min-max", "z-score",
        mean: real number or ndarray, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal,
            useless if `method` is "min-max"
        std: real number or ndarray, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal,
            useless if `method` is "min-max"
        per_channel: bool, default False,
            if True, normalization will be done per channel
        """
        self.method = method.lower()
        assert self.method in ["z-score", "naive", "min-max",]
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
            sampling frequency of the ECG signal,
            not used

        Returns
        -------
        normalized_sig: ndarray,
            the normalized ECG signal
        """
        self.__check_sig(sig)
        normalized_sig = normalize(
            sig=sig,
            method=self.method,
            mean=self.mean,
            std=self.std,
            sig_fmt="channel_first",
            per_channel=self.per_channel,
        )
        return normalized_sig


class MinMaxNormalize(Normalize):
    """
    Min-Max normalization, defined as

        .. math::
            \frac{sig - \min(sig)}{\max(sig) - \min(sig)}
    """
    __name__ = "MinMaxNormalize"

    def __init__(self, per_channel:bool=False,) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        per_channel: bool, default False,
            if True, normalization will be done per channel
        """
        super().__init__(method="min-max", per_channel=per_channel)


class NaiveNormalize(Normalize):
    """
    Naive normalization via

        .. math::
            \frac{sig - m}{s}
    """
    __name__ = "NaiveNormalize"

    def __init__(self,
                 mean:Union[Real,np.ndarray]=0.0,
                 std:Union[Real,np.ndarray]=1.0,
                 per_channel:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: real number or ndarray, default 0.0,
            value(s) to be subtracted
        std: real number or ndarray, default 1.0,
            value(s) to be divided
        per_channel: bool, default False,
            if True, normalization will be done per channel
        """
        super().__init__(
            method="naive",
            mean=mean,
            std=std,
            per_channel=per_channel,
        )


class ZScoreNormalize(Normalize):
    """
    Z-score normalization via

        .. math::
            \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m
    """
    __name__ = "ZScoreNormalize"

    def __init__(self,
                 mean:Union[Real,np.ndarray]=0.0,
                 std:Union[Real,np.ndarray]=1.0,
                 per_channel:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: real number or ndarray, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal,
        std: real number or ndarray, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal,
        per_channel: bool, default False,
            if True, normalization will be done per channel
        """
        super().__init__(
            method="z-score",
            mean=mean,
            std=std,
            per_channel=per_channel,
        )
