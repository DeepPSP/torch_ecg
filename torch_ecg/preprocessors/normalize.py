"""
"""

from numbers import Real
from typing import Any, Iterable, NoReturn, Union

import torch

from ..utils.utils_signal_t import normalize as normalize_t

__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
]


class Normalize(torch.nn.Module):
    r"""
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

    def __init__(
        self,
        method: str = "z-score",
        mean: Union[Real, Iterable[Real]] = 0.0,
        std: Union[Real, Iterable[Real]] = 1.0,
        per_channel: bool = False,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        method: str, default "z-score",
            normalization method, case insensitive, can be one of
            "naive", "min-max", "z-score",
        mean: real number or array_like, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal, if `method` is "z-score";
            mean values to be subtracted from the original signal, if `method` is "naive";
            useless if `method` is "min-max"
        std: real number or array_like, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal, if `method` is "z-score";
            std to be divided from the original signal, if `method` is "naive";
            useless if `method` is "min-max"
        per_channel: bool, default False,
            if True, normalization will be done per channel
        inplace: bool, default True,
            if True, normalization will be done inplace (on the signal)

        """
        super().__init__()
        self.method = method.lower()
        assert self.method in [
            "z-score",
            "naive",
            "min-max",
        ]
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.inplace = inplace

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        sig: Tensor,
            the Tensor ECG signal to be normalized,
            of shape (..., n_leads, siglen)

        Returns
        -------
        sig: Tensor,
            the normalized Tensor ECG signal

        """
        sig = normalize_t(
            sig=sig,
            method=self.method,
            mean=self.mean,
            std=self.std,
            per_channel=self.per_channel,
            inplace=self.inplace,
        )
        return sig


class MinMaxNormalize(Normalize):
    r"""
    Min-Max normalization, defined as

        .. math::
            \frac{sig - \min(sig)}{\max(sig) - \min(sig)}

    """

    __name__ = "MinMaxNormalize"

    def __init__(
        self,
        per_channel: bool = False,
    ) -> NoReturn:
        """

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

    def __init__(
        self,
        mean: Union[Real, Iterable[Real]] = 0.0,
        std: Union[Real, Iterable[Real]] = 1.0,
        per_channel: bool = False,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        mean: real number or array_like, default 0.0,
            value(s) to be subtracted
        std: real number or array_like, default 1.0,
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
    r"""
    Z-score normalization via

        .. math::
            \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m

    """

    __name__ = "ZScoreNormalize"

    def __init__(
        self,
        mean: Union[Real, Iterable[Real]] = 0.0,
        std: Union[Real, Iterable[Real]] = 1.0,
        per_channel: bool = False,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        mean: real number or array_like, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal,
        std: real number or array_like, default 1.0,
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
