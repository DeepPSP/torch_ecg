"""Normalization of the signals."""

from numbers import Real
from typing import Any, List, Tuple, Union

import numpy as np

from ..utils.utils_signal import normalize
from .base import PreProcessor

__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
]


class Normalize(PreProcessor):
    r"""
    Perform z-score normalization on ``sig``,
    to make it has fixed mean and standard deviation;
    or perform min-max normalization on ``sig``,
    or normalize ``sig`` using ``mean`` and ``std`` via
    :math:`(sig - mean) / std`.
    More precisely,

    .. math::

        \begin{align*}
        \text{Min-Max normalization:} & \quad \frac{sig - \min(sig)}{\max(sig) - \min(sig)} \\
        \text{Naive normalization:} & \quad \frac{sig - m}{s} \\
        \text{Z-score normalization:} & \quad \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m
        \end{align*}

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> sig = DEFAULTS.RNG.randn(1000)
    >>> pp = Normalize(method="z-score", mean=0.0, std=1.0)
    >>> sig, _ = pp(sig, 500)

    """

    __name__ = "Normalize"

    def __init__(
        self,
        method: str = "z-score",
        mean: Union[Real, np.ndarray] = 0.0,
        std: Union[Real, np.ndarray] = 1.0,
        per_channel: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Normalize preprocessor.

        Parameters
        ----------
        method : str, default "z-score"
            Normalization method, case insensitive, can be one of
            "naive", "min-max", "z-score".
        mean : real number or numpy.ndarray, default 0.0
            Mean value of the normalized signal,
            or mean values for each lead of the normalized signal.
            Useless if `method` is "min-max".
        std : real number or numpy.ndarray, default 1.0
            Standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal.
            Useless if `method` is "min-max".
        per_channel : bool, default False
            If True, normalization will be done per channel.

        """
        self.method = method.lower()
        assert self.method in [
            "z-score",
            "naive",
            "min-max",
        ]
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        if isinstance(std, Real):
            assert std > 0, "standard deviation should be positive"
        else:
            assert (std > 0).all(), "standard deviations should all be positive"
        if not per_channel:
            assert isinstance(mean, Real) and isinstance(
                std, Real
            ), "mean and std should be real numbers in the non per-channel setting"

    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        """
        Apply the preprocessor to ``sig``.

        Parameters
        ----------
        sig : numpy.ndarray
            The ECG signal, can be
            1d array, which is a single-lead ECG;
            2d array, which is a multi-lead ECG of "lead_first" format;
            3d array, which is a tensor of several ECGs, of shape (batch, lead, siglen).
        fs : real number
            Sampling frequency of the ECG signal.
            **NOT** used currently.

        Returns
        -------
        normalized_sig : numpy.ndarray,
            The normalized ECG signal
        fs : int,
            The sampling frequency of the normalized ECG signal

        """
        self._check_sig(sig)
        normalized_sig = normalize(
            sig=sig,
            method=self.method,
            mean=self.mean,
            std=self.std,
            sig_fmt="channel_first",
            per_channel=self.per_channel,
        )
        return normalized_sig, fs

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "method",
            "mean",
            "std",
            "per_channel",
        ] + super().extra_repr_keys()


class MinMaxNormalize(Normalize):
    r"""
    Min-Max normalization, defined as

    .. math::

        \frac{sig - \min(sig)}{\max(sig) - \min(sig)}

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> sig = DEFAULTS.RNG.randn(1000)
    >>> pp = MinMaxNormalize()
    >>> sig, _ = pp(sig, 500)

    """

    __name__ = "MinMaxNormalize"

    def __init__(
        self,
        per_channel: bool = False,
    ) -> None:
        """Initialize the MinMaxNormalize preprocessor.

        Parameters
        ----------
        per_channel : bool, default False
            If True, normalization will be done per channel

        """
        super().__init__(method="min-max", per_channel=per_channel)

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "per_channel",
        ] + super().extra_repr_keys()


class NaiveNormalize(Normalize):
    r"""
    Naive normalization via

    .. math::

        \frac{sig - m}{s}

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> sig = DEFAULTS.RNG.randn(1000)
    >>> pp = NaiveNormalize()
    >>> sig, _ = pp(sig, 500)

    """

    __name__ = "NaiveNormalize"

    def __init__(
        self,
        mean: Union[Real, np.ndarray] = 0.0,
        std: Union[Real, np.ndarray] = 1.0,
        per_channel: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the NaiveNormalize preprocessor.

        Parameters
        ----------
        mean : real number or numpy.ndarray, default 0.0
            Value(s) to be subtracted.
        std : real number or numpy.ndarray, default 1.0
            Value(s) to be divided.
        per_channel : bool, default False
            If True, normalization will be done per channel.

        """
        super().__init__(
            method="naive",
            mean=mean,
            std=std,
            per_channel=per_channel,
        )

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "mean",
            "std",
            "per_channel",
        ] + super().extra_repr_keys()


class ZScoreNormalize(Normalize):
    r"""
    Z-score normalization via

    .. math::

        \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> sig = DEFAULTS.RNG.randn(1000)
    >>> pp = ZScoreNormalize()
    >>> sig, _ = pp(sig, 500)

    """

    __name__ = "ZScoreNormalize"

    def __init__(
        self,
        mean: Union[Real, np.ndarray] = 0.0,
        std: Union[Real, np.ndarray] = 1.0,
        per_channel: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the ZScoreNormalize preprocessor.

        Parameters
        ----------
        mean : real number or numpy.ndarray, default 0.0
            Mean value of the normalized signal,
            or mean values for each lead of the normalized signal.
        std : real number or numpy.ndarray, default 1.0
            Standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal.
        per_channel : bool, default False
            If True, normalization will be done per channel.

        """
        super().__init__(
            method="z-score",
            mean=mean,
            std=std,
            per_channel=per_channel,
        )

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "mean",
            "std",
            "per_channel",
        ] + super().extra_repr_keys()
