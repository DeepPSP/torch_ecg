"""
"""

from numbers import Real
from typing import Any, Iterable, Union

import torch

from ..utils.utils_signal_t import normalize as normalize_t

__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
]


class Normalize(torch.nn.Module):
    """Normalization preprocessor.

    This preprocessor
    performs z-score normalization on `sig`,
    to make it has fixed mean and standard deviation,
    or performs min-max normalization on `sig`,
    or normalizes `sig` using `mean` and `std`
    :math:`via (sig - mean) / std`. More precisely,

    .. math::

        \\begin{align*}
        \\text{Min-Max normalization:} & \\frac{sig - \\min(sig)}{\\max(sig) - \\min(sig)} \\\\
        \\text{Naive normalization:} & \\frac{sig - m}{s} \\\\
        \\text{Z-score normalization:} & \\left(\\frac{sig - mean(sig)}{std(sig)}\\right) \\cdot s + m
        \\end{align*}

    Parameters
    ----------
    method : {"naive", "min-max", "z-score"}, default "z-score",
        Normalization method, by default "z-score", case-insensitive.
    mean : numbers.Real or array_like, default 0.0
        If `method` is "z-score", then `mean is the mean value
        of the normalized signal,
        or mean values for each lead of the normalized signal.
        If `method` is "naive", then `mean` is the mean value
        to be subtracted from the original signal.
        Useless if `method` is ``"min-max"``.
    std : numbers.Real or array_like, default 1.0
        If `method` is "z-score", then `std` is the standard deviation
        of the normalized signal,
        or standard deviations for each lead of the normalized signal.
        If `method` is "naive", then `std` is the standard deviation
        to be divided from the original signal.
        Useless if `method` is ``"min-max"``.
    per_channel : bool, default False
        Whether to perform the normalization per channel.
    inplace : bool, default True
        Whether to perform the normalization in-place.

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
    ) -> None:
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
        """Apply the preprocessor to the signal tensor.

        Parameters
        ----------
        sig : torch.Tensor
            The input signal tensor,
            of shape ``(batch_size, num_leads, num_samples)``.

        Returns
        -------
        torch.Tensor
            The normalized signal tensor.

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
    """Min-Max normalization.

    Min-Max normalization is defined as

    .. math::

        \\frac{sig - \\min(sig)}{\\max(sig) - \\min(sig)}

    Parameters
    ----------
    per_channel : bool, default False
        Whether to perform the normalization per channel.
    inplace : bool, default True
        Whether to perform the normalization in-place.

    """

    __name__ = "MinMaxNormalize"

    def __init__(
        self, per_channel: bool = False, inplace: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(
            method="min-max", per_channel=per_channel, inplace=inplace, **kwargs
        )


class NaiveNormalize(Normalize):
    """Naive normalization

    Naive normalization is done via

    .. math::

        \\frac{sig - mean}{std}

    Parameters
    ----------
    mean : numbers.Real or array_like, default 0.0
        Value(s) to be subtracted.
    std : numbers.Real or array_like, default 1.0
        Value(s) to be divided.
    per_channel : bool, default False
        Whether to perform the normalization per channel.
    inplace : bool, default True
        Whether to perform the normalization in-place.

    """

    __name__ = "NaiveNormalize"

    def __init__(
        self,
        mean: Union[Real, Iterable[Real]] = 0.0,
        std: Union[Real, Iterable[Real]] = 1.0,
        per_channel: bool = False,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            method="naive",
            mean=mean,
            std=std,
            per_channel=per_channel,
            inplace=inplace,
        )


class ZScoreNormalize(Normalize):
    """Z-score normalization.

    Z-score normalization is defined as

    .. math::

        \\left(\\frac{sig - mean(sig)}{std(sig)}\\right) \\cdot s + m

    Parameters
    ----------
    mean : numbers.Real or array_like, default 0.0
        Mean value of the normalized signal,
        or mean values for each lead of the normalized signal.
    std : numbers.Real or array_like, default 1.0
        Standard deviation of the normalized signal,
        or standard deviations for each lead of the normalized signal.
    per_channel : bool, default False
        Whether to perform the normalization per channel.
    inplace : bool, default True
        Whether to perform the normalization in-place.

    """

    __name__ = "ZScoreNormalize"

    def __init__(
        self,
        mean: Union[Real, Iterable[Real]] = 0.0,
        std: Union[Real, Iterable[Real]] = 1.0,
        per_channel: bool = False,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            method="z-score",
            mean=mean,
            std=std,
            per_channel=per_channel,
            inplace=inplace,
            **kwargs
        )
