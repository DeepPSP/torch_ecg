"""
"""

from typing import NoReturn, Sequence, Iterable, Union, Any
from numbers import Real

import torch


__all__ = [
    "Normalize",
    "MinMaxNormalize",
    "NaiveNormalize",
    "ZScoreNormalize",
    "normalize",
]


class Normalize(torch.nn.Module):
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

    def __init__(self,
                 method:str,
                 mean:Union[Real,Iterable[Real]]=0.0,
                 std:Union[Real,Iterable[Real]]=1.0,
                 per_channel:bool=True,
                 inplace:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        method: str,
            normalization method, case insensitive, can be one of
            "naive", "min-max", "z-score",
        mean: real number or array_like, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal,
            useless if `method` is "min-max"
        std: real number or array_like, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal,
            useless if `method` is "min-max"
        per_channel: bool, default True,
            if True, normalization will be done per channel
        inplace: bool, default False,
            if True, normalization will be done inplace (on the signal)
        """
        super().__init__()
        self.method = method.lower()
        assert self.method in ["z-score", "naive", "min-max",]
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.inplace = inplace

    def forward(self, sig:torch.Tensor) -> torch.Tensor:
        """ finished, checked,

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
        sig = normalize(
            sig=sig,
            method=self.method,
            mean=self.mean, std=self.std,
            per_channel=self.per_channel, inplace=self.inplace,
        )
        return sig


class MinMaxNormalize(Normalize):
    """
    Min-Max normalization, defined as

        .. math::
            \frac{sig - \min(sig)}{\max(sig) - \min(sig)}
    """
    __name__ = "MinMaxNormalize"

    def __init__(self, per_channel:bool=True,) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        per_channel: bool, default True,
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
                 mean:Union[Real,Iterable[Real]]=0.0,
                 std:Union[Real,Iterable[Real]]=1.0,
                 per_channel:bool=True,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: real number or array_like, default 0.0,
            value(s) to be subtracted
        std: real number or array_like, default 1.0,
            value(s) to be divided
        per_channel: bool, default True,
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
                 mean:Union[Real,Iterable[Real]]=0.0,
                 std:Union[Real,Iterable[Real]]=1.0,
                 per_channel:bool=True,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: real number or array_like, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal,
        std: real number or array_like, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal,
        per_channel: bool, default True,
            if True, normalization will be done per channel
        """
        super().__init__(
            method="z-score",
            mean=mean,
            std=std,
            per_channel=per_channel,
        )


def normalize(sig:torch.Tensor,
              method:str,
              mean:Union[Real,Iterable[Real]]=0.0,
              std:Union[Real,Iterable[Real]]=1.0,
              per_channel:bool=True,
              inplace:bool=False,) -> torch.Tensor:
    """ finished, checked,
    
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

    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    mean: real number or array_like, default 0.0,
        mean value of the normalized signal,
        or mean values for each lead of the normalized signal
    std: real number or array_like, default 1.0,
        standard deviation of the normalized signal,
        or standard deviations for each lead of the normalized signal
    per_channel: bool, default False,
        if True, normalization will be done per channel
    inplace: bool, default False,
        if True, normalization will be done inplace (on `sig`)
        
    Returns
    -------
    sig: Tensor,
        the normalized signal
        
    NOTE
    ----
    in cases where normalization is infeasible (std = 0),
    only the mean value will be shifted
    """
    _method = method.lower()
    assert _method in ["z-score", "naive", "min-max",]
    if not inplace:
        sig = sig.clone()
    n_leads, siglen = sig.shape[-2:]
    device = sig.device
    dtype = sig.dtype
    if not per_channel:
        assert isinstance(mean, Real) and isinstance(std, Real), \
            f"mean and std should be real numbers in the non per-channel setting"
    if isinstance(std, Real):
        assert std > 0, "standard deviation should be positive"
        _std = torch.as_tensor([std for _ in range(n_leads)], dtype=dtype, device=device).view((n_leads,1))
    else:
        _std = torch.as_tensor(std, dtype=dtype, device=device).view((n_leads,1))
        assert (_std > 0).all(), "standard deviations should all be positive"
    if isinstance(mean, Real):
        _mean = torch.as_tensor([mean for _ in range(n_leads)], dtype=dtype, device=device).view((n_leads,1))
    else:
        _mean = torch.as_tensor(mean, dtype=dtype, device=device).view((n_leads,1))

    if _method == "naive":
        sig = sig.sub_(_mean).div_(_std)
        return sig

    eps = 1e-7  # to avoid dividing by zero
    if not per_channel:
        options = dict(dim=(-1,-2), keepdims=True)
    else:
        options = dict(dim=-1, keepdims=True)

    if _method == "z-score":
        ori_mean, ori_std = sig.mean(**options), sig.std(**options).add_(eps)
        sig = sig.sub_(ori_mean).div_(ori_std).mul_(_std).add_(_mean)
    elif _method == "min-max":
        ori_min, ori_max = sig.amin(**options), sig.amax(**options)
        sig = sig.sub_(ori_min).div_(ori_max.sub(ori_min).add(eps))
    return sig
