"""
utilities for signal processing on PyTorch tensors
"""

from typing import Union, Iterable, Optional
from numbers import Real

import torch


__all__ = [
    "normalize",
    "resample",
]


def normalize(sig:torch.Tensor,
              method:str,
              mean:Union[Real,Iterable[Real]]=0.0,
              std:Union[Real,Iterable[Real]]=1.0,
              per_channel:bool=False,
              inplace:bool=True,) -> torch.Tensor:
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


def resample(sig:torch.Tensor,
             src_fs:Optional[int]=None,
             fs:Optional[int]=None,
             siglen:Optional[int]=None,
             inplace:bool=False,) -> torch.Tensor:
    """ finished, checked,

    resample signal tensors to a new sampling frequency or a new signal length,

    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    src_fs: int, optional,
        sampling frequency of the source signal to be resampled
    fs: int, optional,
        sampling frequency of the resampled ECG
    siglen: int, optional,
        number of samples in the resampled ECG,
        one of only one of `fs` (with `src_fs`) and `siglen` should be specified
    inplace: bool, default False,
        if True, normalization will be done inplace (on the signal)
    """
    assert sum([bool(fs), bool(siglen)]) == 1, \
        "one and only one of `fs` and `siglen` should be set"
    if fs is not None:
        assert src_fs is not None, \
            "if `fs` is set, `src_fs` should also be set"
        scale_factor = fs / src_fs
    if not inplace:
        sig = sig.clone()
    if sig.ndim == 2:
        sig = torch.nn.functional.interpolate(
            sig.unsqueeze(0),
            size=siglen,
            scale_factor=scale_factor,
            mode="linear",
            align_corners=True,
        ).squeeze(0)
    else:
        sig = torch.nn.functional.interpolate(
            sig,
            size=siglen,
            scale_factor=scale_factor,
            mode="linear",
            align_corners=True,
        )

    return sig
