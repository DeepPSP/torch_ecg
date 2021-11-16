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
              method:str="z-score",
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
    method: str, default "z-score",
        normalization method, one of "z-score", "min-max", "naive", case insensitive
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
        if True, normalization will be done per channel, not strictly required per channel;
        if False, normalization will be done per sample, strictly required per sample
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

    feasible shapes of `sig` and `std`, `mean` are as follows
    | shape of `sig` | `per_channel` |                shape of `std` or `mean                                     |
    |----------------|---------------|----------------------------------------------------------------------------|
    |    (b,l,s)     |     False     | scalar, (b,), (b,1), (b,1,1)                                               |
    |    (b,l,s)     |     True      | scalar, (b,), (l,), (b,1), (b,l), (l,1), (1,l), (b,1,1), (b,l,1), (1,l,1,) |
    |    (l,s)       |     False     | scalar,                                                                    |
    |    (l,s)       |     True      | scalar, (l,), (l,1), (1,l)                                                 |
    `scalar` includes native scalar or scalar tensor. One can check by
    ```python
    (b,l,s) = 2,12,20
    for shape in [(b,), (l,), (b,1), (b,l), (l,1), (1,l), (b,1,1), (b,l,1), (1,l,1,)]:
        nm_sig = normalize(torch.rand(b,l,s), per_channel=True, mean=torch.rand(*shape))
    for shape in [(b,), (b,1), (b,1,1)]:
        nm_sig = normalize(torch.rand(b,l,s), per_channel=False, mean=torch.rand(*shape))
    for shape in [(l,), (l,1), (1,l)]:
        nm_sig = normalize(torch.rand(l,s), per_channel=True, mean=torch.rand(*shape))
    ```
    """
    _method = method.lower()
    assert _method in ["z-score", "naive", "min-max",]
    ori_shape = sig.shape
    if not inplace:
        sig = sig.clone()
    n_leads, siglen = sig.shape[-2:]
    sig = sig.reshape((-1, n_leads, siglen))  # add batch dim if necessary
    device = sig.device
    dtype = sig.dtype
    if isinstance(std, Real):
        assert std > 0, "standard deviation should be positive"
        _std = torch.full((sig.shape[0], 1, 1), std, dtype=dtype, device=device)
    else:
        _std = torch.as_tensor(std, dtype=dtype, device=device)
        assert (_std > 0).all(), "standard deviations should all be positive"
        if _std.shape[0] == sig.shape[0]:
            # of shape (batch, n_leads, 1) or (batch, 1, 1), or (batch, n_leads,) or (batch, 1) or (batch,)
            _std = _std.view((sig.shape[0], -1, 1))
        elif _std.shape[0] == sig.shape[1] or (_std.shape[:2] == (1, sig.shape[1])):
            # of shape (n_leads, 1) or (n_leads,) or (1, n_leads) or (1, n_leads, 1)
            _std = _std.view((-1, sig.shape[1], 1))
        else:
            raise ValueError(f"shape of `sig` = {sig.shape} and `std` = {_std.shape} mismatch")
    if isinstance(mean, Real):
        _mean = torch.full((sig.shape[0], 1, 1), mean, dtype=dtype, device=device)
    else:
        _mean = torch.as_tensor(mean, dtype=dtype, device=device)
        if _mean.shape[0] == sig.shape[0]:
            # of shape (batch, n_leads, 1) or (batch, 1, 1), or (batch, n_leads,) or (batch, 1) or (batch,)
            _mean = _mean.view((sig.shape[0], -1, 1))
        elif _mean.shape[0] == sig.shape[1] or (_mean.shape[:2] == (1, sig.shape[1])):
            # of shape (n_leads, 1) or (n_leads,) or (1, n_leads) or (1, n_leads, 1)
            _mean = _mean.view((-1, sig.shape[1], 1))
        else:
            raise ValueError("shape of `sig` and `mean` mismatch")

    if not per_channel:
        assert _std.shape[1] == 1 and _mean.shape[1] == 1, \
            f"if `per_channel` is False, `std` and `mean` should be scalars, " \
                "or of shape (batch, 1), or (batch, 1, 1), or (1,)"

    # print(f"sig.shape = {sig.shape}, _mean.shape = {_mean.shape}, _std.shape = {_std.shape}")

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
        sig = sig.sub_(ori_mean).div_(ori_std).mul_(_std).add_(_mean).reshape(ori_shape)
    elif _method == "min-max":
        ori_min, ori_max = sig.amin(**options), sig.amax(**options)
        sig = sig.sub_(ori_min).div_(ori_max.sub(ori_min).add(eps)).reshape(ori_shape)
    return sig


def resample(sig:torch.Tensor,
             fs:Optional[int]=None,
             dst_fs:Optional[int]=None,
             siglen:Optional[int]=None,
             inplace:bool=False,) -> torch.Tensor:
    """ finished, checked,

    resample signal tensors to a new sampling frequency or a new signal length,

    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    fs: int, optional,
        sampling frequency of the source signal to be resampled
    dst_fs: int, optional,
        sampling frequency of the resampled ECG
    siglen: int, optional,
        number of samples in the resampled ECG,
        one of only one of `dst_fs` (with `fs`) and `siglen` should be specified
    inplace: bool, default False,
        if True, normalization will be done inplace (on the signal)
    """
    assert sum([bool(dst_fs), bool(siglen)]) == 1, \
        "one and only one of `fs` and `siglen` should be set"
    if dst_fs is not None:
        assert fs is not None, \
            "if `dst_fs` is set, `fs` should also be set"
        scale_factor = dst_fs / fs
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
