"""
"""

from typing import NoReturn, Sequence, Iterable

import torch


__all__ = ["Normalize", "normalize",]


class Normalize(torch.nn.Module):
    """
    """

    def __init__(self,
                 mean:Union[Real,Iterable[Real]]=0.0,
                 std:Union[Real,Iterable[Real]]=1.0,
                 per_channel:bool=True,
                 fixed:bool=True,
                 inplace:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: real number or array_like, default 0.0,
            mean value of the normalized signal,
            or mean values for each lead of the normalized signal
        std: real number or array_like, default 1.0,
            standard deviation of the normalized signal,
            or standard deviations for each lead of the normalized signal
        per_channel: bool, default False,
            if True, normalization will be done per channel
        fixed: bool, default True,
            if True, the normalized signal will have fixed mean (equals to `mean`)
            and fixed standard deviation (equals to `std`),
            otherwise, the signal will be normalized as (signal - mean) / std
        inplace: bool, default False,
            if True, normalization will be done inplace (on the signal)
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.fixed = fixed
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
            mean=self.mean, std=self.std,
            per_channel=self.per_channel, fixed=self.fixed, inplace=self.inplace,
        )
        return sig


def normalize(sig:torch.Tensor,
              mean:Union[Real,Iterable[Real]]=0.0,
              std:Union[Real,Iterable[Real]]=1.0,
              per_channel:bool=True,
              fixed:bool=True,
              inplace:bool=False) -> torch.Tensor:
    """ finished, checked,
    
    perform normalization on `sig`, to make it has fixed mean and standard deviation,
    or normalize `sig` using `mean` and `std` via (sig - mean) / std

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
    fixed: bool, default True,
        if True, the normalized signal will have fixed mean (equals to `mean`)
        and fixed standard deviation (equals to `std`),
        otherwise, the signal will be normalized as (sig - mean) / std
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

    eps = 1e-7  # to avoid dividing by zero

    if fixed:
        if not per_channel:
            options = dict(dim=(-1,-2), keepdims=True)
        else:
            options = dict(dim=-1, keepdims=True)
        ori_mean, ori_std = sig.mean(**options), sig.std(**options).add_(eps)
        sig = sig.sub_(ori_mean).div_(ori_std).mul_(_std).add_(_mean)
    else:
        sig = sig.sub_(_mean).div_(_std)
    return sig
