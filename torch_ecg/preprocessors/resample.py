"""
"""

from typing import NoReturn, Optional, Any

import torch


__all__ = ["Resample", "resample",]


class Resample(torch.nn.Module):
    """
    resample the signal into fixed sampling frequency or length

    TODO: consider vectorized `scipy.signal.resample`?
    """

    def __init__(self,
                 src_fs:Optional[int]=None,
                 fs:Optional[int]=None,
                 siglen:Optional[int]=None,
                 inplace:bool=False,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        src_fs: int, optional,
            sampling frequency of the source signal to be resampled
        fs: int, optional,
            sampling frequency of the resampled ECG
        siglen: int, optional,
            number of samples in the resampled ECG
        inplace: bool, default False,
            if True, normalization will be done inplace (on the signal)

        NOTE that one and only one of `fs` and `siglen` should be set,
        if `fs` is set, `src_fs` should also be set
        """
        super().__init__()
        self.src_fs = src_fs
        self.fs = fs
        self.siglen = siglen
        self.inplace = inplace
        assert sum([bool(self.fs), bool(self.siglen)]) == 1, \
            "one and only one of `fs` and `siglen` should be set"
        if self.fs is not None:
            assert self.src_fs is not None, \
                "if `fs` is set, `src_fs` should also be set"
            self.scale_factor = self.fs / self.src_fs

    def forward(self, sig:torch.Tensor) -> torch.Tensor:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the Tensor ECG signal to be resampled,
            of shape (..., n_leads, siglen)

        Returns
        -------
        sig: Tensor,
            the resampled Tensor ECG signal
        """
        sig = resample(
            sig=sig,
            src_fs=self.src_fs,
            fs=self.fs,
            siglen=self.siglen,
            inplace=self.inplace,
        )
        return sig


def resample(sig:torch.Tensor,
             src_fs:Optional[int]=None,
             fs:Optional[int]=None,
             siglen:Optional[int]=None,
             inplace:bool=False,) -> torch.Tensor:
    """
    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    src_fs: int, optional,
        sampling frequency of the source signal to be resampled
    fs: int, optional,
        sampling frequency of the resampled ECG
    siglen: int, optional,
        number of samples in the resampled ECG
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
