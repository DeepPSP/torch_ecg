"""
"""

from typing import Any, NoReturn, Optional

import torch

from ..utils.utils_signal_t import resample as resample_t

__all__ = [
    "Resample",
]


class Resample(torch.nn.Module):
    """
    resample the signal into fixed sampling frequency or length

    TODO: consider vectorized `scipy.signal.resample`?

    """

    __name__ = "Resample"

    def __init__(
        self,
        fs: Optional[int] = None,
        dst_fs: Optional[int] = None,
        siglen: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: int, optional,
            sampling frequency of the source signal to be resampled
        dst_fs: int, optional,
            sampling frequency of the resampled ECG
        siglen: int, optional,
            number of samples in the resampled ECG
        inplace: bool, default False,
            if True, normalization will be done inplace (on the signal)

        NOTE that one and only one of `fs` and `siglen` should be set,
        if `fs` is set, `src_fs` should also be set

        """
        super().__init__()
        self.dst_fs = dst_fs
        self.fs = fs
        self.siglen = siglen
        self.inplace = inplace
        assert (
            sum([bool(self.fs), bool(self.siglen)]) == 1
        ), "one and only one of `fs` and `siglen` should be set"
        if self.dst_fs is not None:
            assert self.fs is not None, "if `dst_fs` is set, `fs` should also be set"
            self.scale_factor = self.dst_fs / self.fs

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """

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
        sig = resample_t(
            sig=sig,
            fs=self.fs,
            dst_fs=self.dst_fs,
            siglen=self.siglen,
            inplace=self.inplace,
        )
        return sig
