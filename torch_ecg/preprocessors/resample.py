"""
"""

from typing import Any, Optional

import torch

from ..utils.utils_signal_t import resample as resample_t

__all__ = [
    "Resample",
]


class Resample(torch.nn.Module):
    """Resample the signal into fixed sampling frequency or length.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency of the source signal to be resampled.
    dst_fs : int, optional
        Sampling frequency of the resampled ECG.
    siglen : int, optional
        Number of samples in the resampled ECG.
    inplace : bool, default False
        Whether to perform the resampling in-place.

    NOTE
    ----
    One and only one of `fs` and `siglen` should be set.
    If `fs` is set, `src_fs` should also be set.


    TODO
    ----
    Consider vectorized :func:`scipy.signal.resample`?

    """

    __name__ = "Resample"

    def __init__(
        self,
        fs: Optional[int] = None,
        dst_fs: Optional[int] = None,
        siglen: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dst_fs = dst_fs
        self.fs = fs
        self.siglen = siglen
        self.inplace = inplace
        assert sum([bool(self.fs), bool(self.siglen)]) == 1, "one and only one of `fs` and `siglen` should be set"
        if self.dst_fs is not None:
            assert self.fs is not None, "if `dst_fs` is set, `fs` should also be set"
            self.scale_factor = self.dst_fs / self.fs

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Apply the resampling to the signal tensor.

        Parameters
        ----------
        sig : torch.Tensor
            The signal tensor to be resampled,
            of shape ``(..., n_leads, siglen)``.

        Returns
        -------
        torch.Tensor
            The resampled signal tensor,
            of shape ``(..., n_leads, siglen)``.

        """
        sig = resample_t(
            sig=sig,
            fs=self.fs,
            dst_fs=self.dst_fs,
            siglen=self.siglen,
            inplace=self.inplace,
        )
        return sig
