""" """

from typing import Any, Optional, Union

import numpy as np
import torch

from ..utils.utils_signal_t import resample as resample_t
from .registry import PREPROCESSORS

__all__ = [
    "Resample",
]


@PREPROCESSORS.register(name="resample")
@PREPROCESSORS.register()
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
        **kwargs: Any,
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

    def forward(self, sig: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply the resampling to the signal.

        Parameters
        ----------
        sig : numpy.ndarray or torch.Tensor
            The signal to be resampled,
            of shape ``(..., n_leads, siglen)``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The resampled signal, of same type as `sig`.

        """
        if isinstance(sig, torch.Tensor):
            return self._forward_torch(sig)
        else:
            return self._forward_numpy(sig)

    def _forward_torch(self, sig: torch.Tensor) -> torch.Tensor:
        return resample_t(
            sig=sig,
            fs=self.fs,
            dst_fs=self.dst_fs,
            siglen=self.siglen,
            inplace=self.inplace,
        )

    def _forward_numpy(self, sig: np.ndarray) -> np.ndarray:
        _sig = torch.as_tensor(sig, dtype=torch.float32)
        _sig = self._forward_torch(_sig)
        return _sig.cpu().numpy().astype(sig.dtype)
