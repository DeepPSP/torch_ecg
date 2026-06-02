""" """

import warnings
from typing import Any, Union

import numpy as np
import torch

from .._preprocessors.base import preprocess_multi_lead_signal
from ..utils.utils_signal_t import baseline_removal
from .registry import PREPROCESSORS

__all__ = [
    "BaselineRemove",
]


@PREPROCESSORS.register(name="baseline_remove")
@PREPROCESSORS.register()
class BaselineRemove(torch.nn.Module):
    """Baseline removal using sliding average (median filter alternative).

    Parameters
    ----------
    fs : int
        Sampling frequency of the ECG signal to be filtered.
    window1 : float, default 0.2
        The smaller window size, with units in seconds.
    window2 : float, default 0.6
        The larger window size, with units in seconds.
    inplace : bool, default True
        Whether to perform the filtering in-place.
    kwargs : dict, optional
        Other keyword arguments for :class:`torch.nn.Module`.

    """

    __name__ = "BaselineRemove"

    def __init__(self, fs: int, window1: float = 0.2, window2: float = 0.6, inplace: bool = True, **kwargs: Any) -> None:
        super().__init__()
        self.fs = fs
        self.window1 = window1
        self.window2 = window2
        if self.window2 < self.window1:
            self.window1, self.window2 = self.window2, self.window1
            warnings.warn("values of `window1` and `window2` are switched", RuntimeWarning)
        self.inplace = inplace

    def forward(self, sig: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply the preprocessor to the signal.

        Parameters
        ----------
        sig : numpy.ndarray or torch.Tensor
            The ECG signal,
            of shape ``(batch, lead, siglen)`` or ``(lead, siglen)``.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The baseline removed ECG signals,
            of same shape and type as `sig`.

        """
        if isinstance(sig, torch.Tensor):
            return self._forward_torch(sig)
        else:
            return self._forward_numpy(sig)

    def _forward_torch(self, sig: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            sig = sig.clone()
        return baseline_removal(
            sig=sig,
            fs=self.fs,
            window1=self.window1,
            window2=self.window2,
        )

    def _forward_numpy(self, sig: np.ndarray) -> np.ndarray:
        # original implementation for numpy arrays
        return preprocess_multi_lead_signal(
            raw_sig=sig,
            fs=self.fs,
            bl_win=[self.window1, self.window2],
        )
