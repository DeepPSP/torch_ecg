"""
"""

import warnings
from numbers import Real
from typing import Any, NoReturn

import torch

from .._preprocessors.base import preprocess_multi_lead_signal

__all__ = [
    "BaselineRemove",
]


class BaselineRemove(torch.nn.Module):
    """ """

    __name__ = "BaselineRemove"

    def __init__(
        self,
        fs: Real,
        window1: float = 0.2,
        window2: float = 0.6,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: real number,
            sampling frequency of the ECG signal to be filtered
        window1: float, default 0.2,
            the smaller window size of the median filter, with units in seconds
        highcut: float, default 0.6,
            the larger window size of the median filter, with units in seconds
        inplace: bool, default True,
            if True, the preprocessor will modify the input signal
        kwargs: keyword arguments,

        """
        super().__init__()
        self.fs = fs
        self.window1 = window1
        self.window2 = window2
        if self.window2 < self.window1:
            self.window1, self.window2 = self.window2, self.window1
            warnings.warn("values of window1 and window2 are switched")
        self.inplace = inplace

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """

        apply the preprocessor to `sig`

        Parameters
        ----------
        sig: Tensor,
            the ECG signals, of shape (batch, lead, siglen)

        Returns
        -------
        sig: Tensor,
            the median filtered (hence baseline removed) ECG signals,
            of shape (batch, lead, siglen)

        """
        if not self.inplace:
            sig = sig.clone()
        sig = torch.as_tensor(
            preprocess_multi_lead_signal(
                raw_sig=sig.cpu().numpy(),
                fs=self.fs,
                bl_win=[self.window1, self.window2],
            ).copy(),
            dtype=sig.dtype,
            device=sig.device,
        )
        return sig
