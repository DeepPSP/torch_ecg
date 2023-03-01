"""
"""

import warnings
from numbers import Real
from typing import Any

import torch

from .._preprocessors.base import preprocess_multi_lead_signal


__all__ = [
    "BaselineRemove",
]


class BaselineRemove(torch.nn.Module):
    """Baseline removal using median filtering.

    Parameters
    ----------
    fs : numbers.Real
        Sampling frequency of the ECG signal to be filtered.
    window1 : float, default 0.2
        The smaller window size of the median filter,
        with units in seconds.
    window2 : float, default 0.6
        The larger window size of the median filter,
        with units in seconds.
    inplace : bool, default True
        Whether to perform the filtering in-place.
    kwargs : dict, optional
        Other keyword arguments for :class:`torch.nn.Module`.

    """

    __name__ = "BaselineRemove"

    def __init__(
        self,
        fs: Real,
        window1: float = 0.2,
        window2: float = 0.6,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.fs = fs
        self.window1 = window1
        self.window2 = window2
        if self.window2 < self.window1:
            self.window1, self.window2 = self.window2, self.window1
            warnings.warn(
                "values of `window1` and `window2` are switched", RuntimeWarning
            )
        self.inplace = inplace

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Apply the preprocessor to the signal tensor.

        Parameters
        ----------
        sig : torch.Tensor
            The ECG signal tensor,
            of shape ``(batch, lead, siglen)``.

        Returns
        -------
        torch.Tensor
            The median filtered (hence baseline removed) ECG signals,
            of shape ``(batch, lead, siglen)``.

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
