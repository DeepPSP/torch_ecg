"""
"""

from numbers import Real
from typing import Any, Optional

import torch

from .._preprocessors.base import preprocess_multi_lead_signal


__all__ = [
    "BandPass",
]


class BandPass(torch.nn.Module):
    """Bandpass filtering preprocessor.

    Parameters
    ----------
    fs : numbers.Real
        Sampling frequency of the ECG signal to be filtered.
    lowcut : numbers.Real, optional
        Low cutoff frequency.
    highcut : numbers.Real, optional
        High cutoff frequency.
    inplace : bool, default True
        Whether to perform the filtering in-place.
    kwargs : dict, optional
        Other keyword arguments for :class:`torch.nn.Module`.

    """

    __name__ = "BandPass"

    def __init__(
        self,
        fs: Real,
        lowcut: Optional[Real] = 0.5,
        highcut: Optional[Real] = 45,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        assert any(
            [self.lowcut is not None, self.highcut is not None]
        ), "At least one of lowcut and highcut should be set"
        if not self.lowcut:
            self.lowcut = 0
        if not self.highcut:
            self.highcut = float("inf")
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
            The bandpass filtered ECG signal tensor,
            of shape ``(batch, lead, siglen)``.

        """
        if not self.inplace:
            sig = sig.clone()
        sig = torch.as_tensor(
            preprocess_multi_lead_signal(
                raw_sig=sig.cpu().numpy(),
                fs=self.fs,
                band_fs=[self.lowcut, self.highcut],
            ).copy(),
            dtype=sig.dtype,
            device=sig.device,
        )
        return sig
