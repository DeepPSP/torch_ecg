"""
"""

from numbers import Real
from typing import Any, NoReturn, Optional

import torch

from .._preprocessors.base import preprocess_multi_lead_signal

__all__ = [
    "BandPass",
]


class BandPass(torch.nn.Module):
    """ """

    __name__ = "BandPass"

    def __init__(
        self,
        fs: Real,
        lowcut: Optional[Real] = 0.5,
        highcut: Optional[Real] = 45,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: real number,
            sampling frequency of the ECG signal to be filtered
        lowcut: real number, optional,
            low cutoff frequency
        highcut: real number, optional,
            high cutoff frequency
        inplace: bool, default True,
            if True, the preprocessor will modify the input signal
        kwargs: keyword arguments,

        """
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
        """

        apply the preprocessor to `sig`

        Parameters
        ----------
        sig: Tensor,
            the ECG signals, of shape (batch, lead, siglen)

        Returns
        -------
        filtered_sig: Tensor,
            the bandpass filtered ECG signals, of shape (batch, lead, siglen)

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
