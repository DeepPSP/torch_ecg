""" """

from typing import Any, Optional, Union

import numpy as np
import torch

from .._preprocessors.base import preprocess_multi_lead_signal
from ..utils.utils_signal_t import bandpass_filter
from .registry import PREPROCESSORS

__all__ = [
    "BandPass",
]


@PREPROCESSORS.register(name="bandpass")
@PREPROCESSORS.register()
class BandPass(torch.nn.Module):
    """Bandpass filtering preprocessor.

    Parameters
    ----------
    fs : int
        Sampling frequency of the ECG signal to be filtered.
    lowcut : float, optional
        Low cutoff frequency.
    highcut : float, optional
        High cutoff frequency.
    inplace : bool, default True
        Whether to perform the filtering in-place.
    kwargs : dict, optional
        Other keyword arguments for :class:`torch.nn.Module`.

    """

    __name__ = "BandPass"

    def __init__(
        self,
        fs: int,
        lowcut: Optional[float] = 0.5,
        highcut: Optional[float] = 45,
        inplace: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        assert any([self.lowcut is not None, self.highcut is not None]), "At least one of lowcut and highcut should be set"
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
            The bandpass filtered ECG signal,
            of same shape and type as `sig`.

        """
        if isinstance(sig, torch.Tensor):
            return self._forward_torch(sig)
        else:
            return self._forward_numpy(sig)

    def _forward_torch(self, sig: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            sig = sig.clone()
        return bandpass_filter(
            sig=sig,
            fs=self.fs,
            lowcut=self.lowcut,
            highcut=self.highcut,
        )

    def _forward_numpy(self, sig: np.ndarray) -> np.ndarray:
        # original implementation for numpy arrays
        return preprocess_multi_lead_signal(
            raw_sig=sig,
            fs=self.fs,
            band_fs=[self.lowcut, self.highcut],
        )
