"""
Use median filter to remove baseline.
Note that highpass filters also have the effect of baseline removal
"""

import warnings
from numbers import Real
from typing import Any, List, Tuple

import numpy as np

from .base import PreProcessor, preprocess_multi_lead_signal


__all__ = [
    "BaselineRemove",
]


class BaselineRemove(PreProcessor):
    """Baseline removal using median filter.

    Parameters
    ----------
    window1 : float, default 0.2
        The smaller window size of the median filter, with units in seconds.
    highcut : float, default 0.6
        The larger window size of the median filter, with units in seconds.

    Examples
    --------
    .. code-block:: python

        from torch_ecg.cfg import DEFAULTS
        sig = DEFAULTS.RNG.randn(1000)
        pp = BaselineRemove(window1=0.2, window2=0.6)
        sig, _ = pp(sig, 500)

    """

    __name__ = "BaselineRemove"

    def __init__(
        self, window1: float = 0.2, window2: float = 0.6, **kwargs: Any
    ) -> None:
        self.window1 = window1
        self.window2 = window2
        if self.window2 < self.window1:
            self.window1, self.window2 = self.window2, self.window1
            warnings.warn(
                "values of `window1` and `window2` are switched", RuntimeWarning
            )

    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        """Apply the preprocessor to `sig`.

        Parameters
        ----------
        sig : numpy.ndarray
            The ECG signal, can be
                - 1d array, which is a single-lead ECG;
                - 2d array, which is a multi-lead ECG of "lead_first" format;
                - 3d array, which is a tensor of several ECGs, of shape ``(batch, lead, siglen)``.
        fs : numbers.Real
            Sampling frequency of the ECG signal.

        Returns
        -------
        filtered_sig : :class:`numpy.ndarray`
            The median filtered (hence baseline removed) ECG signal.
        fs : :class:`int`
            Sampling frequency of the filtered ECG signal.

        """
        self._check_sig(sig)
        filtered_sig = preprocess_multi_lead_signal(
            raw_sig=sig,
            fs=fs,
            bl_win=[self.window1, self.window2],
        )
        return filtered_sig, fs

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return ["window1", "window2"] + super().extra_repr_keys()
