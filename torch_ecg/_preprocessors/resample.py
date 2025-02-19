"""Resample the signal into fixed sampling frequency or length."""

from numbers import Real
from typing import Any, List, Optional, Tuple

import numpy as np
import scipy.signal as SS

from ..cfg import DEFAULTS
from .base import PreProcessor

__all__ = [
    "Resample",
]


class Resample(PreProcessor):
    """Resample the signal into fixed sampling frequency or length.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency of the resampled ECG.
    siglen : int, optional
        Number of samples in the resampled ECG.

    NOTE
    ----
    One and only one of `fs` and `siglen` should be set.

    Examples
    --------
    .. code-block:: python

        from torch_ecg.cfg import DEFAULTS
        sig = DEFAULTS.RNG.randn(1000)
        pp = Resample(fs=500)
        sig, _ = pp(sig, 250)

    """

    __name__ = "Resample"

    def __init__(self, fs: Optional[int] = None, siglen: Optional[int] = None, **kwargs: Any) -> None:
        self.fs = fs
        self.siglen = siglen
        assert sum([bool(self.fs), bool(self.siglen)]) == 1, "one and only one of `fs` and `siglen` should be set"

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
        rsmp_sig : :class:`numpy.ndarray`
            The resampled ECG signal.
        new_fs : :class:`int`,
            Sampling frequency of the resampled ECG signal.

        """
        self._check_sig(sig)
        if self.fs is not None:
            rsmp_sig = SS.resample_poly(sig.astype(DEFAULTS.np_dtype), up=self.fs, down=fs, axis=-1)
            new_fs = self.fs
        else:  # self.siglen is not None
            rsmp_sig = SS.resample(sig.astype(DEFAULTS.np_dtype), num=self.siglen, axis=-1)
            new_fs = int(round(self.siglen / sig.shape[-1] * fs))
        return rsmp_sig, new_fs

    def extra_repr_keys(self) -> List[str]:
        return [
            "fs",
            "siglen",
        ] + super().extra_repr_keys()
