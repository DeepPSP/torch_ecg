"""BandPass filter preprocessor."""

from numbers import Real
from typing import Any, List, Literal, Optional, Tuple

import numpy as np

from .base import PreProcessor, preprocess_multi_lead_signal

__all__ = [
    "BandPass",
]


class BandPass(PreProcessor):
    """Bandpass filtering preprocessor.

    Parameters
    ----------
    lowcut : numbers.Real, optional
        Low cutoff frequency
    highcut : numbers.Real, optional
        High cutoff frequency.
    filter_type : {"butter", "fir"}, , default "butter"
        Type of the bandpass filter.
    filter_order : int, optional
        Order of the bandpass filter.
    **kwargs : dict, optional
        Other arguments for :class:`PreProcessor`.

    Examples
    --------
    .. code-block:: python

        from torch_ecg.cfg import DEFAULTS
        sig = DEFAULTS.RNG.randn(1000)
        pp = BandPass(lowcut=0.5, highcut=45, filter_type="butter", filter_order=4)
        sig, _ = pp(sig, 500)

    """

    __name__ = "BandPass"

    def __init__(
        self,
        lowcut: Optional[Real] = 0.5,
        highcut: Optional[Real] = 45,
        filter_type: Literal["butter", "fir"] = "butter",
        filter_order: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.lowcut = lowcut
        self.highcut = highcut
        assert any([self.lowcut is not None, self.highcut is not None]), "At least one of lowcut and highcut should be set"
        if not self.lowcut:
            self.lowcut = 0
        if not self.highcut:
            self.highcut = float("inf")
        self.filter_type = filter_type
        self.filter_order = filter_order

    def apply(self, sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        """Apply the preprocessor to `sig`.

        Parameters
        ----------
        sig : numpy.ndarray
            The ECG signal, can be
                - 1d array, which is a single-lead ECG;
                - 2d array, which is a multi-lead ECG of "lead_first" format;
                - 3d array, which is a tensor of several ECGs, of shape ``(batch, lead, siglen)``.
        fs : int
            Sampling frequency of the ECG signal.

        Returns
        -------
        filtered_sig : :class:`numpy.ndarray`
            Bandpass filtered ECG signal.
        fs : :class:`int`
            Sampling frequency of the filtered ECG signal.

        """
        self._check_sig(sig)
        filtered_sig = preprocess_multi_lead_signal(
            raw_sig=sig,
            fs=fs,
            band_fs=[self.lowcut, self.highcut],
            filter_type=self.filter_type,
            filter_order=self.filter_order,
        )
        return filtered_sig, fs

    def extra_repr_keys(self) -> List[str]:
        return [
            "lowcut",
            "highcut",
            "filter_type",
            "filter_order",
        ] + super().extra_repr_keys()
