"""
"""

from numbers import Real
from typing import Any, List, NoReturn, Optional, Tuple

import numpy as np

from .base import PreProcessor, preprocess_multi_lead_signal

__all__ = [
    "BandPass",
]


class BandPass(PreProcessor):
    """ """

    __name__ = "BandPass"

    def __init__(
        self,
        lowcut: Optional[Real] = 0.5,
        highcut: Optional[Real] = 45,
        filter_type: str = "butter",
        filter_order: Optional[int] = None,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        lowcut: real number, optional,
            low cutoff frequency
        highcut: real number, optional,
            high cutoff frequency
        filter_type: str, default "butter",
            type of the bandpass filter, can be "butter" or "fir"
        filter_order: int, optional,
            order of the bandpass filter,
        """
        self.lowcut = lowcut
        self.highcut = highcut
        assert any(
            [self.lowcut is not None, self.highcut is not None]
        ), "At least one of lowcut and highcut should be set"
        if not self.lowcut:
            self.lowcut = 0
        if not self.highcut:
            self.highcut = float("inf")
        self.filter_type = filter_type
        self.filter_order = filter_order

    def apply(self, sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        """

        apply the preprocessor to `sig`

        Parameters
        ----------
        sig: ndarray,
            the ECG signal, can be
            1d array, which is a single-lead ECG
            2d array, which is a multi-lead ECG of "lead_first" format
            3d array, which is a tensor of several ECGs, of shape (batch, lead, siglen)
        fs: int,
            sampling frequency of the ECG signal

        Returns
        -------
        filtered_sig: ndarray,
            the bandpass filtered ECG signal
        fs: int,
            the sampling frequency of the filtered ECG signal
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
        """
        return the extra keys for `__repr__`
        """
        return [
            "lowcut",
            "highcut",
            "filter_type",
            "filter_order",
        ] + super().extra_repr_keys()
