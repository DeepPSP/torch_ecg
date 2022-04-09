"""
"""

from random import sample
from typing import List, NoReturn, Optional, Tuple

import numpy as np

from ..utils.misc import ReprMixin
from .bandpass import BandPass
from .base import PreProcessor
from .baseline_remove import BaselineRemove
from .normalize import Normalize
from .resample import Resample

__all__ = [
    "PreprocManager",
]


class PreprocManager(ReprMixin):
    """

    Examples
    --------
    ```python
    import torch
    from torch_ecg.cfg import CFG
    from torch_ecg._preprocessors import PreprocManager

    config = CFG(
        random=False,
        resample={"fs": 500},
        bandpass={"filter_type": "fir"},
        normalize={"method": "min-max"},
    )
    ppm = PreprocManager.from_config(config)
    sig = torch.rand(12,80000).numpy()
    sig, fs = ppm(sig, 200)
    ```
    """

    __name__ = "PreprocManager"

    def __init__(
        self, *pps: Optional[Tuple[PreProcessor, ...]], random: bool = False
    ) -> NoReturn:
        """

        Parameters
        ----------
        pps: tuple of `PreProcessor`, optional,
            the sequence of preprocessors to be added to the manager
        random: bool, default False,
            whether to apply the augmenters in random order
        """
        super().__init__()
        self.random = random
        self._preprocessors = list(pps)

    def _add_bandpass(self, **config: dict) -> NoReturn:
        """ """
        self._preprocessors.append(BandPass(**config))

    def _add_baseline_remove(self, **config: dict) -> NoReturn:
        """ """
        self._preprocessors.append(BaselineRemove(**config))

    def _add_normalize(self, **config: dict) -> NoReturn:
        """ """
        self._preprocessors.append(Normalize(**config))

    def _add_resample(self, **config: dict) -> NoReturn:
        """ """
        self._preprocessors.append(Resample(**config))

    def __call__(self, sig: np.ndarray, fs: int) -> Tuple[np.ndarray, int]:
        """

        Parameters
        ----------
        sig: np.ndarray,
            the signal to be preprocessed
        fs: int,
            the sampling frequency of the signal

        Returns
        -------
        pp_sig: np.ndarray,
            the preprocessed signal
        new_fs: int,
            the sampling frequency of the preprocessed signal
        """
        if len(self.preprocessors) == 0:
            # raise ValueError("No preprocessors added to the manager.")
            return sig, fs
        ordering = list(range(len(self.preprocessors)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        pp_sig = sig
        new_fs = fs
        for idx in ordering:
            pp_sig, new_fs = self.preprocessors[idx](pp_sig, new_fs)
        return pp_sig, new_fs

    @classmethod
    def from_config(cls, config: dict) -> "PreprocManager":
        """

        Parameters
        ----------
        config: dict,
            the configuration of the preprocessors,
            better to be an `OrderedDict`

        Returns
        -------
        ppm: PreprocManager,
            a new instance of `PreprocManager`
        """
        ppm = cls(random=config.get("random", False))
        _mapping = {
            "bandpass": ppm._add_bandpass,
            "baseline_remove": ppm._add_baseline_remove,
            "normalize": ppm._add_normalize,
            "resample": ppm._add_resample,
        }
        for pp_name, pp_config in config.items():
            if pp_name in [
                "random",
                "fs",
            ]:
                continue
            if pp_name in _mapping and isinstance(pp_config, dict):
                _mapping[pp_name](**pp_config)
            else:
                # just ignore the other items
                pass
                # raise ValueError(f"Unknown preprocessor: {pp_name}")
        return ppm

    def rearrange(self, new_ordering: List[str]) -> NoReturn:
        """

        Parameters
        ----------
        new_ordering: list of str,
            the new ordering of the preprocessors
        """
        _mapping = {
            "Resample": "resample",
            "BandPass": "bandpass",
            "BaselineRemove": "baseline_remove",
            "Normalize": "normalize",
        }
        for k in new_ordering:
            if k not in _mapping:
                _mapping.update({k: k})
        self._preprocessors.sort(
            key=lambda aug: new_ordering.index(_mapping[aug.__name__])
        )

    def add_(self, pp: PreProcessor, pos: int = -1) -> NoReturn:
        """

        add a (custom) preprocessor to the manager,
        this method is preferred against directly manipulating
        the internal list of preprocessors via `PreprocManager.preprocessors.append(pp)`

        Parameters
        ----------
        pp: PreProcessor,
            the preprocessor to be added
        pos: int, default -1,
            the position to insert the preprocessor,
            should be >= -1, with -1 the indicator of the end
        """
        assert isinstance(pp, PreProcessor)
        assert pp.__class__.__name__ not in [
            p.__class__.__name__ for p in self.preprocessors
        ], f"Preprocessor {pp.__class__.__name__} already exists."
        assert (
            isinstance(pos, int) and pos >= -1
        ), f"pos must be an integer >= -1, but got {pos}."
        if pos == -1:
            self._preprocessors.append(pp)
        else:
            self._preprocessors.insert(pos, pp)

    @property
    def preprocessors(self) -> List[PreProcessor]:
        return self._preprocessors

    def extra_repr_keys(self) -> List[str]:
        """
        return the extra keys for `__repr__`
        """
        return super().extra_repr_keys() + [
            "random",
            "preprocessors",
        ]
