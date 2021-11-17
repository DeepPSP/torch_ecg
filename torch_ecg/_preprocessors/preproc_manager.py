"""
"""

from typing import NoReturn, Optional, Any, Tuple, List

import numpy as np

from .base import PreProcessor
from .bandpass import BandPass
from .baseline_remove import BaselineRemove
from .normalize import Normalize
from .resample import Resample
from ..utils.misc import default_class_repr


__all__ = ["PreprocManager",]


class PreprocManager:
    """

    Examples
    --------
    ```python
    import torch
    from easydict import EasyDict as ED
    from torch_ecg._preprocessors import PreprocManager

    config = ED(
        random=False,
        resample={"fs": 500},
        bandpass={},
        normalize={},
    )
    ppm = PreprocManager.from_config(config)
    sig = torch.rand(12,80000).numpy()
    sig, fs = ppm(sig, 200)
    ```
    """
    __name__ = "PreprocManager"

    def __init__(self, *pps:Optional[Tuple[PreProcessor,...]], random:bool=False) -> NoReturn:
        """ finished, checked,

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

    def _add_bandpass(self, **config:dict) -> NoReturn:
        """
        """
        self._preprocessors.append(BandPass(**config))

    def _add_baseline_remove(self, **config:dict) -> NoReturn:
        """
        """
        self._preprocessors.append(BaselineRemove(**config))

    def _add_normalize(self, **config:dict) -> NoReturn:
        """
        """
        self._preprocessors.append(Normalize(**config))

    def _add_resample(self, **config:dict) -> NoReturn:
        """
        """
        self._preprocessors.append(Resample(**config))

    def __call__(self, sig:np.ndarray, fs:int) -> Tuple[np.ndarray, int]:
        """ finished, checked,

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
            raise ValueError("No preprocessors added to the manager.")
        ordering = list(range(len(self.preprocessors)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        pp_sig = sig.copy()
        new_fs = fs
        for idx in ordering:
            pp_sig, new_fs = self.preprocessors[idx](pp_sig, new_fs)
        return pp_sig, new_fs

    @classmethod
    def from_config(cls, config:dict) -> "PreprocManager":
        """ finished, checked,

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
            if pp_name in ["random", "fs",]:
                continue
            if pp_name in _mapping and isinstance(pp_config, dict):
                _mapping[pp_name](**pp_config)
            else:
                raise ValueError(f"Unknown preprocessor: {k}")
        return ppm

    def rearrange(self, new_ordering:List[str]) -> NoReturn:
        """ finished, checked,

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
        self._preprocessors.sort(key=lambda aug: new_ordering.index(_mapping[aug.__name__]))

    @property
    def preprocessors(self) -> List[PreProcessor]:
        return self._preprocessors

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """
        return the extra keys for `__repr__`
        """
        return ["random", "preprocessors",]