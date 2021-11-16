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

    def __init__(self, random:bool=False) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        random: bool, default False,
            whether to apply the augmenters in random order
        """
        super().__init__()
        self.random = random
        self._preprocessors = []

    def _add_bandpass(self, **config:Any) -> NoReturn:
        """
        """
        self._preprocessors.append(BandPass(**config))

    def _add_baseline_remove(self, **config:Any) -> NoReturn:
        """
        """
        self._preprocessors.append(BaselineRemove(**config))

    def _add_normalize(self, **config:Any) -> NoReturn:
        """
        """
        self._preprocessors.append(Normalize(**config))

    def _add_resample(self, **config:Any) -> NoReturn:
        """
        """
        self._preprocessors.append(Resample(**config))

    def __call__(self, sig:np.ndarray, fs:int) -> Tuple[np.ndarray, int]:
        """
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
        """
        """
        ppm = cls(random=config.get("random", False))
        if "resample" in config:
            ppm._add_resample(**config["resample"])
        if "bandpass" in config:
            ppm._add_bandpass(**config["bandpass"])
        if "baseline_remove" in config:
            ppm._add_baseline_remove(**config["baseline_remove"])
        if "normalize" in config:
            ppm._add_normalize(**config["normalize"])
        return ppm

    def rearrange(self, new_ordering:List[str]) -> NoReturn:
        """
        """
        _mapping = {
            "Resample": "resample",
            "BandPass": "bandpass",
            "BaselineRemove": "baseline_remove",
            "Normalize": "normalize",
        }
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
