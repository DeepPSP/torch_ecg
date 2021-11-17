"""
"""

from typing import NoReturn, Optional, Any, Tuple, List

import torch
import torch.nn as nn

from .bandpass import BandPass
from .baseline_remove import BaselineRemove
from .normalize import Normalize
from .resample import Resample
from ..utils.misc import default_class_repr


__all__ = ["PreprocManager",]


class PreprocManager(nn.Module):
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

    def __init__(self, *pps:Optional[Tuple[nn.Module,...]], random:bool=False, inplace:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        pps: tuple of `nn.Module`, optional,
            the sequence of preprocessors to be added to the manager
        random: bool, default False,
            whether to apply the augmenters in random order
        inplace: bool, default True,
            whether to apply the preprocessors in-place
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

    def forward(self, sig:torch.Tensor) -> torch.Tensor:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the signal tensor to be preprocessed
        
        Returns
        -------
        sig: Tensor,
            the preprocessed signal tensor
        """
        if len(self.preprocessors) == 0:
            raise ValueError("No preprocessors added to the manager.")
        ordering = list(range(len(self.preprocessors)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        for idx in ordering:
            sig = self.preprocessors[idx](sig)
        return sig

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
        ppm = cls(random=config.get("random", False), inplace=config.get("inplace", True))
        _mapping = {
            "bandpass": ppm._add_bandpass,
            "baseline_remove": ppm._add_baseline_remove,
            "normalize": ppm._add_normalize,
            "resample": ppm._add_resample,
        }
        for pp_name, pp_config in config.items():
            if pp_name in ["random", "inplace", "fs",]:
                continue
            if pp_name in _mapping and isinstance(pp_config, dict):
                _mapping[pp_name](fs=config["fs"], **pp_config)
            else:
                # just ignore the other items
                pass
                # raise ValueError(f"Unknown preprocessor: {pp_name}")
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
    def preprocessors(self) -> List[nn.Module]:
        return self._preprocessors

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """
        return the extra keys for `__repr__`
        """
        return ["random", "preprocessors",]
