"""
"""

from random import sample
from typing import List, NoReturn, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.misc import ReprMixin
from .bandpass import BandPass
from .baseline_remove import BaselineRemove
from .normalize import Normalize
from .resample import Resample

__all__ = [
    "PreprocManager",
]


class PreprocManager(ReprMixin, nn.Module):
    """

    Examples
    --------
    ```python
    import torch
    from torch_ecg.cfg import CFG
    from torch_ecg.preprocessors import PreprocManager

    config = CFG(
        random=False,
        bandpass={"fs":500},
        normalize={"method": "min-max"},
    )
    ppm = PreprocManager.from_config(config)
    sig = torch.rand(2,12,8000)
    sig = ppm(sig)
    ```

    """

    __name__ = "PreprocManager"

    def __init__(
        self,
        *pps: Optional[Tuple[nn.Module, ...]],
        random: bool = False,
        inplace: bool = True,
    ) -> NoReturn:
        """

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

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """

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
        ppm = cls(
            random=config.get("random", False), inplace=config.get("inplace", True)
        )
        _mapping = {
            "bandpass": ppm._add_bandpass,
            "baseline_remove": ppm._add_baseline_remove,
            "normalize": ppm._add_normalize,
            "resample": ppm._add_resample,
        }
        for pp_name, pp_config in config.items():
            if pp_name in [
                "random",
                "inplace",
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

    def add_(self, pp: nn.Module, pos: int = -1) -> NoReturn:
        """

        add a (custom) preprocessor to the manager,
        this method is preferred against directly manipulating
        the internal list of preprocessors via `PreprocManager.preprocessors.append(pp)`

        Parameters
        ----------
        pp: Module,
            the preprocessor to be added
        pos: int, default -1,
            the position to insert the preprocessor,
            should be >= -1, with -1 the indicator of the end

        """
        assert isinstance(pp, nn.Module)
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
    def preprocessors(self) -> List[nn.Module]:
        return self._preprocessors

    def extra_repr_keys(self) -> List[str]:
        """
        return the extra keys for `__repr__`

        """
        return super().extra_repr_keys() + [
            "random",
            "preprocessors",
        ]
