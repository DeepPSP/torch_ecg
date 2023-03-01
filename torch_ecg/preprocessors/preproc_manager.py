"""
"""

import warnings
from random import sample
from typing import List, Optional, Tuple

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
    """Manager class for preprocessors.

    Parameters
    ----------
    pps : Tuple[torch.nn.Module], optional
        The sequence of preprocessors to be added to the manager.
    random : bool, default False
        Whether to apply the preprocessors in random order.
    inplace : bool, default True
        Whether to apply the preprocessors in-place.

    Examples
    --------
    .. code-block:: python

        import torch
        from torch_ecg.cfg import CFG
        from torch_ecg.preprocessors import PreprocManager

        config = CFG(
            random=False,
            bandpass={"fs":500},
            normalize={"method": "min-max"},
        )
        ppm = PreprocManager.from_config(config)
        sig = torch.randn(2, 12, 8000)
        sig = ppm(sig)

    """

    __name__ = "PreprocManager"

    def __init__(
        self,
        *pps: Optional[Tuple[nn.Module, ...]],
        random: bool = False,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        self.random = random
        self._preprocessors = list(pps)

    def _add_bandpass(self, **config: dict) -> None:
        """Add a bandpass filter to the manager.

        Parameters
        ----------
        config : dict
            The configuration of the bandpass filter.

        Returns
        -------
        None

        """
        self._preprocessors.append(BandPass(**config))

    def _add_baseline_remove(self, **config: dict) -> None:
        """Add a median filter for baseline removal to the manager.

        Parameters
        ----------
        config : dict
            The configuration of the median filter.

        Returns
        -------
        None

        """
        self._preprocessors.append(BaselineRemove(**config))

    def _add_normalize(self, **config: dict) -> None:
        """Add a normalizer to the manager.

        Parameters
        ----------
        config : dict
            The configuration of the normalizer.

        Returns
        -------
        None

        """
        self._preprocessors.append(Normalize(**config))

    def _add_resample(self, **config: dict) -> None:
        """Add a resampler to the manager.

        Parameters
        ----------
        config : dict
            The configuration of the resampler.

        Returns
        -------
        None

        """
        self._preprocessors.append(Resample(**config))

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        """Apply the preprocessors to the signal tensor.

        Parameters
        ----------
        sig : torch.Tensor
            The signal tensor to be preprocessed.

        Returns
        -------
        torch.Tensor
            The preprocessed signal tensor.

        """
        if len(self.preprocessors) == 0:
            # raise ValueError("No preprocessors added to the manager.")
            # allow dummy (empty) preprocessors
            return sig
        ordering = list(range(len(self.preprocessors)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        for idx in ordering:
            sig = self.preprocessors[idx](sig)
        return sig

    @classmethod
    def from_config(cls, config: dict) -> "PreprocManager":
        """Initialize a :class:`PreprocManager` instance from a configuration.

        Parameters
        ----------
        config : dict
            The configuration of the preprocessors,
            better to be an :class:`~collections.OrderedDict`.

        Returns
        -------
        ppm : PreprocManager
            A new instance of :class:`PreprocManager`.

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
        if ppm.empty:
            warnings.warn(
                "No preprocessors added to the manager. You are using a dummy preprocessor.",
                RuntimeWarning,
            )
        return ppm

    def rearrange(self, new_ordering: List[str]) -> None:
        """Rearrange the preprocessors.

        Parameters
        ----------
        new_ordering : List[str]
            The new ordering of the preprocessors.

        Returns
        -------
        None

        """
        if self.random:
            warnings.warn(
                "The preprocessors are applied in random order, "
                "rearranging the preprocessors will not take effect.",
                RuntimeWarning,
            )
        _mapping = {  # built-in preprocessors
            "Resample": "resample",
            "BandPass": "bandpass",
            "BaselineRemove": "baseline_remove",
            "Normalize": "normalize",
        }
        _mapping.update({v: k for k, v in _mapping.items()})
        for k in new_ordering:
            if k not in _mapping:
                # allow custom preprocessors
                assert k in [
                    item.__class__.__name__ for item in self._preprocessors
                ], f"Unknown preprocessor name: `{k}`"
                _mapping.update({k: k})
        assert len(new_ordering) == len(
            set(new_ordering)
        ), "Duplicate preprocessor names."
        assert len(new_ordering) == len(
            self._preprocessors
        ), "Number of preprocessors mismatch."
        self._preprocessors.sort(
            key=lambda item: new_ordering.index(_mapping[item.__class__.__name__])
        )

    def add_(self, pp: nn.Module, pos: int = -1) -> None:
        """Add a (custom) preprocessor to the manager.

        This method is preferred against directly manipulating
        the internal list of preprocessors via
        ``PreprocManager.preprocessors.append(pp)``.

        Parameters
        ----------
        pp : torch.nn.Module
            The preprocessor to be added.
        pos : int, default -1
            The position to insert the preprocessor.
            Should be >= -1, with -1 being the indicator of the end.

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

    @property
    def empty(self) -> bool:
        return len(self.preprocessors) == 0

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "random",
            "preprocessors",
        ]
