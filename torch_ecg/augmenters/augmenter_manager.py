"""Manger for the augmenters"""

import warnings
from random import sample
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..utils.misc import add_docstring, default_class_repr
from .base import Augmenter, _augmenter_forward_doc
from .baseline_wander import BaselineWanderAugmenter
from .label_smooth import LabelSmooth
from .mixup import Mixup
from .random_flip import RandomFlip
from .random_masking import RandomMasking
from .random_renormalize import RandomRenormalize
from .stretch_compress import StretchCompress

__all__ = [
    "AugmenterManager",
]


class AugmenterManager(torch.nn.Module):
    """The :class:`~torch.nn.Module` to manage the augmenters.

    Parameters
    ----------
    aug : Tuple[Augmenter], optional
        The augmenters to be added to the manager.
    random : bool, default False
        Whether to apply the augmenters in random order.

    Examples
    --------
    .. code-block:: python

        import torch
        from torch_ecg.cfg import CFG
        from torch_ecg.augmenters import AugmenterManager

        config = CFG(
            random=False,
            fs=500,
            baseline_wander={},
            label_smooth={},
            mixup={},
            random_flip={},
            random_masking={},
            random_renormalize={},
            stretch_compress={},
        )
        am = AugmenterManager.from_config(config)
        sig = torch.randn(32, 12, 5000)
        label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
        mask1 = torch.randint(0, 2, (32, 5000, 3), dtype=torch.float32)
        mask2 = torch.randint(0, 3, (32, 5000), dtype=torch.long)
        sig, label, mask1, mask2 = am(sig, label, mask1, mask2)

    """

    __name__ = "AugmenterManager"

    def __init__(self, *augs: Optional[Tuple[Augmenter, ...]], random: bool = False) -> None:
        super().__init__()
        self.random = random
        self._augmenters = list(augs)

    def _add_baseline_wander(self, **config: dict) -> None:
        """Add the baseline wander augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the baseline wander augmenter.

        """
        self._augmenters.append(BaselineWanderAugmenter(**config))

    def _add_label_smooth(self, **config: dict) -> None:
        """Add the label smooth augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the label smooth augmenter.

        """
        self._augmenters.append(LabelSmooth(**config))

    def _add_mixup(self, **config: dict) -> None:
        """Add the mixup augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the mixup augmenter.

        """
        self._augmenters.append(Mixup(**config))

    def _add_random_flip(self, **config: dict) -> None:
        """Add the random flip augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the random flip augmenter.

        """
        self._augmenters.append(RandomFlip(**config))

    def _add_random_masking(self, **config: dict) -> None:
        """Add the random masking augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the random masking augmenter.

        """
        self._augmenters.append(RandomMasking(**config))

    def _add_random_renormalize(self, **config: dict) -> None:
        """Add the random renormalize augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the random renormalize augmenter.

        """
        self._augmenters.append(RandomRenormalize(**config))

    def _add_stretch_compress(self, **config: dict) -> None:
        """Add the stretch compress augmenter to the manager.

        Parameters
        ----------
        **config : dict
            The configuration for the stretch compress augmenter.

        """
        self._augmenters.append(StretchCompress(**config))

    @add_docstring(
        _augmenter_forward_doc.replace(
            "Forward method of the augmenter.",
            "Forward the input ECGs through the augmenters.",
        )
    )
    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor]]:
        if len(self.augmenters) == 0:
            # raise ValueError("No augmenters added to the manager.")
            return (sig, label, *extra_tensors)
        ordering = list(range(len(self.augmenters)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        for idx in ordering:
            sig, label, *extra_tensors = self.augmenters[idx](sig, label, *extra_tensors, **kwargs)
        return (sig, label, *extra_tensors)

    @property
    def augmenters(self) -> List[Augmenter]:
        """The list of augmenters in the manager."""
        return self._augmenters

    def extra_repr(self) -> str:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        indent = 4 * " "
        s = (
            f"augmenters = [\n{indent}"
            + f",\n{2*indent}".join(default_class_repr(aug, depth=2) for aug in self.augmenters)
            + f"{indent}\n]"
        )
        return s

    @classmethod
    def from_config(cls, config: dict) -> "AugmenterManager":
        """Create an :class:`AugmenterManager` from a configuration.

        Parameters
        ----------
        config : dict
            The configuration of the augmenters,
            better to be an :class:`~collections.OrderedDict`.

        Returns
        -------
        am : :class:`AugmenterManager`
            A new instance of :class:`AugmenterManager`.

        """
        am = cls(random=config.get("random", False))
        _mapping = {
            "baseline_wander": am._add_baseline_wander,
            "label_smooth": am._add_label_smooth,
            "mixup": am._add_mixup,
            "random_flip": am._add_random_flip,
            "random_masking": am._add_random_masking,
            "random_renormalize": am._add_random_renormalize,
            "stretch_compress": am._add_stretch_compress,
        }
        for aug_name, aug_config in config.items():
            if aug_name in [
                "fs",
                "random",
            ]:
                continue
            elif aug_name in _mapping and isinstance(aug_config, dict):
                _mapping[aug_name](fs=config["fs"], **aug_config)
            else:
                # just ignore the other items
                pass
                # raise ValueError(f"Unknown augmenter name: {aug_name}")
        return am

    def rearrange(self, new_ordering: List[str]) -> None:
        """Rearrange the augmenters in the manager.

        Parameters
        ----------
        new_ordering : List[str]
            The list of augmenter names in the new order.

        """
        if self.random:
            warnings.warn(
                "The augmenters are applied in random order, " "rearranging the augmenters will not take effect.",
                RuntimeWarning,
            )
        _mapping = {  # built-in augmenters
            "".join([w.capitalize() for w in k.split("_")]): k
            for k in "label_smooth,mixup,random_flip,random_masking,random_renormalize,stretch_compress".split(",")
        }
        _mapping.update({"BaselineWanderAugmenter": "baseline_wander"})
        _mapping.update({v: k for k, v in _mapping.items()})
        for k in new_ordering:
            if k not in _mapping:
                # allow custom augmenters
                assert k in [am.__class__.__name__ for am in self._augmenters], f"Unknown augmenter name: `{k}`"
                _mapping.update({k: k})
        assert len(new_ordering) == len(set(new_ordering)), "Duplicate augmenter names."
        assert len(new_ordering) == len(self._augmenters), "Number of augmenters mismatch."

        self._augmenters.sort(key=lambda aug: new_ordering.index(_mapping[aug.__class__.__name__]))
