"""Manger for the augmenters"""

import warnings
from random import sample
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..utils.misc import add_docstring, default_class_repr
from .base import Augmenter, _augmenter_forward_doc
from .registry import AUGMENTERS

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
        self._augmenters = torch.nn.ModuleList(list(augs))

    def add_(self, aug: Union[Augmenter, str, dict], pos: int = -1, **kwargs: Any) -> None:
        """Add an augmenter to the manager.

        Parameters
        ----------
        aug : Augmenter or str or dict
            The augmenter to be added.
            If it's a string or a dict, it will be built using the registry.
        pos : int, default -1
            The position to insert the augmenter.
            Should be >= -1, with -1 being the indicator of the end.
        **kwargs : Any
            Additional keyword arguments for building the augmenter.

        """
        if isinstance(aug, (str, dict)):
            aug = AUGMENTERS.build(aug, **kwargs)
        assert isinstance(aug, Augmenter)
        assert aug.__class__.__name__ not in [
            a.__class__.__name__ for a in self.augmenters
        ], f"Augmenter {aug.__class__.__name__} already exists."
        assert isinstance(pos, int) and pos >= -1, f"pos must be an integer >= -1, but got {pos}."
        if pos == -1:
            self._augmenters.append(aug)
        else:
            self._augmenters.insert(pos, aug)

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
    def augmenters(self) -> torch.nn.ModuleList:
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
        for aug_name, aug_config in config.items():
            if aug_name in [
                "fs",
                "random",
            ]:
                continue
            if aug_name in AUGMENTERS or aug_name in [
                "".join([w.capitalize() for w in k.split("_")]) for k in AUGMENTERS.list_all()
            ]:
                if isinstance(aug_config, dict):
                    # add default fs from config if not specified in aug_config
                    if "fs" not in aug_config and "fs" in config:
                        aug_config["fs"] = config["fs"]
                    am.add_(aug_name, **aug_config)
                else:
                    am.add_(aug_name)
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
        # TODO: use a more robust way to map names to classes
        _mapping = {  # built-in augmenters
            "".join([w.capitalize() for w in k.split("_")]): k
            for k in "label_smooth,mixup,random_flip,random_masking,random_renormalize,stretch_compress,cutmix".split(",")
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

        augs = list(self._augmenters)
        augs.sort(key=lambda aug: new_ordering.index(_mapping[aug.__class__.__name__]))
        self._augmenters = torch.nn.ModuleList(augs)
