"""
"""

from random import sample
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Any, Sequence, NoReturn

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter
from .baseline_wander import BaselineWanderAugmenter
from .label_smooth import LabelSmooth
from .mixup import Mixup
from .random_flip import RandomFlip
from .random_masking import RandomMasking
from .random_renormalize import RandomRenormalize
from .stretch_compress import StretchCompress
from ..utils.misc import default_class_repr


__all__ = ["AugmenterManager",]


class AugmenterManager(torch.nn.Module):
    """

    The `Module` to manage the augmenters

    Examples
    --------
    ```python
    import torch
    from easydict import EasyDict as ED
    from torch_ecg.augmenters import AugmenterManager

    config = ED(
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
    sig, label, mask = torch.rand(2,12,5000), torch.rand(2,26), torch.rand(2,5000,1)
    sig, label, mask = am(sig, label, mask)
    ```
    """
    __name__ = "AugmenterManager"

    def __init__(self, random:bool=False) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        random: bool, default False,
            whether to apply the augmenters in random order
        """
        super().__init__()
        self.random = random
        self._augmenters = []

    def _add_baseline_wander(self, **config:Any) -> NoReturn:
        self._augmenters.append(BaselineWanderAugmenter(**config))

    def _add_label_smooth(self, **config:Any) -> NoReturn:
        self._augmenters.append(LabelSmooth(**config))

    def _add_mixup(self, **config:Any) -> NoReturn:
        self._augmenters.append(Mixup(**config))

    def _add_random_flip(self, **config:Any) -> NoReturn:
        self._augmenters.append(RandomFlip(**config))

    def _add_random_masking(self, **config:Any) -> NoReturn:
        self._augmenters.append(RandomMasking(**config))

    def _add_random_renormalize(self, **config:Any) -> NoReturn:
        self._augmenters.append(RandomRenormalize(**config))

    def _add_stretch_compress(self, **config:Any) -> NoReturn:
        self._augmenters.append(StretchCompress(**config))

    def forward(self, sig:Tensor, label:Optional[Tensor], *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Union[Tensor,Tuple[Tensor]]:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor, optional,
            labels of the ECGs
        extra_tensors: Tensor(s), optional,
            extra tensors to be augmented, e.g. masks for custom loss functions, etc.
        kwargs: keyword arguments

        Returns
        -------
        Tensor(s), the augmented ECGs, labels, and optional extra tensors
        """
        if len(self.augmenters) == 0:
            raise ValueError("No augmenters added to the manager.")
        ordering = list(range(len(self.augmenters)))
        if self.random:
            ordering = sample(ordering, len(ordering))
        for idx in ordering:
            sig, label, *extra_tensors = self.augmenters[idx](sig, label, *extra_tensors, **kwargs)
        return (sig, label, *extra_tensors)

    @property
    def augmenters(self) -> List[Augmenter]:
        return self._augmenters

    def extra_repr(self) -> str:
        indent = 4*" "
        s = f"augmenters = [\n{indent}" + \
            f",\n{2*indent}".join(default_class_repr(aug, depth=2) for aug in self.augmenters) + \
            f"{indent}\n]"
        return s

    @classmethod
    def from_config(cls, config:dict) -> "AugmenterManager":
        """
        """
        am = cls(random=config.get("random", False))
        if "baseline_wander" in config:
            am._add_baseline_wander(fs=config["fs"], **config["baseline_wander"])
        if "label_smooth" in config:
            am._add_label_smooth(fs=config["fs"], **config["label_smooth"])
        if "mixup" in config:
            am._add_mixup(fs=config["fs"], **config["mixup"])
        if "random_flip" in config:
            am._add_random_flip(fs=config["fs"], **config["random_flip"])
        if "random_masking" in config:
            am._add_random_masking(fs=config["fs"], **config["random_masking"])
        if "random_renormalize" in config:
            am._add_random_renormalize(fs=config["fs"], **config["random_renormalize"])
        if "stretch_compress" in config:
            am._add_stretch_compress(fs=config["fs"], **config["stretch_compress"])
        return am

    def rearrange(self, new_ordering:List[str]) -> NoReturn:
        """
        """
        _mapping = {
            "".join([w.capitalize() for w in k.split("_")]): k \
                for k in "baseline_wander,label_smooth,mixup,random_flip,random_masking,random_renormalize,stretch_compress".split(",")
        }
        self._augmenters.sort(key=lambda aug: new_ordering.index(_mapping[aug.__name__]))
