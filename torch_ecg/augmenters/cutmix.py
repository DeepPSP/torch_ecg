"""
Cutmix augmentation, for segmentation tasks.
This technique is very successful in CPSC2021 challenge of paroxysmal AF events detection.
"""

from copy import deepcopy
from numbers import Real
from random import shuffle
from typing import Any, List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter
from ..cfg import DEFAULTS

__all__ = [
    "CutMix",
]


class CutMix(Augmenter):
    """
    CutMix augmentation, for segmentation tasks.

    References
    ----------
    1. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6023-6032).
    2. https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    3. https://github.com/ildoonet/cutmix/blob/master/cutmix/cutmix.py
    """

    __name__ = "CutMix"

    def __init__(
        self,
        fs: Optional[int] = None,
        alpha: Real = 0.5,
        beta: Optional[Real] = None,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """ """
        super().__init__()
        self.fs = fs
        self.alpha = alpha
        self.beta = beta or self.alpha
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace

    def forward(
        self,
        sig: Optional[Tensor],
        label: Tensor,
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """ """
        batch, lead, siglen = sig.shape
        lam = torch.from_numpy(
            DEFAULTS.RNG.beta(self.alpha, self.beta, size=batch),
            dtype=sig.dtype,
            device=sig.device,
        )
        indices = np.arange(batch, dtype=int)
        ori = self.get_indices(prob=self.prob, pop_size=batch)
        perm = deepcopy(ori)
        shuffle(perm)
        indices[ori] = perm
        indices = torch.from_numpy(indices).long()
        intervals = self._make_intervals(lam, siglen)

        if not self.inplace:
            sig = sig.clone()
            label = label.clone()
            extra_tensors = [t.clone() for t in extra_tensors]

        raise NotImplementedError

    def _make_intervals(self, lam: Tensor, siglen: int) -> np.ndarray:
        """ """
        _lam = (lam.numpy() * siglen).astype(int)
        intervals = np.zeros((lam.shape[0], 2), dtype=int)
        intervals[:, 0] = np.minimum(
            DEFAULTS.RNG_randint(0, siglen, size=lam.shape[0]), siglen - _lam
        )
        intervals[:, 1] = intervals[:, 0] + _lam
        return intervals

    def extra_repr_keys(self) -> List[str]:
        return [
            "alpha",
            "beta",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
