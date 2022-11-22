"""
Cutmix augmentation, for segmentation tasks.
This technique is very successful in CPSC2021 challenge of paroxysmal AF events detection.
"""

from copy import deepcopy
from numbers import Real
from random import shuffle
from typing import Any, List, Optional, Sequence, Tuple

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
    CutMix augmentation

    Examples
    --------
    ```python
    cm = CutMix(prob=0.7)
    sig = torch.randn(32, 12, 5000)
    lb = torch.randint(0, 2, (32, 5000, 2), dtype=torch.float32)  # 2 classes mask
    sig, lb = cm(sig, lb)
    ```

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
    ) -> None:
        """
        Parameters
        ----------
        fs: int, optional,
            Sampling frequency, by default None.
        alpha: float, default 0.5,
            Beta distribution parameter.
        beta: float, optional,
            Beta distribution parameter, by default equal to `alpha`.
        prob: float, default 0.5,
            Probability of applying this augmenter.
        inplace: bool, default True,
            Whether to perform this augmentation in-place.
        kwargs: Any,
            Other arguments for `Augmenter`.

        """
        super().__init__()
        self.fs = fs
        self.alpha = alpha
        self.beta = beta or self.alpha
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace

    def forward(
        self,
        sig: Tensor,
        label: Tensor,
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """
        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor,
            class labels, of shape (batch, num_classes) or (batch,);
            or segmentation masks, of shape (batch, siglen, num_classes)
        extra_tensors: Sequence[Tensor], optional,
            other tensors to be augmented, by default None.
        kwargs: Any,
            other arguments.

        Returns
        -------
        Tuple[Tensor, ...],
            augmented tensors.

        """
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

        # TODO: perform cutmix in batch

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
