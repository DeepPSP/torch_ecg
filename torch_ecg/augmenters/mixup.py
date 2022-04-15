"""
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
    "Mixup",
]


class Mixup(Augmenter):
    """
    Mixup augmentor.

    References
    ----------
    1. Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." International Conference on Learning Representations. 2018.
    2. https://arxiv.org/abs/1710.09412
    3. https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """

    __name__ = "Mixup"

    def __init__(
        self,
        fs: Optional[int] = None,
        alpha: Real = 0.5,
        beta: Optional[Real] = None,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: int, optional,
            sampling frequency of the ECGs to be augmented
        alpha: real number, default 0.5,
            alpha parameter of the Beta distribution used in Mixup.
        beta: real number, optional,
            beta parameter of the Beta distribution used in Mixup,
            default to alpha.
        prob: float, default 0.5,
            probability of applying Mixup.
        inplace: bool, default True,
            if True, ECG signal tensors will be modified inplace
        kwargs: Keyword arguments.
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
            label tensor of the ECGs
        extra_tensors: sequence of Tensors, optional,
            not used, but kept for consistency with other augmenters
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

        Returns
        -------
        sig: Tensor,
            the augmented ECGs
        label: Tensor,
            the augmented label
        extra_tensors: sequence of Tensors, optional,
            if set in the input arguments, unchanged
        """
        batch, lead, siglen = sig.shape
        # TODO: make `lam` different for each batch element, using
        # lam = torch.from_numpy(DEFAULTS.RNG.beta(self.alpha, self.beta, batch), dtype=sig.dtype, device=sig.device)
        # of shape (batch,)
        lam = DEFAULTS.RNG.beta(self.alpha, self.beta)
        indices = np.arange(batch, dtype=int)
        ori = self.get_indices(prob=self.prob, pop_size=batch)
        # print(f"ori = {ori}, len(ori) = {len(ori)}")
        perm = deepcopy(ori)
        shuffle(perm)
        indices[ori] = perm
        indices = torch.from_numpy(indices).long()

        if not self.inplace:
            sig = sig.clone()
            label = label.clone()

        sig = lam * sig + (1 - lam) * sig[indices]
        label = lam * label + (1 - lam) * label[indices]
        # TODO: if `lam` is a Tensor of shape (batch,) instead of a scalar,
        # sig = lam.view(batch, 1, 1) * sig + (1 - lam).view(batch, 1, 1) * sig[indices]
        # label = lam.view(batch, 1) * label + (1 - lam).view(batch, 1) * label[indices]

        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "alpha",
            "beta",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
