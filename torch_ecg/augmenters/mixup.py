"""
"""

from copy import deepcopy
from numbers import Real
from random import shuffle
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..cfg import DEFAULTS
from .base import Augmenter

__all__ = [
    "Mixup",
]


class Mixup(Augmenter):
    """Mixup augmentor.

    Mixup is a data augmentation technique originally proposed in [1]_.
    The PDF file of the paper can be found on arXiv [2]_.
    The official implementation is provided in [3]_. This technique was designed
    for image classification tasks, but it is also widely used for ECG tasks.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency of the ECGs to be augmented.
    alpha : numbers.Real, default 0.5
        alpha parameter of the Beta distribution used in Mixup.
    beta : numbers.Real, optional
        beta parameter of the Beta distribution used in Mixup,
        defaults to `alpha`.
    prob : float, default 0.5
        Probability of applying Mixup.
    inplace : bool, default True
        If True, ECG signal tensors will be modified inplace.
    **kwargs : dict, optional
        Additional keyword arguments, not used.

    Examples
    --------
    .. code-block:: python

        mixup = Mixup(alpha=0.3, beta=0.6, prob=0.7)
        sig = torch.randn(32, 12, 5000)
        label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
        sig, label = mixup(sig, label)

    References
    ----------
    .. [1] Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization."
           International Conference on Learning Representations. 2018.
    .. [2] https://arxiv.org/abs/1710.09412
    .. [3] https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py

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
    ) -> None:
        super().__init__()
        self.fs = fs
        self.alpha = alpha
        self.beta = beta or self.alpha
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace

    def forward(self, sig: Tensor, label: Tensor, *extra_tensors: Sequence[Tensor], **kwargs: Any) -> Tuple[Tensor, ...]:
        """Forward method of the Mixup augmenter.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor
            Label tensor of the ECGs.
        extra_tensors: Sequence[torch.Tensor], optional
            Not used, but kept for consistency with other augmenters.
        **kwargs : dict, optional
            Not used, but kept for consistency with other augmenters.

        Returns
        -------
        sig : torch.Tensor,
            The augmented ECGs.
        label : torch.Tensor
            The augmented labels.
        extra_tensors : Sequence[torch.Tensor], optional
            Unchanged extra tensors.

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
        return [
            "alpha",
            "beta",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
