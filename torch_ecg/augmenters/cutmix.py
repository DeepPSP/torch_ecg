"""
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
    """CutMix augmentation.

    CutMix is a data augmentation technique originally proposed in
    :cite:p:`yun2019cutmix`, with official implementation in
    `clovaai/CutMix-PyTorch <https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py>`_,
    and an unofficial implementation in
    `ildoonet/cutmix <https://github.com/ildoonet/cutmix/blob/master/cutmix/cutmix.py>`_.

    This technique was designed for image classification tasks, but it can also be used
    for ECG tasks. This technique was very successful
    in CPSC2021 challenge of paroxysmal AF events detection.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency, by default None.
    num_mix : int, default 1
        Number of mixtures.
    alpha : float, default 0.5
        Beta distribution parameter.
    beta : float, optional
        Beta distribution parameter, by default equal to `alpha`.
    prob : float, default 0.5
        Probability of applying this augmenter.
    inplace : bool, default True
        Whether to perform this augmentation in-place.
    **kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        cm = CutMix(prob=0.7)
        sig = torch.randn(32, 12, 5000)
        lb = torch.randint(0, 2, (32, 5000, 2), dtype=torch.float32)  # 2 classes mask
        sig, lb = cm(sig, lb)

    .. bibliography::
        :filter: docname in docnames

    """

    __name__ = "CutMix"

    def __init__(
        self,
        fs: Optional[int] = None,
        num_mix: int = 1,
        alpha: Real = 0.5,
        beta: Optional[Real] = None,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fs = fs
        self.num_mix = num_mix
        assert (
            isinstance(self.num_mix, int) and self.num_mix > 0
        ), f"`num_mix` must be a positive integer, but got `{self.num_mix}`"
        self.alpha = alpha
        self.beta = beta or self.alpha
        assert (
            self.alpha > 0 and self.beta > 0
        ), f"`alpha` and `beta` must be positive, but got `{self.alpha}` and `{self.beta}`"
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace

    def forward(
        self,
        sig: Tensor,
        label: Tensor,
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any,
    ) -> Tuple[Tensor, ...]:
        """Forward method to perform CutMix augmentation.

        Parameters
        ----------
        sig: torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        label: torch.Tensor
            Class (one-hot) labels, of shape ``(batch, num_classes)``;
            or segmentation masks, of shape ``(batch, siglen, num_classes)``.
        extra_tensors: Sequence[torch.Tensor], optional
            Other tensors to be augmented, by default None.
        **kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        Tuple[torch.Tensor]
            Augmented tensors.

        """
        assert label.ndim != 1, "`label` should NOT be categorical labels"

        if not self.inplace:
            sig = sig.clone()
            label = label.clone()
            extra_tensors = [t.clone() for t in extra_tensors]
        else:
            extra_tensors = list(extra_tensors)  # make it mutable
        batch, lead, siglen = sig.shape

        for _ in range(self.num_mix):
            indices = np.arange(batch, dtype=int)
            # original indices chosen by probability
            ori = self.get_indices(prob=self.prob, pop_size=batch)
            # permuted indices
            perm = deepcopy(ori)
            shuffle(perm)
            indices[ori] = perm
            indices = torch.from_numpy(indices).long()

            lam = torch.from_numpy(
                # DEFAULTS.RNG.beta(self.alpha, self.beta, size=len(ori)),
                DEFAULTS.RNG.beta(self.alpha, self.beta, size=batch),
            ).to(
                dtype=sig.dtype, device=sig.device
            )  # shape: (batch,)
            intervals = _make_intervals(lam, siglen)

            # perform cutmix in batch
            # set values of sig enclosed by intervals to 0
            mask = torch.ones_like(sig)
            for i, (start, end) in enumerate(intervals):
                mask[i, :, start:end] = 0
            sig = sig * mask + sig[indices] * (1 - mask)

            if label.ndim == 2:
                # one-hot labels, shape (batch, num_classes)
                label = label * lam.view(-1, 1) + label[indices] * (1 - lam.view(-1, 1))
            else:
                # segmentation masks, shape (batch, siglen, num_classes)
                label = label * lam.view(-1, 1, 1) + label[indices] * (
                    1 - lam.view(-1, 1, 1)
                )

            for i, t in enumerate(extra_tensors):
                if t.ndim == 2:
                    extra_tensors[i] = t * lam.view(-1, 1) + t[indices] * (
                        1 - lam.view(-1, 1)
                    )
                else:
                    extra_tensors[i] = t * lam.view(-1, 1, 1) + t[indices] * (
                        1 - lam.view(-1, 1, 1)
                    )

        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        return [
            "alpha",
            "beta",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()


def _make_intervals(lam: Tensor, siglen: int) -> np.ndarray:
    """Make intervals for cutmix.

    Parameters
    ----------
    lam : torch.Tensor
        Parameter ``lambda`` for cutmix, of shape ``(n,)``.
    siglen : int
        Length of the signal.

    Returns
    -------
    numpy.ndarray
        Intervals for cutmix, of shape ``(n, 2)``.

    """
    _lam = (lam.numpy() * siglen).astype(int)
    intervals = np.zeros((lam.shape[0], 2), dtype=int)
    intervals[:, 0] = np.minimum(
        DEFAULTS.RNG_randint(0, siglen, size=lam.shape[0]), siglen - _lam
    )
    intervals[:, 1] = intervals[:, 0] + _lam
    return intervals
