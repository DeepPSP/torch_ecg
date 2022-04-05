"""
"""

from numbers import Real
from random import randint
from typing import Any, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter

__all__ = [
    "RandomMasking",
]


class RandomMasking(Augmenter):
    """
    Randomly mask ECGs with a probability.
    """

    __name__ = "RandomMasking"

    def __init__(
        self,
        fs: int,
        mask_value: Real = 0.0,
        mask_width: Sequence[Real] = [0.08, 0.18],
        prob: Union[Sequence[Real], Real] = [0.3, 0.15],
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: int,
            sampling frequency of the ECGs to be augmented
        mask_value: real number, default 0.0,
            value to mask with.
        mask_width: sequence of real numbers, default [0.08,0.18],
            width range of the masking window, with units in seconds
        prob: sequence of real numbers or real number, default [0.3,0.15],
            probabilities of masking ECG signals,
            the first probality is for the batch dimension,
            the second probability is for the lead dimension.
            note that 0.15 is approximately the proportion of QRS complexes in ECGs.
        inplace: bool, default True,
            whether to mask inplace or not
        kwargs: Keyword arguments.
        """
        super().__init__()
        self.fs = fs
        self.prob = prob
        if isinstance(self.prob, Real):
            self.prob = np.array([self.prob, self.prob])
        else:
            self.prob = np.array(self.prob)
        assert (self.prob >= 0).all() and (
            self.prob <= 1
        ).all(), "Probability must be between 0 and 1"
        self.mask_value = mask_value
        self.mask_width = (np.array(mask_width) * self.fs).round().astype(int)
        self.inplace = inplace

    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        critical_points: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor,
            label tensor of the ECGs,
            not used, but kept for compatibility with other augmenters
        extra_tensors: sequence of Tensors, optional,
            not used, but kept for consistency with other augmenters
        critical_points: sequence of sequences of integers,
            if given, random masking will be performed in windows centered at these points,
            this is useful for example when one wants to randomly mask QRS complexes
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

        Returns
        -------
        sig: Tensor,
            the augmented ECGs, of shape (batch, lead, siglen)
        label: Tensor,
            the label tensor of the augmented ECGs, unchanged
        extra_tensors: sequence of Tensors, optional,
            if set in the input arguments, unchanged
        """
        batch, lead, siglen = sig.shape
        if not self.inplace:
            sig = sig.clone()
        if self.prob[0] == 0:
            return (sig, label, *extra_tensors)
        sig_mask_prob = self.prob[1] / self.mask_width[1]
        sig_mask_scale_ratio = min(self.prob[1] / 4, 0.1) / self.mask_width[1]
        mask = torch.full_like(sig, 1, dtype=sig.dtype, device=sig.device)
        for batch_idx in self.get_indices(prob=self.prob[0], pop_size=batch):
            if critical_points is not None:
                indices = self.get_indices(
                    prob=self.prob[1], pop_size=len(critical_points[batch_idx])
                )
                indices = np.arange(siglen)[indices]
            else:
                indices = np.array(
                    self.get_indices(
                        prob=sig_mask_prob,
                        pop_size=siglen - self.mask_width[1],
                        scale_ratio=sig_mask_scale_ratio,
                    )
                )
                indices += self.mask_width[1] // 2
            for j in indices:
                masked_radius = randint(self.mask_width[0], self.mask_width[1]) // 2
                mask[
                    batch_idx, :, j - masked_radius : j + masked_radius
                ] = self.mask_value
            # print(f"batch_idx = {batch_idx}, len(indices) = {len(indices)}")
        sig = sig.mul_(mask)
        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "fs",
            "mask_value",
            "mask_width",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
