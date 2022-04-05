"""
"""

from numbers import Real
from typing import Any, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter

__all__ = [
    "RandomFlip",
]


class RandomFlip(Augmenter):
    """ """

    __name__ = "RandomFlip"

    def __init__(
        self,
        fs: Optional[int] = None,
        per_channel: bool = True,
        prob: Union[Sequence[float], float] = [0.4, 0.2],
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: int, optional,
            sampling frequency of the ECGs to be augmented
        per_channel: bool, default True,
            whether to flip each channel independently.
        prob: sequence of float or float, default [0.4,0.2],
            probability of performing flip,
            the first probality is for the batch dimension,
            the second probability is for the lead dimension.
        inplace: bool, default True,
            if True, ECG signal tensors will be modified inplace
        kwargs: Keyword arguments.
        """
        super().__init__()
        self.fs = fs
        self.per_channel = per_channel
        self.inplace = inplace
        self.prob = prob
        if isinstance(self.prob, Real):
            self.prob = np.array([self.prob, self.prob])
        else:
            self.prob = np.array(self.prob)
        assert (self.prob >= 0).all() and (
            self.prob <= 1
        ).all(), "Probability must be between 0 and 1"

    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor, optional,
            label tensor of the ECGs,
            not used, but kept for consistency with other augmenters
        extra_tensors: sequence of Tensors, optional,
            not used, but kept for consistency with other augmenters
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

        Returns
        -------
        sig: Tensor,
            the augmented ECGs
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
        if self.per_channel:
            flip = torch.ones((batch, lead, 1), dtype=sig.dtype, device=sig.device)
            for i in self.get_indices(prob=self.prob[0], pop_size=batch):
                flip[i, self.get_indices(prob=self.prob[1], pop_size=lead), ...] = -1
            sig = sig.mul_(flip)
        else:
            flip = torch.ones((batch, 1, 1), dtype=sig.dtype, device=sig.device)
            flip[self.get_indices(prob=self.prob[0], pop_size=batch), ...] = -1
            sig = sig.mul_(flip)
        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "per_channel",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
