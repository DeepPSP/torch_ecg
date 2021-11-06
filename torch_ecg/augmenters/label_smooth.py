"""
"""

from typing import Any, NoReturn, Sequence, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["LabelSmooth",]


class LabelSmooth(Augmenter):
    """
    Label smoothing augmentation.
    """
    __name__ = "LabelSmooth"

    def __init__(self, smoothing:float=0.1, prob:float=0.5, inplace:bool=True, **kwargs: Any) -> None:
        """ finished, checked,

        Parameters
        ----------
        smoothing: float, default 0.1,
            the smoothing factor
        prob: float, default 0.5,
            the probability of applying label smoothing
        inplace: bool, default True,
            if True, the input tensor will be modified inplace
        kwargs: keyword arguments
        """
        super().__init__(**kwargs)
        self.smoothing = smoothing
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace

    def generate(self, label:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        label: Tensor,
            of shape (batch_size, n_classes) or (batch_size, seq_len, n_classes),
            the input label tensor

        Returns
        -------
        label: Tensor,
            of shape (batch_size, n_classes) or (batch_size, seq_len, n_classes),
            the output label tensor
        """
        if not self.inplace:
            label = label.clone()
        if self.prob == 0 or self.smoothing == 0:
            return label
        n_classes = label.shape[-1]
        batch_size = label.shape[0]
        eps = self.smoothing / max(1, n_classes)
        indices = self.get_indices(prob=self.prob, pop_size=batch_size)
        # print(f"indices = {indices}, len(indices) = {len(indices)}")
        label[indices, ...] = (1 - self.smoothing) * label[indices, ...] \
            + torch.full_like(label[indices, ...], eps)
        return label

    def __call__(self, label:Tensor) -> Tensor:
        """
        alias of `self.generate`
        """
        return self.generate(label=label)
