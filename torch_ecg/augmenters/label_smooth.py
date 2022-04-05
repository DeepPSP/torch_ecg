"""
"""

from typing import Any, List, NoReturn, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .base import Augmenter

__all__ = [
    "LabelSmooth",
]


class LabelSmooth(Augmenter):
    """
    Label smoothing augmentation.
    """

    __name__ = "LabelSmooth"

    def __init__(
        self,
        fs: Optional[int] = None,
        smoothing: float = 0.1,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """

        Parameters
        ----------
        fs: int, optional,
            sampling frequency of the ECGs to be augmented
        smoothing: float, default 0.1,
            the smoothing factor
        prob: float, default 0.5,
            the probability of applying label smoothing
        inplace: bool, default True,
            if True, the input tensor will be modified inplace
        kwargs: keyword arguments
        """
        super().__init__()
        self.fs = fs
        self.smoothing = smoothing
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
        """

        Parameters
        ----------
        sig: Tensor,
            the input ECG tensor,
            not used, but kept for compatibility with other augmenters
        label: Tensor,
            of shape (batch_size, n_classes) or (batch_size, seq_len, n_classes),
            the input label tensor
        extra_tensors: sequence of Tensors, optional,
            not used, but kept for consistency with other augmenters
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

        Returns
        -------
        sig: Tensor,
            the input ECG tensor, unchanged
        label: Tensor,
            of shape (batch_size, n_classes) or (batch_size, seq_len, n_classes),
            the output label tensor
        extra_tensors: sequence of Tensors, optional,
            if set in the input arguments, unchanged
        """
        if not self.inplace:
            label = label.clone()
        if self.prob == 0 or self.smoothing == 0:
            return (sig, label, *extra_tensors)
        n_classes = label.shape[-1]
        batch_size = label.shape[0]
        eps = self.smoothing / max(1, n_classes)
        indices = self.get_indices(prob=self.prob, pop_size=batch_size)
        # print(f"indices = {indices}, len(indices) = {len(indices)}")
        label[indices, ...] = (1 - self.smoothing) * label[
            indices, ...
        ] + torch.full_like(label[indices, ...], eps)
        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "smoothing",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
