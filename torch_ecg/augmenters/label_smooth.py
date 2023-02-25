"""
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .base import Augmenter

__all__ = [
    "LabelSmooth",
]


class LabelSmooth(Augmenter):
    """Label smoothing augmentation.

    Parameters
    ----------
    fs : int, optional
        Sampling frequency of the ECGs to be augmented.
    smoothing : float, default 0.1
        The smoothing factor.
    prob : float, default 0.5
        Probability of applying label smoothing.
    inplace : bool, default True
        If True, the input tensor will be modified inplace.
    **kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        ls = LabelSmooth()
        label = torch.randint(0, 2, (32, 26), dtype=torch.float32)
        _, label = ls(None, label)

    """

    __name__ = "LabelSmooth"

    def __init__(
        self,
        fs: Optional[int] = None,
        smoothing: float = 0.1,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
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
        """Forward method to perform label smoothing.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
            Not used, but kept for compatibility with other augmenters.
        label : torch.Tensor
            The input label tensor,
            of shape ``(batch_size, n_classes)``
            or ``(batch_size, seq_len, n_classes)``.
        extra_tensors : Sequence[torch.Tensor], optional
            Not used, but kept for consistency with other augmenters.
        **kwargs : dict, optional
            Not used, but kept for consistency with other augmenters.

        Returns
        -------
        sig : torch.Tensor
            The input ECG tensor, unchanged.
        label : torch.Tensor
            The output label tensor
            of shape ``(batch_size, n_classes)``
            or ``(batch_size, seq_len, n_classes)``.
        extra_tensors : Sequence[torch.Tensor], optional
            Unchanged extra tensors.

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
        return [
            "smoothing",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
