"""
"""

from numbers import Real
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor

from ..cfg import DEFAULTS
from ..utils.utils_signal_t import normalize as normalize_t
from .base import Augmenter

__all__ = [
    "RandomRenormalize",
]


class RandomRenormalize(Augmenter):
    """Randomly re-normalize the ECG tensor,
    using the Z-score normalization method.

    Parameters
    ----------
    mean : array_like, default ``[-0.05, 0.1]``
        Range of mean value of the re-normalized signal, of shape ``(2,)``;
        or range of mean values for each lead of the re-normalized signal,
        of shape ``(lead, 2)``.
    std : array_like, default ``[0.08, 0.32]``
        Range of standard deviation of the re-normalized signal, of shape ``(2,)``;
        or range of standard deviations for each lead of the re-normalized signal,
        of shape ``(lead, 2)``.
    per_channel : bool, default False
        If True, re-normalization will be done per channel.
    prob : float, default 0.5
        Probability of applying the random re-normalization augmenter.
    inplace : bool, default True
        Whether to apply the random re-normalization augmenter in-place.
    kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        rrn = RandomRenormalize()
        sig = torch.randn(32, 12, 5000)
        sig, _ = rrn(sig, None)

    """

    __name__ = "RandomRenormalize"

    def __init__(
        self,
        mean: Iterable[Real] = [-0.05, 0.1],
        std: Iterable[Real] = [0.08, 0.32],
        per_channel: bool = False,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.mean = np.array(mean)
        self.mean_mean = self.mean.mean(axis=-1, keepdims=True)
        self.mean_scale = (self.mean[..., -1] - self.mean_mean) * 0.3
        self.std = np.array(std)
        self.std_mean = self.std.mean(axis=-1, keepdims=True)
        self.std_scale = (self.std[..., -1] - self.std_mean) * 0.3
        self.per_channel = per_channel
        if not self.per_channel:
            assert self.mean.ndim == 1 and self.std.ndim == 1
        self.prob = prob
        self.inplace = inplace

    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """Forward function of the RandomRenormalize augmenter.

        Parameters
        ----------
        sig : torch.Tensor
            The input ECG tensor, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor, optional
            The input ECG label tensor.
            Not used, but kept for compatibility with other augmenters.
        extra_tensors : Sequence[torch.Tensor], optional,
            Not used, but kept for consistency with other augmenters.
        kwargs : dict, optional
            Not used, but kept for consistency with other augmenters.

        Returns
        -------
        sig : torch.Tensor
            The randomly re-normalized ECG tensor.
        label : torch.Tensor
            The label tensor of the augmented ECGs, unchanged.
        extra_tensors: Sequence[torch.Tensor], optional,
            Unchanged extra tensors.

        """
        batch, lead, siglen = sig.shape
        if self.mean.ndim == 2:
            assert self.mean.shape[0] == lead
        if self.std.ndim == 2:
            assert self.std.shape[0] == lead
        if not self.inplace:
            sig = sig.clone()
        if self.prob == 0:
            return (sig, label, *extra_tensors)
        indices = self.get_indices(self.prob, pop_size=batch)
        if self.per_channel:
            mean = DEFAULTS.RNG.normal(
                self.mean_mean, self.mean_scale, size=(len(indices), lead, 1)
            )
            std = DEFAULTS.RNG.normal(
                self.std_mean, self.std_scale, size=(len(indices), lead, 1)
            )
        else:
            mean = DEFAULTS.RNG.normal(
                self.mean_mean, self.mean_scale, size=(len(indices), 1, 1)
            )
            std = DEFAULTS.RNG.normal(
                self.std_mean, self.std_scale, size=(len(indices), 1, 1)
            )
        sig[indices, ...] = normalize_t(
            sig[indices, ...],
            method="z-score",
            mean=mean,
            std=std,
            per_channel=self.per_channel,
            inplace=True,
        )
        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "mean",
            "std",
            "per_channel",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()
