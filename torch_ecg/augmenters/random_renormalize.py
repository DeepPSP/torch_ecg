"""
"""

from typing import Any, NoReturn, Sequence, Union, Optional, Iterable, List
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter

from ..utils.utils_signal_t import normalize as normalize_t


__all__ = ["RandomRenormalize",]


class RandomRenormalize(Augmenter):
    """
    Randomly re-normalize the ECG tensor,
    using the Z-score normalization method.
    """
    __name__ = "RandomRenormalize"

    def __init__(self,
                 mean:Iterable[Real]=[-0.05,0.1],
                 std:Iterable[Real]=[0.08,0.32],
                 per_channel:bool=False,
                 prob:float=0.5,
                 inplace:bool=True,
                 **kwargs: Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        mean: array_like, default [-0.05,0.1],
            range of mean value of the re-normalized signal, of shape (2,);
            or range of mean values for each lead of the re-normalized signal, of shape (lead, 2).
        std: array_like, default [0.08,0.32],
            range of standard deviation of the re-normalized signal, of shape (2,);
            or range of standard deviations for each lead of the re-normalized signal, of shape (lead, 2).
        per_channel: bool, default False,
            if True, re-normalization will be done per channel
        prob: float, default 0.5,
            Probability of applying the random re-normalization augmenter.
        inplace: bool, default True,
            Whether to apply the random re-normalization augmenter in-place.
        kwargs: keyword arguments
        """
        super().__init__(**kwargs)
        self.mean = np.array(mean)
        self.mean_mean = self.mean.mean(axis=-1, keepdims=True)
        self.mean_scale = (self.mean[...,-1] - self.mean_mean) * 0.3
        self.std = np.array(std)
        self.std_mean = self.std.mean(axis=-1, keepdims=True)
        self.std_scale = (self.std[...,-1] - self.std_mean) * 0.3
        self.per_channel = per_channel
        if not self.per_channel:
            assert self.mean.ndim == 1 and self.std.ndim == 1
        self.prob = prob
        self.inplace = inplace

    def generate(self, sig:Tensor, label:Optional[Tensor]=None) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            The input ECG tensor, of shape (batch, lead, siglen).
        label: Tensor, optional,
            The input ECG label tensor, not used.

        Returns
        -------
        sig: Tensor,
            The randomly re-normalized ECG tensor.
        """
        batch, lead, siglen = sig.shape
        if self.mean.ndim == 2:
            assert self.mean.shape[0] == lead
        if self.std.ndim == 2:
            assert self.std.shape[0] == lead
        if not self.inplace:
            sig = sig.clone()
        indices = self.get_indices(self.prob, pop_size=batch)
        if self.per_channel:
            mean = np.random.normal(self.mean_mean, self.mean_scale, size=(len(indices), lead, 1))
            std = np.random.normal(self.std_mean, self.std_scale, size=(len(indices), lead, 1))
        else:
            mean = np.random.normal(self.mean_mean, self.mean_scale, size=(len(indices), 1, 1))
            std = np.random.normal(self.std_mean, self.std_scale, size=(len(indices), 1, 1))
        sig = normalize_t(
            sig,
            method="z-score",
            mean=mean,
            std=std,
            per_channel=self.per_channel,
            inplace=True,
        )
        return sig

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["mean", "std", "per_channel", "prob", "inplace",] + super().extra_repr_keys()
