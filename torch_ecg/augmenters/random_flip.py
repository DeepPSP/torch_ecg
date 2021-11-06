"""
"""

from typing import Any, NoReturn, Sequence, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["RandomFlip",]


class RandomFlip(Augmenter):
    """
    """
    __name__ = "RandomFlip"

    def __init__(self,
                 per_channel:bool=True,
                 prob:Union[Sequence[float],float]=[0.4,0.2],
                 inplace:bool=True,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        per_channel: bool, default True,
            whether to flip each channel independently.
        prob: sequence of float or float, default [0.4,0.2],
            probability of performing flip.
        inplace: bool, default True,
            if True, ECG signal tensors will be modified inplace
        kwargs: Keyword arguments.
        """
        super().__init__(**kwargs)
        self.per_channel = per_channel
        self.inplace = inplace
        self.prob = prob
        if isinstance(self.prob, Real):
            self.prob = np.array([self.prob, self.prob])
        else:
            self.prob = np.array(self.prob)
        assert (self.prob >= 0).all() and (self.prob <= 1).all(), \
            "Probability must be between 0 and 1"

    def generate(self, sig:Tensor, fs:Optional[int]=None, label:Optional[Tensor]=None) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        fs: int, optional,
            sampling frequency of the ECGs, not used
        label: Tensor, optional,
            labels of the ECGs, not used

        Returns
        -------
        sig: Tensor,
            the augmented ECGs
        """
        batch, lead, siglen = sig.shape
        if not self.inplace:
            sig = sig.clone()
        if self.per_channel:
            flip = torch.ones((batch,lead,1), dtype=sig.dtype, device=sig.device)
            for i in self.get_indices(prob=self.prob[0], pop_size=batch):
                flip[i, self.get_indices(prob=self.prob[1], pop_size=lead), ...] = -1
            sig = sig.mul_(flip)
        else:
            flip = torch.ones((batch,1,1), dtype=sig.dtype, device=sig.device)
            flip[self.get_indices(prob=self.prob[0], pop_size=batch), ...] = -1
            sig = sig.mul_(flip)
        return sig
