"""
"""

from random import sample
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Any, Sequence

import numpy as np
from torch import Tensor
import torch.nn as nn

from ..utils.misc import default_class_repr


__all__ = ["Augmenter",]


class Augmenter(nn.Module, ABC):
    """
    An Augmentor do data augmentation for ECGs and labels
    """
    __name__ = "Augmentor"

    @abstractmethod
    def forward(self, sig:Tensor, label:Optional[Tensor]=None, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
        """ generates a new signal and label using corresponding augmentation method,

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor, optional,
            labels of the ECGs
        extra_tensors: Tensor(s), optional,
            extra tensors to be augmented, e.g. masks for custom loss functions, etc.
        kwargs: keyword arguments

        Returns
        -------
        Tensor(s), the augmented ECGs, labels, and optional extra tensors
        """
        raise NotImplementedError

    # def __call__(self, sig:Tensor, label:Optional[Tensor]=None, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
    #     """
    #     alias of `self.generate`
    #     """
    #     return self.generate(sig, label, *extra_tensors, **kwargs)

    def get_indices(self, prob:float, pop_size:int, scale_ratio:float=0.1) -> List[int]:
        """ finished, checked

        compute a random list of indices in the range [0, pop_size-1]

        Parameters
        ----------
        prob: float,
            the probability of each index to be selected
        pop_size: int,
            the size of the population
        scale_ratio: float, default 0.1,
            scale ratio of std of the normal distribution to the population size

        Returns
        -------
        indices: List[int],
            a list of indices

        TODO
        ----
        add parameter `min_dist` so that any 2 selected indices are at least `min_dist` apart
        """
        k = np.random.normal(pop_size * prob, scale_ratio*pop_size)
        # print(pop_size * prob, scale_ratio*pop_size)
        k = int(round(np.clip(k, 0, pop_size)))
        indices = sample(list(range(pop_size)), k=k)
        return indices

    def __repr__(self) -> str:
        return default_class_repr(self)

    def __str__(self) -> str:
        return repr(self)

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return []
