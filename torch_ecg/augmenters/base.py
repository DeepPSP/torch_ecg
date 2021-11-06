"""
"""

from random import sample
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List

import numpy as np
from torch import Tensor


__all__ = ["Augmenter",]


class Augmenter(ABC):
    """
    An Augmentor do data augmentation for ECGs and labels
    """
    __name__ = "Augmentor"

    @abstractmethod
    def generate(self, sig:Tensor, fs:Optional[int]=None, label:Optional[Tensor]=None) -> Union[Tensor,Tuple[Tensor]]:
        """
        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        fs: int, optional,
            sampling frequency of the ECGs
        label: Tensor, optional,
            labels of the ECGs

        Returns
        -------
        Tensor(s), the augmented ECGs
        """
        raise NotImplementedError

    def __call__(self, sig:Tensor, fs:Optional[int]=None, label:Optional[Tensor]=None) -> Union[Tensor,Tuple[Tensor]]:
        """
        alias of `self.generate`
        """
        return self.generate(sig=sig, fs=fs, label=label)

    def get_indices(self, prob:float, pop_size:int) -> List[int]:
        """ finished, checked

        compute a random list of indices in the range [0, pop_size-1]

        Parameters
        ----------
        prob: float,
            the probability of each index to be selected
        pop_size: int,
            the size of the population

        Returns
        -------
        indices: List[int],
            a list of indices
        """
        k = np.random.normal(pop_size * prob, 0.1*pop_size)
        k = int(round(np.clip(k, 0, pop_size)))
        indices = sample(list(range(pop_size)), k=k)
        return indices
