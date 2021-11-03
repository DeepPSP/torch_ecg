"""
"""

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


__all__ = ["",]


class Augmentor(ABC):
    """
    An Augmentor do data augmentation for ECGs and labels
    """
    __name__ = "Augmentor"

    @abstractmethod
    def generate(self, sig:Tensor, fs:int, label:Optional[Tensor]=None) -> Tensor:
        """
        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        fs: int,
            sampling frequency of the ECGs
        label: Tensor, optional,
            labels of the ECGs

        Returns
        -------
        Tensor, the augmented ECGs
        """
        raise NotImplementedError

    def __call__(self, sig:Tensor, fs:int, label:Optional[Tensor]=None) -> Tensor:
        """
        alias of `self.generate`
        """
        return self.generate(sig=sig, fs=fs, label=label)
