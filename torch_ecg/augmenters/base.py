"""
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

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
