"""
"""

from abc import ABC, abstractmethod

from torch import Tensor


__all__ = ["",]


class Augmentor(ABC):
    """
    An Augmentor do data augmentation for ECGs and labels
    """
    __name__ = "Augmentor"


    @abstractmethod
    def generate(self, sig:Tensor, label:Tensor) -> Tensor:
        """
        """
        raise NotImplementedError

    def __call__(self, sig:Tensor, label:Tensor) -> Tensor:
        """
        """
        return self.generate(sig, label)
