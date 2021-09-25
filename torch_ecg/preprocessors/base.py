"""
"""

from abc import ABC, abstractmethod

import numpy as np


__all__ = ["PreProcessor",]


class PreProcessor(ABC):
    """
    a preprocessor do preprocessing for ECGs
    """
    __name__ = "PreProcessor"

    @abstractmethod
    def apply(self, sig:np.ndarray) -> np.ndarray:
        """
        """
        raise NotImplementedError

    def __call__(self, sig:np.ndarray) -> np.ndarray:
        """
        """
        return self.apply(sig)
