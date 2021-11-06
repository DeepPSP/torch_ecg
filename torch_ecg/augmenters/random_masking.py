"""
"""

from random import choices
from typing import Any, NoReturn, Sequence, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["RandomMasking",]


class RandomMasking(Augmenter):
    """
    Randomly mask ECGs with a probability.
    """
    __name__ = "RandomMasking"

    def __init__(self,
                 prob:Union[Sequence[Real],Real]=[0.3,0.1],
                 mask_value:Real=0.0,
                 mask_width:Real=0.12,
                 inplace:bool=True,
                 **kwargs: Any) -> None:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        prob: sequence of real numbers or real number, default [0.3,0.1],
            probabilities of masking a ECG signal, and of masking sample poins in the signal
        mask_value: real number, default 0.0,
            value to mask with.
        mask_width: real number, default 0.12,
            width of the masking window, with units in seconds
        inplace: bool, default True,
            whether to mask inplace or not
        kwargs: Keyword arguments.
        """
        raise NotImplementedError

    def generate(self,
                 sig:Tensor,
                 fs:Optional[int]=None,
                 label:Optional[Tensor]=None,
                 critical_points:Optional[Sequence[Sequence[int]]]=None) -> Tensor:
        """ NOT finished, NOT checked,
        """
        raise NotImplementedError

    def __call__(self,
                 sig:Tensor,
                 fs:Optional[int]=None,
                 label:Optional[Tensor]=None,
                 critical_points:Optional[Sequence[Sequence[int]]]=None) -> Tensor:
        """
        alias of `self.generate`
        """
        raise NotImplementedError
