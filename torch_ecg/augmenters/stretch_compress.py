"""
"""

from typing import Any, NoReturn, Sequence, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["StretchCompress", "StretchCompress_",]


class StretchCompress(Augmenter):
    """
    stretch-or-compress augmenter on ECG tensors
    """
    __name__ = "StretchCompress"

    def __init__(self,) -> NoReturn:
        """
        """
        raise NotImplementedError


class StretchCompress_(object):
    """
    stretch-or-compress augmenter on orginal length-varying ECG signals (in the form of numpy arrays)
    """
    def __init__(self,) -> NoReturn:
        """
        """
        raise NotImplementedError
