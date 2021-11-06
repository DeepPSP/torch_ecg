"""
"""

from random import choices
from typing import Any, NoReturn, Sequence, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["LabelSmooth",]


class LabelSmooth(Augmenter):
    """
    Label smoothing augmentation.
    """
    __name__ = "LabelSmooth"

    def __init__(self,) -> NoReturn:
        """
        """
        raise NotImplementedError
