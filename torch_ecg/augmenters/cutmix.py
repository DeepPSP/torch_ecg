"""
Cutmix augmentation, for segmentation tasks.
This technique is very successful in CPSC2021 challenge of paroxysmal AF events detection.
"""

from typing import Any, NoReturn, Sequence, Union, Optional, List, Tuple
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmenter


__all__ = ["Cutmix",]


class Cutmix(Augmenter):
    """
    Cutmix augmentation, for segmentation tasks.

    References
    ----------
    1. Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6023-6032).
    """
    __name__ = "Cutmix"

    def __init__(self,
                 fs:Optional[int]=None,
                 len_range:Optional[Sequence[Real]]=None,
                 prob:float=0.5,
                 inplace:bool=True,
                 **kwargs:Any) -> NoReturn:
        """
        """
        raise NotImplementedError

    def forward(self, sig:Optional[Tensor], label:Tensor, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
        """
        """
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        raise NotImplementedError
