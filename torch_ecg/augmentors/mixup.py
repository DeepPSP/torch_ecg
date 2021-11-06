"""
"""

from random import choices, shuffle
from copy import deepcopy
from typing import Any, NoReturn, Sequence, Union, Tuple, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from .base import Augmentor


__all__ = ["Mixup",]


class Mixup(Augmentor):
    """
    Mixup augmentor.

    References
    ----------
    1. Zhang, Hongyi, et al. "mixup: Beyond Empirical Risk Minimization." International Conference on Learning Representations. 2018.
    2. https://arxiv.org/abs/1710.09412
    3. https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    __name__ = "Mixup"

    def __init__(self,
                 alpha:Real=0.5,
                 beta:Optional[Real]=None,
                 prob:float=0.5,
                 inplace:bool=True,
                 **kwargs:Any) -> NoReturn:
        """
        Parameters
        ----------
        alpha: real number, default 0.5,
            alpha parameter of the Beta distribution used in Mixup.
        beta: real number, optional,
            beta parameter of the Beta distribution used in Mixup,
            default to alpha.
        prob: float, default 0.5,
            probability of applying Mixup.
        inplace: bool, default True,
            if True, ECG signal tensors will be modified inplace
        kwargs: Keyword arguments.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta or self.alpha
        self.prob = prob
        self.inplace = inplace

    def generate(self, sig:Tensor, label:Tensor) -> Tuple[Tensor,Tensor]:
        """

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor, optional,
            labels of the ECGs

        Returns
        -------
        sig: Tensor,
            the augmented ECGs
        label: Tensor,
            the augmented label
        """
        batch, lead, siglen = sig.shape
        lam = np.random.beta(self.alpha, self.beta)
        indices = np.arange(batch, dtype=int)
        ori = choices(range(batch), k=int(round(self.prob*batch)))
        perm = deepcopy(ori)
        shuffle(perm)
        indices[ori] = perm
        indices = torch.from_numpy(indices).long()
        print(indices)

        if not self.inplace:
            sig = sig.clone()
            label = label.clone()

        sig = lam * sig + (1 - lam) * sig[indices]
        label = lam * label + (1 - lam) * label[indices]

        return sig, label

    def __call__(self, sig:Tensor, label:Tensor) -> Tuple[Tensor,Tensor]:
        """
        alias of `self.generate`
        """
        return self.generate(sig=sig, label=label)
