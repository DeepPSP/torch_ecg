"""
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn
from torch import Tensor

from ..cfg import DEFAULTS
from ..utils.misc import ReprMixin, add_docstring

__all__ = [
    "Augmenter",
]


_augmenter_forward_doc = """Forward method of the augmenter.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor, optional
            Batched labels of the ECGs.
        *extra_tensors : Sequence[torch.Tensor], optional
            Extra tensors to be augmented, e.g. masks for custom loss functions, etc.
        **kwargs: dict, optional
            Additional keyword arguments to be passed to the augmenters.

        Returns
        -------
        Sequence[torch.Tensor]
            The augmented ECGs, labels, and optional extra tensors.

        """


class Augmenter(ReprMixin, nn.Module, ABC):
    """Base class for augmenters.

    An Augmentor performs data augmentation on the input ECGs,
    labels, and optional extra tensors.

    """

    __name__ = "Augmentor"

    @add_docstring(_augmenter_forward_doc)
    @abstractmethod
    def forward(
        self, sig: Tensor, label: Optional[Tensor] = None, *extra_tensors: Sequence[Tensor], **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    # def __call__(self, sig:Tensor, label:Optional[Tensor]=None, *extra_tensors:Sequence[Tensor], **kwargs:Any) -> Tuple[Tensor, ...]:
    #     """
    #     alias of `self.generate`
    #     """
    #     return self.generate(sig, label, *extra_tensors, **kwargs)

    def get_indices(self, prob: float, pop_size: int, scale_ratio: float = 0.1) -> List[int]:
        """Get a list of indices to be selected.

        A random list of indices in the range ``[0, pop_size-1]``
        is generated, with the probability of each index to be selected.

        Parameters
        ----------
        prob : float
            The probability of each index to be selected.
        pop_size : int
            The size of the population.
        scale_ratio : float, default 0.1
            Scale ratio of std of the normal distribution to the population size.

        Returns
        -------
        indices : List[int],
            A list of indices.

        TODO
        ----
        Add parameter `min_dist` so that
        any 2 selected indices are at least `min_dist` apart.

        """
        k = DEFAULTS.RNG.normal(pop_size * prob, scale_ratio * pop_size)
        # print(pop_size * prob, scale_ratio*pop_size)
        k = int(round(np.clip(k, 0, pop_size)))
        indices = DEFAULTS.RNG_sample(list(range(pop_size)), k).tolist()
        return indices
