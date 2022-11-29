"""
Audio data augmentation using `torch_audiomentations`

The drawbacks of this approach:
1. the forward function only accepts the input tensor and the sample rate, without the labels.
If labels contains segmentation masks, the forward function will not be able to process them,
which is needed for example for `Shift`, `TimeInversion`, etc.
"""

from typing import Sequence

import torch_audiomentations as TA
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_ecg.utils.misc import get_kwargs


__all__ = ["AugmenterManager"]


class AugmenterManager(TA.SomeOf):
    """Audio data augmenters"""

    def __init__(
        self,
        transforms: Sequence[BaseWaveformTransform],
        p: float = 1.0,
        p_mode="per_batch",
    ) -> None:
        """ """
        kwargs = dict(p=p, p_mode=p_mode)
        if "output_type" in get_kwargs(TA.SomeOf.__init__):
            # newer versions of `torch_audiomentations`
            kwargs["output_type"] = "tensor"
        super().__init__((1, None), transforms, **kwargs)

    @classmethod
    def from_config(cls, config: dict) -> "AugmenterManager":
        """ """
        transforms = [TA.from_dict(item) for item in config["augmentations"]]
        return cls(transforms, **config["augmentations_kw"])

    def __len__(self) -> int:
        """ """
        return len(self.transforms)
