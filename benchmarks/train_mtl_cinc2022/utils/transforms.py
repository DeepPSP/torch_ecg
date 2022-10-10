"""
Transforms for data augmentation.

deprecated and overrided by ./augmentations.py

"""

import re
from typing import NoReturn, Union
from collections import OrderedDict

from torch.nn import Module, Sequential
from torchaudio import transforms as TT

try:
    from torchaudio.transforms import PitchShift  # noqa: F401
except ImportError:
    from ._transforms import PitchShift  # noqa: F401


__all__ = [
    "Transforms",
]


class Transforms(Sequential):
    """

    Composition of transforms for audio data augmentation.

    Supported transforms:
        - :class:`torchaudio.transforms.Fade`
            domain: time
            applies to: waveform
            args:
                fade_in_len (int, optional): Length of fade-in (time frames). (Default: ``0``)
                fade_out_len (int, optional): Length of fade-out (time frames). (Default: ``0``)
                fade_shape (str, optional): Shape of fade. Must be one of: "quarter_sine",
                    "half_sine", "linear", "logarithmic", "exponential". (Default: ``"linear"``)
        - :class:`torchaudio.transforms.TimeStretch`
            domain: time
            applies to: complex specgram
            args:
                hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
                n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
                fixed_rate (float or None, optional): rate to speed up or slow down by.
                    If None is provided, rate must be passed to the forward method. (Default: ``None``)
        - :class:`torchaudio.transforms.TimeMasking`
            domain: time
            applies to: specgram
            args:
                time_mask_param (int): maximum possible length of the mask.
                    Indices uniformly sampled from [0, time_mask_param).
                iid_masks (bool, optional): whether to apply different masks to each
                    example/channel in the batch. (Default: ``False``)
                    This option is applicable only when the input tensor is 4D.
        - :class:`torchaudio.transforms.FrequencyMasking`
            domain: frequency
            applies to: specgram
            args:
                freq_mask_param (int): maximum possible length of the mask.
                    Indices uniformly sampled from [0, freq_mask_param).
                iid_masks (bool, optional): whether to apply different masks to each
                    example/channel in the batch. (Default: ``False``)
                    This option is applicable only when the input tensor is 4D.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> import torch
    >>> t = Transforms.from_config(
    ...     OrderedDict(
    ...         TimeMasking = dict(time_mask_param=10),
    ...         FrequencyMasking = dict(freq_mask_param=10),
    ...     )
    ... )
    >>> t(torch.rand(8, 1, 201, 433))

    """

    __name__ = "Transforms"

    def __init__(self, *args: Union[Module, OrderedDict]) -> NoReturn:
        """ """
        super().__init__(*args)

    @classmethod
    def from_config(cls, config: OrderedDict) -> "Transforms":
        """ """
        assert isinstance(config, OrderedDict), "config must be an OrderedDict"
        assert (
            len(set([_applies_to(tn) for tn in config.keys()])) == 1
        ), "all transforms must be applied to the same form of data"
        transforms = []
        for tn, cfg in config.items():
            transforms.append(_normalize_transform_name(tn)(**cfg))
        return cls(*transforms)


def _normalize_transform_name(tn: str) -> Module:
    """ """
    return dict(
        fade=TT.Fade,  # waveform
        # pitchshift=TT.PitchShift,  # waveform
        timestretch=TT.TimeStretch,  # complex spectrogram
        timemasking=TT.TimeMasking,  # spectrogram
        frequencymasking=TT.FrequencyMasking,  # spectrogram
    )[re.sub("[\\s\\-\\_]+", "", tn).lower()]


def _applies_to(tn: str) -> str:
    """ """
    return dict(
        fade="waveform",
        # pitchshift="waveform",
        timestretch="complex_specgram",
        timemasking="specgram",
        frequencymasking="specgram",
    )[re.sub("[\\s\\-\\_]+", "", tn).lower()]
