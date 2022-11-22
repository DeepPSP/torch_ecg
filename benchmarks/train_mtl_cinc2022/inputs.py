"""
"""

import inspect
import math
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Union, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torchaudio import transforms as TT
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_nn import compute_conv_output_shape


__all__ = [
    "InputConfig",
    "WaveformInput",
    "SpectrogramInput",
    "MelSpectrogramInput",
    "MFCCInput",
    "SpectralInput",
]


class InputConfig(CFG):
    """ """

    __name__ = "InputConfig"

    def __init__(
        self,
        *args: Union[CFG, dict],
        input_type: str,
        n_channels: int,
        n_samples: int = -1,
        **kwargs: dict
    ) -> None:
        """

        Parameters
        ----------
        input_type : str,
            the type of the input, can be
            - "waveform"
            - "spectrogram"
            - "mel_spectrogram" (with aliases `mel`, `melspectrogram`)
            - "mfcc"
            - "spectral" (concatenates the "spectrogram", the "mel_spectrogram" and the "mfcc")
        n_channels : int,
            the number of channels of the input
        n_samples : int,
            the number of samples of the input

        """
        super().__init__(
            *args,
            input_type=input_type,
            n_channels=n_channels,
            n_samples=n_samples,
            **kwargs
        )
        assert "n_channels" in self and self.n_channels > 0
        assert "n_samples" in self and (self.n_samples > 0 or self.n_samples == -1)
        assert "input_type" in self and self.input_type.lower() in [
            "waveform",
            "spectrogram",
            "mel_spectrogram",
            "melspectrogram",
            "mel",
            "mfcc",
            "spectral",
        ]
        self.input_type = self.input_type.lower()
        if self.input_type in [
            "spectrogram",
            "mel_spectrogram",
            "melspectrogram",
            "mel",
            "mfcc",
            "spectral",
        ]:
            assert "n_bins" in self


class BaseInput(ReprMixin, ABC):
    """ """

    __name__ = "BaseInput"

    def __init__(self, config: InputConfig) -> None:
        """ """
        assert isinstance(config, InputConfig)
        self._config = deepcopy(config)
        self._values = None
        self._dtype = self._config.get("dtype", DEFAULTS.torch_dtype)
        self._device = self._config.get("device", DEFAULTS.device)
        self._post_init()

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        return self.from_waveform(waveform)

    @abstractmethod
    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """ """
        raise NotImplementedError

    @abstractmethod
    def _post_init(self) -> None:
        """ """
        raise NotImplementedError

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @property
    def n_channels(self) -> int:
        return self._config.n_channels

    @property
    def n_samples(self) -> int:
        if self.values is not None:
            return self.values.shape[-1]
        return self._config.n_samples

    @property
    def input_type(self) -> str:
        return self._config.input_type

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def compute_input_shape(
        self, waveform_shape: Union[Sequence[int], torch.Size]
    ) -> Tuple[Union[type(None), int], ...]:
        """

        computes the input shape of the model based on the input type and the waveform shape

        Parameters
        ----------
        waveform_shape : sequence of int or torch.Size,
            the shape of the waveform

        Returns
        -------
        tuple of int or None,
            the input shape of the model

        """
        if self.input_type == "waveform":
            return tuple(waveform_shape)
        n_samples = compute_conv_output_shape(
            waveform_shape
            if len(waveform_shape) == 3
            else [None] + list(waveform_shape),
            kernel_size=self.win_length,
            stride=self.hop_length,
            padding=[self.hop_length, self.win_length - self.hop_length],
        )[-1]
        if self.feature_fs is not None:
            n_samples = math.floor(n_samples * self.feature_fs / self.fs)
        if len(waveform_shape) == 3:
            # with a batch dimension
            return (waveform_shape[0], self.n_channels, self.n_bins, n_samples)
        else:
            return (self.n_channels, self.n_bins, n_samples)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["input_type", "n_channels", "n_samples", "dtype", "device"]


class WaveformInput(BaseInput):
    """ """

    __name__ = "WaveformInput"

    def _post_init(self) -> None:
        """ """
        assert self.input_type == "waveform"

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        self._values = torch.as_tensor(waveform).to(self.device, self.dtype)
        return self._values


class _SpectralInput(BaseInput):
    """

    Inputs from the spectro-temporal domain.

    One has to set the following parameters for initialization:
        - n_bins : int,
            the number of frequency bins
        - fs (or sample_rate) : int,
            the sample rate of the waveform
    with the following optional parameters with default values:
        - window_size: float, default: 1 / 40
            the size of the window in seconds
        - overlap_size : float, default: 1 / 80
            the overlap of the windows in seconds
        - feature_fs : None or float,
            the sample rate of the features,
            if specified, the features will be resampled
            against `fs` to this sample rate

    """

    __name__ = "_SpectralInput"

    def _post_init(self) -> None:
        """ """
        assert "n_bins" in self._config
        self.fs = self._config.get("fs", self._config.get("sample_rate", None))
        assert self.fs is not None
        self.feature_fs = self._config.get("feature_fs", None)
        if "window_size" not in self._config:
            self._config.window_size = 1 / 40
        assert 0 < self._config.window_size < 0.1
        if "overlap_size" not in self._config:
            self._config.overlap_size = 1 / 80
        assert 0 < self._config.overlap_size < self._config.window_size

    @property
    def n_bins(self) -> int:
        return self._config.n_bins

    @property
    def window_size(self) -> int:
        return round(self._config.window_size * self.fs)

    @property
    def win_length(self) -> int:
        return self.window_size

    @property
    def overlap_size(self) -> int:
        return round(self._config.overlap_size * self.fs)

    @property
    def hop_length(self) -> int:
        return self.window_size - self.overlap_size

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + [
            "n_bins",
            "win_length",
            "hop_length",
            "fs",
            "feature_fs",
        ]


class SpectrogramInput(_SpectralInput):

    __doc__ = _SpectralInput.__doc__ + """"""
    __name__ = "SpectrogramInput"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        assert self.input_type in ["spectrogram"]
        assert self.n_channels == 1
        args = inspect.getfullargspec(TT.Spectrogram.__init__).args
        for k in ["self", "n_fft", "win_length", "hop_length"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_fft"] = (self.n_bins - 1) * 2
        kwargs["win_length"] = self.win_length
        kwargs["hop_length"] = self.hop_length
        self._transform = TT.Spectrogram(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        self._values = self._transform(
            torch.as_tensor(waveform).to(self.device, self.dtype)
        )
        if self.feature_fs is not None:
            scale_factor = [1] * (self.values.ndim - 3) + [self.feature_fs / self.fs]
            self._values = F.interpolate(self._values, scale_factor=scale_factor)
        return self._values


class MelSpectrogramInput(_SpectralInput):

    __doc__ = _SpectralInput.__doc__ + """"""
    __name__ = "MelSpectrogramInput"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        assert self.input_type in [
            "mel_spectrogram",
            "mel",
            "melspectrogram",
        ]
        assert self.n_channels == 1
        args = inspect.getfullargspec(TT.MelSpectrogram.__init__).args
        for k in ["self", "sample_rate", "n_fft", "n_mels", "win_length", "hop_length"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_fft"] = (self.n_bins - 1) * 2
        kwargs["sample_rate"] = self.fs
        kwargs["n_mels"] = self.n_bins
        kwargs["win_length"] = self.win_length
        kwargs["hop_length"] = self.hop_length
        self._transform = TT.MelSpectrogram(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        self._values = self._transform(
            torch.as_tensor(waveform).to(self.device, self.dtype)
        )
        if self.feature_fs is not None:
            scale_factor = [1] * (self.values.ndim - 3) + [self.feature_fs / self.fs]
            self._values = F.interpolate(self._values, scale_factor=scale_factor)
        return self._values


class MFCCInput(_SpectralInput):

    __doc__ = _SpectralInput.__doc__ + """"""
    __name__ = "MFCCInput"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        assert self.input_type in [
            "mfcc",
        ]
        assert self.n_channels == 1
        args = inspect.getfullargspec(TT.MFCC.__init__).args
        for k in ["self", "sample_rate", "n_mfcc"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_mfcc"] = self.n_bins
        kwargs["sample_rate"] = self.fs
        kwargs["melkwargs"] = kwargs.get("melkwargs", {})
        kwargs["melkwargs"].update(
            dict(
                n_fft=(self.n_bins - 1) * 2,
                n_mels=self.n_bins,
                win_length=self.win_length,
                hop_length=self.hop_length,
            )
        )
        self._transform = TT.MFCC(**kwargs).to(self.device, self.dtype)

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        self._values = self._transform(
            torch.as_tensor(waveform).to(self.device, self.dtype)
        )
        if self.feature_fs is not None:
            scale_factor = [1] * (self.values.ndim - 3) + [self.feature_fs / self.fs]
            self._values = F.interpolate(self._values, scale_factor=scale_factor)
        return self._values


class SpectralInput(_SpectralInput):
    __doc__ = (
        _SpectralInput.__doc__
        + """

    Concatenation of 3 different types of spectrograms:
    - Spectrogram
    - MelSpectrogram
    - MFCC

    Example
    -------
    >>> input_config = InputConfig(
    ...     input_type="spectral",
    ...     n_bins=224,
    ...     n_channels=3,
    ...     fs=1000,
    ...     normalized=True
    ... )
    >>> inputer = SpectralInput(input_config)
    >>> inputer
    ... SpectralInput(
    ...     fs         = 1000,
    ...     feature_fs = None,
    ...     n_bins     = 224,
    ...     win_length = 25,
    ...     hop_length = 13,
    ...     n_channels = 3,
    ...     n_samples  = -1,
    ...     input_type = 'spectral',
    ...     dtype      = torch.float32,
    ...     device     = device(type='cuda')
    ... )
    >>> inputer(torch.rand(1, 1000*30).to(inputer.device)).shape
    torch.Size([3, 224, 2308])
    >>> inputer.compute_input_shape((1, 1000*30))
    (3, 224, 2308)
    >>> inputer.compute_input_shape((32, 1, 1000*30))
    (32, 3, 224, 2308)

    """
    )

    __name__ = "SpectralInput"

    def _post_init(self) -> None:
        """ """
        super()._post_init()
        assert self.input_type in [
            "spectral",
        ]
        assert self.n_channels == 3
        self._transforms = []
        # spectrogram
        args = inspect.getfullargspec(TT.Spectrogram.__init__).args
        for k in ["self", "n_fft", "win_length", "hop_length"]:
            args.remove(k)
        spectro_kwargs = {k: self._config[k] for k in args if k in self._config}
        spectro_kwargs["n_fft"] = (self.n_bins - 1) * 2
        spectro_kwargs["win_length"] = self.win_length
        spectro_kwargs["hop_length"] = self.hop_length
        self._transforms.append(
            TT.Spectrogram(**spectro_kwargs).to(self.device, self.dtype)
        )
        # mel spectrogram
        args = inspect.getfullargspec(TT.MelSpectrogram.__init__).args
        for k in ["self", "sample_rate", "n_fft", "n_mels", "win_length", "hop_length"]:
            args.remove(k)
        mel_kwargs = {k: self._config[k] for k in args if k in self._config}
        mel_kwargs["n_fft"] = (self.n_bins - 1) * 2
        mel_kwargs["sample_rate"] = self.fs
        mel_kwargs["n_mels"] = self.n_bins
        mel_kwargs["win_length"] = self.win_length
        mel_kwargs["hop_length"] = self.hop_length
        self._transforms.append(
            TT.MelSpectrogram(**mel_kwargs).to(self.device, self.dtype)
        )
        # MFCC
        args = inspect.getfullargspec(TT.MFCC.__init__).args
        for k in ["self", "sample_rate", "n_mfcc"]:
            args.remove(k)
        mfcc_kwargs = {k: self._config[k] for k in args if k in self._config}
        mfcc_kwargs["n_mfcc"] = self.n_bins
        mfcc_kwargs["sample_rate"] = self.fs
        mfcc_kwargs["melkwargs"] = mfcc_kwargs.get("melkwargs", {})
        mfcc_kwargs["melkwargs"].update(deepcopy(mel_kwargs))
        mfcc_kwargs["melkwargs"].pop("sample_rate")
        self._transforms.append(TT.MFCC(**mfcc_kwargs).to(self.device, self.dtype))

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """

        Parameters
        ----------
        waveform : np.ndarray or torch.Tensor,
            the waveform to be transformed

        Returns
        -------
        torch.Tensor,
            the transformed waveform

        """
        self._values = torch.as_tensor(waveform).to(self.device, self.dtype)
        cat_dim = 0 if self.values.ndim == 2 else 1
        self._values = torch.cat(
            [transform(self._values.clone()) for transform in self._transforms],
            dim=cat_dim,
        )
        if self.feature_fs is not None:
            scale_factor = [1] * (self.values.ndim - 3) + [self.feature_fs / self.fs]
            self._values = F.interpolate(self._values, scale_factor=scale_factor)
        return self._values
