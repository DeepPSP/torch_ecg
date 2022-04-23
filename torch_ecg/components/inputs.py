"""
TODO: add examples for the classes
"""

import inspect
import math
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import NoReturn, Union, List, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from einops.layers.torch import Rearrange

from ..cfg import CFG, DEFAULTS
from ..utils.misc import ReprMixin
from ..utils.utils_nn import compute_conv_output_shape
from ..utils.utils_signal_t import Spectrogram


__all__ = [
    "InputConfig",
    "WaveformInput",
    "FFTInput",
    "SpectrogramInput",
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
    ) -> NoReturn:
        """

        Parameters
        ----------
        input_type : str,
            the type of the input, can be
            - "waveform"
            - "fft"
            - "spectrogram"
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
            "fft",
            "spectrogram",
        ]
        self.input_type = self.input_type.lower()
        if self.input_type in [
            "spectrogram",
        ]:
            assert "n_bins" in self


class BaseInput(ReprMixin, ABC):
    """ """

    __name__ = "BaseInput"

    def __init__(self, config: InputConfig) -> NoReturn:
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
    def _post_init(self) -> NoReturn:
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
        if self.input_type == "fft":
            nfft = self.nfft or waveform_shape[-1]
            seq_len = torch.fft.rfftfreq(nfft).shape[0]
            if self.drop_dc:
                seq_len -= 1
            return (*waveform_shape[:-2], 2 * waveform_shape[-2], nfft)
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
        if self.to1d:
            mid_dims = (self.n_channels * self.n_bins,)
        else:
            mid_dims = (self.n_channels, self.n_bins)
        return (*waveform_shape[:-2], *mid_dims, n_samples)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["input_type", "n_channels", "n_samples", "dtype", "device"]


class WaveformInput(BaseInput):
    """ """

    __name__ = "WaveformInput"

    def _post_init(self) -> NoReturn:
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


class FFTInput(BaseInput):
    """

    Inputs from the FFT, via concatenating the amplitudes and the phases.

    One can set the following optional parameters for initialization:
        - nfft : int,
            the number of FFT bins
            if nfft is None, the number of FFT bins is computed from the input shape
        - drop_dc: bool, default True,
            whether to drop the zero frequency bin (the DC component)
        - norm: str, optional,
            the normalization of the FFT, can be
            - "forward"
            - "backward"
            - "ortho"

    """

    __name__ = "FFTInput"

    def _post_init(self) -> NoReturn:
        """ """
        assert self.input_type == "fft"
        self.nfft = self._config.get("nfft", None)
        if self.nfft is None and self.n_samples > 0:
            self.nfft = self.n_samples
        self.drop_dc = self._config.get("drop_dc", True)
        self.norm = self._config.get("norm", None)
        if self.norm is not None:
            assert self.norm in ["forward", "backward", "ortho"]

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
        self._values = torch.fft.rfft(
            torch.as_tensor(waveform).to(self.device, self.dtype),
            n=self.nfft,
            dim=-1,
            norm=self.norm,
        )
        if self.drop_dc:
            self._values = self._values[..., 1:]
        self._values = torch.cat(
            [torch.abs(self._values), torch.angle(self._values)], dim=1
        )
        return self._values

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + ["nfft", "drop_dc", "norm"]


class _SpectralInput(BaseInput):
    """

    Inputs from the spectro-temporal domain.

    One has to set the following parameters for initialization:
        - n_bins : int,
            the number of frequency bins
        - fs (or sample_rate) : int,
            the sample rate of the waveform
    with the following optional parameters with default values:
        - window_size: float, default: 1 / 20
            the size of the window in seconds
        - overlap_size : float, default: 1 / 40
            the overlap of the windows in seconds
        - feature_fs : None or float,
            the sample rate of the features,
            if specified, the features will be resampled
            against `fs` to this sample rate
        - to1d : bool, default: False
            whether to convert the features to 1D.
            NOTE that if `to1d` is True,
            then if the convolutions with `groups=1` applied to the `input`
            acts on all the bins, which is "global"
            w.r.t. the `bins` dimension of the corresponding 2d input.

    """

    __name__ = "_SpectralInput"

    def _post_init(self) -> NoReturn:
        """ """
        assert "n_bins" in self._config
        self.fs = self._config.get("fs", self._config.get("sample_rate", None))
        assert self.fs is not None
        self.feature_fs = self._config.get("feature_fs", None)
        if "window_size" not in self._config:
            self._config.window_size = 1 / 20
        assert 0 < self._config.window_size < 0.2
        if "overlap_size" not in self._config:
            self._config.overlap_size = 1 / 40
        assert 0 < self._config.overlap_size < self._config.window_size
        self.to1d = self._config.get("to1d", False)

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
            "to1d",
        ]


class SpectrogramInput(_SpectralInput):

    __doc__ = _SpectralInput.__doc__ + """"""
    __name__ = "SpectrogramInput"

    def _post_init(self) -> NoReturn:
        """ """
        super()._post_init()
        assert self.input_type in ["spectrogram"]
        args = inspect.getfullargspec(Spectrogram.__init__).args
        for k in ["self", "n_fft", "win_length", "hop_length"]:
            args.remove(k)
        kwargs = {k: self._config[k] for k in args if k in self._config}
        kwargs["n_fft"] = (self.n_bins - 1) * 2
        kwargs["win_length"] = self.win_length
        kwargs["hop_length"] = self.hop_length
        self._transform = torch.nn.Sequential()
        self._transform.add_module(
            "spectrogram", Spectrogram(**kwargs).to(self.device, self.dtype)
        )
        if self.to1d:
            self._transform.add_module(
                "to1d",
                Rearrange("... channel n_bins time -> ... (channel n_bins) time"),
            )

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
