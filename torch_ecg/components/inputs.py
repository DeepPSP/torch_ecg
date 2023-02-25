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
from einops.layers.torch import Rearrange

from ..cfg import CFG, DEFAULTS
from ..utils.misc import ReprMixin, add_docstring
from ..utils.utils_nn import compute_conv_output_shape
from ..utils.utils_signal_t import Spectrogram


__all__ = [
    "InputConfig",
    "WaveformInput",
    "FFTInput",
    "SpectrogramInput",
]


class InputConfig(CFG):
    """A Class to store the configuration of the input.

    Parameters
    ----------
    input_type : {"waveform", "fft", "spectrogram"}, optional
        Type of the input.
    n_channels : int
        Number of channels of the input.
    n_samples : int
        Number of samples of the input.
    ensure_batch_dim : bool
        Whether to ensure the transformed input has a batch dimension.

    Examples
    --------
    .. code-block:: python

        input_config = InputConfig(
            input_type="waveform",
            n_channels=12,
            n_samples=5000,
        )

    """

    __name__ = "InputConfig"

    def __init__(
        self,
        *args: Union[CFG, dict],
        input_type: str,
        n_channels: int,
        n_samples: int = -1,
        ensure_batch_dim: bool = True,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            *args,
            input_type=input_type,
            n_channels=n_channels,
            n_samples=n_samples,
            ensure_batch_dim=ensure_batch_dim,
            **kwargs,
        )
        assert (
            "n_channels" in self and self.n_channels > 0
        ), f"`n_channels` must be positive, got {self.n_channels}"
        assert "n_samples" in self and (
            self.n_samples > 0 or self.n_samples == -1
        ), f"`n_samples` must be positive or -1, got {self.n_samples}"
        assert "input_type" in self and self.input_type.lower() in [
            "waveform",
            "fft",
            "spectrogram",
        ], f"`input_type` must be one of ['waveform', 'fft', 'spectrogram'], got {self.input_type}"
        self.input_type = self.input_type.lower()
        if self.input_type in [
            "spectrogram",
        ]:
            assert (
                "n_bins" in self
            ), f"`n_bins` must be specified for {self.input_type} input"
            assert (
                "fs" in self or "sample_rate" in self
            ), f"`fs` or `sample_rate` must be specified for {self.input_type} input"


class BaseInput(ReprMixin, ABC):
    """Base class for all input classes.

    Parameters
    ----------
    config : InputConfig
        The configuration of the input.

    """

    __name__ = "BaseInput"

    def __init__(self, config: InputConfig) -> None:
        """ """
        assert isinstance(
            config, InputConfig
        ), "`config` must be an instance of `InputConfig`"
        self._config = deepcopy(config)
        self._values = None
        self._dtype = self._config.get("dtype", DEFAULTS.DTYPE.TORCH)
        self._device = self._config.get("device", DEFAULTS.device)
        self._post_init()

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Method to transform the waveform to the input tensor.

        Parameters
        ----------
        waveform : numpy.ndarray or torch.Tensor
            The waveform to be transformed.

        Returns
        -------
        torch.Tensor
            The transformed waveform.

        """
        return self.from_waveform(waveform)

    @abstractmethod
    def _from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Internal method to convert the waveform to the input tensor."""
        raise NotImplementedError

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transform the waveform to the input tensor.

        Parameters
        ----------
        waveform : numpy.ndarray or torch.Tensor
            The waveform to be transformed.

        Returns
        -------
        torch.Tensor
            The transformed waveform.

        """
        assert waveform.shape[-2:] == (self.n_channels, self.n_samples,), (
            f"`waveform` shape must be `(batch_size, {self.n_channels}, {self.n_samples})` "
            f"or `({self.n_channels}, {self.n_samples})`, got `{waveform.shape}`"
        )
        input_tensors = self._from_waveform(waveform)
        if waveform.ndim == 2 and self._config.ensure_batch_dim:
            input_tensors = input_tensors.unsqueeze(0)
        return input_tensors

    @abstractmethod
    def _post_init(self) -> None:
        """Method to be called after initialization"""
        raise NotImplementedError

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @property
    def n_channels(self) -> int:
        return self._config.n_channels

    @property
    def n_samples(self) -> int:
        return self._config.n_samples

    @property
    def input_channels(self) -> int:
        channel_dim = {
            "waveform": -2,
            "fft": -2,
            # "spectrogram": -3,  # implemented in `SpectrogramInput`
        }
        if self.values is not None:
            return self.values.shape[channel_dim[self.input_type]]
        return self.compute_input_shape((self.n_channels, self.n_samples))[
            channel_dim[self.input_type]
        ]

    @property
    def input_samples(self) -> int:
        if self.values is not None:
            return self.values.shape[-1]
        return self.compute_input_shape((self.n_channels, self.n_samples))[-1]

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
        """Computes the input shape of the model based on
        the input type and the waveform shape.

        Parameters
        ----------
        waveform_shape : Sequence[int] or torch.Size
            The shape of the waveform.

        Returns
        -------
        Tuple[int] or None
            The input shape of the model.

        """
        if self.input_type == "waveform":
            input_shape = tuple(waveform_shape)
        elif self.input_type == "fft":
            nfft = self.nfft or waveform_shape[-1]
            seq_len = torch.fft.rfftfreq(nfft).shape[0]
            if self.drop_dc:
                seq_len -= 1
            input_shape = (*waveform_shape[:-2], 2 * waveform_shape[-2], seq_len)
        elif self.input_type == "spectrogram":
            n_samples = compute_conv_output_shape(
                waveform_shape
                if len(waveform_shape) == 3
                else [None] + list(waveform_shape),
                kernel_size=self.win_length,
                stride=self.hop_length,
                asymmetric_padding=[self.hop_length, self.win_length - self.hop_length],
            )[-1]
            if self.feature_fs is not None:
                n_samples = math.floor(n_samples * self.feature_fs / self.fs)
            if self.to1d:
                mid_dims = (self.n_channels * self.n_bins,)
            else:
                mid_dims = (self.n_channels, self.n_bins)
            input_shape = (*waveform_shape[:-2], *mid_dims, n_samples)

        if len(waveform_shape) == 2 and self._config.ensure_batch_dim:
            input_shape = (1, *input_shape)

        return input_shape

    def extra_repr_keys(self) -> List[str]:
        return ["input_type", "n_channels", "n_samples", "dtype", "device"]


class WaveformInput(BaseInput):
    """Waveform input.

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> BATCH_SIZE = 32
    >>> N_CHANNELS = 12
    >>> N_SAMPLES = 5000
    >>> input_config = InputConfig(
    ...     input_type="waveform",
    ...     n_channels=N_CHANNELS,
    ...     n_samples=N_SAMPLES,
    ... )
    >>> inputer = WaveformInput(input_config)
    >>> waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    >>> inputer(waveform).shape
    torch.Size([32, 12, 5000])
    >>> waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    >>> inputer(waveform).shape
    torch.Size([1, 12, 5000])
    >>> input_config = InputConfig(
    ...     input_type="waveform",
    ...     n_channels=N_CHANNELS,
    ...     n_samples=N_SAMPLES,
    ...     ensure_batch_dim=False,
    ... )
    >>> inputer = WaveformInput(input_config)
    >>> waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    >>> inputer(waveform).shape
    torch.Size([12, 5000])

    """

    __name__ = "WaveformInput"

    def _post_init(self) -> None:
        """Make sure the input type is `waveform`."""
        assert self.input_type == "waveform", "`input_type` must be `waveform`"

    def _from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Internal method to convert the waveform to the input tensor."""
        self._values = torch.as_tensor(waveform).to(self.device, self.dtype)
        return self._values

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Converts the input :class:`~numpy.ndarray` or
        :class:`~torch.Tensor` waveform to a :class:`~torch.Tensor`.

        Parameters
        ----------
        waveform : numpy.ndarray or torch.Tensor
            The waveform to be transformed,
            of shape ``(batch_size, n_channels, n_samples)``
            or ``(n_channels, n_samples)``.

        Returns
        -------
        torch.Tensor
            The transformed waveform,
            of shape ``(batch_size, n_channels, n_samples)``.

        NOTE
        ----
        If the input is a 2D tensor,
        then the batch dimension is added (batch_size = 1).

        """
        return super().from_waveform(waveform)

    @add_docstring(from_waveform.__doc__)
    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """ """
        return self.from_waveform(waveform)


class FFTInput(BaseInput):
    """Inputs from the FFT, via concatenating the amplitudes and the phases.

    One can set the following optional parameters for initialization:
        - nfft: int
            the number of FFT bins.
            If nfft is None, the number of FFT bins is computed from the input shape.
        - drop_dc: bool, default True
            Whether to drop the zero frequency bin (the DC component).
        - norm: str, optional
            The normalization of the FFT, can be
                - "forward"
                - "backward"
                - "ortho"

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> BATCH_SIZE = 32
    >>> N_CHANNELS = 12
    >>> N_SAMPLES = 5000
    >>> input_config = InputConfig(
    ...     input_type="fft",
    ...     n_channels=N_CHANNELS,
    ...     n_samples=N_SAMPLES,
    ...     n_fft=200,
    ...     drop_dc=True,
    ...     norm="ortho",
    ... )
    >>> inputer = FFTInput(input_config)
    >>> waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    >>> inputer(waveform).ndim
    3
    >>> inputer(waveform).shape == inputer.compute_input_shape(waveform.shape)
    True
    >>> waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    >>> inputer(waveform).ndim
    3
    >>> inputer(waveform).shape == inputer.compute_input_shape(waveform.shape)
    True
    >>> input_config = InputConfig(
    ...     input_type="fft",
    ...     n_channels=N_CHANNELS,
    ...     n_samples=N_SAMPLES,
    ...     n_fft=None,
    ...     drop_dc=False,
    ...     norm="forward",
    ...     ensure_batch_dim=False,
    ... )
    >>> inputer = FFTInput(input_config)
    >>> waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    >>> inputer(waveform).ndim
    2
    >>> inputer(waveform).shape == inputer.compute_input_shape(waveform.shape)
    True

    """

    __name__ = "FFTInput"

    def _post_init(self) -> None:
        """Make sure the input type is `fft` and set the parameters."""
        assert self.input_type == "fft", "`input_type` must be `fft`"
        self.nfft = self._config.get("nfft", None)
        if self.nfft is None and self.n_samples > 0:
            self.nfft = self.n_samples
        self.drop_dc = self._config.get("drop_dc", True)
        self.norm = self._config.get("norm", None)
        if self.norm is not None:
            assert self.norm in [
                "forward",
                "backward",
                "ortho",
            ], f"`norm` must be one of [`forward`, `backward`, `ortho`], got {self.norm}"

    def _from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Internal method to convert the waveform to the input tensor."""
        self._values = torch.fft.rfft(
            torch.as_tensor(waveform).to(self.device, self.dtype),
            n=self.nfft,
            dim=-1,
            norm=self.norm,
        )
        if self.drop_dc:
            self._values = self._values[..., 1:]
        self._values = torch.cat(
            [torch.abs(self._values), torch.angle(self._values)], dim=-2
        )
        return self._values

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Converts the input :class:`~numpy.ndarray` or
        :class:`~torch.Tensor` waveform to a :class:`~torch.Tensor` of FFTs.

        Parameters
        ----------
        waveform : numpy.ndarray or torch.Tensor
            The waveform to be transformed,
            of shape ``(batch_size, n_channels, n_samples)``
            or ``(n_channels, n_samples)``.

        Returns
        -------
        torch.Tensor
            The transformed waveform,
            of shape ``(batch_size, 2 * n_channels, seq_len)``,
            where `seq_len` is computed via :code:`torch.fft.rfftfreq(nfft).shape[0]`,
            if `drop_dc` is True, then seq_len is reduced by 1

        NOTE
        ----
        If the input is a 2D tensor,
        then the batch dimension is added (batch_size = 1).

        """
        return super().from_waveform(waveform)

    @add_docstring(from_waveform.__doc__)
    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.from_waveform(waveform)

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + ["nfft", "drop_dc", "norm"]


class _SpectralInput(BaseInput):
    """Inputs from the spectro-temporal domain.

    One has to set the following parameters for initialization:
        - n_bins : int
            The number of frequency bins.
        - fs (or sample_rate) : int
            The sample rate of the waveform.
    with the following optional parameters with default values:
        - window_size : float, default: 1 / 20
            The size of the window in seconds.
        - overlap_size : float, default: 1 / 40
            The overlap of the windows in seconds.
        - feature_fs : None or float,
            The sample rate of the features.
            If specified, the features will be resampled
            against `fs` to this sample rate.
        - to1d : bool, default False
            Whether to convert the features to 1D.
            NOTE that if `to1d` is True,
            then if the convolutions with ``groups=1`` applied to the `input`
            acts on all the bins, which is "global"
            w.r.t. the `bins` dimension of the corresponding 2d input.

    """

    __name__ = "_SpectralInput"

    def _post_init(self) -> None:
        """Make sure the input type is `spectral` and set the parameters."""
        self.to1d = self._config.get("to1d", False)
        self.fs = self._config.get("fs", self._config.get("sample_rate"))
        self.feature_fs = self._config.get("feature_fs", None)
        if "window_size" not in self._config:
            self._config.window_size = 1 / 20
        assert (
            0 < self._config.window_size < 0.2
        ), f"`window_size` must be in (0, 0.2), got {self._config.window_size}"
        if "overlap_size" not in self._config:
            self._config.overlap_size = 1 / 40
        # TODO: consider negative overlap_size, i.e. positive gaps between windows
        assert 0 < self._config.overlap_size < self._config.window_size, (
            f"`overlap_size` must be in `(0, window_size)` = {(0, self._config.window_size)}, "
            f"got {self._config.overlap_size}"
        )

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

    @property
    def input_channels(self) -> int:
        channel_dim = -2 if self.to1d else -3
        if self.values is not None:
            return self.values.shape[channel_dim]
        return self.compute_input_shape((self.n_channels, self.n_samples))[channel_dim]

    @property
    def input_samples(self) -> Tuple[int, ...]:
        sample_dim = (-1,) if self.to1d else (-2, -1)
        if self.values is not None:
            input_shape = self.values.shape
        input_shape = self.compute_input_shape((self.n_channels, self.n_samples))
        return tuple(input_shape[dim] for dim in sample_dim)

    def extra_repr_keys(self) -> List[str]:
        return super().extra_repr_keys() + [
            "n_bins",
            "win_length",
            "hop_length",
            "fs",
            "feature_fs",
            "to1d",
        ]


class SpectrogramInput(_SpectralInput):

    __doc__ = (
        _SpectralInput.__doc__
        + """

    Examples
    --------
    >>> from torch_ecg.cfg import DEFAULTS
    >>> BATCH_SIZE = 32
    >>> N_CHANNELS = 12
    >>> N_SAMPLES = 5000
    >>> input_config = InputConfig(
    ...     name="spectrogram",
    ...     n_channels=N_CHANNELS,
    ...     n_samples=N_SAMPLES,
    ...     n_bins=128,
    ...     fs=500,
    ...     window_size=1 / 20,
    ...     overlap_size=1 / 40,
    ...     feature_fs=100,
    ...     to1d=True,
    ... )
    >>> inputer = SpectrogramInput(input_config)
    >>> waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    >>> spectrogram = inputer(waveform)
    >>> spectrogram.shape == inputer.compute_input_shape(waveform.shape)
    True

    """
    )
    __name__ = "SpectrogramInput"

    def _post_init(self) -> None:
        """Make sure the input type is `spectrogram` and set the parameters."""
        super()._post_init()
        assert self.input_type in [
            "spectrogram"
        ], f"`input_type` must be one of [`spectrogram`], got {self.input_type}"
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

    def _from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Internal method to convert the waveform to the input tensor."""
        self._values = self._transform(
            torch.as_tensor(waveform).to(self.device, self.dtype)
        )
        if self.feature_fs is not None:
            # self.values.ndim can be 2, 3, or 4
            scale_factor = [1] * (self.values.ndim - 3) + [self.feature_fs / self.fs]
            if self.values.ndim == 2:
                self._values = F.interpolate(
                    self._values.unsqueeze(0),
                    scale_factor=scale_factor,
                    recompute_scale_factor=True,
                ).squeeze(0)
            else:
                self._values = F.interpolate(
                    self._values, scale_factor=scale_factor, recompute_scale_factor=True
                )
        return self._values

    def from_waveform(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        r"""Converts the input :class:`~numpy.ndarray` or
        :class:`~torch.Tensor` waveform to a :class:`~torch.Tensor` of spectrograms.

        Parameters
        ----------
        waveform : numpy.ndarray or torch.Tensor
            The waveform to be transformed,
            of shape ``(batch_size, n_channels, n_samples)``
            or ``(n_channels, n_samples)``.

        Returns
        -------
        torch.Tensor
            The transformed waveform,
            of shape ``(batch_size, n_channels, n_bins, n_frames)``, where

            .. math::

                n\_frames = (n\_samples - win\_length) // hop\_length + 1

        NOTE
        ----
        If the input is a 2D tensor,
        then the batch dimension is added (batch_size = 1).

        """
        return super().from_waveform(waveform)

    @add_docstring(from_waveform.__doc__)
    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return self.from_waveform(waveform)
