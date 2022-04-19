"""
utilities for signal processing on PyTorch tensors

"""

import warnings
from numbers import Real
from typing import Iterable, Optional, Union, Callable

import torch

__all__ = [
    "normalize",
    "resample",
]


def normalize(
    sig: torch.Tensor,
    method: str = "z-score",
    mean: Union[Real, Iterable[Real]] = 0.0,
    std: Union[Real, Iterable[Real]] = 1.0,
    per_channel: bool = False,
    inplace: bool = True,
) -> torch.Tensor:
    r"""

    perform z-score normalization on `sig`,
    to make it has fixed mean and standard deviation,
    or perform min-max normalization on `sig`,
    or normalize `sig` using `mean` and `std` via (sig - mean) / std.
    More precisely,

        .. math::
            \begin{align*}
            \text{Min-Max normalization:} & \frac{sig - \min(sig)}{\max(sig) - \min(sig)} \\
            \text{Naive normalization:} & \frac{sig - m}{s} \\
            \text{Z-score normalization:} & \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m
            \end{align*}

    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    method: str, default "z-score",
        normalization method, one of "z-score", "min-max", "naive", case insensitive
    mean: real number or array_like, default 0.0,
        mean value of the normalized signal,
        or mean values for each lead of the normalized signal, if `method` is "z-score";
        mean values to be subtracted from the original signal, if `method` is "naive";
        useless if `method` is "min-max"
    std: real number or array_like, default 1.0,
        standard deviation of the normalized signal,
        or standard deviations for each lead of the normalized signal, if `method` is "z-score";
        std to be divided from the original signal, if `method` is "naive";
        useless if `method` is "min-max"
    per_channel: bool, default False,
        if True, normalization will be done per channel, not strictly required per channel;
        if False, normalization will be done per sample, strictly required per sample
    inplace: bool, default True,
        if True, normalization will be done inplace (on `sig`)

    Returns
    -------
    sig: Tensor,
        the normalized signal

    NOTE
    ----
    in cases where normalization is infeasible (std = 0),
    only the mean value will be shifted

    feasible shapes of `sig` and `std`, `mean` are as follows
    | shape of `sig` | `per_channel` |                shape of `std` or `mean                                     |
    |----------------|---------------|----------------------------------------------------------------------------|
    |    (b,l,s)     |     False     | scalar, (b,), (b,1), (b,1,1)                                               |
    |    (b,l,s)     |     True      | scalar, (b,), (l,), (b,1), (b,l), (l,1), (1,l), (b,1,1), (b,l,1), (1,l,1,) |
    |    (l,s)       |     False     | scalar,                                                                    |
    |    (l,s)       |     True      | scalar, (l,), (l,1), (1,l)                                                 |
    `scalar` includes native scalar or scalar tensor. One can check by
    ```python
    (b,l,s) = 2,12,20
    for shape in [(b,), (l,), (b,1), (b,l), (l,1), (1,l), (b,1,1), (b,l,1), (1,l,1,)]:
        nm_sig = normalize(torch.rand(b,l,s), per_channel=True, mean=torch.rand(*shape))
    for shape in [(b,), (b,1), (b,1,1)]:
        nm_sig = normalize(torch.rand(b,l,s), per_channel=False, mean=torch.rand(*shape))
    for shape in [(l,), (l,1), (1,l)]:
        nm_sig = normalize(torch.rand(l,s), per_channel=True, mean=torch.rand(*shape))
    ```

    """
    _method = method.lower()
    assert _method in [
        "z-score",
        "naive",
        "min-max",
    ]
    ori_shape = sig.shape
    if not inplace:
        sig = sig.clone()
    n_leads, siglen = sig.shape[-2:]
    sig = sig.reshape((-1, n_leads, siglen))  # add batch dim if necessary
    device = sig.device
    dtype = sig.dtype
    if isinstance(std, Real):
        assert std > 0, "standard deviation should be positive"
        _std = torch.full((sig.shape[0], 1, 1), std, dtype=dtype, device=device)
    else:
        _std = torch.as_tensor(std, dtype=dtype, device=device)
        assert (_std > 0).all(), "standard deviations should all be positive"
        if _std.shape[0] == sig.shape[0]:
            # of shape (batch, n_leads, 1) or (batch, 1, 1), or (batch, n_leads,) or (batch, 1) or (batch,)
            _std = _std.view((sig.shape[0], -1, 1))
        elif _std.shape[0] == sig.shape[1] or (_std.shape[:2] == (1, sig.shape[1])):
            # of shape (n_leads, 1) or (n_leads,) or (1, n_leads) or (1, n_leads, 1)
            _std = _std.view((-1, sig.shape[1], 1))
        else:
            raise ValueError(
                f"shape of `sig` = {sig.shape} and `std` = {_std.shape} mismatch"
            )
    if isinstance(mean, Real):
        _mean = torch.full((sig.shape[0], 1, 1), mean, dtype=dtype, device=device)
    else:
        _mean = torch.as_tensor(mean, dtype=dtype, device=device)
        if _mean.shape[0] == sig.shape[0]:
            # of shape (batch, n_leads, 1) or (batch, 1, 1), or (batch, n_leads,) or (batch, 1) or (batch,)
            _mean = _mean.view((sig.shape[0], -1, 1))
        elif _mean.shape[0] == sig.shape[1] or (_mean.shape[:2] == (1, sig.shape[1])):
            # of shape (n_leads, 1) or (n_leads,) or (1, n_leads) or (1, n_leads, 1)
            _mean = _mean.view((-1, sig.shape[1], 1))
        else:
            raise ValueError("shape of `sig` and `mean` mismatch")

    if not per_channel:
        assert _std.shape[1] == 1 and _mean.shape[1] == 1, (
            "if `per_channel` is False, `std` and `mean` should be scalars, "
            "or of shape (batch, 1), or (batch, 1, 1), or (1,)"
        )

    # print(f"sig.shape = {sig.shape}, _mean.shape = {_mean.shape}, _std.shape = {_std.shape}")

    if _method == "naive":
        sig = sig.sub_(_mean).div_(_std)
        return sig

    eps = 1e-7  # to avoid dividing by zero
    if not per_channel:
        options = dict(dim=(-1, -2), keepdims=True)
    else:
        options = dict(dim=-1, keepdims=True)

    if _method == "z-score":
        ori_mean, ori_std = sig.mean(**options), sig.std(**options).add_(eps)
        sig = sig.sub_(ori_mean).div_(ori_std).mul_(_std).add_(_mean).reshape(ori_shape)
    elif _method == "min-max":
        ori_min, ori_max = sig.amin(**options), sig.amax(**options)
        sig = sig.sub_(ori_min).div_(ori_max.sub(ori_min).add(eps)).reshape(ori_shape)
    return sig


def resample(
    sig: torch.Tensor,
    fs: Optional[int] = None,
    dst_fs: Optional[int] = None,
    siglen: Optional[int] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """

    resample signal tensors to a new sampling frequency or a new signal length,

    Parameters
    ----------
    sig: Tensor,
        signal to be normalized, assumed to have shape (..., n_leads, siglen)
    fs: int, optional,
        sampling frequency of the source signal to be resampled
    dst_fs: int, optional,
        sampling frequency of the resampled ECG
    siglen: int, optional,
        number of samples in the resampled ECG,
        one of only one of `dst_fs` (with `fs`) and `siglen` should be specified
    inplace: bool, default False,
        if True, normalization will be done inplace (on the signal)

    """
    assert (
        sum([bool(dst_fs), bool(siglen)]) == 1
    ), "one and only one of `fs` and `siglen` should be set"
    if dst_fs is not None:
        assert fs is not None, "if `dst_fs` is set, `fs` should also be set"
        scale_factor = dst_fs / fs
    if not inplace:
        sig = sig.clone()
    if sig.ndim == 2:
        sig = torch.nn.functional.interpolate(
            sig.unsqueeze(0),
            size=siglen,
            scale_factor=scale_factor,
            mode="linear",
            align_corners=True,
        ).squeeze(0)
    else:
        sig = torch.nn.functional.interpolate(
            sig,
            size=siglen,
            scale_factor=scale_factor,
            mode="linear",
            align_corners=True,
        )

    return sig


def spectrogram(
    waveform: torch.Tensor,
    pad: int,
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: Optional[float],
    normalized: bool,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: bool = True,
) -> torch.Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw signal.
    The spectrogram can be either magnitude-only or complex.

    Args:
        waveform (Tensor): Tensor of dimension `(..., time)`
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool): Whether to normalize by magnitude after stft
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``
        return_complex (bool, optional):
            Indicates whether the resulting complex-valued Tensor should be represented with
            native complex dtype, such as `torch.cfloat` and `torch.cdouble`, or real dtype
            mimicking complex value with an extra dimension for real and imaginary parts.
            (See also ``torch.view_as_real``.)
            This argument is only effective when ``power=None``. It is ignored for
            cases where ``power`` is a number as in those cases, the returned tensor is
            power spectrogram, which is a real-valued tensor.

    Returns:
        Tensor: Dimension `(..., freq, time)`, freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).

    This function is borrowed from torchaudio.functional.spectrogram

    """
    if power is None and not return_complex:
        warnings.warn(
            "The use of pseudo complex type in spectrogram is now deprecated."
            "Please migrate to native complex type by providing `return_complex=True`. "
            "Please refer to https://github.com/pytorch/audio/issues/1337 "
            "for more details about torchaudio's plan to migrate to native complex type."
        )

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if normalized:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    if not return_complex:
        return torch.view_as_real(spec_f)
    return spec_f


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a signal.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Indicates whether the resulting complex-valued Tensor should be represented with
            native complex dtype, such as `torch.cfloat` and `torch.cdouble`, or real dtype
            mimicking complex value with an extra dimension for real and imaginary parts.
            (See also ``torch.view_as_real``.)
            This argument is only effective when ``power=None``. It is ignored for
            cases where ``power`` is a number as in those cases, the returned tensor is
            power spectrogram, which is a real-valued tensor.

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)

    This class is borrowed from torchaudio.transforms.Spectrogram

    """
    __constants__ = ["n_fft", "win_length", "hop_length", "pad", "power", "normalized"]

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        return_complex: bool = True,
    ) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = (
            window_fn(self.win_length)
            if wkwargs is None
            else window_fn(self.win_length, **wkwargs)
        )
        self.register_buffer("window", window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex,
        )
