"""
Transforms that does not exist in old versions of torchaudio.

Added transforms:
- ``PitchShift`` (added in ``torchaudio`` 0.9.x)

NOTE:
- ``PitchShift`` seems to raise the following error:
``"arange_cuda" not implemented for 'ComplexFloat'`` or
``"arange_cpu" not implemented for 'ComplexFloat'``

"""

import math
from typing import Optional, NoReturn, Callable

import torch
from torch import Tensor

# from torchaudio.functional import phase_vocoder


__all__ = [
    "PitchShift",
    "pitch_shift",
]


class PitchShift(torch.nn.Module):
    r"""Shift the pitch of a waveform by ``n_steps`` steps.
    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default : ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then ``win_length // 4``
            is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).
    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.PitchShift(sample_rate, 4)
        >>> waveform_shift = transform(waveform)  # (channel, time)

    """
    __constants__ = [
        "sample_rate",
        "n_steps",
        "bins_per_octave",
        "n_fft",
        "win_length",
        "hop_length",
    ]

    def __init__(
        self,
        sample_rate: int,
        n_steps: int,
        bins_per_octave: int = 12,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
    ) -> NoReturn:
        super().__init__()
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        window = (
            window_fn(self.win_length)
            if wkwargs is None
            else window_fn(self.win_length, **wkwargs)
        )
        self.register_buffer("window", window)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension `(..., time)`.
        Returns:
            Tensor: The pitch-shifted audio of shape `(..., time)`.

        """

        return pitch_shift(
            waveform,
            self.sample_rate,
            self.n_steps,
            self.bins_per_octave,
            self.n_fft,
            self.win_length,
            self.hop_length,
            self.window,
        )


def pitch_shift(
    waveform: Tensor,
    sample_rate: int,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
    """
    Shift the pitch of a waveform by ``n_steps`` steps.
    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default: ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then
            ``win_length // 4`` is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).
    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.

    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length, device=waveform.device)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    ori_len = shape[-1]
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(
        0, math.pi * hop_length, spec_f.shape[-2], device=spec_f.device
    )[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    waveform_stretch = torch.istft(
        spec_stretch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=len_stretch,
    )
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)
    shift_len = waveform_shift.size()[-1]
    if shift_len > ori_len:
        waveform_shift = waveform_shift[..., :ori_len]
    else:
        waveform_shift = torch.nn.functional.pad(
            waveform_shift, [0, ori_len - shift_len]
        )

    # unpack batch
    waveform_shift = waveform_shift.view(shape[:-1] + waveform_shift.shape[-1:])
    return waveform_shift


def phase_vocoder(
    complex_specgrams: Tensor, rate: float, phase_advance: Tensor
) -> Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.
    Args:
        complex_specgrams (Tensor): Dimension of `(..., freq, time, complex=2)`
        rate (float): Speed-up factor
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of (freq, 1)
    Returns:
        Tensor: Complex Specgrams Stretch with dimension of `(..., freq, ceil(time/rate), complex=2)`
    Example
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time, complex=2)
        >>> complex_specgrams = torch.randn(2, freq, 300, 2)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231, 2])

    """

    # pack batch
    shape = complex_specgrams.size()
    complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-3:]))

    time_steps = torch.arange(
        0,
        complex_specgrams.size(-2),
        rate,
        device=complex_specgrams.device,
        dtype=complex_specgrams.dtype,
    )

    alphas = time_steps % 1.0
    phase_0 = angle(complex_specgrams[..., :1, :])

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 0, 0, 2])

    # (new_bins, freq, 2)
    complex_specgrams_0 = complex_specgrams.index_select(-2, time_steps.long())
    complex_specgrams_1 = complex_specgrams.index_select(-2, (time_steps + 1).long())

    angle_0 = angle(complex_specgrams_0)
    angle_1 = angle(complex_specgrams_1)

    norm_0 = torch.norm(complex_specgrams_0, p=2, dim=-1)
    norm_1 = torch.norm(complex_specgrams_1, p=2, dim=-1)

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    real_stretch = mag * torch.cos(phase_acc)
    imag_stretch = mag * torch.sin(phase_acc)

    complex_specgrams_stretch = torch.stack([real_stretch, imag_stretch], dim=-1)

    # unpack batch
    complex_specgrams_stretch = complex_specgrams_stretch.reshape(
        shape[:-3] + complex_specgrams_stretch.shape[1:]
    )

    return complex_specgrams_stretch


def angle(complex_tensor: Tensor) -> Tensor:
    r"""Compute the angle of complex tensor input.
    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
    Return:
        Tensor: Angle of a complex tensor. Shape of `(..., )`
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int,
    rolloff: float,
    resampling_method: str,
    beta: Optional[float],
    device: torch.device = torch.device("cpu"),
    dtype: Optional[torch.dtype] = None,
):

    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method not in ["sinc_interpolation", "kaiser_window"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    assert lowpass_filter_width > 0
    kernels = []
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64
    idx = torch.arange(-width, width + orig_freq, device=device, dtype=idx_dtype)

    for i in range(new_freq):
        t = (-i / new_freq + idx / orig_freq) * base_freq
        t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

        # we do not use built in torch windows here as we need to evaluate the window
        # at specific positions, not over a regular grid.
        if resampling_method == "sinc_interpolation":
            window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        else:
            # kaiser_window
            if beta is None:
                beta = 14.769656459379492
            beta_tensor = torch.tensor(float(beta))
            window = torch.i0(
                beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)
            ) / torch.i0(beta_tensor)
        t *= math.pi
        kernel = torch.where(t == 0, torch.tensor(1.0).to(t), torch.sin(t) / t)
        kernel.mul_(window)
        kernels.append(kernel)

    scale = base_freq / orig_freq
    kernels = torch.stack(kernels).view(new_freq, 1, -1).mul_(scale)
    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)
    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = int(math.ceil(new_freq * length / orig_freq))
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interpolation",
    beta: Optional[float] = None,
) -> Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation.
    https://ccrma.stanford.edu/~jos/resample/Theory_Ideal_Bandlimited_Interpolation.html
    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.
    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interpolation``, ``kaiser_window``] (Default: ``'sinc_interpolation'``)
        beta (float or None, optional): The shape parameter used for kaiser window.
    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    assert orig_freq > 0.0 and new_freq > 0.0

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(
        waveform, orig_freq, new_freq, gcd, kernel, width
    )
    return resampled
