"""
"""

import itertools

import pytest
import torch

from torch_ecg.cfg import DEFAULTS
from torch_ecg.components.inputs import BaseInput, FFTInput, InputConfig, SpectrogramInput, WaveformInput, _SpectralInput

BATCH_SIZE = 32
N_CHANNELS = 12
N_SAMPLES = 5000


def test_input_config():
    input_config = InputConfig(
        input_type="waveform",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
    )

    with pytest.raises(AssertionError, match="`n_channels` must be positive"):
        input_config = InputConfig(
            input_type="waveform",
            n_channels=0,
            n_samples=N_SAMPLES,
        )
    with pytest.raises(AssertionError, match="`n_samples` must be positive or -1"):
        input_config = InputConfig(
            input_type="waveform",
            n_channels=N_CHANNELS,
            n_samples=0,
        )
    with pytest.raises(AssertionError, match="`input_type` must be one of"):
        input_config = InputConfig(
            input_type="invalid",
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
        )
    with pytest.raises(AssertionError, match="`n_bins` must be specified for spectrogram input"):
        input_config = InputConfig(
            input_type="spectrogram",
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            fs=500,
        )
    with pytest.raises(
        AssertionError,
        match="`fs` or `sample_rate` must be specified for spectrogram input",
    ):
        input_config = InputConfig(
            input_type="spectrogram",
            n_channels=N_CHANNELS,
            n_samples=N_SAMPLES,
            n_bins=128,
        )
        si = SpectrogramInput(input_config)


def test_base_input():
    input_config = InputConfig(
        input_type="waveform",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
    )
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {BaseInput.__name__} with abstract method",
    ):
        bi = BaseInput(input_config)


def test_waveform_input():
    input_config = InputConfig(
        input_type="waveform",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
    )
    wi = WaveformInput(input_config)

    assert (wi.input_channels, wi.input_samples) == wi.compute_input_shape((N_CHANNELS, N_SAMPLES))[-2:]

    waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    waveform_input = wi(waveform)
    assert waveform_input.shape == wi.compute_input_shape(waveform.shape) == (BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    assert isinstance(waveform_input, torch.Tensor)
    assert waveform_input.dtype == wi.dtype
    assert waveform_input.device.type == wi.device.type

    waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    waveform_input = wi.from_waveform(waveform)
    assert waveform_input.shape == wi.compute_input_shape(waveform.shape) == (1, N_CHANNELS, N_SAMPLES)
    assert isinstance(waveform_input, torch.Tensor)
    assert waveform_input.dtype == wi.dtype
    assert waveform_input.device.type == wi.device.type

    input_config = InputConfig(
        input_type="waveform",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        ensure_batch_dim=False,
    )
    wi = WaveformInput(input_config)

    waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
    waveform_input = wi(waveform)
    assert waveform_input.shape == wi.compute_input_shape(waveform.shape) == (BATCH_SIZE, N_CHANNELS, N_SAMPLES)

    waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
    waveform_input = wi.from_waveform(waveform)
    assert waveform_input.shape == wi.compute_input_shape(waveform.shape) == (N_CHANNELS, N_SAMPLES)

    with pytest.raises(
        AssertionError,
        match=(
            f"`waveform` shape must be `\\(batch_size, {wi.n_channels}, {wi.n_samples}\\)` "
            f"or `\\({wi.n_channels}, {wi.n_samples}\\)`"
        ),
    ):
        waveform = torch.randn(BATCH_SIZE, 2, N_SAMPLES)
        waveform_input = wi(waveform)
    with pytest.raises(
        AssertionError,
        match=(
            f"`waveform` shape must be `\\(batch_size, {wi.n_channels}, {wi.n_samples}\\)` "
            f"or `\\({wi.n_channels}, {wi.n_samples}\\)`"
        ),
    ):
        waveform = torch.randn(BATCH_SIZE, N_CHANNELS, 4000)
        waveform_input = wi(waveform)

    assert str(wi) == repr(wi)


def test_fft_input():
    init_config = dict(
        input_type="fft",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
    )
    aux_config_grid = itertools.product(
        [True, False],  # ensure_batch_dim
        [None, 100, 200],  # n_fft
        [True, False],  # drop_dc
        ["forward", "backward", "ortho"],  # norm (normalization)
    )
    for ensure_batch_dim, n_fft, drop_dc, norm in aux_config_grid:
        input_config = InputConfig(
            **init_config,
            ensure_batch_dim=ensure_batch_dim,
            n_fft=n_fft,
            drop_dc=drop_dc,
            norm=norm,
        )
        fi = FFTInput(input_config)

        assert (fi.input_channels, fi.input_samples) == fi.compute_input_shape((N_CHANNELS, N_SAMPLES))[-2:]

        waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
        fft_input = fi(waveform)
        assert fft_input.ndim == 3
        assert fft_input.shape == fi.compute_input_shape(waveform.shape)
        assert isinstance(fft_input, torch.Tensor)
        assert fft_input.dtype == fi.dtype
        assert fft_input.device.type == fi.device.type

        waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
        fft_input = fi.from_waveform(waveform)
        assert fft_input.ndim == 3 if ensure_batch_dim else 2
        assert fft_input.shape == fi.compute_input_shape(waveform.shape)
        assert isinstance(fft_input, torch.Tensor)
        assert fft_input.dtype == fi.dtype
        assert fft_input.device.type == fi.device.type

    with pytest.raises(
        AssertionError,
        match=(
            f"`waveform` shape must be `\\(batch_size, {fi.n_channels}, {fi.n_samples}\\)` "
            f"or `\\({fi.n_channels}, {fi.n_samples}\\)`"
        ),
    ):
        waveform = torch.randn(BATCH_SIZE, 2, N_SAMPLES)
        fft_input = fi(waveform)
    with pytest.raises(
        AssertionError,
        match=(
            f"`waveform` shape must be `\\(batch_size, {fi.n_channels}, {fi.n_samples}\\)` "
            f"or `\\({fi.n_channels}, {fi.n_samples}\\)`"
        ),
    ):
        waveform = torch.randn(BATCH_SIZE, N_CHANNELS, 4000)
        fft_input = fi(waveform)

    assert str(fi) == repr(fi)


def test_spectral_input():
    input_config = InputConfig(
        input_type="spectrogram",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        n_bins=128,
        fs=500,
    )
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {_SpectralInput.__name__} with abstract method",
    ):
        si = _SpectralInput(input_config)


def test_spectrogram_input():
    init_config = dict(
        input_type="spectrogram",
        n_channels=N_CHANNELS,
        n_samples=N_SAMPLES,
        n_bins=128,
        fs=500,
    )
    aux_config_grid = itertools.product(
        [True, False],  # ensure_batch_dim
        [1 / 20, 1 / 40],  # window_size
        [0.1, 0.25, 0.5, 0.8],  # overlap_ratio
        [10, 50, None],  # feature_fs
        [True, False],  # to1d
    )

    for (
        ensure_batch_dim,
        window_size,
        overlap_ratio,
        feature_fs,
        to1d,
    ) in aux_config_grid:
        overlap_size = window_size * overlap_ratio
        input_config = InputConfig(
            **init_config,
            ensure_batch_dim=ensure_batch_dim,
            window_size=window_size,
            overlap_size=overlap_size,
            feature_fs=feature_fs,
            to1d=to1d,
        )

        si = SpectrogramInput(input_config)

        idx = -2 if to1d else -3
        assert (si.input_channels, *si.input_samples) == si.compute_input_shape((N_CHANNELS, N_SAMPLES))[idx:]

        waveform = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES)
        spectrogram_input = si(waveform)
        assert spectrogram_input.ndim == 3 if to1d else 4
        assert spectrogram_input.shape == si.compute_input_shape(waveform.shape)
        assert isinstance(spectrogram_input, torch.Tensor)
        assert spectrogram_input.dtype == si.dtype
        assert spectrogram_input.device.type == si.device.type

        waveform = DEFAULTS.RNG.uniform(size=(N_CHANNELS, N_SAMPLES))
        spectrogram_input = si.from_waveform(waveform)
        if ensure_batch_dim:
            assert spectrogram_input.ndim == 3 if to1d else 4
        else:
            assert spectrogram_input.ndim == 2 if to1d else 3
        assert spectrogram_input.shape == si.compute_input_shape(waveform.shape)
        assert isinstance(spectrogram_input, torch.Tensor)
        assert spectrogram_input.dtype == si.dtype
        assert spectrogram_input.device.type == si.device.type

    with pytest.raises(AssertionError, match="`window_size` must be in \\(0, 0.2\\)"):
        input_config = InputConfig(**init_config, window_size=0.3)
        si = SpectrogramInput(input_config)
    with pytest.raises(AssertionError, match="`overlap_size` must be in `\\(0, window_size\\)`"):
        input_config = InputConfig(**init_config, window_size=0.1, overlap_size=0.3)
        si = SpectrogramInput(input_config)

    with pytest.raises(AssertionError, match="`waveform` shape must be"):
        waveform = torch.randn(BATCH_SIZE, 2, N_SAMPLES)
        spectrogram_input = si(waveform)
    with pytest.raises(AssertionError, match="`waveform` shape must be"):
        waveform = torch.randn(BATCH_SIZE, N_CHANNELS, 4000)
        spectrogram_input = si(waveform)

    assert str(si) == repr(si)
