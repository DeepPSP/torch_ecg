"""
"""

import pytest
import torch

from torch_ecg.utils.utils_signal_t import Spectrogram, normalize, resample


def test_normalize():
    (b, l, s) = 2, 12, 20
    nm_sig = normalize(torch.randn(b, l, s), method="min-max", inplace=True)
    for shape in [
        (b,),
        (l,),
        (b, 1),
        (b, l),
        (l, 1),
        (1, l),
        (b, 1, 1),
        (b, l, 1),
        (1, l, 1),
    ]:
        nm_sig = normalize(
            torch.randn(b, l, s),
            per_channel=True,
            mean=torch.rand(*shape),
            std=torch.rand(*shape),
        )
    for shape in [(b,), (b, 1), (b, 1, 1)]:
        nm_sig = normalize(
            torch.randn(b, l, s),
            method="naive",
            per_channel=False,
            mean=torch.rand(*shape),
        )
    for shape in [(l,), (l, 1), (1, l)]:
        nm_sig = normalize(torch.randn(l, s), per_channel=True, mean=torch.rand(*shape))

    with pytest.raises(
        AssertionError,
        match="method `not-supported` not supported, choose from 'z-score', 'naive', 'min-max'",
    ):
        nm_sig = normalize(torch.randn(b, l, s), method="not-supported")
    with pytest.raises(AssertionError, match="standard deviation should be positive"):
        nm_sig = normalize(torch.randn(b, l, s), std=0)
    with pytest.raises(AssertionError, match="standard deviations should all be positive"):
        nm_sig = normalize(
            torch.randn(b, l, s),
            std=torch.zeros(
                b,
            ),
        )
    with pytest.raises(
        AssertionError,
        match=(
            "if `per_channel` is False, `std` and `mean` should be scalars, "
            "or of shape \\(batch, 1\\), or \\(batch, 1, 1\\), or \\(1,\\)"
        ),
    ):
        nm_sig = normalize(torch.randn(b, l, s), per_channel=False, std=torch.rand(b, l))
    with pytest.raises(ValueError, match="shape of `sig` = .+ and `std` = .+ mismatch"):
        nm_sig = normalize(torch.randn(b, l, s), std=torch.rand(b, l, s), per_channel=True)
    with pytest.raises(ValueError, match="shape of `sig` = .+ and `mean` = .+ mismatch"):
        nm_sig = normalize(torch.randn(b, l, s), mean=torch.rand(b, l, s), per_channel=True)


def test_resample():
    sig = torch.randn(2, 12, 2000)

    assert resample(sig, fs=200, dst_fs=500).shape == (2, 12, 5000)
    assert resample(sig, fs=500, dst_fs=250).shape == (2, 12, 1000)
    assert resample(sig, siglen=5000, inplace=True).shape == (2, 12, 5000)
    assert resample(sig[0], siglen=500).shape == (12, 500)

    with pytest.raises(AssertionError, match="one and only one of `dst_fs` and `siglen` should be set"):
        resample(sig, dst_fs=500, siglen=1000)
    with pytest.raises(AssertionError, match="if `dst_fs` is set, `fs` should also be set"):
        resample(sig, dst_fs=500)


def test_spectrogram():
    waveform = torch.randn(32, 2000)  # (batch, length)
    n_bins = 224
    win_length = 25
    hop_length = 13
    config = dict(
        n_fft=(n_bins - 1) * 2,
        win_length=win_length,
        hop_length=hop_length,
    )
    transform = Spectrogram(**config)
    spec = transform(waveform)
    assert spec.shape == (32, n_bins, 154)
    config["power"] = 1.0
    config["pad"] = 16
    transform = Spectrogram(**config)
    spec = transform(waveform)
    assert spec.shape == (32, n_bins, 157)

    with pytest.warns(
        RuntimeWarning,
        match="The use of pseudo complex type in spectrogram is now deprecated",
    ):
        new_config = config.copy()
        new_config["power"] = None
        new_config["return_complex"] = False
        transform = Spectrogram(**new_config)
        spec = transform(waveform)
