""" """

import numpy as np
import pytest
import torch

from torch_ecg.cfg import CFG
from torch_ecg.preprocessors import (
    BandPass,
    BaselineRemove,
    MinMaxNormalize,
    NaiveNormalize,
    Normalize,
    PreprocManager,
    Resample,
    ZScoreNormalize,
)

test_sig = torch.randn(2, 12, 8000)
test_sig_np = test_sig.cpu().numpy()


class DummyPreProcessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return sig


def test_preproc_manager() -> None:
    ppm = PreprocManager(random=True)
    assert ppm.random
    assert ppm.empty
    sig = test_sig.clone()
    sig = ppm(sig)

    ppm.add_(DummyPreProcessor())
    assert not ppm.empty
    ppm.add_(BandPass(fs=500))
    ppm.add_(BaselineRemove(fs=500))
    ppm.add_(Normalize(method="min-max"))
    ppm.add_(Resample(fs=300, dst_fs=500), pos=0)

    # Test with torch.Tensor
    sig = test_sig.clone()
    sig = ppm(sig)
    assert isinstance(sig, torch.Tensor)

    # Test with np.ndarray
    sig_np = test_sig_np.copy()
    sig_np = ppm(sig_np)
    assert isinstance(sig_np, np.ndarray)

    config = CFG(
        random=False,
        bandpass={"fs": 500},
        normalize={"method": "min-max"},
        resample={"fs": 500, "dst_fs": 300},
        baseline_remove={"fs": 500},
        xxx={"fs": 500},  # ignored by `from_config`
    )
    ppm = PreprocManager.from_config(config)
    assert not ppm.random

    sig = test_sig.clone()
    sig = ppm(sig)

    ppm.rearrange(
        new_ordering=[
            "resample",
            "bandpass",
            "baseline_remove",
            "normalize",
        ]
    )

    ppm.random = True
    with pytest.warns(RuntimeWarning, match="The preprocessors are applied in random order"):
        ppm.rearrange(
            new_ordering=[
                "bandpass",
                "baseline_remove",
                "resample",
                "normalize",
            ]
        )
    ppm.random = False

    with pytest.raises(AssertionError, match="Duplicate preprocessor names"):
        ppm.rearrange(
            new_ordering=[
                "bandpass",
                "baseline_remove",
                "resample",
                "normalize",
                "bandpass",
            ]
        )
    with pytest.raises(AssertionError, match="Number of preprocessors mismatch"):
        ppm.rearrange(
            new_ordering=[
                "bandpass",
                "baseline_remove",
                "resample",
            ]
        )

    with pytest.warns(RuntimeWarning, match="No preprocessors added to the manager"):
        ppm = PreprocManager.from_config({"random": True})

    with pytest.warns(RuntimeWarning, match="No preprocessors added to the manager"):
        ppm = PreprocManager.from_config({"bandpass": False})

    del ppm, sig


def test_bandpass() -> None:
    from torch_ecg.utils.utils_signal_t import bandpass_filter

    bp = BandPass(fs=500)
    # Tensor
    sig = test_sig.clone()
    sig = bp(sig)
    assert isinstance(sig, torch.Tensor)
    # ndarray
    sig_np = test_sig_np.copy()
    sig_np = bp(sig_np)
    assert isinstance(sig_np, np.ndarray)

    # lowcut=0 now warns and disables the high-pass side (treated as lowpass)
    bp = BandPass(fs=500, lowcut=0, highcut=40)
    sig = test_sig.clone()
    with pytest.warns(RuntimeWarning, match="lowcut <= 0"):
        sig = bp(sig)

    bp = BandPass(fs=500, lowcut=1.5, highcut=None, inplace=False)
    sig = test_sig.clone()
    sig = bp(sig)

    # highcut >= nyquist warns and disables the low-pass side (treated as highpass)
    with pytest.warns(RuntimeWarning, match="highcut >= Nyquist"):
        bandpass_filter(test_sig.clone(), fs=500, lowcut=1.0, highcut=250.0)

    # invalid fs
    with pytest.raises(ValueError, match="fs must be a positive real number"):
        bandpass_filter(test_sig.clone(), fs=-1)

    # lowcut >= nyquist raises
    with pytest.raises(ValueError, match="lowcut must be less than Nyquist"):
        bandpass_filter(test_sig.clone(), fs=500, lowcut=300.0)

    # highcut <= 0 raises
    with pytest.raises(ValueError, match="highcut must be positive"):
        bandpass_filter(test_sig.clone(), fs=500, highcut=-5.0)

    # lowcut >= highcut raises
    with pytest.raises(ValueError, match="lowcut must be less than highcut"):
        bandpass_filter(test_sig.clone(), fs=500, lowcut=80.0, highcut=40.0)

    del bp, sig


def test_baseline_remove() -> None:
    br = BaselineRemove(fs=500, inplace=False)
    # Tensor
    sig = test_sig.clone()
    sig = br(sig)
    assert isinstance(sig, torch.Tensor)
    # ndarray
    sig_np = test_sig_np.copy()
    sig_np = br(sig_np)
    assert isinstance(sig_np, np.ndarray)

    br = BaselineRemove(fs=500, window1=0.3, window2=0.7)
    sig = test_sig.clone()
    sig = br(sig)

    with pytest.warns(RuntimeWarning, match="values of `window1` and `window2` are switched"):
        br = BaselineRemove(fs=500, window1=0.7, window2=0.3)

    del br, sig


def test_normalize() -> None:
    norm = Normalize(method="min-max", inplace=False)
    # Tensor
    sig = test_sig.clone()
    sig = norm(sig)
    assert isinstance(sig, torch.Tensor)
    # ndarray
    sig_np = test_sig_np.copy()
    sig_np = norm(sig_np)
    assert isinstance(sig_np, np.ndarray)

    norm = Normalize(method="z-score")
    sig = test_sig.clone()
    sig = norm(sig)

    norm = MinMaxNormalize()
    sig = test_sig.clone()
    sig = norm(sig)

    norm = ZScoreNormalize()
    sig = test_sig.clone()
    sig = norm(sig)

    norm = NaiveNormalize()
    sig = test_sig.clone()
    sig = norm(sig)

    del norm, sig


def test_resample() -> None:
    rsmp = Resample(fs=500, dst_fs=300)
    # Tensor
    sig = test_sig.clone()
    sig = rsmp(sig)
    assert isinstance(sig, torch.Tensor)
    # ndarray
    sig_np = test_sig_np.copy()
    sig_np = rsmp(sig_np)
    assert isinstance(sig_np, np.ndarray)

    del rsmp, sig
