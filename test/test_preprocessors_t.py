"""
"""

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

    sig = test_sig.clone()
    sig = ppm(sig)

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
    bp = BandPass(fs=500)
    sig = test_sig.clone()
    sig = bp(sig)

    bp = BandPass(fs=500, lowcut=0, highcut=40)
    sig = test_sig.clone()
    sig = bp(sig)

    bp = BandPass(fs=500, lowcut=1.5, highcut=None, inplace=False)
    sig = test_sig.clone()
    sig = bp(sig)

    del bp, sig


def test_baseline_remove() -> None:
    br = BaselineRemove(fs=500, inplace=False)
    sig = test_sig.clone()
    sig = br(sig)

    br = BaselineRemove(fs=500, window1=0.3, window2=0.7)
    sig = test_sig.clone()
    sig = br(sig)

    with pytest.warns(RuntimeWarning, match="values of `window1` and `window2` are switched"):
        br = BaselineRemove(fs=500, window1=0.7, window2=0.3)

    del br, sig


def test_normalize() -> None:
    norm = Normalize(method="min-max", inplace=False)
    sig = test_sig.clone()
    sig = norm(sig)

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
