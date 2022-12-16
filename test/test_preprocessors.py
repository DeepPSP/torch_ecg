"""
"""

import itertools
from numbers import Real
from typing import Tuple

import numpy as np
import torch
import pytest

from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import (  # noqa: F401
    PreProcessor,
    PreprocManager,
    BandPass,
    BaselineRemove,
    Normalize,
    MinMaxNormalize,
    NaiveNormalize,
    ZScoreNormalize,
    Resample,
    preprocess_multi_lead_signal,
    preprocess_single_lead_signal,
)  # noqa: F401


test_sig = torch.randn(12, 80000).numpy()


class DummyPreProcessor(PreProcessor):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        return sig, fs


def test_preproc_manager():
    sig = test_sig.copy()

    ppm = PreprocManager()
    assert ppm.empty
    sig, fs = ppm(sig, 200)
    ppm.add_(BandPass(0.5, 40))
    assert not ppm.empty
    ppm.add_(Resample(100), pos=0)
    ppm.add_(ZScoreNormalize())
    ppm.add_(BaselineRemove(), pos=1)
    assert len(ppm.preprocessors) == 4
    del ppm.preprocessors[-1]
    assert len(ppm.preprocessors) == 3
    ppm.add_(NaiveNormalize())
    ppm.add_(DummyPreProcessor(), pos=0)

    sig = test_sig.copy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs

    config = CFG(
        random=True,
        resample={"fs": 500},
        bandpass={"filter_type": "fir"},
        normalize={"method": "min-max"},
        baseline_remove={"window1": 0.3, "window2": 0.7},
    )
    ppm = PreprocManager.from_config(config)

    sig = test_sig.copy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs

    config = {}
    with pytest.warns(
        RuntimeWarning,
        match="No preprocessors added to the manager\\. You are using a dummy preprocessor",
    ):
        ppm = PreprocManager.from_config(config)
    assert ppm.empty
    sig = test_sig.copy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs


def test_bandpass():
    sig = test_sig.copy()
    bp = BandPass(0, 40)
    sig, fs = bp(sig, 200)
    bp = BandPass(0.5, None)
    sig, fs = bp(sig, 200)

    assert str(bp) == repr(bp)


def test_baseline_remove():
    sig = test_sig.copy()
    br = BaselineRemove()
    sig, fs = br(sig, 200)
    br = BaselineRemove(0.3, 0.9)
    sig, fs = br(sig, 200)

    with pytest.warns(
        RuntimeWarning, match="values of `window1` and `window2` are switched"
    ):
        br = BaselineRemove(0.9, 0.3)

    assert str(br) == repr(br)


def test_normalize():
    sig = test_sig.copy()
    std = 0.5 * np.ones(sig.shape[0])
    norm = Normalize(std=std, per_channel=True)
    sig, fs = norm(sig, 200)

    with pytest.raises(AssertionError, match="standard deviation should be positive"):
        norm = Normalize(std=0)
    with pytest.raises(
        AssertionError, match="standard deviations should all be positive"
    ):
        norm = Normalize(std=np.zeros(sig.shape[0]))

    assert str(norm) == repr(norm)

    norm = MinMaxNormalize(per_channel=True)
    sig, fs = norm(sig, 200)

    assert str(norm) == repr(norm)

    norm = NaiveNormalize(per_channel=True)
    sig, fs = norm(sig, 200)

    assert str(norm) == repr(norm)

    norm = ZScoreNormalize(per_channel=True)
    sig, fs = norm(sig, 200)

    assert str(norm) == repr(norm)


def test_resample():
    sig = test_sig.copy()
    rsmp = Resample(fs=500)
    sig, fs = rsmp(sig, 200)
    rsmp = Resample(siglen=5000)
    sig, fs = rsmp(sig, 200)

    with pytest.raises(
        AssertionError, match="one and only one of `fs` and `siglen` should be set"
    ):
        rsmp = Resample(fs=500, siglen=5000)
    with pytest.raises(
        AssertionError, match="one and only one of `fs` and `siglen` should be set"
    ):
        rsmp = Resample()

    assert str(rsmp) == repr(rsmp)


def test_preprocess_multi_lead_signal():
    sig = torch.randn(12, 8000).numpy()
    fs = 200

    grid = itertools.product(
        ["lead_first", "channel_last"],  # sig_fmt
        [None, [0.2, 0.6]],  # bl_win
        [None, [0.5, 45], [-np.inf, 40], [1, fs]],  # band_fs
        ["butter", "fir"],  # filter_type
    )
    for sig_fmt, bl_win, band_fs, filter_type in grid:
        if sig_fmt == "channel_last":
            filt_sig = sig.transpose(1, 0)
        else:
            filt_sig = sig.copy()
        filt_sig = preprocess_multi_lead_signal(
            filt_sig,
            fs,
            sig_fmt=sig_fmt,
            bl_win=bl_win,
            band_fs=band_fs,
            filter_type=filter_type,
        )

    with pytest.raises(AssertionError, match="multi-lead signal should be 2d array"):
        preprocess_multi_lead_signal(sig[0], fs)
    with pytest.raises(AssertionError, match="multi-lead signal should be 2d array"):
        preprocess_multi_lead_signal(sig[np.newaxis, ...], fs)

    with pytest.raises(
        AssertionError, match="multi-lead signal format `xxx` not supported"
    ):
        preprocess_multi_lead_signal(sig, fs, sig_fmt="xxx")

    with pytest.raises(AssertionError, match="Invalid frequency band"):
        preprocess_multi_lead_signal(sig, fs, band_fs=[1, 0.5])
    with pytest.raises(AssertionError, match="Invalid frequency band"):
        preprocess_multi_lead_signal(sig, fs, band_fs=[0, fs])

    with pytest.raises(ValueError, match="Unsupported filter type `xxx`"):
        preprocess_multi_lead_signal(sig, fs, band_fs=[0.5, 45], filter_type="xxx")


def test_preprocess_single_lead_signal():
    sig = torch.randn(8000).numpy()
    fs = 200

    grid = itertools.product(
        [None, [0.2, 0.6]],  # bl_win
        [None, [0.5, 45], [-np.inf, 40], [1, fs]],  # band_fs
        ["butter", "fir"],  # filter_type
    )
    for bl_win, band_fs, filter_type in grid:
        filt_sig = preprocess_single_lead_signal(
            sig,
            fs,
            bl_win=bl_win,
            band_fs=band_fs,
            filter_type=filter_type,
        )

    with pytest.raises(AssertionError, match="single-lead signal should be 1d array"):
        preprocess_single_lead_signal(sig[np.newaxis, ...], fs)

    with pytest.raises(AssertionError, match="Invalid frequency band"):
        preprocess_single_lead_signal(sig, fs, band_fs=[1, 0.5])
    with pytest.raises(AssertionError, match="Invalid frequency band"):
        preprocess_single_lead_signal(sig, fs, band_fs=[0, fs])

    with pytest.raises(ValueError, match="Unsupported filter type `xxx`"):
        preprocess_single_lead_signal(sig, fs, band_fs=[0.5, 45], filter_type="xxx")
