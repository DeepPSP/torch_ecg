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


class DummyPreProcessor(PreProcessor):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        return sig, fs


def test_preproc_manager():
    ppm = PreprocManager()
    assert ppm.empty
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

    sig = torch.randn(12, 80000).numpy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs

    config = CFG(
        random=False,
        resample={"fs": 500},
        bandpass={"filter_type": "fir"},
        normalize={"method": "min-max"},
    )
    ppm = PreprocManager.from_config(config)

    sig = torch.randn(12, 80000).numpy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs


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
