"""
"""

from pathlib import Path

import numpy as np

from torch_ecg.databases import CINC2021
from torch_ecg.utils._preproc import preprocess_multi_lead_signal, preprocess_single_lead_signal, rpeaks_detect_multi_leads

_SAMPLE_DATA_DIR = Path(__file__).resolve().parents[2] / "sample-data" / "cinc2021"

reader = CINC2021(_SAMPLE_DATA_DIR)


def test_preprocess_multi_lead_signal():
    raw_data = reader.load_data(0, leads=["II", "aVR"])
    fs = reader.get_fs(0)
    data = preprocess_multi_lead_signal(
        raw_data,
        fs,
        bl_win=[0.2, 0.6],
        band_fs=[0.5, 45],
        rpeak_fn="xqrs",
        verbose=2,
    )
    assert isinstance(data, dict)
    assert data.keys() == {"filtered_ecg", "rpeaks"}
    assert data["filtered_ecg"].shape == raw_data.shape
    assert data["rpeaks"].ndim == 1
    data = preprocess_multi_lead_signal(
        raw_data.T,
        fs,
        sig_fmt="channel_last",
        bl_win=[0.2, 0.6],
        band_fs=[0.5, 45],
    )
    assert isinstance(data, dict)
    assert data.keys() == {"filtered_ecg", "rpeaks"}
    assert data["filtered_ecg"].shape == raw_data.shape
    assert len(data["rpeaks"]) == 0


def test_preprocess_single_lead_signal():
    raw_data = reader.load_data(0, leads=["II"]).squeeze()
    fs = reader.get_fs(0)
    data = preprocess_single_lead_signal(
        raw_data,
        fs,
        bl_win=[0.2, 0.6],
        band_fs=[0.5, 45],
        rpeak_fn="gqrs",
        verbose=2,
    )
    assert isinstance(data, dict)
    assert data.keys() == {"filtered_ecg", "rpeaks"}
    assert data["filtered_ecg"].shape == raw_data.shape
    assert data["rpeaks"].ndim == 1
    data = preprocess_single_lead_signal(
        raw_data,
        fs,
        bl_win=[0.2, 0.6],
        band_fs=[0.5, 45],
    )
    assert isinstance(data, dict)
    assert data.keys() == {"filtered_ecg", "rpeaks"}
    assert data["filtered_ecg"].shape == raw_data.shape
    assert len(data["rpeaks"]) == 0


def test_rpeaks_detect_multi_leads():
    raw_data = reader.load_data(0, leads=["II", "aVR"])
    fs = reader.get_fs(0)
    rpeaks = rpeaks_detect_multi_leads(raw_data, fs, rpeak_fn="xqrs", verbose=2)
    assert isinstance(rpeaks, np.ndarray)
    assert rpeaks.ndim == 1
