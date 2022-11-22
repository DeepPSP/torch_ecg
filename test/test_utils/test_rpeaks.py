"""
"""

from pathlib import Path

import numpy as np
import wfdb

from torch_ecg.utils.rpeaks import (
    xqrs_detect,
    gqrs_detect,
    hamilton_detect,
    ssf_detect,
    christov_detect,
    engzee_detect,
    gamboa_detect,
)


sample_path = list(
    (Path(__file__).parents[2] / "sample-data" / "cinc2021").resolve().rglob("*.mat")
)[0]
rec = wfdb.rdrecord(str(sample_path).replace(".mat", ""), channels=[0])
sig = rec.p_signal.flatten()
fs = rec.fs


def test_xqrs_detect():
    rpeaks = xqrs_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_gqrs_detect():
    rpeaks = gqrs_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_hamilton_detect():
    rpeaks = hamilton_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_ssf_detect():
    rpeaks = ssf_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_christov_detect():
    rpeaks = christov_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_engzee_detect():
    rpeaks = engzee_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)


def test_gamboa_detect():
    rpeaks = gamboa_detect(sig, fs)
    assert isinstance(rpeaks, np.ndarray)
    print(rpeaks)
