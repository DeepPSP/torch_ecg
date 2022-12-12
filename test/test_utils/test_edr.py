"""
"""

import warnings
from pathlib import Path

import pytest

from torch_ecg.databases import CINC2021
from torch_ecg.utils.rpeaks import xqrs_detect
from torch_ecg.utils._edr import phs_edr


warnings.simplefilter(action="ignore", category=DeprecationWarning)


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"
###############################################################################


reader = CINC2021(_CWD)


def test_edr():
    for rec in reader:
        signal = reader.load_data(rec)[2]
        fs = reader.get_fs(rec)

        rpeaks = xqrs_detect(signal, fs=fs)
        # respiratory_rate
        rsp = phs_edr(signal, fs, rpeaks, return_with_time=False)
        assert rsp.shape == rpeaks.shape
        rsp = phs_edr(signal, fs, rpeaks, return_with_time=False, mode="simple")
        assert rsp.shape == rpeaks.shape
        rsp = phs_edr(signal, fs, rpeaks, return_with_time=True)
        assert rsp.shape[1] == 2 and rsp.shape[0] == rpeaks.shape[0]

    phs_edr(signal, fs, rpeaks, verbose=-1)
    phs_edr(signal, fs, rpeaks, verbose=2)

    with pytest.raises(ValueError, match="No mode named `invalid`!"):
        phs_edr(signal, fs, rpeaks, return_with_time=True, mode="invalid")
