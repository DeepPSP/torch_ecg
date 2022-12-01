"""
methods from the base class, e.g. `load_data`, are comprehensively tested in this file

TestAFDB: accomplished

subsampling: NOT tested
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases import AFDB, DataBaseInfo
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "afdb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


with pytest.warns(RuntimeWarning):
    reader = AFDB(_CWD, verbose=3)
reader.download()


class TestAFDB:
    def test_len(self):
        assert len(reader) == 23

        reader._ls_rec(local=False)
        assert len(reader) == 23

    def test_load_data(self):
        data = reader.load_data(0)
        data_muv = reader.load_data(0, units="μv")
        data_lead_last = reader.load_data(0, data_format="lead_last")
        data_0 = reader.load_data(0, leads=0)
        assert np.allclose(data, data_lead_last.T)
        assert np.allclose(data, data_muv / 1000)
        assert np.allclose(data[0], data_0)
        assert reader.load_data(0, sampfrom=1000, sampto=2000).shape == (2, 1000)
        assert reader.load_data(
            0, sampfrom=1000, sampto=2000, fs=reader.fs * 2
        ).shape == (2, 2000)
        assert reader.load_data(0, units=None).dtype == np.int32

        with pytest.raises(
            AssertionError,
            match="`leads` should be a subset of .+ or non-negative integers less than",
        ):
            reader.load_data(0, leads=3)
        with pytest.raises(
            AssertionError, match="`data_format` should be one of `.+`, but got `.+`"
        ):
            reader.load_data(0, data_format="lead_last_first")
        with pytest.raises(
            AssertionError,
            match="`data_format` should be one of `\\['channel_first', 'lead_first', 'channel_last', 'lead_last'\\]` when the passed number of `leads` is larger than 1",
        ):
            reader.load_data(0, data_format="flat")
        with pytest.raises(
            AssertionError, match="`units` should be one of `.+` or None, but got `.+`"
        ):
            reader.load_data(0, units="kV")

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert isinstance(ann, dict) and ann.keys() == reader.class_map.keys()
        ann = reader.load_ann(0, ann_format="mask")
        assert (
            isinstance(ann, np.ndarray) and ann.shape[0] == reader.load_data(0).shape[1]
        )
        ann = reader.load_ann(0, sampfrom=1000, sampto=2000)
        ann_1 = reader.load_ann(0, sampfrom=1000, sampto=2000, keep_original=True)
        for k, v in ann.items():
            for idx, itv in enumerate(v):
                assert len(ann_1[k][idx]) == len(itv) == 2
                ann_1[k][idx] = [ann_1[k][idx][0] - 1000, ann_1[k][idx][1] - 1000]
        assert ann == ann_1
        ann = reader.load_ann(0, sampfrom=1000, sampto=2000, ann_format="mask")
        ann_1 = reader.load_ann(
            0, sampfrom=1000, sampto=2000, ann_format="mask", keep_original=True
        )
        assert ann.shape == ann_1.shape == (1000,)
        assert np.allclose(ann, ann_1)

    def test_load_beat_ann(self):
        rec = reader.qrsc_records[0]
        beat_ann = reader.load_beat_ann(rec)
        assert isinstance(beat_ann, np.ndarray) and beat_ann.ndim == 1
        beat_ann = reader.load_beat_ann(rec, use_manual=False)
        assert isinstance(beat_ann, np.ndarray) and beat_ann.ndim == 1
        beat_ann = reader.load_beat_ann(rec, sampfrom=1000, sampto=2000)
        beat_ann_1 = reader.load_beat_ann(
            rec, sampfrom=1000, sampto=2000, keep_original=True
        )
        assert beat_ann.shape == beat_ann_1.shape
        assert np.allclose(beat_ann, beat_ann_1 - 1000)

    def test_load_rpeak_indices(self):
        # `load_rpeak_indices` is alias of `load_beat_ann`
        rec = reader.qrsc_records[0]
        beat_ann = reader.load_beat_ann(rec)
        rpeak_indices = reader.load_rpeak_indices(rec)
        assert np.allclose(beat_ann, rpeak_indices)

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert str(reader) == repr(reader)
        assert isinstance(reader.df_all_db_info, pd.DataFrame)
        assert len(reader.df_all_db_info) > 0
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, leads=0, ticks_granularity=2, sampfrom=1000, sampto=2000)
