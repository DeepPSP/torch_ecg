"""
TestLTAFDB: accomplished

subsampling: accomplished
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import LTAFDB, BeatAnn, DataBaseInfo
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN
from torch_ecg.utils.utils_interval import validate_interval


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "ltafdb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


with pytest.warns(RuntimeWarning):
    reader = LTAFDB(_CWD)
reader.download()


class TestLTAFDB:
    def test_len(self):
        assert len(reader) == 84

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = LTAFDB(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = LTAFDB(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LTAFDB(_CWD, subsample=0.0)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LTAFDB(_CWD, subsample=1.01)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LTAFDB(_CWD, subsample=-0.1)

    def test_load_data(self):
        rec = 0
        data = reader.load_data(rec)
        assert data.ndim == 2
        data = reader.load_data(rec, leads=0, data_format="flat")
        assert data.ndim == 1
        data = reader.load_data(
            rec, leads=[0], data_format="flat", sampfrom=1000, sampto=2000
        )
        assert data.shape == (1000,)

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert isinstance(ann, dict) and ann.keys() == {"beat", "rhythm"}
        assert isinstance(ann["beat"], list) and all(
            isinstance(a, BeatAnn) for a in ann["beat"]
        )
        assert (
            isinstance(ann["rhythm"], dict)
            and ann["rhythm"].keys() <= reader.rhythm_types_map.keys()
        )
        for v in ann["rhythm"].values():
            assert isinstance(v, list) and all(validate_interval(i)[0] for i in v)

        ann = reader.load_ann(0, sampfrom=1000, sampto=9000, keep_original=True)
        assert isinstance(ann, dict) and ann.keys() == {"beat", "rhythm"}

        ann = reader.load_ann(0, rhythm_format="mask", beat_format="dict")
        assert isinstance(ann, dict) and ann.keys() == {"beat", "rhythm"}
        assert isinstance(ann["beat"], dict) and ann["beat"].keys() <= set(
            reader.beat_types
        )
        for v in ann["beat"].values():
            assert isinstance(v, np.ndarray) and v.ndim == 1
        assert isinstance(ann["rhythm"], np.ndarray) and ann["rhythm"].ndim == 1
        data = reader.load_data(0, leads=[0], data_format="flat")
        assert ann["rhythm"].shape == data.shape

    def test_load_rhythm_ann(self):
        rhythm_ann = reader.load_rhythm_ann(0)
        assert (
            isinstance(rhythm_ann, dict)
            and rhythm_ann.keys() <= reader.rhythm_types_map.keys()
        )
        for v in rhythm_ann.values():
            assert isinstance(v, list) and all(validate_interval(i)[0] for i in v)

        rhythm_ann = reader.load_rhythm_ann(
            0, sampfrom=1000, sampto=9000, keep_original=True
        )
        assert isinstance(rhythm_ann, dict)

        rhythm_ann = reader.load_rhythm_ann(0, rhythm_format="mask")
        assert isinstance(rhythm_ann, np.ndarray) and rhythm_ann.ndim == 1
        data = reader.load_data(0, leads=[0], data_format="flat")
        assert rhythm_ann.shape == data.shape

    def test_load_beat_ann(self):
        beat_ann = reader.load_beat_ann(0)
        assert isinstance(beat_ann, list) and all(
            isinstance(a, BeatAnn) for a in beat_ann
        )

        beat_ann = reader.load_beat_ann(
            0, sampfrom=1000, sampto=9000, keep_original=True
        )
        assert isinstance(beat_ann, list)

        beat_ann = reader.load_beat_ann(0, beat_format="dict")
        assert isinstance(beat_ann, dict) and beat_ann.keys() <= set(reader.beat_types)
        for v in beat_ann.values():
            assert isinstance(v, np.ndarray) and v.ndim == 1

    def test_load_rpeak_indices(self):
        rpeak_indices = reader.load_rpeak_indices(0)
        assert isinstance(rpeak_indices, np.ndarray) and rpeak_indices.ndim == 1

        rpeak_indices = reader.load_rpeak_indices(
            0, sampfrom=1000, sampto=9000, keep_original=True
        )
        assert isinstance(rpeak_indices, np.ndarray) and rpeak_indices.ndim == 1
        rpeak_indices_1 = reader.load_rpeak_indices(
            0, sampfrom=1000, sampto=9000, keep_original=False
        )
        assert isinstance(rpeak_indices_1, np.ndarray) and rpeak_indices_1.ndim == 1
        assert np.all(rpeak_indices_1 == rpeak_indices - 1000)

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, leads=0, ticks_granularity=2, sampfrom=1000, sampto=3000)
        reader.plot(0, ticks_granularity=0, sampfrom=1000, sampto=3000)
        data = reader.load_data(0, sampfrom=1000, sampto=3000)
        reader.plot(0, data=data, ticks_granularity=1)
