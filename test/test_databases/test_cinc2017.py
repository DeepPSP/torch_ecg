"""
TestCINC2017: accomplished

subsampling: accomplished
"""

import re
import shutil
from pathlib import Path

import pytest

from torch_ecg.databases import CINC2017, DataBaseInfo
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN

###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cinc2017"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


with pytest.warns(RuntimeWarning):
    reader = CINC2017(_CWD)
if len(reader) == 0:
    reader.download()


class TestCINC2017:
    def test_len(self):
        assert len(reader) == 8528

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = CINC2017(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = CINC2017(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CINC2017(_CWD, subsample=0.0)
        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CINC2017(_CWD, subsample=1.01)
        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CINC2017(_CWD, subsample=-0.1)

    def test_load_data(self):
        rec = reader._validation_set[0]
        data = reader.load_data(rec)
        assert data.ndim == 2
        data = reader.load_data(rec, leads=0, data_format="flat")
        assert data.ndim == 1
        data = reader.load_data(rec, leads=[0], data_format="flat", sampfrom=1000, sampto=2000)
        assert data.shape == (1000,)

    def test_load_ann(self):
        rec = 0
        ann = reader.load_ann(rec)
        ann_1 = reader.load_ann(rec, version=1)
        ann_2 = reader.load_ann(rec, ann_format="f")
        assert ann in reader.d_ann_names
        assert ann_1 in reader.d_ann_names
        assert ann_2 in reader.d_ann_names.values()
        with pytest.raises(ValueError, match="Annotation version v100 does not exist! Choose from "):
            reader.load_ann(rec, version=100)

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(PHYSIONET_DB_VERSION_PATTERN, reader.version)
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, ticks_granularity=2)
