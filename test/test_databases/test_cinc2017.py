"""
TestCINC2017: accomplished

subsampling: NOT tested
"""

import re
import shutil
from pathlib import Path

import pytest

from torch_ecg.databases import CINC2017
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
reader.download()


class TestCINC2017:
    def test_len(self):
        assert len(reader) == 8528

    def test_load_data(self):
        rec = reader._validation_set[0]
        data = reader.load_data(rec)
        assert data.ndim == 2
        data = reader.load_data(rec, leads=0, data_format="flat")
        assert data.ndim == 1
        data = reader.load_data(
            rec, leads=[0], data_format="flat", sampfrom=1000, sampto=2000
        )
        assert data.shape == (1000,)

    def test_load_ann(self):
        rec = 0
        ann = reader.load_ann(rec)
        ann_1 = reader.load_ann(rec, original=True)
        ann_2 = reader.load_ann(rec, ann_format="f")
        assert ann in reader.d_ann_names
        assert ann_1 in reader.d_ann_names
        assert ann_2 in reader.d_ann_names.values()

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed

    def test_plot(self):
        reader.plot(0, ticks_granularity=2)
