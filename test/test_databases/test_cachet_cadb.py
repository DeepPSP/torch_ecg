"""
TestCACHET_CADB: accomplished

subsampling: NOT tested
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases import CACHET_CADB
from torch_ecg.utils.download import http_get


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cache-cadb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)

http_get(
    url="https://www.dropbox.com/s/u0qbewjh7zjsdu6/CACHET-CADB-Mini.tar.gz?dl=1",
    dst_dir=_CWD,
    extract=True,
)
###############################################################################


reader = CACHET_CADB(_CWD)


class TestCACHET_CADB:
    def test_len(self):
        assert len(reader) == 2

    def test_load_data(self):
        for rec in reader:
            data = reader.load_data(rec)
            data_1 = reader.load_data(rec, data_format="flat", units="μV")
            assert data.ndim == 2 and data.shape[0] == 1
            assert data_1.ndim == 1 and data_1.shape[0] == data.shape[1]
            assert np.allclose(data, data_1.reshape(1, -1) / 1000, atol=1e-2)
            data_1 = reader.load_data(rec, data_format="flat", fs=2 * reader.fs)
            assert data_1.shape[0] == 2 * data.shape[1]
            data_1 = reader.load_data(rec, sampfrom=1000, sampto=5000)
            assert data_1.shape[1] == 4000
            assert np.allclose(data_1, data[:, 1000:5000])

        with pytest.raises(ValueError, match="Invalid `data_format`: xxx"):
            reader.load_data(0, data_format="xxx")
        with pytest.raises(ValueError, match="Invalid `units`: kV"):
            reader.load_data(0, units="kV")
        with pytest.raises(ValueError, match="short format file not found"):
            reader.load_data(-1)
        with pytest.raises(ValueError, match="Invalid record name: `xxx`"):
            reader.load_data("xxx")

    def test_load_context_data(self):
        for context_name in reader.context_data_ext:
            context_data = reader.load_context_data(0, context_name)
            assert context_data.ndim == 2

        with pytest.raises(ValueError, match="Call `load_data` to load ECG data"):
            reader.load_context_data(0, context_name="ecg")
        with pytest.raises(AssertionError, match="Invalid `context_name`: `xxx`"):
            reader.load_context_data(0, context_name="xxx")
        with pytest.raises(
            AssertionError, match="`units` should be `default` or `.+`, but got `xxx`"
        ):
            reader.load_context_data(0, context_name="acc", units="xxx")
        with pytest.raises(
            AssertionError, match="`channels` should be a subset of `.+`, but got"
        ):
            reader.load_context_data(0, context_name="acc", channels="xxx")
        with pytest.raises(
            AssertionError, match="`channels` should be a subset of `.+`, but got"
        ):
            reader.load_context_data(0, context_name="acc", channels=["xxx"])
        with pytest.raises(
            AssertionError, match="`channels` should be less than `\\d+`, but got"
        ):
            reader.load_context_data(0, context_name="acc", channels=10)

        with pytest.warns(RuntimeWarning, match="duplicate `channels` are removed"):
            reader.load_context_data(0, context_name="acc", channels=["accX", "accX"])
        with pytest.warns(RuntimeWarning, match="duplicate `channels` are removed"):
            reader.load_context_data(0, context_name="acc", channels=[0, "accX"])

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert isinstance(ann, pd.DataFrame)
        assert ann.columns.tolist() == ["Start", "End", "Class"]

        with pytest.raises(ValueError, match="Short format file not found"):
            reader.load_ann(-1)
        with pytest.raises(ValueError, match="Invalid record name: `xxx`"):
            reader.load_ann("xxx")
        with pytest.raises(ValueError, match="`ann_format`: `np` not supported"):
            reader.load_ann(0, ann_format="np")

    def test_load_context_ann(self):
        context_ann = reader.load_context_ann(0)
        assert isinstance(context_ann, dict)
        context_ann = reader.load_context_ann(
            0, sheet_name="movisens DataAnalyzer Parameter"
        )
        assert isinstance(context_ann, pd.DataFrame)
        context_ann = reader.load_context_ann(
            0, sheet_name="movisens DataAnalyzer Results"
        )
        assert isinstance(context_ann, pd.DataFrame)

    def test_get_subject_id(self):
        sid = reader.get_subject_id(0)
        assert isinstance(sid, str)
        assert sid in reader.all_subjects

    def test_get_record_metadata(self):
        metadata = reader.get_record_metadata(0)
        assert isinstance(metadata, dict)

    def test_meta_data(self):
        assert isinstance(reader.url, dict)
        assert reader.get_citation() is None  # printed

    def test_plot(self):
        pass  # `plot` not implemented yet
