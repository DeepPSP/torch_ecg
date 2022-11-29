"""
TestSPH: accomplished

subsampling: NOT tested
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import SPH
from torch_ecg.utils.download import http_get


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "sph"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)

http_get(
    url="https://www.dropbox.com/s/og877l6d4bh4vew/SPH-Mini.tar.gz?dl=1",
    dst_dir=_CWD,
    extract=True,
)
###############################################################################


reader = SPH(_CWD)


class TestSPH:
    def test_len(self):
        assert len(reader) == 100

    def test_load_data(self):
        for rec in reader:
            data = reader.load_data(rec)
            data_1 = reader.load_data(rec, leads=[1, 7])
            assert data.shape[0] == 12
            assert data_1.shape[0] == 2
            assert np.allclose(data[[1, 7], :], data_1)
            data_1 = reader.load_data(rec, units="uV")
            assert np.allclose(data_1, data * 1000)
            data_1 = reader.load_data(rec, data_format="lead_last")
            assert data.shape == data_1.T.shape

        with pytest.raises(AssertionError, match="Invalid data_format: `flat`"):
            reader.load_data(rec, data_format="flat")
        with pytest.raises(AssertionError, match="Invalid units: `kV`"):
            reader.load_data(rec, units="kV")

    def test_load_ann(self):
        for rec in reader:
            ann = reader.load_ann(rec, ann_format="c")
            ann_1 = reader.load_ann(rec, ann_format="f")
            assert isinstance(ann, list) and isinstance(ann_1, list)
            assert len(ann) == len(ann_1)
            ann_1 = reader.load_ann(rec, ann_format="c", ignore_modifier=False)
            assert len(ann) == len(ann_1)
            ann_1 = reader.load_ann(rec, ann_format="f", ignore_modifier=False)
            assert len(ann) == len(ann_1)

        with pytest.raises(ValueError, match="Unknown annotation format: `flat`"):
            reader.load_ann(rec, ann_format="flat")
        with pytest.raises(
            NotImplementedError, match="Abbreviations are not supported yet"
        ):
            reader.load_ann(rec, ann_format="a")

    def test_get_subject_info(self):
        for rec in reader:
            info = reader.get_subject_info(rec)
            assert isinstance(info, dict)
            assert info.keys() == {"age", "sex"}

    def test_get_age(self):
        for rec in reader:
            age = reader.get_age(rec)
            assert isinstance(age, int)
            assert age > 0

    def test_get_sex(self):
        for rec in reader:
            sex = reader.get_sex(rec)
            assert isinstance(sex, str)
            assert sex in ["M", "F"]

    def test_get_siglen(self):
        for rec in reader:
            siglen = reader.get_siglen(rec)
            data = reader.load_data(rec)
            assert isinstance(siglen, int)
            assert siglen == data.shape[1]

    def test_meta_data(self):
        assert isinstance(reader.url, dict)
        assert reader.get_citation() is None  # printed

    def test_plot(self):
        reader.plot(0, leads=["II", 7], ticks_granularity=2)
