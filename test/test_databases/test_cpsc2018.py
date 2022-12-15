"""
TestCPSC2018: accomplished

subsampling: accomplished
"""

from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import CPSC2018, DataBaseInfo
from torch_ecg.databases.cpsc_databases.cpsc2018 import compute_metrics


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cpsc2018"
###############################################################################


with pytest.warns(
    RuntimeWarning,
    match="Annotation file not found\\. Please call method `_download_labels`, and call method `_ls_rec` again",
):
    reader = CPSC2018(_CWD)
reader._download_labels()
reader._ls_rec()


class TestCPSC2018:
    def test_len(self):
        assert len(reader) == 10

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = CPSC2018(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = CPSC2018(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CPSC2018(_CWD, subsample=0.0)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CPSC2018(_CWD, subsample=1.01)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CPSC2018(_CWD, subsample=-0.1)

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

    def test_load_ann(self):
        for rec in reader:
            ann_1 = reader.load_ann(rec, ann_format="n")
            ann_2 = reader.load_ann(rec, ann_format="a")
            ann_3 = reader.load_ann(rec, ann_format="f")
            assert set(reader.diagnosis_num_to_abbr[item] for item in ann_1) == set(
                ann_2
            )
            assert set(reader.diagnosis_abbr_to_full[item] for item in ann_2) == set(
                ann_3
            )

        with pytest.raises(
            ValueError,
            match="`ann_format` should be one of `\\['a', 'f', 'n'\\]`, but got `xxx`",
        ):
            reader.load_ann(0, ann_format="xxx")

    def test_get_labels(self):
        # alias of `load_ann`
        for rec in reader:
            ann_1 = reader.load_ann(rec, ann_format="n")
            ann_2 = reader.get_labels(rec, ann_format="n")
            assert ann_1 == ann_2

    def test_get_subject_info(self):
        for rec in reader:
            info = reader.get_subject_info(rec)
            assert isinstance(info, dict)
            assert info.keys() == {"age", "sex"}

    def test_meta_data(self):
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed

    def test_helper(self):
        assert reader.helper() is None  # printed
        for item in ["attributes", "methods"]:
            assert reader.helper(item) is None  # printed
        assert reader.helper(["attributes", "methods"]) is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, leads=["I", 3, 9], ticks_granularity=2)

    def test_compute_metrics(self):
        with pytest.raises(NotImplementedError):
            compute_metrics()  # not implemented yet
