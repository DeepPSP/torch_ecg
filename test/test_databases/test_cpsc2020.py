"""
TestCPSC2020: accomplished
TestCPSC2020Dataset: NOT accomplished

subsampling: NOT tested
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import CPSC2020, DataBaseInfo
from torch_ecg.databases.cpsc_databases.cpsc2020 import compute_metrics
from torch_ecg.utils import validate_interval


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cpsc2020"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = CPSC2020(_CWD)
reader.download()


class TestCPSC2020:
    def test_len(self):
        assert len(reader) == 10

    def test_load_data(self):
        data_1 = reader.load_data(0, sampfrom=2000, sampto=4000, data_format="flat")
        data_2 = reader.load_data(
            0, sampfrom=2000, sampto=4000, data_format="channel_last"
        )
        data_3 = reader.load_data(0, sampfrom=2000, sampto=4000, units="uV")
        data_4 = reader.load_data(0, sampfrom=2000, sampto=4000, fs=reader.fs * 2)
        assert data_1.shape == (2000,)
        assert data_2.shape == (2000, 1)
        assert np.allclose(data_1, data_2[:, 0])
        assert np.allclose(data_1, data_3 / 1000, atol=1e-2)
        assert data_4.shape == (1, 4000)

    def test_load_ann(self):
        ann = reader.load_ann(0, sampfrom=1000, sampto=9000)
        assert ann.keys() == {"SPB_indices", "PVC_indices"}
        assert all(isinstance(v, np.ndarray) for v in ann.values())

    def test_locate_premature_beats(self):
        premature_beat_intervals = reader.locate_premature_beats(0)
        assert len(premature_beat_intervals) > 0
        premature_beat_intervals_1 = reader.locate_premature_beats(
            0, sampfrom=1000, sampto=90000
        )
        assert len(premature_beat_intervals_1) <= len(premature_beat_intervals)
        premature_beat_intervals = reader.locate_premature_beats(
            0, premature_type="SPB"
        )
        assert (
            len(premature_beat_intervals) == 0
            or validate_interval(premature_beat_intervals)[0]
        )
        premature_beat_intervals = reader.locate_premature_beats(
            0, premature_type="PVC"
        )
        assert (
            len(premature_beat_intervals) == 0
            or validate_interval(premature_beat_intervals)[0]
        )

    def test_train_test_split_rec(self):
        for test_rec_num in range(1, 5):
            split_res = reader.train_test_split_rec(test_rec_num=test_rec_num)
            assert split_res.keys() == {"train", "test"}
            assert len(split_res["train"]) == 10 - test_rec_num
            assert len(split_res["test"]) == test_rec_num
        with pytest.raises(ValueError, match="test data ratio too high"):
            reader.train_test_split_rec(test_rec_num=5)
        with pytest.raises(ValueError, match="Invalid `test_rec_num`"):
            reader.train_test_split_rec(test_rec_num=15)
        with pytest.raises(ValueError, match="Invalid `test_rec_num`"):
            reader.train_test_split_rec(test_rec_num=0)

    def test_meta_data(self):
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, ticks_granularity=2, sampfrom=2000, sampto=4000)

    def test_compute_metrics(self):
        sbp_true_0 = reader.load_ann(0)["SPB_indices"]
        pvc_true_0 = reader.load_ann(0)["PVC_indices"]
        sbp_true_1 = reader.load_ann(1)["SPB_indices"]
        pvc_true_1 = reader.load_ann(1)["PVC_indices"]
        assert compute_metrics(
            [sbp_true_0, sbp_true_1],
            [pvc_true_0, pvc_true_1],
            [sbp_true_0, sbp_true_1],
            [pvc_true_0, pvc_true_1],
        ) == (0, 0)
        Score1, Score2 = compute_metrics(
            [sbp_true_0], [pvc_true_0], [sbp_true_1], [pvc_true_1]
        )
        assert Score1 < 0 and Score2 < 0
        assert compute_metrics(
            [sbp_true_0], [pvc_true_0], [sbp_true_1], [pvc_true_1], verbose=2
        ).keys() == {
            "total_loss",
            "class_loss",
            "true_positive",
            "false_positive",
            "false_negative",
        }
