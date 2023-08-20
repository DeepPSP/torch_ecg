"""
TestCINC2018: accomplished

subsampling: accomplished
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import CINC2018, DataBaseInfo
from torch_ecg.utils.download import http_get, PHYSIONET_DB_VERSION_PATTERN


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cinc2018"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = CINC2018(_CWD)

for file in [
    "training/tr03-0005/tr03-0005-arousal.mat",
    "training/tr03-0005/tr03-0005.arousal",
    "training/tr03-0005/tr03-0005.mat",
    "training/tr03-0005/tr03-0005.hea",
    "training/tr12-0685/tr12-0685-arousal.mat",
    "training/tr12-0685/tr12-0685.arousal",
    "training/tr12-0685/tr12-0685.hea",
    "training/tr12-0685/tr12-0685.mat",
    "test/te06-0293/te06-0293.hea",
    "test/te06-0293/te06-0293.mat",
]:
    url = reader.get_file_download_url(file)
    http_get(url, _CWD, extract=False)

reader._ls_rec()


class TestCINC2018:
    def test_len(self):
        assert len(reader) == 2

    def test_set_subset(self):
        assert reader._subset == "training"
        reader.set_subset("test")
        assert reader._subset == "test"
        assert len(reader) == 1
        reader.set_subset(None)
        assert reader._subset is None
        assert len(reader) == 3
        reader.set_subset("training")
        assert reader._subset == "training"
        assert len(reader) == 2

    def test_subsample(self):
        ss_ratio = 0.6
        reader_ss = CINC2018(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = CINC2018(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CINC2018(_CWD, subsample=0.0)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CINC2018(_CWD, subsample=1.01)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            CINC2018(_CWD, subsample=-0.1)

    def test_get_available_signals(self):
        available_signals = reader.get_available_signals(0)
        assert isinstance(available_signals, list)
        assert len(available_signals) > 0
        assert all([isinstance(s, str) for s in available_signals])

    def test_get_fs(self):
        fs = reader.get_fs(0)
        assert isinstance(fs, int)

    def test_get_siglen(self):
        siglen = reader.get_siglen(0)
        assert isinstance(siglen, int)

    def test_load_psg_data(self):
        available_signals = reader.get_available_signals(0)
        psg_data = reader.load_psg_data(0)
        assert isinstance(psg_data, np.ndarray)
        assert psg_data.shape == (len(available_signals), reader.get_siglen(0))

        psg_data = reader.load_psg_data(
            0,
            channel=available_signals[1],
            data_format="flat",
            sampfrom=10000,
            sampto=50000,
        )
        assert isinstance(psg_data, np.ndarray)
        assert psg_data.shape == (40000,)

        psg_data = reader.load_psg_data(
            0,
            channel=available_signals[:3],
            data_format="channel_last",
            sampfrom=10000,
            sampto=50000,
            physical=False,
            fs=2 * reader.get_fs(0),
        )
        assert isinstance(psg_data, np.ndarray)
        assert psg_data.shape == (80000, 3)

        with pytest.raises(
            AssertionError,
            match="`data_format` should be one of `.+`, but got `.+`",
        ):
            reader.load_psg_data(0, data_format="invalid")

        with pytest.raises(
            AssertionError,
            match=(
                "`data_format` should be one of `.+` when the passed number "
                "of `channel` is larger than 1, but got `.+`"
            ),
        ):
            reader.load_psg_data(0, data_format="plain")

    def test_load_data(self):
        data = reader.load_data(0)
        assert isinstance(data, np.ndarray)
        assert data.shape == (1, reader.get_siglen(0))
        data = reader.load_data(0, data_format="channel_last", units="uv")
        assert isinstance(data, np.ndarray)
        assert data.shape == (reader.get_siglen(0), 1)
        data = reader.load_data(
            0, data_format="flat", sampfrom=10000, sampto=50000, fs=2 * reader.get_fs(0)
        )
        assert isinstance(data, np.ndarray)
        assert data.shape == (80000,)
        data, data_fs = reader.load_data(0, fs=100, return_fs=True)
        assert data_fs == 100

    def test_load_ecg_data(self):
        # alias of `load_data`
        data = reader.load_ecg_data(0)
        assert isinstance(data, np.ndarray)
        assert data.shape == (1, reader.get_siglen(0))

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert isinstance(ann, dict)
        assert ann.keys() == {"arousals", "sleep_stages"}
        assert isinstance(ann["arousals"], dict)
        assert isinstance(ann["sleep_stages"], dict)
        assert set(ann["arousals"].keys()) <= set(reader.arousal_types)
        assert set(ann["sleep_stages"].keys()) <= set(reader.sleep_stage_names)
        for k in ann["arousals"].keys():
            assert isinstance(ann["arousals"][k], list)
            for itv in ann["arousals"][k]:
                assert len(itv) == 2
                assert isinstance(itv[0], int)
                assert isinstance(itv[1], int)
        for k in ann["sleep_stages"].keys():
            assert isinstance(ann["sleep_stages"][k], list)
            for itv in ann["sleep_stages"][k]:
                assert len(itv) == 2
                assert isinstance(itv[0], int)
                assert isinstance(itv[1], int)

        SAMPFROM = 10000
        SAMPTO = reader.get_siglen(0) - 100
        ann = reader.load_ann(0, sampfrom=SAMPFROM, sampto=SAMPTO, keep_original=False)
        ann_1 = reader.load_ann(0, sampfrom=SAMPFROM, sampto=SAMPTO, keep_original=True)
        assert set(ann["arousals"].keys()) == set(ann_1["arousals"].keys())
        assert set(ann["sleep_stages"].keys()) == set(ann_1["sleep_stages"].keys())
        for k in ann["arousals"].keys():
            assert len(ann["arousals"][k]) == len(ann_1["arousals"][k])
            for idx, itv in enumerate(ann["arousals"][k]):
                assert itv[0] == ann_1["arousals"][k][idx][0] - SAMPFROM
                assert itv[1] == ann_1["arousals"][k][idx][1] - SAMPFROM
        for k in ann["sleep_stages"].keys():
            assert len(ann["sleep_stages"][k]) == len(ann_1["sleep_stages"][k])
            for idx, itv in enumerate(ann["sleep_stages"][k]):
                assert itv[0] == ann_1["sleep_stages"][k][idx][0] - SAMPFROM
                assert itv[1] == ann_1["sleep_stages"][k][idx][1] - SAMPFROM

    def test_load_sleep_stages_ann(self):
        sleep_stages_ann = reader.load_sleep_stages_ann(0)
        assert isinstance(sleep_stages_ann, dict)
        assert set(sleep_stages_ann.keys()) <= set(reader.sleep_stage_names)
        for k in sleep_stages_ann.keys():
            assert isinstance(sleep_stages_ann[k], list)
            for itv in sleep_stages_ann[k]:
                assert len(itv) == 2
                assert isinstance(itv[0], int)
                assert isinstance(itv[1], int)

    def test_load_arousals_ann(self):
        arousals_ann = reader.load_arousals_ann(0)
        assert isinstance(arousals_ann, dict)
        assert set(arousals_ann.keys()) <= set(reader.arousal_types)
        for k in arousals_ann.keys():
            assert isinstance(arousals_ann[k], list)
            for itv in arousals_ann[k]:
                assert len(itv) == 2
                assert isinstance(itv[0], int)
                assert isinstance(itv[1], int)

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        pass  # NOTE: not implemented yet

    def test_plot_ann(self):
        reader.plot_ann(0)
