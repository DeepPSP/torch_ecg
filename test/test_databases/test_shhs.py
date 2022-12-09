"""
TestSHHS: NOT accomplished

subsampling: NOT tested
"""

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases import SHHS, DataBaseInfo


###############################################################################
# set paths
# 9 files are downloaded in the following directory using `nsrr`
# ref. the action file .github/workflows/run-pytest.yml
_CWD = Path("~/tmp/nsrr-data/shhs").expanduser().resolve()
###############################################################################


# both `db_dir` and `current_version` will be
# adjusted according to the downloaded files
reader = SHHS(_CWD / "polysomnography", current_version="0.15.0")


class TestSHHS:
    def test_emtpy_db(self):
        directory = Path("~/tmp/test-empty/").expanduser().resolve()
        if directory.exists():
            directory.rmdir()
        with pytest.warns(
            RuntimeWarning, match="`.+` does not exist\\. It is now created"
        ):
            empty_reader = SHHS(directory, logger=reader.logger)
        assert len(empty_reader) == 0
        assert len(empty_reader.all_records) == 0
        assert len(empty_reader.rec_with_event_ann) == 0
        assert len(empty_reader.rec_with_event_profusion_ann) == 0
        assert len(empty_reader.rec_with_hrv_detail_ann) == 0
        assert len(empty_reader.rec_with_hrv_summary_ann) == 0
        assert len(empty_reader.rec_with_rpeaks_ann) == 0
        assert empty_reader.list_table_names() == []
        assert empty_reader._tables == {}
        assert empty_reader._df_records.empty

    def test_len(self):
        assert len(reader) == 10
        assert len(reader.all_records) == 10
        assert len(reader.rec_with_event_ann) == 10
        assert len(reader.rec_with_event_profusion_ann) == 10
        assert len(reader.rec_with_hrv_detail_ann) > 0
        assert len(reader.rec_with_hrv_summary_ann) > 0
        assert len(reader.rec_with_rpeaks_ann) == 2
        assert len(reader.list_table_names()) > 0
        assert isinstance(reader._tables, dict) and len(reader._tables) > 0
        for table_name, df in reader._tables.items():
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

    def test_load_psg_data(self):
        psg_data = reader.load_psg_data(0, physical=False)
        assert isinstance(psg_data, dict)
        for key, value in psg_data.items():
            assert isinstance(key, str)
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], np.ndarray)
            assert isinstance(value[1], Real) and value[1] > 0
        psg_data = reader.load_psg_data(
            0, channel=reader.all_signals[0], physical=False
        )
        assert isinstance(psg_data, tuple)
        assert len(psg_data) == 2
        assert isinstance(psg_data[0], np.ndarray)
        assert isinstance(psg_data[1], Real) and psg_data[1] > 0

    def test_load_data(self):
        data, fs = reader.load_data(0)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert isinstance(fs, Real) and fs > 0
        data_1, fs_1 = reader.load_data(0, fs=500, data_format="flat")
        assert isinstance(data_1, np.ndarray)
        assert data_1.ndim == 1
        assert fs_1 == 500
        data_1, fs_1 = reader.load_data(0, sampfrom=10, sampto=20, data_format="flat")
        assert fs_1 == fs
        assert data_1.shape[0] == int(10 * fs)
        assert np.allclose(data_1, data[0, int(10 * fs) : int(20 * fs)])

    def test_load_ecg_data(self):
        # alias of `load_data`
        pass

    def test_load_ann(self):
        pass

    def test_load_apnea_ann(self):
        pass

    def test_load_event_ann(self):
        pass

    def test_load_event_profusion_ann(self):
        pass

    def test_load_hrv_detailed_ann(self):
        pass

    def test_load_hrv_summary_ann(self):
        pass

    def test_load_wave_delineation(self):
        pass

    def test_load_rpeak_ann(self):
        pass

    def test_load_rr_ann(self):
        pass

    def test_load_nn_ann(self):
        pass

    def test_load_sleep_ann(self):
        pass

    def test_load_sleep_event_ann(self):
        pass

    def test_load_sleep_stage_ann(self):
        pass

    def test_locate_abnormal_beats(self):
        pass

    def test_locate_artifacts(self):
        pass

    def test_get_chn_num(self):
        pass

    def test_get_fs(self):
        pass

    def test_get_nsrrid(self):
        pass

    def test_get_subject_id(self):
        pass

    def test_get_table(self):
        pass

    def test_get_tranche(self):
        pass

    def test_get_visit_number(self):
        pass

    def test_split_rec_name(self):
        pass

    def test_meta_data(self):
        # TODO: add more....
        assert isinstance(reader.database_info, DataBaseInfo)
        assert reader.db_dir == _CWD
        assert reader.current_version >= "0.19.0"

    def test_plot(self):
        pass
