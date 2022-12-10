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
reader = SHHS(_CWD / "polysomnography", current_version="0.15.0", lazy=False, verbose=2)


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
        assert len(empty_reader.rec_with_hrv_detailed_ann) == 0
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
        assert len(reader.rec_with_hrv_detailed_ann) > 0
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
            0, channel=list(reader.all_signals)[0], physical=False
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

        data_2, _ = reader.load_data(
            0, sampfrom=10, sampto=20, data_format="flat", units="uv"
        )
        assert np.allclose(data_2, data_1 * 1e3)

        with pytest.raises(
            AssertionError, match="`data_format` should be one of `.+`, but got `.+`"
        ):
            reader.load_data(0, data_format="invalid")
        with pytest.raises(
            AssertionError, match="`units` should be one of `.+` or None, but got `.+`"
        ):
            reader.load_data(0, units="kV")

    def test_load_ecg_data(self):
        # alias of `load_data`
        data, fs = reader.load_data(0)
        data_1, fs_1 = reader.load_ecg_data(0)
        assert np.allclose(data, data_1)

    def test_load_ann(self):
        rec = reader.rec_with_event_ann[0]
        # fmt: off
        for ann_type in [
            "event",
            "sleep", "sleep_stage", "sleep_event", "apnea", "sleep_apnea"
        ]:
            ann = reader.load_ann(rec, ann_type)
            assert isinstance(ann, (np.ndarray, pd.DataFrame, dict))
        # fmt: on

        rec = reader.rec_with_event_profusion_ann[0]
        ann = reader.load_ann(rec, "event_profusion")
        assert isinstance(ann, dict)

        rec = reader.rec_with_hrv_summary_ann[0]
        ann = reader.load_ann(rec, "hrv_summary")
        assert isinstance(ann, pd.DataFrame)

        rec = reader.rec_with_hrv_detailed_ann[0]
        ann = reader.load_ann(rec, "hrv_detailed")
        assert isinstance(ann, pd.DataFrame)

        rec = reader.rec_with_rpeaks_ann[0]
        for ann_type in ["wave_delineation", "rpeak", "rr", "nn"]:
            ann = reader.load_ann(rec, ann_type)
            assert isinstance(ann, (pd.DataFrame, np.ndarray))

    def test_load_event_ann(self):
        rec = reader.rec_with_event_ann[0]
        ann = reader.load_event_ann(rec, simplify=False)
        assert isinstance(ann, pd.DataFrame) and len(ann) > 0
        ann_1 = reader.load_event_ann(rec, simplify=True)
        assert isinstance(ann_1, pd.DataFrame)
        assert len(ann_1) == len(ann)
        assert (ann_1.columns == ann.columns).all()

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        ann = reader.load_event_ann(rec)
        assert isinstance(ann, pd.DataFrame) and ann.empty

    def test_load_event_profusion_ann(self):
        rec = reader.rec_with_event_profusion_ann[0]
        ann = reader.load_event_profusion_ann(rec)
        assert isinstance(ann, dict) and len(ann) == 2
        assert set(ann.keys()) == {"sleep_stage_list", "df_events"}
        assert (
            isinstance(ann["sleep_stage_list"], list)
            and len(ann["sleep_stage_list"]) > 0
        )
        assert isinstance(ann["df_events"], pd.DataFrame) and len(ann["df_events"]) > 0

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        ann = reader.load_event_profusion_ann(rec)
        assert isinstance(ann, dict) and len(ann) == 2
        assert set(ann.keys()) == {"sleep_stage_list", "df_events"}
        assert (
            isinstance(ann["sleep_stage_list"], list)
            and len(ann["sleep_stage_list"]) == 0
        )
        assert isinstance(ann["df_events"], pd.DataFrame) and ann["df_events"].empty

    def test_load_hrv_detailed_ann(self):
        rec = reader.rec_with_hrv_detailed_ann[0]
        ann = reader.load_hrv_detailed_ann(rec)
        assert isinstance(ann, pd.DataFrame) and len(ann) > 0

        rec = list(set(reader.all_records) - set(reader.rec_with_hrv_detailed_ann))[0]
        ann = reader.load_hrv_detailed_ann(rec)
        assert isinstance(ann, pd.DataFrame) and ann.empty

    def test_load_hrv_summary_ann(self):
        rec = reader.rec_with_hrv_summary_ann[0]
        ann = reader.load_hrv_summary_ann(rec)
        assert isinstance(ann, pd.DataFrame) and len(ann) > 0

        rec = list(set(reader.all_records) - set(reader.rec_with_hrv_summary_ann))[0]
        ann = reader.load_hrv_summary_ann(rec)
        assert isinstance(ann, pd.DataFrame) and ann.empty

        ann = reader.load_hrv_summary_ann(rec=None)
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) == len(reader.get_table("shhs1-hrv-summary")) + len(
            reader.get_table("shhs2-hrv-summary")
        )

    def test_load_wave_delineation_ann(self):
        rec = reader.rec_with_rpeaks_ann[0]
        ann = reader.load_wave_delineation_ann(rec)
        assert isinstance(ann, pd.DataFrame) and len(ann) > 0

        rec = list(set(reader.all_records) - set(reader.rec_with_rpeaks_ann))[0]
        ann = reader.load_wave_delineation_ann(rec)
        assert isinstance(ann, pd.DataFrame) and ann.empty

    def test_load_rpeak_ann(self):
        rec = reader.rec_with_rpeaks_ann[0]
        ann = reader.load_rpeak_ann(rec)
        assert isinstance(ann, np.ndarray)
        assert ann.ndim == 1 and len(ann) > 0
        assert ann.dtype == np.int64
        ann_1 = reader.load_rpeak_ann(rec, units="s")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == ann.shape
        assert ann_1.dtype == np.float64
        ann_2 = reader.load_rpeak_ann(rec, units="ms")
        assert isinstance(ann_2, np.ndarray)
        assert ann_2.shape == ann.shape
        assert ann_2.dtype == np.int64
        assert np.allclose(ann_2 / 1000, ann_1, atol=1e-2)  # ann_2 is rounded

        ann_3 = reader.load_rpeak_ann(rec, exclude_artifacts=False)
        assert isinstance(ann_3, np.ndarray)
        assert ann_3.ndim == 1 and len(ann_3) >= len(ann)
        ann_3 = reader.load_rpeak_ann(rec, exclude_abnormal_beats=False)
        assert isinstance(ann_3, np.ndarray)
        assert ann_3.ndim == 1 and len(ann_3) >= len(ann)

        rec = list(set(reader.all_records) - set(reader.rec_with_rpeaks_ann))[0]
        ann = reader.load_rpeak_ann(rec)
        assert isinstance(ann, np.ndarray)
        assert ann.ndim == 1 and len(ann) == 0

        rec = list(set(reader.all_records) - set(reader.rec_with_rpeaks_ann))[0]
        ann = reader.load_rpeak_ann(rec)
        assert isinstance(ann, np.ndarray) and ann.ndim == 1 and len(ann) == 0

        rec = reader.rec_with_rpeaks_ann[0]
        with pytest.raises(
            ValueError,
            match="`units` should be one of 's', 'ms', case insensitive, or None",
        ):
            reader.load_rpeak_ann(rec, units="invalid")

    def test_load_rr_ann(self):
        rec = reader.rec_with_rpeaks_ann[0]
        rpeaks = reader.load_rpeak_ann(rec)

        ann = reader.load_rr_ann(rec)
        assert isinstance(ann, np.ndarray)
        assert ann.shape == (len(rpeaks) - 1, 2)
        ann_1 = reader.load_rr_ann(rec, units="ms")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == ann.shape
        assert np.allclose(ann_1 / 1000, ann, atol=1e-2)  # ann_1 is rounded
        ann_2 = reader.load_rr_ann(rec, units=None)
        assert isinstance(ann_2, np.ndarray)
        assert ann_2.shape == ann.shape
        assert np.allclose(
            ann_2 / reader.get_fs(rec, "rpeak"), ann, atol=1e-2
        )  # ann_2 is rounded

        rec = list(set(reader.all_records) - set(reader.rec_with_rpeaks_ann))[0]
        ann = reader.load_rr_ann(rec)
        assert isinstance(ann, np.ndarray) and ann.ndim == 2 and len(ann) == 0

        rec = reader.rec_with_rpeaks_ann[0]
        with pytest.raises(
            ValueError,
            match="`units` should be one of 's', 'ms', case insensitive, or None",
        ):
            reader.load_rr_ann(rec, units="invalid")

    def test_load_nn_ann(self):
        rec = reader.rec_with_rpeaks_ann[0]
        rpeaks = reader.load_rpeak_ann(rec)

        ann = reader.load_nn_ann(rec)
        assert isinstance(ann, np.ndarray)
        assert ann.ndim == 2 and ann.shape[1] == 2
        assert ann.shape[0] < len(rpeaks) - 1

        ann_1 = reader.load_nn_ann(rec, units="ms")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == ann.shape
        assert np.allclose(ann_1 / 1000, ann, atol=1e-2)  # ann_1 is rounded
        ann_2 = reader.load_nn_ann(rec, units=None)
        assert isinstance(ann_2, np.ndarray)
        assert ann_2.shape == ann.shape
        assert np.allclose(
            ann_2 / reader.get_fs(rec, "rpeak"), ann, atol=1e-2
        )  # ann_2 is rounded

        rec = list(set(reader.all_records) - set(reader.rec_with_rpeaks_ann))[0]
        ann = reader.load_nn_ann(rec)
        assert isinstance(ann, np.ndarray) and ann.ndim == 2 and len(ann) == 0

        rec = reader.rec_with_rpeaks_ann[0]
        with pytest.raises(
            ValueError,
            match="`units` should be one of 's', 'ms', case insensitive, or None",
        ):
            reader.load_nn_ann(rec, units="invalid")

    def test_load_sleep_ann(self):
        rec = reader.rec_with_event_ann[0]
        ann = reader.load_sleep_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = reader.rec_with_event_profusion_ann[0]
        ann = reader.load_sleep_ann(rec, source="event_profusion")
        assert isinstance(ann, dict)
        assert len(ann) == 2 and set(ann.keys()) == {"sleep_stage_list", "df_events"}
        assert (
            isinstance(ann["sleep_stage_list"], list)
            and len(ann["sleep_stage_list"]) > 0
        )
        assert isinstance(ann["df_events"], pd.DataFrame) and len(ann["df_events"]) > 0

        rec = reader.rec_with_hrv_detailed_ann[0]
        ann = reader.load_sleep_ann(rec, source="hrv")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = "shhs2-200001"  # a record (both event and hrv) without sleep stage
        ann = reader.load_sleep_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0
        ann = reader.load_sleep_ann(rec, source="hrv")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0
        ann = reader.load_sleep_ann(rec, source="event_profusion")
        assert isinstance(ann, dict) and len(ann) == 2
        assert (
            isinstance(ann["sleep_stage_list"], list)
            and len(ann["sleep_stage_list"]) == 0
        )
        assert isinstance(ann["df_events"], pd.DataFrame) and len(ann["df_events"]) == 0

        with pytest.raises(ValueError, match="Source `.+` not supported, "):
            reader.load_sleep_ann(rec, source="invalid")

    def test_load_apnea_ann(self):
        rec = reader.rec_with_event_ann[0]
        for apnea_types in [None, ["CSA", "OSA"], ["MSA", "Hypopnea"]]:
            ann = reader.load_apnea_ann(rec, source="event", apnea_types=apnea_types)
            assert isinstance(ann, pd.DataFrame)
            assert len(ann) > 0
        rec = reader.rec_with_event_profusion_ann[0]
        for apnea_types in [None, ["OSA", "Hypopnea"], ["CSA", "MSA", "Hypopnea"]]:
            ann = reader.load_apnea_ann(
                rec, source="event_profusion", apnea_types=apnea_types
            )
            assert isinstance(ann, pd.DataFrame)
            assert len(ann) > 0

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        ann = reader.load_apnea_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame) and ann.empty
        ann = reader.load_apnea_ann(rec, source="event_profusion")
        assert isinstance(ann, pd.DataFrame) and ann.empty

        with pytest.raises(
            ValueError, match="Source `hrv` contains no apnea annotations"
        ):
            reader.load_apnea_ann(rec, source="hrv")

    def test_load_sleep_event_ann(self):
        rec = reader.rec_with_event_ann[0]
        ann = reader.load_sleep_event_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = reader.rec_with_event_profusion_ann[0]
        ann = reader.load_sleep_event_ann(rec, source="event_profusion")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = reader.rec_with_hrv_detailed_ann[0]
        ann = reader.load_sleep_event_ann(rec, source="hrv")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = "shhs2-200001"  # a record (both event and hrv) without sleep stage
        ann = reader.load_sleep_event_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0
        ann = reader.load_sleep_event_ann(rec, source="hrv")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0
        ann = reader.load_sleep_event_ann(rec, source="event_profusion")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0

        with pytest.raises(ValueError, match="Source `.+` not supported, "):
            reader.load_sleep_event_ann(rec, source="invalid")

    def test_load_sleep_stage_ann(self):
        rec = reader.rec_with_event_ann[0]
        ann = reader.load_sleep_stage_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = reader.rec_with_event_profusion_ann[0]
        ann = reader.load_sleep_stage_ann(rec, source="event_profusion")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = reader.rec_with_hrv_detailed_ann[0]
        ann = reader.load_sleep_stage_ann(rec, source="hrv")
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 0

        rec = "shhs2-200001"  # a record (both event and hrv) without sleep stage
        ann = reader.load_sleep_stage_ann(rec, source="event")
        assert isinstance(ann, pd.DataFrame) and len(ann) == 0

        with pytest.raises(ValueError, match="Source `.+` not supported, "):
            reader.load_sleep_stage_ann(rec, source="invalid")

    def test_locate_abnormal_beats(self):
        rec = reader.rec_with_rpeaks_ann[0]
        abn_beats = reader.locate_abnormal_beats(rec)
        assert isinstance(abn_beats, dict)
        assert set(abn_beats.keys()) == {"VE", "SVE"}
        assert isinstance(abn_beats["VE"], np.ndarray)
        assert isinstance(abn_beats["SVE"], np.ndarray)
        assert abn_beats["VE"].ndim == 1 and abn_beats["SVE"].ndim == 1
        assert len(abn_beats["VE"]) > 0 and len(abn_beats["SVE"]) > 0

        ann_1 = reader.locate_abnormal_beats(rec, abnormal_type="VE")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == abn_beats["VE"].shape
        assert np.all(ann_1 == abn_beats["VE"])

        ann_2 = reader.locate_abnormal_beats(rec, abnormal_type="VE", units="s")
        assert isinstance(ann_2, np.ndarray)
        assert ann_2.shape == abn_beats["VE"].shape
        assert np.allclose(
            ann_2, abn_beats["VE"] / reader.get_fs(rec, "rpeak"), atol=1e-2
        )
        ann_2 = reader.locate_abnormal_beats(rec, abnormal_type="VE", units="ms")
        assert isinstance(ann_2, np.ndarray)
        assert ann_2.shape == abn_beats["VE"].shape
        assert np.allclose(
            ann_2, abn_beats["VE"] / reader.get_fs(rec, "rpeak") * 1000, atol=1e-2
        )

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        abn_beats = reader.locate_abnormal_beats(rec)
        assert isinstance(abn_beats, dict)
        assert set(abn_beats.keys()) == {"VE", "SVE"}
        assert isinstance(abn_beats["VE"], np.ndarray)
        assert isinstance(abn_beats["SVE"], np.ndarray)
        assert abn_beats["VE"].ndim == 1 and abn_beats["SVE"].ndim == 1
        assert len(abn_beats["VE"]) == 0 and len(abn_beats["SVE"]) == 0

        rec = reader.rec_with_rpeaks_ann[0]
        with pytest.raises(ValueError, match="No abnormal type of `.+`"):
            reader.locate_abnormal_beats(rec, abnormal_type="AF")
        with pytest.raises(
            ValueError,
            match="`units` should be one of 's', 'ms', case insensitive, or None",
        ):
            reader.locate_abnormal_beats(rec, units="invalid")

    def test_locate_artifacts(self):
        rec = reader.rec_with_rpeaks_ann[0]
        artifacts = reader.locate_artifacts(rec)
        assert isinstance(artifacts, np.ndarray)
        assert artifacts.ndim == 1
        assert len(artifacts) > 0

        ann_1 = reader.locate_artifacts(rec, units="s")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == artifacts.shape
        assert np.allclose(ann_1, artifacts / reader.get_fs(rec, "rpeak"), atol=1e-2)
        ann_1 = reader.locate_artifacts(rec, units="ms")
        assert isinstance(ann_1, np.ndarray)
        assert ann_1.shape == artifacts.shape
        assert np.allclose(
            ann_1, artifacts / reader.get_fs(rec, "rpeak") * 1000, atol=1.0
        )

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        artifacts = reader.locate_artifacts(rec)
        assert isinstance(artifacts, np.ndarray)
        assert artifacts.ndim == 1
        assert len(artifacts) == 0

        rec = reader.rec_with_rpeaks_ann[0]
        with pytest.raises(
            ValueError,
            match="`units` should be one of 's', 'ms', case insensitive, or None",
        ):
            reader.locate_artifacts(rec, units="invalid")

    def test_get_available_signals(self):
        assert reader.get_available_signals(None) is None  # no return
        available_signals = reader.get_available_signals(0)
        assert isinstance(available_signals, list)
        assert set() < set(available_signals) <= set(reader.all_signals)

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        assert reader.get_available_signals(rec) == []

    def test_get_chn_num(self):
        available_signals = reader.get_available_signals(0)
        for sig in available_signals:
            chn_num = reader.get_chn_num(0, sig)
            assert isinstance(chn_num, int)
            assert 0 <= chn_num < len(available_signals)

    def test_match_channel(self):
        available_signals = reader.get_available_signals(0)
        for sig in available_signals:
            assert sig == reader.match_channel(sig.lower())
            assert sig in reader.all_signals

        assert reader.match_channel("rpeak", raise_error=False) == "rpeak"

    def test_get_fs(self):
        available_signals = reader.get_available_signals(0)
        for sig in available_signals:
            fs = reader.get_fs(0, sig)
            assert isinstance(fs, Real) and fs > 0

        rec = reader.rec_with_rpeaks_ann[0]
        fs = reader.get_fs(rec, "rpeak")
        assert isinstance(fs, Real) and fs > 0

        rec = (
            "shhs2-200001"  # a record (both signal and ann. files) that does not exist
        )
        fs = reader.get_fs(rec)
        assert fs == -1
        fs = reader.get_fs(rec, "rpeak")
        assert fs == -1

    def test_get_nsrrid(self):
        nsrrid = reader.get_nsrrid(0)
        assert isinstance(nsrrid, int)
        nsrrid = reader.get_nsrrid("shhs1-200001")
        assert isinstance(nsrrid, int) and nsrrid == 200001
        for rec in reader:
            nsrrid = reader.get_nsrrid(rec)
            assert isinstance(nsrrid, int)

    def test_get_subject_id(self):
        sid = reader.get_subject_id(0)
        assert isinstance(sid, int)
        sid = reader.get_subject_id("shhs1-200001")
        assert isinstance(sid, int)

    def test_get_table(self):
        for table_name in reader.list_table_names():
            table = reader.get_table(table_name)
            assert isinstance(table, pd.DataFrame)
            assert len(table) > 0

    def test_get_tranche(self):
        for rec in reader:
            tranche = reader.get_tranche(rec)
            assert isinstance(tranche, str)
            assert tranche in {"shhs1", "shhs2"}

    def test_get_visitnumber(self):
        visitnumber = reader.get_visitnumber(0)
        assert isinstance(visitnumber, int)
        visitnumber = reader.get_visitnumber("shhs1-200001")
        assert isinstance(visitnumber, int) and visitnumber == 1
        for rec in reader:
            visitnumber = reader.get_visitnumber(rec)
            assert isinstance(visitnumber, int)

    def test_split_rec_name(self):
        split_result = reader.split_rec_name(0)
        assert isinstance(split_result, dict)
        assert split_result.keys() == {"nsrrid", "tranche", "visitnumber"}
        assert isinstance(split_result["nsrrid"], int)
        assert isinstance(split_result["tranche"], str)
        assert isinstance(split_result["visitnumber"], int)

        split_result = reader.split_rec_name("shhs1-200001")
        assert isinstance(split_result, dict)
        assert split_result.keys() == {"nsrrid", "tranche", "visitnumber"}
        assert split_result["nsrrid"] == 200001
        assert split_result["tranche"] == "shhs1"
        assert split_result["visitnumber"] == 1

        with pytest.raises(AssertionError, match="Invalid record name: `.+`"):
            reader.split_rec_name("shhs1-200001-1")

    def test_meta_data(self):
        # TODO: add more....
        assert isinstance(reader.database_info, DataBaseInfo)
        assert reader.db_dir == _CWD
        assert reader.current_version >= "0.19.0"
        assert reader.show_rec_stats(0) is None  # printed to stdout

    def test_plot(self):
        pass
