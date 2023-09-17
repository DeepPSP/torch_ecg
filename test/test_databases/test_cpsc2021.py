"""
TestCPSC2021: accomplished
TestCPSC2021Dataset: accomplished

subsampling: accomplished
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb
from scipy.io import savemat

from torch_ecg.databases import CPSC2021, DataBaseInfo
from torch_ecg.databases.cpsc_databases.cpsc2021 import RefInfo, compute_metrics, load_ans
from torch_ecg.databases.cpsc_databases.cpsc2021 import score as score_func
from torch_ecg.databases.datasets import CPSC2021Dataset, CPSC2021TrainCfg
from torch_ecg.utils.misc import list_sum
from torch_ecg.utils.utils_interval import generalized_interval_len

# from torch_ecg.utils.misc import dicts_equal


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cpsc2021"
###############################################################################


reader = CPSC2021(_CWD)


_ANS_JSON_FILE = Path(__file__).absolute().parents[2] / "tmp" / "cpsc2021_test" / "cpsc2021_ans.json"
_ANS_MAT_FILE = Path(__file__).absolute().parents[2] / "tmp" / "cpsc2021_test" / "cpsc2021_ans.mat"

_ANS_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)

_ANS_JSON_DICT = {"predict_endpoints": [[1000, 2000], [3000, 4000], [5000, 6000]]}
_ANS_MAT_DICT = {"predict_endpoints": [[1001, 2001], [3001, 4001], [5001, 6001]]}

_ANS_JSON_FILE.write_text(json.dumps(_ANS_JSON_DICT))
savemat(str(_ANS_MAT_FILE), _ANS_MAT_DICT)

rec = reader.diagnoses_records_list["AFp"][0]
data_path = reader.get_absolute_path(rec)
stem = data_path.stem
ans_json_file = _ANS_JSON_FILE.parent / f"{stem}.json"
_ANS_JSON_FILE.rename(ans_json_file)
_ANS_JSON_FILE = ans_json_file
rec = reader.diagnoses_records_list["AFf"][0]
data_path = reader.get_absolute_path(rec)
ans_mat_file = _ANS_MAT_FILE.parent / f"{stem}.mat"
_ANS_MAT_FILE.rename(ans_mat_file)
_ANS_MAT_FILE = ans_mat_file


class TestCPSC2021:
    def test_len(self):
        assert len(reader) == 18

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = CPSC2021(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = CPSC2021(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CPSC2021(_CWD, subsample=0.0)
        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CPSC2021(_CWD, subsample=1.01)
        with pytest.raises(AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"):
            CPSC2021(_CWD, subsample=-0.1)

    def test_load_data(self):
        data = reader.load_data(0)
        data_1 = reader.load_data(0, leads=0, sampfrom=1000, sampto=5000, data_format="plain", units="μV")
        assert data.ndim == 2
        assert data_1.shape == (4000,)
        assert np.allclose(data[0, 1000:5000], data_1 / 1000, atol=1e-2)
        data_1 = reader.load_data(0, leads=0, data_format="channel_last", fs=2 * reader.fs)
        assert data_1.shape == (2 * data.shape[1], 1)

    def test_load_ann(self):
        rec = reader.diagnoses_records_list["AFp"][0]
        ann = reader.load_ann(rec)
        assert isinstance(ann, dict) and ann.keys() == {
            "rpeaks",
            "af_episodes",
            "label",
        }

        # field: "rpeaks", "af_episodes", "label", "raw", "wfdb"
        ann = reader.load_ann(rec, field="rpeaks", sampfrom=1000, sampto=5000)
        ann_1 = reader.load_ann(rec, field="rpeaks", sampfrom=1000, sampto=5000, keep_original=True)
        ann_2 = reader.load_ann(
            rec,
            field="rpeaks",
            sampfrom=1000,
            sampto=5000,
            keep_original=True,
            fs=2 * reader.fs,
        )
        ann_3 = reader.load_ann(rec, field="rpeaks", sampfrom=1000, sampto=5000, valid_only=False)
        assert ann.ndim == 1
        assert ann.shape == ann_1.shape == ann_2.shape
        assert np.allclose(ann, ann_1 - 1000)
        assert np.allclose(ann_1 * 2, ann_2)
        assert set(ann_3) >= set(ann)

        ann = reader.load_ann(rec, field="af_episodes", sampfrom=1000, sampto=5000)
        ann_1 = reader.load_ann(rec, field="af_episodes", sampfrom=1000, sampto=5000, keep_original=True)
        ann_2 = reader.load_ann(
            rec,
            field="af_episodes",
            sampfrom=1000,
            sampto=5000,
            keep_original=True,
            fs=2 * reader.fs,
        )
        ann_3 = reader.load_ann(
            rec,
            field="af_episodes",
            sampfrom=1000,
            sampto=5000,
            keep_original=True,
            fmt="mask",
        )
        ann_4 = reader.load_ann(rec, field="af_episodes", fmt="c_intervals")
        assert len(ann) == len(ann_1) == len(ann_2)
        assert np.allclose(np.array(ann), np.array(ann_1) - 1000)
        assert ann_3.shape == (4000,)
        assert ann_3.sum() == generalized_interval_len(ann_1)
        assert len(ann_4) >= len(ann)

        ann = reader.load_ann(rec, field="label", sampfrom=1000, sampto=5000, fmt="f")
        ann_1 = reader.load_ann(rec, field="label", sampfrom=1000, sampto=5000, fmt="a")
        ann_2 = reader.load_ann(rec, field="label", sampfrom=1000, sampto=5000, fmt="n")
        assert isinstance(ann, str)
        assert isinstance(ann_1, str)
        assert isinstance(ann_2, int)
        assert reader._labels_f2a[ann] == ann_1
        assert reader._labels_f2n[ann] == ann_2

        ann = reader.load_ann(rec, field="raw")
        ann_1 = reader.load_ann(rec, field="wfdb")
        assert isinstance(ann, wfdb.io.annotation.Annotation)
        assert ann == ann_1

        with pytest.raises(ValueError, match="Invalid `field`: xxx"):
            reader.load_ann(rec, field="xxx")
        with pytest.raises(
            ValueError,
            match="when `fmt` is `c_intervals`, `sampfrom` and `sampto` should never be used",
        ):
            reader.load_ann(rec, field="af_episodes", sampfrom=1000, sampto=5000, fmt="c_intervals")
        with pytest.raises(ValueError, match="format `xxx` of labels is not supported"):
            reader.load_ann(rec, field="label", fmt="xxx")

        with pytest.warns(
            RuntimeWarning,
            match="key word arguments `.+` ignored when `field` is not specified",
        ):
            reader.load_ann(rec, fmt="c_intervals")

        sig_len = reader.df_stats[reader.df_stats.record == rec].iloc[0].sig_len
        with pytest.warns(
            RuntimeWarning,
            match="the end index \\d+ is larger than the signal length \\d+, so it is set to \\d+",
        ):
            reader.load_ann(rec, sampto=sig_len + 1000)

        with pytest.raises(ValueError, match="Invalid `sampfrom` and `sampto`"):
            reader.load_ann(0, sampfrom=1000, sampto=500)

        # aliases
        assert (reader.load_rpeaks(rec) == reader.load_ann(rec, field="rpeaks")).all()
        assert (reader.load_rpeak_indices(rec) == reader.load_ann(rec, field="rpeaks")).all()
        assert reader.load_af_episodes(rec) == reader.load_ann(rec, field="af_episodes")
        assert reader.load_label(rec) == reader.load_ann(rec, field="label")

    def test_gen_endpoint_score_mask(self):
        rec = reader.diagnoses_records_list["AFp"][0]
        data = reader.load_data(rec)
        onset_score_mask, offset_score_mask = reader.gen_endpoint_score_mask(rec)
        assert onset_score_mask.shape == offset_score_mask.shape == (data.shape[1],)
        onset_score_mask_1, offset_score_mask_1 = reader.gen_endpoint_score_mask(rec, bias={1: 0.5, 2: 0.5})
        assert onset_score_mask.sum() > onset_score_mask_1.sum()
        assert offset_score_mask.sum() > offset_score_mask_1.sum()
        onset_score_mask_1, offset_score_mask_1 = reader.gen_endpoint_score_mask(rec, bias={1: 1, 2: 1}, verbose=2)
        assert onset_score_mask.sum() < onset_score_mask_1.sum()
        assert offset_score_mask.sum() < offset_score_mask_1.sum()

    def test_aggregate_stats(self):
        new_reader = CPSC2021(_CWD)
        stats_file = "stats.csv"
        stats_file_fp = new_reader.db_dir_base / stats_file
        if stats_file_fp.is_file():
            stats_file_fp.unlink()
        new_reader._stats = pd.DataFrame()
        new_reader._aggregate_stats()

        assert not new_reader.df_stats.empty

        del new_reader

    def test_ls_diagnoses_records(self):
        new_reader = CPSC2021(_CWD)
        fn = "diagnoses_records_list.json"
        dr_fp = new_reader.db_dir_base / fn
        if dr_fp.is_file():
            dr_fp.unlink()
        new_reader._stats = pd.DataFrame()
        new_reader._diagnoses_records_list = None

        assert new_reader.diagnoses_records_list is not None

        del new_reader

    def test_meta_data(self):
        assert isinstance(reader.diagnoses_records_list, dict)
        assert all(isinstance(v, list) for v in reader.diagnoses_records_list.values())
        assert all(set(v) <= set(reader.all_records) for v in reader.diagnoses_records_list.values())
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_helper(self):
        assert reader.helper() is None  # printed
        for item in ["attributes", "methods"]:
            assert reader.helper(item) is None  # printed
        assert reader.helper(["attributes", "methods"]) is None  # printed

    def test_plot(self):
        waves = {
            "p_onsets": [100, 1100],
            "p_offsets": [110, 1110],
            "q_onsets": [115, 1115],
            "s_offsets": [130, 1130],
            "t_onsets": [150, 1150],
            "t_offsets": [190, 1190],
        }
        reader.plot(0, leads=[1], sampfrom=1000, sampto=3000, ticks_granularity=2, waves=waves)
        waves = {
            "p_peaks": [105, 1105],
            "q_peaks": [120, 1120],
            "s_peaks": [125, 1125],
            "t_peaks": [170, 1170],
        }
        reader.plot(0, leads=0, sampfrom=1000, sampto=3000, ticks_granularity=1, waves=waves)
        waves = {
            "p_peaks": [105, 1105],
            "r_peaks": [122, 1122],
            "t_peaks": [170, 1170],
        }
        data = reader.load_data(0, sampfrom=1000, sampto=3000)
        reader.plot(0, data=data, ticks_granularity=0, waves=waves)

    def test_compute_metric(self):
        rec = reader.diagnoses_records_list["AFp"][0]
        class_true = reader.load_ann(rec, field="label", fmt="n")
        class_pred = reader.load_ann(rec, field="label", fmt="n")
        endpoints_true = reader.load_ann(rec, field="af_episodes")
        endpoints_pred = reader.load_ann(rec, field="af_episodes")
        onset_score_mask, offset_score_mask = reader.gen_endpoint_score_mask(rec)
        score = compute_metrics(
            class_true=class_true,
            class_pred=class_pred,
            endpoints_true=endpoints_true,
            endpoints_pred=endpoints_pred,
            onset_score_range=onset_score_mask,
            offset_score_range=offset_score_mask,
        )
        assert score >= 2 + 1

        rec = reader.diagnoses_records_list["N"][0]
        class_pred = reader.load_ann(rec, field="label", fmt="n")
        endpoints_pred = reader.load_ann(rec, field="af_episodes")
        score = compute_metrics(
            class_true=class_true,
            class_pred=class_pred,
            endpoints_true=endpoints_true,
            endpoints_pred=endpoints_pred,
            onset_score_range=onset_score_mask,
            offset_score_range=offset_score_mask,
        )
        assert score < 0

    def test_RefInfo(self):
        rec = reader.diagnoses_records_list["AFp"][0]
        path = str(reader.get_absolute_path(rec))
        ref_info = RefInfo(path)
        ref_info._gen_endpoint_score_range(verbose=2)

        assert ref_info.fs == reader.fs
        assert ref_info.len_sig == reader.load_data(rec).shape[1]

        assert np.allclose(ref_info.beat_loc, reader.load_ann(rec, field="rpeaks", valid_only=False))

        af_episodes = np.array(reader.load_ann(rec, field="af_episodes", fmt="c_intervals"))
        assert np.allclose(af_episodes[:, 0], ref_info.af_starts)
        assert np.allclose(af_episodes[:, 1], ref_info.af_ends)

        assert ref_info.class_true == reader.load_ann(rec, field="label", fmt="n")

        onset_score_mask, offset_score_mask = reader.gen_endpoint_score_mask(rec)
        assert np.allclose(onset_score_mask, ref_info.onset_score_range)
        assert np.allclose(offset_score_mask, ref_info.offset_score_range)

        rec = reader.diagnoses_records_list["N"][0]
        path = str(reader.get_absolute_path(rec))
        ref_info = RefInfo(path)
        assert ref_info.class_true == reader.load_ann(rec, field="label", fmt="n")
        assert ref_info.onset_score_range is None
        assert ref_info.offset_score_range is None

        rec = reader.diagnoses_records_list["AFf"][0]
        path = str(reader.get_absolute_path(rec))
        ref_info = RefInfo(path)
        assert ref_info.class_true == reader.load_ann(rec, field="label", fmt="n")

        onset_score_mask, offset_score_mask = reader.gen_endpoint_score_mask(rec)
        assert np.allclose(onset_score_mask, ref_info.onset_score_range)
        assert np.allclose(offset_score_mask, ref_info.offset_score_range)

    def test_load_ans(self):
        ans = load_ans(str(_ANS_JSON_FILE))
        # assert dicts_equal(ans, _ANS_JSON_DICT)
        assert np.array_equal(ans, _ANS_JSON_DICT["predict_endpoints"])

        ans = load_ans(str(_ANS_MAT_FILE))
        # assert dicts_equal(ans, _ANS_JSON_DICT)  # NOT _ANS_MAT_DICT
        assert np.array_equal(ans, _ANS_JSON_DICT["predict_endpoints"])

    def test_score_func(self):
        score = score_func(
            data_path=str(_CWD),
            ans_path=str(_ANS_JSON_FILE.parent),
        )
        assert isinstance(score.item(), float)


config = deepcopy(CPSC2021TrainCfg)
config.db_dir = _CWD
config.stretch_compress = 5  # 5%

with pytest.warns(RuntimeWarning, match="`db_dir` is specified in both config and reader_kwargs"):
    ds = CPSC2021Dataset(config, task="main", training=False, lazy=False, db_dir=_CWD)
ds.persistence(verbose=2)


config_1 = deepcopy(config)
ds_1 = CPSC2021Dataset(config_1, task="rr_lstm", training=False, lazy=False)

config_2 = deepcopy(config)
ds_2 = CPSC2021Dataset(config_2, task="rr_lstm", training=False, lazy=False)
ds_2.reset_task(task="qrs_detection", lazy=True)


class TestCPSC2021Dataset:
    def test_len(self):
        assert len(ds) > 0
        assert len(ds_1) > 0
        assert len(ds_2) > 0

    def test_getitem(self):
        input_len = config[ds.task].input_len
        for i in range(len(ds)):
            data, af_mask, weight_mask = ds[i]
            assert data.shape == (config.n_leads, input_len)
            assert af_mask.shape == (input_len, 1)
            assert weight_mask.shape == (input_len, 1)

        input_len = config_1[ds_1.task].input_len
        for i in range(len(ds_1)):
            rr_seq, rr_af_mask, rr_weight_mask = ds_1[i]
            assert rr_seq.shape == (input_len, 1)
            assert rr_af_mask.shape == (input_len, 1)
            assert rr_weight_mask.shape == (input_len, 1)

        input_len = config_2[ds_2.task].input_len
        for i in range(len(ds_2)):
            data, qrs_mask = ds_2[i]
            assert data.shape == (config.n_leads, input_len)
            assert qrs_mask.shape == (input_len, 1)

    def test_properties(self):
        assert ds.task == "main"
        assert ds_1.task == "rr_lstm"
        assert ds_2.task == "qrs_detection"
        assert isinstance(ds.all_segments, dict) and isinstance(ds_1.all_segments, dict) and isinstance(ds_2.all_segments, dict)
        assert set(ds.all_segments) == set(ds_2.all_segments) == set(ds_1.all_rr_seq) > set()
        assert set(ds.subjects) == set(ds_2.subjects) == set(ds_1.subjects) > set()
        assert set(ds.all_segments) > set(ds.subjects)
        assert len(ds.all_rr_seq) == len(ds_2.all_rr_seq) == len(ds_1.all_segments) == 0
        assert len(ds) == len(ds.segments) < len(list_sum(ds.all_segments.values()))
        assert len(ds_1) == len(ds_1.rr_seq) < len(list_sum(ds_1.all_rr_seq.values()))
        assert len(ds_2) == len(ds_2.segments) < len(list_sum(ds_2.all_segments.values()))
        assert str(ds) == repr(ds)

    def test_load_seg_seq_lab(self):
        seg_data = ds._load_seg_data(ds.segments[0])
        seg_seq_lab = ds._load_seg_seq_lab(ds.segments[0], reduction=1)
        assert seg_seq_lab.shape == (seg_data.shape[1], 1)
        seg_seq_lab = ds._load_seg_seq_lab(ds.segments[0], reduction=8)
        assert seg_seq_lab.shape == (seg_data.shape[1] // 8, 1)

    def test_load_rr_seq(self):
        rr_seq = ds_1._load_rr_seq(ds_1.rr_seq[0])
        assert isinstance(rr_seq, dict)
        assert rr_seq.keys() == {"rr", "label", "interval"}
        assert rr_seq["rr"].shape == rr_seq["label"].shape == (ds_1.seglen, 1)
        assert rr_seq["interval"].shape == (2,)

    def test_plot_seg(self):
        ds.plot_seg(ds.segments[0], ticks_granularity=2)

    def test_clear_cached_segments(self):
        ds._clear_cached_segments(recs=[ds.reader.all_records[0]])
        ds._clear_cached_segments()

    def test_clear_cached_rr_seq(self):
        ds_1._clear_cached_rr_seq(recs=[ds_1.reader.all_records[0]])
        ds_1._clear_cached_rr_seq()
