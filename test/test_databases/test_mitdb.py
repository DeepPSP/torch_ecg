"""
TestMITDB: accomplished
TestMITDBDataset: partially accomplished

subsampling: accomplished
"""

import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases import MITDB, WFDB_Rhythm_Annotations, DataBaseInfo
from torch_ecg.databases.datasets import MITDBDataset, MITDBTrainCfg
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "mitdb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = MITDB(_CWD)
reader.download()


class TestMITDB:
    def test_len(self):
        assert len(reader) == 48

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = MITDB(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = MITDB(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            MITDB(_CWD, subsample=0.0)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            MITDB(_CWD, subsample=1.01)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            MITDB(_CWD, subsample=-0.1)

    def test_load_data(self):
        data = reader.load_data(0)
        assert data.ndim == 2
        data_1 = reader.load_data(0, leads=0, data_format="flat", sampto=1000)
        assert np.allclose(data[0][:1000], data_1)

    def test_load_ann(self):
        data = reader.load_data(0)
        ann = reader.load_ann(0)
        assert ann.keys() == {"beat", "rhythm"}
        assert isinstance(ann["beat"], list)
        assert isinstance(ann["rhythm"], dict)
        ann = reader.load_ann(0, beat_format="dict", rhythm_format="mask")
        assert isinstance(ann["beat"], dict)
        assert isinstance(ann["rhythm"], np.ndarray)
        assert ann["rhythm"].shape == (data.shape[1],)
        ann = reader.load_ann(0, rhythm_types=list(WFDB_Rhythm_Annotations)[:5])
        assert isinstance(ann["beat"], list)
        assert isinstance(ann["rhythm"], dict)

    def test_load_rhythm_ann(self):
        # part of test_load_ann
        rhythm_ann = reader.load_rhythm_ann(0)
        ann = reader.load_ann(0)
        assert ann["rhythm"].keys() == rhythm_ann.keys()
        for k, v in ann["rhythm"].items():
            assert np.allclose(v, rhythm_ann[k])

    def test_load_beat_ann(self):
        # part of test_load_ann
        beat_ann = reader.load_beat_ann(0)
        ann = reader.load_ann(0)
        assert ann["beat"] == beat_ann

    def test_load_rpeak_indices(self):
        rpeaks = reader.load_rpeak_indices(0)
        assert rpeaks.ndim == 1
        rpeaks = reader.load_rpeak_indices(0, sampfrom=2000, sampto=4000)
        rpeaks_1 = reader.load_rpeak_indices(
            0, sampfrom=2000, sampto=4000, keep_original=True
        )
        assert np.allclose(rpeaks, rpeaks_1 - 2000)

    def test_get_lead_names(self):
        lead_names = reader._get_lead_names(0)
        assert isinstance(lead_names, list)
        assert all(isinstance(lead_name, str) for lead_name in lead_names)

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)
        assert isinstance(reader.df_stats, pd.DataFrame)
        assert not reader.df_stats.empty
        assert isinstance(reader.df_stats_expanded, pd.DataFrame)
        assert not reader.df_stats_expanded.empty
        assert isinstance(reader.df_stats_expanded_boolean, pd.DataFrame)
        assert not reader.df_stats_expanded_boolean.empty
        assert isinstance(reader.db_stats, dict)
        assert isinstance(reader.beat_types_records, dict)
        assert isinstance(reader.rhythm_types_records, dict)

    def test_plot(self):
        pass  # `plot` not implemented yet


config = deepcopy(MITDBTrainCfg)
config.db_dir = _CWD
config.stretch_compress = 5  # 5%

# tasks: "qrs_detection", "rhythm_segmentation", "af_event", "beat_classification", "rr_lstm"
TASK = "qrs_detection"

with pytest.warns(
    RuntimeWarning, match="`db_dir` is specified in both config and reader_kwargs"
):
    ds = MITDBDataset(
        config, task=TASK, training=True, lazy=True, subsample=0.2, db_dir=_CWD
    )
ds.persistence(verbose=2)
ds.reset_task(TASK, lazy=False)

ds_rhythm = MITDBDataset(
    config, task="rhythm_segmentation", training=True, lazy=False, subsample=0.2
)

ds_af = MITDBDataset(config, task="af_event", training=True, lazy=False, subsample=0.2)

ds_beat = MITDBDataset(
    config, task="beat_classification", training=True, lazy=False, subsample=0.2
)

ds_rr = MITDBDataset(config, task="rr_lstm", training=True, lazy=False, subsample=0.2)


class TestMITDBDataset:
    def test_len(self):
        assert len(ds) > 0
        assert len(ds_rhythm) > 0
        assert len(ds_af) > 0
        assert len(ds_beat) > 0
        assert len(ds_rr) > 0

    def test_getitem(self):
        data, ann = ds[0]
        assert data.ndim == ann.ndim == 2
        assert data.shape == (config.n_leads, config[TASK].input_len)
        assert ann.shape == (config[TASK].input_len, 1)

        data, ann = ds_beat[0]
        assert data.ndim == 2 and ann.ndim == 1
        assert data.shape == (config.n_leads, config.beat_classification.input_len)
        assert ann.shape == (len(config.beat_classification.classes),)

        rr, ann, wt_mask = ds_rr[0]
        assert rr.shape == ann.shape == wt_mask.shape == (config.rr_lstm.input_len, 1)

        # `ds_rhythm` and `ds_af` have bugs now

    def test_load_seg_data(self):
        seg = ds.all_segments[list(ds.all_segments)[0]][0]
        data = ds._load_seg_data(seg)
        assert data.ndim == 2
        assert data.shape == (config.n_leads, config[TASK].input_len)

    def test_load_seg_ann(self):
        seg = ds.all_segments[list(ds.all_segments)[0]][0]
        ann = ds._load_seg_ann(seg)
        assert isinstance(ann, dict)
        for k, v in ann.items():
            assert isinstance(v, np.ndarray) and v.ndim == 1

    def test_load_seg_mask(self):
        seg = ds.all_segments[list(ds.all_segments)[0]][0]
        mask = ds._load_seg_mask(seg)
        assert isinstance(mask, np.ndarray) and mask.ndim == 2

    def test_load_seg_seq_lab(self):
        seg = ds.all_segments[list(ds.all_segments)[0]][0]
        mask = ds._load_seg_mask(seg)
        seq_lab = ds._load_seg_seq_lab(seg, reduction=8)
        assert isinstance(seq_lab, np.ndarray) and seq_lab.ndim == 2
        assert mask.shape[0] == seq_lab.shape[0] * 8

    def test_load_rr_seq(self):
        rr = ds_rr.all_rr_seq[list(ds_rr.all_rr_seq)[0]][0]
        data = ds_rr._load_rr_seq(rr)
        assert isinstance(data, dict) and len(data) > 0
        for k, v in data.items():
            assert isinstance(v, np.ndarray)

    def test_properties(self):
        assert str(ds) == repr(ds)
        assert isinstance(ds.all_segments, dict) and len(ds.all_segments) > 0
        assert isinstance(ds_rr.all_rr_seq, dict) and len(ds_rr.all_rr_seq) > 0

    def test_plot_seg(self):
        # `plot_seg` not implemented yet
        seg = ds.all_segments[list(ds.all_segments)[0]][0]
        with pytest.raises(NotImplementedError):
            ds.plot_seg(seg)

    def test_clear_cached_segments(self):
        rec = list(ds.all_segments)[0]
        ds._clear_cached_segments([rec])
        ds._clear_cached_segments()

    def test_clear_cached_rr_seq(self):
        rec = list(ds_rr.all_rr_seq)[0]
        ds_rr._clear_cached_rr_seq([rec])
        ds_rr._clear_cached_rr_seq()
