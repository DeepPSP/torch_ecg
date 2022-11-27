"""
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import wfdb
import pytest

from torch_ecg.databases import CPSC2021
from torch_ecg.databases.cpsc_databases.cpsc2021 import compute_metrics
from torch_ecg.databases.datasets import CPSC2021Dataset, CPSC2021TrainCfg
from torch_ecg.utils.utils_interval import generalized_interval_len


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cpsc2021"
###############################################################################


reader = CPSC2021(_CWD)


class TestCPSC2021:
    def test_len(self):
        assert len(reader) == 18

    def test_load_data(self):
        data = reader.load_data(0)
        data_1 = reader.load_data(
            0, leads=0, sampfrom=1000, sampto=5000, data_format="plain", units="μV"
        )
        assert data.ndim == 2
        assert data_1.shape == (4000,)
        assert np.allclose(data[0, 1000:5000], data_1 / 1000, atol=1e-2)
        data_1 = reader.load_data(
            0, leads=0, data_format="channel_last", fs=2 * reader.fs
        )
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
        ann_1 = reader.load_ann(
            rec, field="rpeaks", sampfrom=1000, sampto=5000, keep_original=True
        )
        ann_2 = reader.load_ann(
            rec,
            field="rpeaks",
            sampfrom=1000,
            sampto=5000,
            keep_original=True,
            fs=2 * reader.fs,
        )
        assert ann.ndim == 1
        assert ann.shape == ann_1.shape == ann_2.shape
        assert np.allclose(ann, ann_1 - 1000)
        assert np.allclose(ann_1 * 2, ann_2)

        ann = reader.load_ann(rec, field="af_episodes", sampfrom=1000, sampto=5000)
        ann_1 = reader.load_ann(
            rec, field="af_episodes", sampfrom=1000, sampto=5000, keep_original=True
        )
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
            reader.load_ann(
                rec, field="af_episodes", sampfrom=1000, sampto=5000, fmt="c_intervals"
            )
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

        # aliases
        assert (reader.load_rpeaks(rec) == reader.load_ann(rec, field="rpeaks")).all()
        assert (
            reader.load_rpeak_indices(rec) == reader.load_ann(rec, field="rpeaks")
        ).all()
        assert reader.load_af_episodes(rec) == reader.load_ann(rec, field="af_episodes")
        assert reader.load_label(rec) == reader.load_ann(rec, field="label")

    def test_gen_endpoint_score_mask(self):
        rec = reader.diagnoses_records_list["AFp"][0]
        data = reader.load_data(rec)
        onset_score_mask, offset_score_mask = reader.gen_endpoint_score_mask(rec)
        assert onset_score_mask.shape == offset_score_mask.shape == (data.shape[1],)
        onset_score_mask_1, offset_score_mask_1 = reader.gen_endpoint_score_mask(
            rec, bias={1: 0.5, 2: 0.5}
        )
        assert onset_score_mask.sum() > onset_score_mask_1.sum()
        assert offset_score_mask.sum() > offset_score_mask_1.sum()
        onset_score_mask_1, offset_score_mask_1 = reader.gen_endpoint_score_mask(
            rec, bias={1: 1, 2: 1}
        )
        assert onset_score_mask.sum() < onset_score_mask_1.sum()
        assert offset_score_mask.sum() < offset_score_mask_1.sum()

    def test_meta_data(self):
        assert isinstance(reader.diagnoses_records_list, dict)
        assert all(isinstance(v, list) for v in reader.diagnoses_records_list.values())
        assert all(
            set(v) <= set(reader.all_records)
            for v in reader.diagnoses_records_list.values()
        )

    def test_plot(self):
        reader.plot(0, leads=[1], sampfrom=1000, sampto=5000, ticks_granularity=2)

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


config = deepcopy(CPSC2021TrainCfg)
config.db_dir = _CWD

ds = CPSC2021Dataset(config, task="main", training=False, lazy=False)
ds.persistence(verbose=2)


config_1 = deepcopy(config)
ds_1 = CPSC2021Dataset(config_1, task="rr_lstm", training=False, lazy=False)


class TestCPSC2021Dataset:
    def test_len(self):
        assert len(ds) > 0
        assert len(ds_1) > 0

    def test_getitem(self):
        pass

    def test_plot_seg(self):
        ds.plot_seg(ds.segments[0], ticks_granularity=2)
