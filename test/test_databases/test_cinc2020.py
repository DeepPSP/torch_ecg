"""
TestCINC2020: accomplished
TestCINC2020Dataset: accomplished

subsampling: NOT tested
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.cfg import DEFAULTS
from torch_ecg.databases import CINC2020
from torch_ecg.databases.physionet_databases.cinc2020 import compute_all_metrics
from torch_ecg.databases.aux_data.cinc2020_aux_data import dx_mapping_scored
from torch_ecg.databases.datasets import CINC2020Dataset, CINC2020TrainCfg
from torch_ecg.utils import dicts_equal


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"
###############################################################################


reader = CINC2020(_CWD)


class TestCINC2020:
    def test_len(self):
        assert len(reader) == 30
        for db in list("ABCD"):
            assert len(reader.all_records[db]) == 0
        assert len(reader.all_records["E"]) == 10
        assert len(reader.all_records["F"]) == 20

    def test_load_data(self):
        for rec in reader:
            data = reader.load_data(rec)
            data_1 = reader.load_data(rec, leads=[1, 7])
            assert data.shape[0] == 12
            assert data_1.shape[0] == 2
            assert np.allclose(data[[1, 7], :], data_1)
            data_1 = reader.load_data(rec, units="uV")
            assert np.allclose(data_1, data * 1000)
            data_1 = reader.load_data(rec, units=None)
            assert data.shape == data_1.shape
            data_1 = reader.load_data(rec, data_format="lead_last")
            assert data.shape == data_1.T.shape
            data_1 = reader.load_data(rec, fs=2 * reader.get_fs(rec))
            assert data_1.shape[1] == 2 * data.shape[1]
            data_1 = reader.load_data(rec, backend="scipy")
            assert np.allclose(data_1, data)

        with pytest.raises(AssertionError, match="Invalid data_format: `flat`"):
            reader.load_data(rec, data_format="flat")
        with pytest.raises(
            ValueError, match="backend `numpy` not supported for loading data"
        ):
            reader.load_data(rec, backend="numpy")

    def test_load_ann(self):
        for rec in reader:
            ann_1 = reader.load_ann(rec, backend="wfdb")
            ann_2 = reader.load_ann(rec, backend="naive")
            ann_3 = reader.load_ann(rec, raw=True)
            assert isinstance(ann_1, dict)
            assert dicts_equal(ann_1, ann_2)
            assert isinstance(ann_3, str)

        with pytest.raises(
            ValueError, match="backend `numpy` not supported for loading annotations"
        ):
            reader.load_ann(0, backend="numpy")

    def test_load_header(self):
        # alias for `load_ann`
        for rec in reader:
            header = reader.load_header(rec)
            assert dicts_equal(header, reader.load_ann(rec))

    def test_get_labels(self):
        for rec in reader:
            labels_1 = reader.get_labels(rec)
            labels_2 = reader.get_labels(rec, fmt="f")
            labels_3 = reader.get_labels(rec, fmt="a")
            labels_4 = reader.get_labels(rec, scored_only=False)
            assert len(labels_1) == len(labels_2) == len(labels_3) <= len(labels_4)
            assert set(labels_1) <= set(labels_4)

    def test_get_fs(self):
        for rec in reader:
            assert reader.get_fs(rec) in reader.fs.values()

    def test_get_subject_info(self):
        for rec in reader:
            info = reader.get_subject_info(rec)
            assert isinstance(info, dict)
            assert info.keys() == {
                "age",
                "sex",
                "medical_prescription",
                "history",
                "symptom_or_surgery",
            }
            info_1 = reader.get_subject_info(rec, items=["age", "sex"])
            assert info_1.keys() <= info.keys()
            for k, v in info_1.items():
                assert info[k] == v

    def test_get_tranche_class_distribution(self):
        dist = reader.get_tranche_class_distribution(list("ABCDE"))
        assert isinstance(dist, dict)
        dist_1 = reader.get_tranche_class_distribution(list("ABCDE"), scored_only=False)
        assert isinstance(dist_1, dict)
        assert set(dist.keys()) <= set(dist_1.keys())
        for k, v in dist.items():
            assert v == dist_1[k]

    def test_load_resampled_data(self):
        for rec in reader:
            data = reader.load_resampled_data(rec)
            assert data.ndim == 2 and data.shape[0] == 12
            data_1 = reader.load_resampled_data(rec, data_format="lead_last")
            assert np.allclose(data, data_1.T)
            data_1 = reader.load_resampled_data(rec, siglen=2000)
            assert data_1.ndim == 3 and data_1.shape[1:] == (12, 2000)

    def test_load_raw_data(self):
        for rec in reader:
            data_1 = reader.load_raw_data(rec, backend="wfdb")  # lead-last
            data_2 = reader.load_raw_data(rec, backend="scipy")  # lead-first
            assert data_1.ndim == 2 and data_1.shape[1] == 12
            assert data_2.ndim == 2 and data_2.shape[0] == 12
            assert np.allclose(data_1, data_2.T)

    def test_meta_data(self):
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert isinstance(reader.url, list) and len(reader.url) == len(
            reader.all_records
        ) == len(reader.tranche_names) == len(reader.db_tranches)
        assert reader.get_citation() is None  # printed
        assert set(reader.diagnoses_records_list.keys()) >= set(
            dx_mapping_scored.Abbreviation
        )

    def test_plot(self):
        reader.plot(0, leads=["II", 7], ticks_granularity=2)

    def test_compute_all_metrics(self):
        classes = dx_mapping_scored.Abbreviation.tolist()
        n_records, n_classes = 32, len(classes)
        truth = DEFAULTS.RNG_randint(0, 1, size=(n_records, n_classes))
        probs = DEFAULTS.RNG.uniform(n_records, n_classes)
        thresholds = DEFAULTS.RNG.uniform(n_classes)
        binary_pred = (probs > thresholds).astype(int)
        metrics = compute_all_metrics(
            classes=classes,
            truth=truth,
            binary_pred=binary_pred,
            scalar_pred=probs,
        )
        assert isinstance(metrics, tuple)
        assert all(isinstance(m, float) for m in metrics)


config = deepcopy(CINC2020TrainCfg)
config.db_dir = _CWD

ds = CINC2020Dataset(config, training=False, lazy=False)


class TestCINC2020Dataset:
    def test_len(self):
        assert len(ds) == len(ds.records) > 0

    def test_getitem(self):
        for i in range(len(ds)):
            data, target = ds[i]
            assert data.ndim == 2 and data.shape == (
                len(config.leads),
                config.input_len,
            )
            assert target.ndim == 1 and target.shape == (len(config.classes),)

    def test_load_one_record(self):
        for rec in ds.records:
            data, target = ds._load_one_record(rec)
            assert data.shape == (1, len(config.leads), config.input_len)
            assert target.shape == (1, len(config.classes))

    def test_properties(self):
        assert ds.signals.shape == (
            len(ds.records),
            len(config.leads),
            config.input_len,
        )
        assert ds.labels.shape == (len(ds.records), len(config.classes))

    def test_persistence(self):
        ds.persistence()

    def test_check_nan(self):
        ds._check_nan()
