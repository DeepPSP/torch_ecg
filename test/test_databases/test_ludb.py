"""
methods from the base class, e.g. `load_data`, are tested in a simple way in this file,
since they are comprehensively tested `test_afdb.py`.

TestLUDB: accomplished
TestLUDBDataset: accomplished

subsampling: NOT tested
"""

import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import LUDB, DataBaseInfo
from torch_ecg.databases.datasets import LUDBDataset, LUDBTrainCfg
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "ludb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = LUDB(_CWD)
reader.download()


class TestLUDB:
    def test_len(self):
        assert len(reader) == 200

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = LUDB(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1)
        ss_ratio = 0.1 / len(reader)
        reader_ss = LUDB(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1

        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LUDB(_CWD, subsample=0.0)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LUDB(_CWD, subsample=1.01)
        with pytest.raises(
            AssertionError, match="`subsample` must be in \\(0, 1\\], but got `.+`"
        ):
            LUDB(_CWD, subsample=-0.1)

    def test_load_data(self):
        data = reader.load_data(0)
        data_1 = reader.load_data(0, leads=[1, 7])
        assert data.shape[0] == 12
        assert data_1.shape[0] == 2
        assert np.allclose(data[[1, 7], :], data_1)

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert ann.keys() == {"waves"}
        assert ann["waves"].keys() == set(reader.all_leads)
        ann = reader.load_ann(0, leads=["II", "aVR"])
        assert ann["waves"].keys() == {"II", "aVR"}
        ann = reader.load_ann(0, metadata=True)
        assert ann.keys() > {"waves"}

    def test_load_diagnoses(self):
        diagnoses = reader.load_diagnoses(0)
        assert all(isinstance(item, str) for item in diagnoses)

    def test_load_masks(self):
        data = reader.load_data(0)
        masks = reader.load_masks(0)
        assert masks.shape == data.shape
        data = reader.load_data(0, leads=[1, 7])
        masks = reader.load_masks(0, leads=[1, 7], mask_format="lead_last")
        assert masks.shape == data.T.shape

    def test_load_subject_info(self):
        subject_info = reader.load_subject_info(0)
        assert isinstance(subject_info, dict)
        subject_info = reader.load_subject_info(0, fields=["Sex", "Age"])
        assert isinstance(subject_info, dict)
        assert subject_info.keys() == {"Sex", "Age"}
        subject_info = reader.load_subject_info(0, fields="Sex")
        assert isinstance(subject_info, str)

    def test_get_subject_id(self):
        assert isinstance(reader.get_subject_id(0), int)

    def test_from_masks(self):
        ann = reader.from_masks(reader.load_masks(0), leads=reader.all_leads)
        ann_1 = reader.load_ann(0)["waves"]
        for lead in reader.all_leads:
            assert len(ann[lead]) == len(ann_1[lead])
            for i in range(len(ann[lead])):
                assert ann[lead][i].name == ann_1[lead][i].name
                assert ann[lead][i].onset == ann_1[lead][i].onset
                assert ann[lead][i].offset == ann_1[lead][i].offset

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        reader.plot(0, leads=["I", 5], ticks_granularity=2)
        data = reader.load_data(0, leads="III", data_format="flat")
        reader.plot(0, data=data, leads="III")


config = deepcopy(LUDBTrainCfg)
config.db_dir = _CWD

ds = LUDBDataset(config, training=False, lazy=False)

config_1 = deepcopy(config)
ds_1 = LUDBDataset(config_1, training=False, lazy=True)


class TestLUDBDataset:
    def test_len(self):
        assert len(ds) == len(ds_1) > 0

    def test_getitem(self):
        for i in range(len(ds)):
            signals, labels = ds[i]
            assert signals.shape == (config.n_leads, config.input_len)
            assert labels.shape == (config.input_len, len(config.classes))

        for i in range(len(ds_1)):
            signals, labels = ds[i]
            assert signals.shape == (config.n_leads, config.input_len)
            assert labels.shape == (config.input_len, len(config.classes))

    def test_properties(self):
        signals_shape = ds.signals.shape  # (n_samples, n_leads, signal_len)
        labels_shape = ds.labels.shape  # (n_samples, n_leads, signal_len, n_classes)
        assert signals_shape[:2] == labels_shape[:2] == (len(ds), config.n_leads)
        assert signals_shape[2] == labels_shape[2] >= config.input_len
        assert labels_shape[3] == len(config.classes)

        assert ds_1.signals is None
        assert ds_1.labels is None

        assert str(ds) == repr(ds)
