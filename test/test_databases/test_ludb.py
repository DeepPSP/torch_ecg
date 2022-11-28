"""
methods from the base class, e.g. `load_data`, are tested in a simple way in this file,
since they are comprehensively tested `test_afdb.py`
"""

import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from torch_ecg.databases import LUDB
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

    def test_plot(self):
        reader.plot(0, leads=["I", 5], ticks_granularity=2)


config = deepcopy(LUDBTrainCfg)
config.db_dir = _CWD

ds = LUDBDataset(config, training=False, lazy=True)


class TestLUDBDataset:
    def test_len(self):
        pass
