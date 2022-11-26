"""
"""

import re
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from torch_ecg.databases import MITDB, WFDB_Rhythm_Annotations
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

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed

    def test_plot(self):
        pass  # `plot` not implemented yet


config = deepcopy(MITDBTrainCfg)
config.db_dir = _CWD

ds = MITDBDataset(config, task="qrs_detection", training=False, lazy=False)
