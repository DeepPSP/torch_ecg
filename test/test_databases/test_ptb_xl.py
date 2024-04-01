"""
TestPTBXL: not accomplished
TestPTBXLDataset: PTBXLDataset not implemented yet

subsampling: not accomplished
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import PTBXL, DataBaseInfo
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN

###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "ptb-xl"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = PTBXL(_CWD)
if len(reader) == 0:
    reader.download()


class TestPTBXL:
    def test_len(self):
        assert len(reader) > 0

    def test_subsample(self):
        ss_ratio = 0.3
        reader_ss = PTBXL(_CWD, subsample=ss_ratio, verbose=0)
        assert len(reader_ss) == pytest.approx(len(reader) * ss_ratio, abs=1), f"{len(reader_ss)=} != {len(reader) * ss_ratio}"
        ss_ratio = 0.1 / len(reader)
        reader_ss = PTBXL(_CWD, subsample=ss_ratio)
        assert len(reader_ss) == 1, f"{len(reader_ss)=} != 1"

    def test_load_data(self):
        data = reader.load_data(0)
        assert data.ndim == 2 and data.shape[0] == 12
        data_1 = reader.load_data(0, leads=0, data_format="flat", sampto=1000)
        assert np.allclose(data[0][:1000], data_1)

    def test_reset_fs(self):
        reader.reset_fs(100)
        assert reader.fs == 100, f"{reader.fs=}"
        reader.reset_fs(500)
        assert reader.fs == 500, f"{reader.fs=}"

    def test_load_metadata(self):
        metadata = reader.load_metadata(0)
        assert isinstance(metadata, dict), f"{type(metadata)=}"
        assert len(metadata) > 0, f"{metadata=}"
        metadata_items = ["age", "sex", "height", "weight", "report"]
        metadata = reader.load_metadata(0, items=metadata_items)
        assert set(metadata.keys()) == set(metadata_items), f"{set(metadata.keys())=}, {set(metadata_items)=}"
        metadata = reader.load_metadata(0, items="patient_id")
        assert isinstance(metadata, int), f"{type(metadata)=}"

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert isinstance(ann, dict), f"{type(ann)=}"
        for k, v in ann.items():
            assert k in reader._df_scp_statements.index
            assert isinstance(v, float) and 0 <= v <= 100, f"{type(v)=}, {v=}"
        ann_1 = reader.load_ann(0, with_interpretation=True)
        assert set(ann.keys()) == set(ann_1.keys()), f"{set(ann.keys())=}, {set(ann_1.keys())=}"
        for k, v in ann_1.items():
            assert isinstance(v, dict), f"{type(v)=}"
            assert "likelihood" in v and v["likelihood"] == ann[k], f"{v=}, {ann[k]=}"

    def test_properties(self):
        data_split_dict = reader.default_train_val_test_split
        assert len(data_split_dict) == 3, f"{len(data_split_dict)=}"
        for k, v in data_split_dict.items():
            assert isinstance(v, list), f"{type(v)=}"
            assert all(item in reader._all_records for item in v), f"{set(v) - set(reader._all_records)=}"

        data_split_dict = reader.default_train_val_split
        assert len(data_split_dict) == 2, f"{len(data_split_dict)=}"
        for k, v in data_split_dict.items():
            assert isinstance(v, list)
            assert all(item in reader._all_records for item in v), f"{set(v) - set(reader._all_records)=}"

        assert isinstance(reader.version, str) and re.match(PHYSIONET_DB_VERSION_PATTERN, reader.version)
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)


# TODO: implement PTBXLDataset and test it
