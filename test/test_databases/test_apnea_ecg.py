"""
methods from the base class, e.g. `load_data`, are tested in a simple way in this file,
since they are comprehensively tested `test_afdb.py`.

TestApneaECG: accomplished

subsampling: NOT tested
"""

import re
import shutil
from pathlib import Path

import pytest

from torch_ecg.databases import ApneaECG
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "apnea-ecg"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


with pytest.warns(RuntimeWarning):
    reader = ApneaECG(_CWD)
reader.download()


class TestApneaECG:
    def test_len(self):
        assert len(reader) == 70

    def test_load_data(self):
        data = reader.load_data(0)
        assert data.ndim == 2
        data = reader.load_data(0, leads=0, data_format="flat")
        assert data.ndim == 1
        data = reader.load_data(
            0, leads=0, data_format="flat", sampfrom=1000, sampto=2000
        )
        assert data.shape == (1000,)

    def test_load_ecg_data(self):
        rec = reader.ecg_records[0]
        data = reader.load_ecg_data(rec)
        assert data.ndim == 2

        rec = reader.rsp_records[0]
        with pytest.raises(ValueError, match=f"`{rec}` is not a record of ECG signals"):
            reader.load_ecg_data(rec)

    def test_load_rsp_data(self):
        rec = reader.rsp_records[0]
        data = reader.load_rsp_data(rec)
        assert data.ndim == 2

        rec = reader.ecg_records[0]
        with pytest.raises(ValueError, match=f"`{rec}` is not a record of RSP signals"):
            reader.load_rsp_data(rec)

    def test_load_ann(self):
        ann = reader.load_ann(0)
        assert all(len(item) == 2 for item in ann)
        assert all(isinstance(item[0], int) for item in ann)
        assert all(isinstance(item[1], str) for item in ann)

    def test_load_apnea_event(self):
        df_apnea_event = reader.load_apnea_event(0)
        assert df_apnea_event.columns.tolist() == reader.sleep_event_keys

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed

    def test_plot_ann(self):
        reader.plot_ann(0)
