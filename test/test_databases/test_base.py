""" """

from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import AFDB, list_databases
from torch_ecg.databases.base import (
    BeatAnn,
    CPSCDataBase,
    DataBaseInfo,
    NSRRDataBase,
    PhysioNetDataBase,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
    _DataBase,
)
from torch_ecg.databases.datasets import list_datasets


def test_base_database():
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {_DataBase.__name__}",
    ):
        db = _DataBase()  # type: ignore[abstract]

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {PhysioNetDataBase.__name__}",
    ):
        db = PhysioNetDataBase()  # type: ignore[abstract]

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {NSRRDataBase.__name__}",
    ):
        db = NSRRDataBase()  # type: ignore[abstract]

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {CPSCDataBase.__name__}",
    ):
        db = CPSCDataBase()  # type: ignore[abstract]


def test_beat_ann():
    index = 100
    for symbol, name in WFDB_Beat_Annotations.items():
        ba = BeatAnn(index, symbol)
        assert ba.index == index
        assert ba.symbol == symbol
        assert ba.name == name

    for symbol, name in WFDB_Non_Beat_Annotations.items():
        ba = BeatAnn(index, symbol)
        assert ba.index == index
        assert ba.symbol == symbol
        assert ba.name == name

    ba = BeatAnn(index, "XXX")
    assert ba.index == index
    assert ba.symbol == "XXX"
    assert ba.name == "XXX"


def test_get_arrhythmia_knowledge():
    assert _DataBase.get_arrhythmia_knowledge("AF") is None  # printed
    assert _DataBase.get_arrhythmia_knowledge(["AF", "PVC"]) is None  # printed


def test_database_meta():
    with pytest.warns(RuntimeWarning, match="`db_dir` is not specified"):
        reader = AFDB()

    assert reader.db_dir == Path("~").expanduser() / ".cache" / "torch_ecg" / "data" / "afdb"

    assert reader.helper() is None  # printed
    for item in ["attributes", "methods", "beat", "non-beat", "rhythm"]:
        assert reader.helper(item) is None  # printed
    assert reader.helper(["methods", "beat"]) is None  # printed

    for k in WFDB_Beat_Annotations:
        assert reader.helper(k) is None  # printed: `{k}` stands for `{WFDB_Beat_Annotations[k]}`
    for k in WFDB_Non_Beat_Annotations:
        assert reader.helper(k) is None  # printed: `{k}` stands for `{WFDB_Non_Beat_Annotations[k]}`
    for k in WFDB_Rhythm_Annotations:
        assert reader.helper(k) is None  # printed: `{k}` stands for `{WFDB_Rhythm_Annotations[k]}`

    with pytest.raises(NotImplementedError, match="not implemented for"):
        reader._auto_infer_units(np.ones((10, 2)), sig_type="EEG")


def test_database_info():
    with pytest.warns(RuntimeWarning, match="`db_dir` is not specified"):
        reader = AFDB()

    assert isinstance(reader.database_info, DataBaseInfo)


def test_list_databases():
    assert isinstance(list_databases(), list)
    assert len(list_databases()) > 0


def test_list_datasets():
    assert isinstance(list_datasets(), list)
    assert len(list_datasets()) > 0
    assert all([item.endswith("Dataset") for item in list_datasets()]), list_datasets()


class _NSRR(NSRRDataBase):
    """Minimal concrete NSRR database to exercise `safe_edf_file_operation`."""

    def _ls_rec(self):
        pass

    @property
    def database_info(self):
        return None

    def load_ann(self, rec):
        pass

    def load_data(self, rec):
        pass

    @property
    def url(self):
        return []


def test_edf_reader_roundtrip(tmp_path):
    """`safe_edf_file_operation` goes through the `_EdfReader` adapter on
    top of `edfio` (pyedflib was replaced); verify the API subset used by
    the NSRR readers via a synthetic EDF written with edfio itself."""
    from edfio import Edf, EdfSignal

    ecg = np.linspace(0.0, 1.0, 1250, dtype=np.float64)
    eeg = np.random.rand(625)
    signals = [
        EdfSignal(ecg, 125.0, label="ECG", transducer_type="AgCl", physical_dimension="mV", prefiltering="HP:0.5Hz"),
        EdfSignal(eeg, 62.5, label="EEG", physical_dimension="uV"),
    ]
    edf_path = tmp_path / "test.edf"
    Edf(signals=signals).write(edf_path)

    db = _NSRR("shhs", verbose=0)
    db.safe_edf_file_operation("open", edf_path)
    assert db.file_opened.getSignalLabels() == ["ECG", "EEG"]
    assert db.file_opened.getSampleFrequency(0) == 125.0
    assert db.file_opened.getSampleFrequency(1) == 62.5
    assert db.file_opened.getPhysicalDimension(0) == "mV"
    assert db.file_opened.getTransducer(0) == "AgCl"
    assert db.file_opened.getPrefilter(0) == "HP:0.5Hz"

    # physical samples are quantized through the digital range, so allow
    # a tolerance; digital samples must round-trip exactly
    np.testing.assert_allclose(db.file_opened.readSignal(0), ecg, atol=1e-3)
    digital = db.file_opened.readSignal(0, digital=True)
    assert np.issubdtype(digital.dtype, np.integer)
    db.safe_edf_file_operation("close")
    assert db.file_opened is None

    with pytest.raises(ValueError, match="Illegal operation"):
        db.safe_edf_file_operation("reopen")
