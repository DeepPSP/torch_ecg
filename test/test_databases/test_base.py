""" """

import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import requests

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
    try:
        with pytest.warns(RuntimeWarning, match="`db_dir` is not specified"):
            reader = AFDB()
    except requests.exceptions.RequestException as err:
        # `AFDB()` with no `db_dir` populates the default cache dir,
        # which needs physionet.org; transient outages should skip,
        # not fail the lane
        pytest.skip(f"physionet.org unreachable: {err}")

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
    try:
        with pytest.warns(RuntimeWarning, match="`db_dir` is not specified"):
            reader = AFDB()
    except requests.exceptions.RequestException as err:
        pytest.skip(f"physionet.org unreachable: {err}")

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


def _make_afdb(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AFDB(tmp_path / "afdb", verbose=0)


def test_version_fallback_to_s3(monkeypatch, tmp_path):
    """physionet.org down -> version inferred from the S3 bucket layout."""
    import torch_ecg.databases.base as tb

    monkeypatch.setattr(tb, "wfdb_get_version", lambda db_name: (_ for _ in ()).throw(ConnectionError("site down")))
    monkeypatch.setattr(tb, "get_latest_version_from_physionet_s3", lambda db_name: "1.0.1")
    reader = _make_afdb(tmp_path)
    with pytest.warns(RuntimeWarning, match="Inferred version 1.0.1 from the S3 bucket"):
        assert reader.version == "1.0.1"


def test_version_fallback_default(monkeypatch, tmp_path):
    """physionet.org and S3 both down -> default version '1.0.0'."""
    import torch_ecg.databases.base as tb

    monkeypatch.setattr(tb, "wfdb_get_version", lambda db_name: (_ for _ in ()).throw(ConnectionError("site down")))
    monkeypatch.setattr(
        tb, "get_latest_version_from_physionet_s3", lambda db_name: (_ for _ in ()).throw(ConnectionError("s3 down"))
    )
    reader = _make_afdb(tmp_path)
    with pytest.warns(RuntimeWarning, match="Defaults to '1.0.0'"):
        assert reader.version == "1.0.0"


def test_url_get_zip_fallback(monkeypatch, tmp_path):
    """url_ falls back to the get-zip endpoint when the static zip probe
    fails (timeout) or reports a non-zip content type, and caches it."""
    import torch_ecg.databases.base as tb

    monkeypatch.setattr(tb, "wfdb_get_version", lambda db_name: "1.0.0")
    reader = _make_afdb(tmp_path)

    # probe crashes (e.g. physionet.org unreachable)
    monkeypatch.setattr(
        tb.requests, "head", lambda url, **kwargs: (_ for _ in ()).throw(requests.exceptions.ConnectTimeout("down"))
    )
    url = reader.url_
    assert url is not None and url.endswith("/get-zip/1.0.0"), url

    # probe answers, but the static zip does not exist
    reader._url_compressed = None
    monkeypatch.setattr(tb.requests, "head", lambda url, **kwargs: types.SimpleNamespace(headers={"Content-Type": "text/html"}))
    url = reader.url_
    assert url is not None and url.endswith("/get-zip/1.0.0"), url


def test_download_without_awscli_stays_on_s3(monkeypatch, tmp_path):
    """Without the aws binary, download() keeps use_s3 and lets http_get
    route to the boto3 backend (instead of dropping back to physionet)."""
    import torch_ecg.databases.base as tb

    calls = []
    monkeypatch.setattr(tb, "http_get", lambda url, dst_dir, **kwargs: calls.append(url))
    monkeypatch.setattr(tb.shutil, "which", lambda name: None)
    reader = _make_afdb(tmp_path)
    monkeypatch.setattr(tb, "wfdb_get_version", lambda db_name: "1.0.0")
    reader.download()
    assert calls == [reader.s3_url]
