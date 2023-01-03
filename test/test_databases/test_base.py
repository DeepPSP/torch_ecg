"""
"""

from pathlib import Path

import pytest

from torch_ecg.databases.base import (
    _DataBase,
    PhysioNetDataBase,
    NSRRDataBase,
    CPSCDataBase,
    BeatAnn,
    DataBaseInfo,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
)
from torch_ecg.databases import AFDB, list_databases
from torch_ecg.databases.datasets import list_datasets


def test_base_database():
    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {_DataBase.__name__} with abstract methods",
    ):
        db = _DataBase()

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {PhysioNetDataBase.__name__} with abstract method",
    ):
        db = PhysioNetDataBase()

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {NSRRDataBase.__name__} with abstract methods",
    ):
        db = NSRRDataBase()

    with pytest.raises(
        TypeError,
        match=f"Can't instantiate abstract class {CPSCDataBase.__name__} with abstract methods",
    ):
        db = CPSCDataBase()


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

    assert (
        reader.db_dir
        == Path("~").expanduser() / ".cache" / "torch_ecg" / "data" / "afdb"
    )

    assert reader.helper() is None  # printed
    for item in ["attributes", "methods", "beat", "non-beat", "rhythm"]:
        assert reader.helper(item) is None  # printed
    assert reader.helper(["methods", "beat"]) is None  # printed

    for k in WFDB_Beat_Annotations:
        assert (
            reader.helper(k) is None
        )  # printed: `{k}` stands for `{WFDB_Beat_Annotations[k]}`
    for k in WFDB_Non_Beat_Annotations:
        assert (
            reader.helper(k) is None
        )  # printed: `{k}` stands for `{WFDB_Non_Beat_Annotations[k]}`
    for k in WFDB_Rhythm_Annotations:
        assert (
            reader.helper(k) is None
        )  # printed: `{k}` stands for `{WFDB_Rhythm_Annotations[k]}`


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
    assert all(item.endswith("Dataset") for item in list_datasets())
