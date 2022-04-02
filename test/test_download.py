"""
"""

import shutil
from pathlib import Path

import pytest

try:
    import torch_ecg
except:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    import torch_ecg
from torch_ecg.databases import (
    AFDB,
    ApneaECG,
    CINC2017,
    CINC2018,
    CINC2020,
    CINC2021,
    LTAFDB,
    LUDB,
    MITDB,
)


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parent.parent / "tmp" / "download"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
except Exception as e:
    raise e
_CWD.mkdir(parents=True, exist_ok=True)

###############################################################################


def test_download_ludb():
    """ """
    # download ludb
    db_dir = _CWD / "ludb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db = LUDB(db_dir)
    assert len(db) == 0
    db.download(compressed=True)
    db._ls_rec()
    assert len(db) == 200
    shutil.rmtree(db_dir)


def test_download_mitdb():
    """ """
    # download mitdb
    db_dir = _CWD / "mitdb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db = MITDB(db_dir)
    assert len(db) == 0
    db.download(compressed=True)
    db._ls_rec()
    assert len(db) == 48
    shutil.rmtree(db_dir)


def test_download_afdb():
    """ """
    # download afdb
    db_dir = _CWD / "afdb"
    db_dir.mkdir(parents=True, exist_ok=True)
    db = AFDB(db_dir)
    assert len(db) == 0
    db.download(compressed=True)
    db._ls_rec()
    assert len(db) == 23
    shutil.rmtree(db_dir)


# other databases are very large, hence currently not being tested
