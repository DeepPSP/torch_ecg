"""
"""

import shutil
from pathlib import Path

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.databases import AFDB  # noqa: F401; noqa: F401
from torch_ecg.databases import CINC2017  # noqa: F401
from torch_ecg.databases import CINC2018  # noqa: F401
from torch_ecg.databases import CINC2020  # noqa: F401
from torch_ecg.databases import CINC2021  # noqa: F401
from torch_ecg.databases import LTAFDB  # noqa: F401
from torch_ecg.databases import LUDB  # noqa: F401
from torch_ecg.databases import MITDB  # noqa: F401
from torch_ecg.databases import ApneaECG  # noqa: F401
from torch_ecg.databases import CPSC2019  # noqa: F401


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[1] / "tmp" / "download"
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


def test_download_cpsc2019():
    """ """
    # download cpsc2019
    db_dir = _CWD / "cpsc2019"
    db_dir.mkdir(parents=True, exist_ok=True)
    db = CPSC2019(db_dir)
    assert len(db) == 0
    db.download()
    db._ls_rec()
    assert len(db) == 2000
    shutil.rmtree(db_dir)


# other databases are very large, hence currently not being tested


if __name__ == "__main__":
    test_download_afdb()
    test_download_ludb()
    test_download_mitdb()
    test_download_cpsc2019()
