""" """

from pathlib import Path

import pytest

from benchmarks.train_unet_ludb.train import main

_CWD = Path(__file__).absolute().parents[1] / "tmp" / "test_unet_ludb_pipeline"
_DB_DIR = _CWD / "ludb"


@pytest.fixture(scope="session")
def db_dir():
    """ """
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from torch_ecg.databases.physionet_databases.ludb import LUDB

        dr = LUDB(_DB_DIR)
        dr.download(compressed=True)
    except Exception as err:
        pytest.skip(f"failed to download LUDB: {err}")
    return _DB_DIR


@pytest.mark.db
def test_unet_ludb_pipeline(db_dir, tmp_path):
    """

    thin wrapper over the benchmarks.train_unet_ludb example,
    trained briefly on a downloaded copy of LUDB.

    """
    main(
        [
            "--db-dir",
            str(db_dir),
            "--smoke",
            "--num-workers",
            "0",
            "--outdir",
            str(tmp_path / "ludb"),
        ]
    )
