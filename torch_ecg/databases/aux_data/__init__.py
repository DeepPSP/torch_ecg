"""
"""

from pathlib import Path

import pandas as pd
import wfdb

from ...utils.misc import timeout
from . import aha, cinc2020_aux_data, cinc2021_aux_data

__all__ = [
    "cinc2020_aux_data",
    "cinc2021_aux_data",
    "get_physionet_dbs",
    "aha",
]


def get_physionet_dbs() -> pd.DataFrame:
    """
    load the list of PhysioNet databases,
    locally stored in the file "./physionet_dbs.csv.gz"
    """
    try:
        with timeout(3):
            return pd.DataFrame(wfdb.get_dbs(), columns=["db_name", "db_description"])
    except Exception:  # TimeoutError, ConnectionError, ReadTimeout, ...
        pass
    return pd.read_csv(Path(__file__).absolute().parent / "physionet_dbs.csv.gz")
