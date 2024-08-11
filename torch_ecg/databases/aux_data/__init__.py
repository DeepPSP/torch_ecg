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


def get_physionet_dbs(local: bool = True) -> pd.DataFrame:
    """
    load the list of PhysioNet databases,
    locally stored in the file "./physionet_dbs.csv.gz"
    """
    if not local:
        try:
            with timeout(2):
                return (
                    pd.DataFrame(wfdb.get_dbs(), columns=["db_name", "db_description"]).drop_duplicates().reset_index(drop=True)
                )
        except Exception:  # TimeoutError, ConnectionError, ReadTimeout, ...
            print("Failed to fetch the list of PhysioNet databases from the remote server.")
    return pd.read_csv(Path(__file__).absolute().parent / "physionet_dbs.csv.gz")
