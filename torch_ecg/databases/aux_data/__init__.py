"""
"""

from pathlib import Path

import pandas as pd

from . import cinc2020_aux_data, cinc2021_aux_data

__all__ = [
    "cinc2020_aux_data",
    "cinc2021_aux_data",
    "get_physionet_dbs",
]


def get_physionet_dbs() -> pd.DataFrame:
    """
    load the list of PhysioNet databases,
    locally stored in the file "./physionet_dbs.csv.tar.gz"
    """
    return pd.read_csv(Path(__file__).absolute().parent / "physionet_dbs.csv.tar.gz")
