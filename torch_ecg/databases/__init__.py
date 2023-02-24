"""
ECG Database readers
====================

This module contains ECG database readers.

.. currentmodule:: torch_ecg.databases

Base classes
------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    PhysioNetDataBase
    NSRRDataBase
    CPSCDataBase
    PSGDataBaseMixin

PhysioNet database readers
--------------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    AFDB
    ApneaECG
    CINC2017
    CINC2018
    CINC2020
    CINC2021
    LTAFDB
    LUDB
    MITDB
    QTDB

CPSC database readers
---------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    CPSC2018
    CPSC2019
    CPSC2020
    CPSC2021

NSRR database readers
---------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    SHHS

Other database readers
----------------------
.. autosummary::
    :toctree: generated/
    :recursive:

    CACHET_CADB
    SPH

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/
    :recursive:

    BeatAnn

"""

from . import aux_data
from .base import (
    PhysioNetDataBase,
    NSRRDataBase,
    CPSCDataBase,
    DataBaseInfo,
    BeatAnn,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
    PSGDataBaseMixin,
)
from .cpsc_databases import CPSC2018, CPSC2019, CPSC2020, CPSC2021
from .nsrr_databases import SHHS
from .other_databases import CACHET_CADB, SPH
from .physionet_databases import (
    AFDB,
    ApneaECG,
    CINC2017,
    CINC2018,
    CINC2020,
    CINC2021,
    LTAFDB,
    LUDB,
    MITDB,
    QTDB,
)


__all__ = [
    # from physionet
    "AFDB",
    "ApneaECG",
    "CINC2017",
    "CINC2018",
    "CINC2020",
    "CINC2021",
    "LTAFDB",
    "LUDB",
    "MITDB",
    "QTDB",
    # from CPSC
    "CPSC2018",
    "CPSC2019",
    "CPSC2020",
    "CPSC2021",
    # from NSRR
    "SHHS",
    # other databases
    "CACHET_CADB",
    "SPH",
    # base classes
    "PhysioNetDataBase",
    "NSRRDataBase",
    "CPSCDataBase",
    "PSGDataBaseMixin",
    # auxilliary data, functions and classes
    "aux_data",
    "WFDB_Beat_Annotations",
    "WFDB_Non_Beat_Annotations",
    "WFDB_Rhythm_Annotations",
    "BeatAnn",
    "DataBaseInfo",
    "list_databases",
]


def list_databases() -> list:
    return [
        db
        for db in __all__
        if db
        not in [
            "PhysioNetDataBase",
            "NSRRDataBase",
            "CPSCDataBase",
            "PSGDataBaseMixin",
            "aux_data",
            "WFDB_Beat_Annotations",
            "WFDB_Non_Beat_Annotations",
            "WFDB_Rhythm_Annotations",
            "BeatAnn",
            "DataBaseInfo",
            "list_databases",
        ]
    ]
