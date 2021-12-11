# -*- coding: utf-8 -*-
"""
facilities for easy reading of `official` databases from physionet

data from `official` physionet databases can be loaded from its server at use,
or downloaded using `wfdb` easily beforehand

About the header (.hea) files:
https://physionet.org/physiotools/wag/header-5.htm
"""

from .afdb import AFDB
from .apnea_ecg import ApneaECG
from .cinc2017 import CINC2017
from .cinc2018 import CINC2018
from .cinc2020 import CINC2020
from .cinc2021 import CINC2021
from .ltafdb import LTAFDB
from .ludb import LUDB
from .mitdb import MITDB

__all__ = [
    "AFDB",
    "ApneaECG",
    "CINC2017",
    "CINC2018",
    "CINC2020",
    "CINC2021",
    "LTAFDB",
    "LUDB",
    "MITDB",
]
