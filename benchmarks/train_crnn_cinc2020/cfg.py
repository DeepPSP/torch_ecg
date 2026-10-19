"""Configs for the CINC2020 benchmark example.

CINC2020 is the 12-lead-only predecessor of CINC2021, and its data is a
strict subset of CINC2021's; the model and trainer machinery are shared
with (imported from) the CINC2021 benchmark, which see.
"""

from copy import deepcopy

from torch_ecg.databases.datasets.cinc2020 import CINC2020TrainCfg

__all__ = [
    "TrainCfg",
]


TrainCfg = deepcopy(CINC2020TrainCfg)
