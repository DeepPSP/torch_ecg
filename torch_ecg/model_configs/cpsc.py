"""
the best model of CPSC2018

this model keeps number of channels constantly 12, without raising it at any step;
the basic blocks of this model is a combination of 2 small kernel layer with 1 large kernel layer,
following the pattern of
    baby vision --> baby vision --> giant vision
"""
from copy import deepcopy

from easydict import EasyDict as ED

from .cnn import (
    cpsc_block_basic,
    cpsc_2018,
)


__all__ = [
    "CPSC_CONFIG",
]


CPSC_CONFIG = ED()


# cnn part
CPSC_CONFIG.cnn = ED()
CPSC_CONFIG.cnn.name = "cpsc_2018"

if CPSC_CONFIG.cnn.name == "cpsc_2018":
    CPSC_CONFIG.cnn.cpsc_block = deepcopy(cpsc_block_basic)
    CPSC_CONFIG.cnn.cpsc = deepcopy(cpsc_2018)
else:
    pass


CPSC_CONFIG.rnn = ED()
CPSC_CONFIG.rnn.activation = "leaky"
CPSC_CONFIG.rnn.kw_activation = ED(negative_slope=0.2, inplace=True)
CPSC_CONFIG.rnn.dropout = 0.2
