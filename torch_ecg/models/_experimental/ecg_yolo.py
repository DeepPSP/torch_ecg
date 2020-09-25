"""
3rd place (entry 0436) of CPSC2019
"""

import sys
from copy import deepcopy
from collections import OrderedDict
from itertools import repeat
from typing import Union, Optional, Sequence, NoReturn
from numbers import Real

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from ...cfg import Cfg
from ...utils.utils_nn import compute_deconv_output_shape
from ...utils.misc import dict_to_str
from ..nets import (
    Conv_Bn_Activation,
    DownSample, ZeroPadding,
)

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)







