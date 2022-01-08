"""
Segmentation using Fully-Convolutional Network

References
----------
[1] https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, List, Sequence, NoReturn, Any
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
