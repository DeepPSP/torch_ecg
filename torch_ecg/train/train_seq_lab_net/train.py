"""
"""
import os
import sys
import time
import logging
import argparse
from copy import deepcopy
from collections import deque
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

from tqdm import tqdm
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

from ...models.ecg_crnn import ECG_CRNN
from ...models.nets import (
    BCEWithLogitsWithClassWeightLoss,
    default_collate_fn as collate_fn,
)
from ...model_configs import ECG_SEQ_LAB_NET_CONFIG
from ...utils.misc import init_logger, get_date_str, dict_to_str, str2bool
from .cfg import ModelCfg, TrainCfg
from .dataset import CINC2020

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


__all__ = [
    "train",
]


def train(model:nn.Module, device:torch.device, config:dict, log_step:int=20, logger:Optional[logging.Logger]=None, debug:bool=False):
    """ finished, checked,

    Parameters:
    -----------
    model: Module,
    device: torch.device,
    config: dict,
    log_step: int, default 20,
    logger: Logger, optional,
    debug: bool, default False,
    """
    print(f"training configurations are as follows:\n{dict_to_str(config)}")

    train_dataset = CINC2020(config=config, training=True)

    if debug:
        val_train_dataset = CINC2020(config=config, training=True)
        val_train_dataset.disable_data_augmentation()
    val_dataset = CINC2020(config=config, training=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    
    raise NotImplementedError
