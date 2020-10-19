"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
from random import shuffle, randint
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from easydict import EasyDict as ED

from ...utils.misc import ensure_siglen, dict_to_str
from ..database_reader.database_reader.other_databases import CPSC2019 as CR
from .cfg import TrainCfg, ModelCfg


if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2019",
]


class CPSC2019(Dataset):
    """
    """
    def __init__(self, config:ED, training:bool=True) -> NoReturn:
        """

        Parameters:
        -----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        self.training = training
        if ModelCfg.torch_dtype.lower() == 'double':
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.__data_aug = self.training
        raise NotImplementedError


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        rec = self.records[index]
        raise NotImplementedError
        # return values, labels


    def __len__(self) -> int:
        """
        """
        raise NotImplementedError
        # return len(self.records)


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    
    def _train_test_split(self, train_ratio:float=0.8, force_recompute:bool=False) -> List[str]:
        """

        do train test split,
        it is ensured that both the train and the test set contain all classes

        Parameters:
        -----------
        train_ratio: float, default 0.8,
            ratio of the train set in the whole dataset (or the whole tranche(s))
        force_recompute: bool, default False,
            if True, force redo the train-test split,
            regardless of the existing ones stored in json files

        Returns:
        --------
        records: list of str,
            list of the records split for training or validation
        """
        raise NotImplementedError


    def persistence(self) -> NoReturn:
        """

        make the dataset persistent w.r.t. the tranches and the ratios in `self.config`
        """
        raise NotImplementedError


    def _check_nan(self) -> NoReturn:
        """
        """
        raise NotImplementedError
