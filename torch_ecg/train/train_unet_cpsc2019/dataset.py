"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
import math
from random import shuffle, randint, uniform, sample
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

# from torch_ecg.utils.misc import ensure_siglen, dict_to_str
from torch_ecg.train.database_reader.database_reader.other_databases import CPSC2019 as CR
from .cfg import TrainCfg, ModelCfg
from .utils import gen_baseline_wander


if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2019",
]


class CPSC2019(Dataset):
    """
    """
    __DEBUG__ = False
    __name__ = "CPSC2019"

    def __init__(self, config:ED, training:bool=True) -> NoReturn:
        """ finished, checked,

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
        self.n_classes = 1
        if ModelCfg.torch_dtype.lower() == "double":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.__data_aug = self.training

        self.siglen = self.config.input_len  # alias, for simplicity
        self.records = []
        self._train_test_split(
            train_ratio=self.config.train_ratio,
            force_recompute=False,
        )

        if self.config.bw:
            self._n_bw_choices = len(self.config.bw_ampl_ratio)
            self._n_gn_choices = len(self.config.bw_gaussian)


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        rec_name = self.records[index]
        ann_name = rec_name.replace("data", "R")
        values = self.reader.load_data(rec_name, units='mV', keep_dim=False)
        rpeaks = self.reader.load_ann(rec_name, keep_dim=False)
        labels = np.zeros((self.siglen,))
        # rpeak indices to mask
        for r in rpeaks:
            if r < self.config.skip_dist or r >= self.siglen - self.config.skip_dist:
                continue
            start_idx = r - self.config.bias_thr
            end_idx = r + self.config.bias_thr
            labels[start_idx:end_idx] = 1

        # data augmentation, finished yet
        sig_ampl = self._get_ampl(values)
        if self.__data_aug:
            if self.config.bw:
                ar = self.config.bw_ampl_ratio[randint(0, self._n_bw_choices-1)]
                gm, gs = self.config.bw_gaussian[randint(0, self._n_gn_choices-1)]
                bw_ampl = ar * sig_ampl
                g_ampl = gm * sig_ampl
                bw = gen_baseline_wander(
                    siglen=self.siglen,
                    fs=self.config.fs,
                    bw_fs=self.config.bw_fs,
                    amplitude=bw_ampl,
                    amplitude_mean=gm,
                    amplitude_std=gs,
                )
                values = values + bw
            if len(self.config.flip) > 0:
                sign = sample(self.config.flip, 1)[0]
                values *= sign
            if self.config.random_normalize:
                rn_mean = uniform(
                    self.config.random_normalize_mean[0],
                    self.config.random_normalize_mean[1],
                )
                rn_std = uniform(
                    self.config.random_normalize_std[0],
                    self.config.random_normalize_std[1],
                )
                values = (values-np.mean(values)+rn_mean) / np.std(values) * rn_std
            if self.config.label_smoothing > 0:
                label = (1 - self.config.label_smoothing) * label \
                    + self.config.label_smoothing / self.n_classes

        if self.__DEBUG__:
            self.reader.plot(
                rec="",  # unnecessary indeed
                data=values,
                ann=rpeaks,
                ticks_granularity=2,
            )

        values = values.reshape((self.config.n_leads, self.siglen))
        labels = labels[..., np.newaxis]

        return values, labels


    def __len__(self) -> int:
        """
        """
        return len(self.records)


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False


    def enable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = True


    def _get_ampl(self, values:np.ndarray, window:int=100) -> float:
        """ finished, checked,

        get amplitude of a segment

        Parameters:
        -----------
        values: ndarray,
            data of the segment
        window: int, default 100 (corr. to 200ms),
            window length of a window for computing amplitude, with units in number of sample points

        Returns:
        --------
        ampl: float,
            amplitude of `values`
        """
        half_window = window // 2
        ampl = 0
        for idx in range(len(values)//half_window-1):
            s = values[idx*half_window: idx*half_window+window]
            ampl = max(ampl, np.max(s)-np.min(s))
        return ampl

    
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
        assert 0 < train_ratio < 100
        _train_ratio = train_ratio if train_ratio < 1 else train_ratio/100
        split_fn = os.path.join(self.reader.db_dir, f"train_test_split_{_train_ratio:.2f}.json")
        if os.path.isfile(split_fn) and not force_recompute:
            with open(split_fn, "r") as f:
                split_res = json.load(f)
                if self.training:
                    self.records = split_res["train"]
                    shuffle(self.records)
                else:
                    self.records = split_res["test"]
            return
        records = deepcopy(self.reader.all_records)
        shuffle(records)
        split_num = int(_train_ratio*len(records))
        train = sorted(records[:split_num])
        test = sorted(records[split_num:])
        split_res = {"train":train, "test":test}
        with open(split_fn, "w") as f:
            json.dump(split_res, f, ensure_ascii=False)
        if self.training:
            self.records = train
            shuffle(self.records)
        else:
            self.records = test


    # def _check_nan(self) -> NoReturn:
    #     """
    #     """
    #     raise NotImplementedError
