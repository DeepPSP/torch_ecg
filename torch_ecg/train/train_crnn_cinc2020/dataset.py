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
from easydict import EasyDict as ED
# try:
#     from tqdm.auto import tqdm
# except ModuleNotFoundError:
#     from tqdm import tqdm
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

from torch_ecg.utils.utils_signal import butter_bandpass_filter
from torch_ecg.utils.misc import ensure_siglen, dict_to_str
# from .data_reader import CINC2020Reader as CR
from torch_ecg.train.database_reader.database_reader.physionet_databases import CINC2020 as CR
from .cfg import TrainCfg, ModelCfg


if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CINC2020",
]


class CINC2020(Dataset):
    """
    """
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
        self._TRANCHES = self.config.tranche_classes.keys()  # ["A", "B", "AB", "E", "F"]
        self.reader = CR(db_dir=config.db_dir)
        self.tranches = config.tranches_for_training
        self.training = training
        if ModelCfg.torch_dtype.lower() == "double":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        assert not self.tranches or self.tranches in self._TRANCHES
        if self.tranches:
            self.all_classes = self.config.tranche_classes[self.tranches]
            self.class_weights = self.config.tranche_class_weights[self.tranches]
        else:
            self.all_classes = self.config.classes
            self.class_weights = self.config.class_weights
        self.n_classes = len(self.all_classes)
        # print(f"tranches = {self.tranches}, all_classes = {self.all_classes}")
        # print(f"class_weights = {dict_to_str(self.class_weights)}")
        cw = np.zeros((len(self.class_weights),), dtype=self.dtype)
        for idx, c in enumerate(self.all_classes):
            cw[idx] = self.class_weights[c]
        self.class_weights = torch.from_numpy(cw.astype(self.dtype)).view(1, self.n_classes)
        # if self.training:
        #     self.siglen = self.config.siglen
        # else:
        #     self.siglen = None
        # validation also goes in batches, hence length has to be fixed
        self.siglen = self.config.input_len

        self.records = self._train_test_split(config.train_ratio, force_recompute=False)
        self.records = [r for r in self.records if r not in self.reader.exceptional_records]

        self.__data_aug = self.training

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        rec = self.records[index]
        # values = self.reader.load_data(
        #     rec,
        #     data_format='channel_first', units='mV', backend='wfdb'
        # )
        # values = self.reader.load_resampled_data(rec, data_format="channel_first", siglen=self.siglen)
        values = self.reader.load_resampled_data(rec, data_format="channel_first", siglen=None)
        if self.config.bandpass is not None:
            values = butter_bandpass_filter(
                values,
                lowcut=self.config.bandpass[0],
                highcut=self.config.bandpass[1],
                order=self.config.bandpass_order,
                fs=self.config.fs,
            )
        values = ensure_siglen(values, siglen=self.siglen, fmt="channel_first")
        if self.config.normalize_data:
            values = (values - np.mean(values)) / np.std(values)
        labels = self.reader.get_labels(
            rec, scored_only=True, fmt='a', normalize=True
        )
        labels = np.isin(self.all_classes, labels).astype(int)

        if self.__data_aug:
            # data augmentation for input
            if self.config.random_mask > 0:
                mask_len = randint(0, self.config.random_mask)
                mask_start = randint(0, self.siglen-mask_len-1)
                values[...,mask_start:mask_start+mask_len] = 0
            if self.config.stretch_compress != 1:
                pass  # not implemented
            # data augmentation for labels
            labels = (1 - self.config.label_smoothing) * labels \
                + self.config.label_smoothing / self.n_classes

        if self.config.data_format.lower() in ['channel_last', 'lead_last']:
            values = values.T

        return values, labels


    def __len__(self) -> int:
        """
        """
        return len(self.records)


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    
    def _train_test_split(self, train_ratio:float=0.8, force_recompute:bool=False) -> List[str]:
        """ finished, checked,

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
        _TRANCHES = list("ABEF")
        _train_ratio = int(train_ratio*100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        file_suffix = f"_siglen_{self.siglen}.json"
        train_file = os.path.join(self.reader.db_dir_base, f"train_ratio_{_train_ratio}{file_suffix}")
        test_file = os.path.join(self.reader.db_dir_base, f"test_ratio_{_test_ratio}{file_suffix}")

        if force_recompute or not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            tranche_records = {t: [] for t in _TRANCHES}
            train_set = {t: [] for t in _TRANCHES}
            test_set = {t: [] for t in _TRANCHES}
            for t in _TRANCHES:
                with tqdm(self.reader.all_records[t], total=len(self.reader.all_records[t])) as bar:
                    for rec in bar:
                        if rec in self.reader.exceptional_records:
                            # skip exceptional records
                            continue
                        rec_labels = self.reader.get_labels(rec, scored_only=True, fmt='a', normalize=True)
                        rec_labels = [c for c in rec_labels if c in TrainCfg.tranche_classes[t]]
                        if len(rec_labels) == 0:
                            # skip records with no scored class
                            continue
                        rec_samples = self.reader.load_resampled_data(rec).shape[1]
                        if rec_samples < self.siglen:
                            continue
                        tranche_records[t].append(rec)
                    print(f"tranche {t} has {len(tranche_records[t])} valid records for training")
            for t in _TRANCHES:
                is_valid = False
                while not is_valid:
                    shuffle(tranche_records[t])
                    split_idx = int(len(tranche_records[t])*train_ratio)
                    train_set[t] = tranche_records[t][:split_idx]
                    test_set[t] = tranche_records[t][split_idx:]
                    is_valid = _check_train_test_split_validity(train_set[t], test_set[t], set(TrainCfg.tranche_classes[t]))
            with open(train_file, "w") as f:
                json.dump(train_set, f, ensure_ascii=False)
            with open(test_file, "w") as f:
                json.dump(test_set, f, ensure_ascii=False)
        else:
            with open(train_file, "r") as f:
                train_set = json.load(f)
            with open(test_file, "r") as f:
                test_set = json.load(f)

        add = lambda a,b:a+b
        _tranches = list(self.tranches or "ABEF")
        if self.training:
            records = reduce(add, [train_set[k] for k in _tranches])
        else:
            records = reduce(add, [test_set[k] for k in _tranches])
        return records


    def _check_train_test_split_validity(self, train_set:List[str], test_set:List[str], all_classes:Set[str]) -> bool:
        """ finished, checked,

        the train-test split is valid iff
        records in both `train_set` and `test` contain all classes in `all_classes`

        Parameters:
        -----------
        train_set: list of str,
            list of the records in the train set
        test_set: list of str,
            list of the records in the test set
        all_classes: set of str,
            the set of all classes for training

        Returns:
        --------
        is_valid: bool,
            the split is valid or not
        """
        add = lambda a,b:a+b
        train_classes = set(reduce(add, [self.reader.get_labels(rec, fmt='a') for rec in train_set]))
        train_classes.intersection_update(all_classes)
        test_classes = set(reduce(add, [self.reader.get_labels(rec, fmt='a') for rec in test_set]))
        test_classes.intersection_update(all_classes)
        is_valid = (len(all_classes) == len(train_classes) == len(test_classes))
        print(f"all_classes = {all_classes}\ntrain_classes = {train_classes}\ntest_classes = {test_classes}\nis_valid = {is_valid}")
        return is_valid


    def persistence(self) -> NoReturn:
        """ finished, checked,

        make the dataset persistent w.r.t. the tranches and the ratios in `self.config`
        """
        _TRANCHES = "ABEF"
        prev_state = self.__data_aug
        self.disable_data_augmentation()
        if self.training:
            ratio = int(self.config.train_ratio*100)
        else:
            ratio = 100 - int(self.config.train_ratio*100)
        fn_suffix = f"tranches_{self.tranches or _TRANCHES}_ratio_{ratio}"
        if self.config.bandpass is not None:
            bp_low = max(0, self.config.bandpass[0])
            bp_high = min(self.config.bandpass[1], self.config.fs//2)
            fn_suffix = fn_suffix + f"_bp_{bp_low:.1f}_{bp_high:.1f}"
        fn_suffix = fn_suffix + f"_siglen_{self.siglen}"

        X, y = [], []
        with tqdm(range(self.__len__()), total=self.__len__()) as bar:
            for idx in bar:
                values, labels = self.__getitem__(idx)
                X.append(values)
                y.append(labels)
        X, y = np.array(X), np.array(y)
        print(f"X.shape = {X.shape}, y.shape = {y.shape}")
        filename = f"{'train' if self.training else 'test'}_X_{fn_suffix}.npy"
        np.save(os.path.join(self.reader.db_dir_base, filename), X)
        print(f"X saved to {filename}")
        filename = f"{'train' if self.training else 'test'}_y_{fn_suffix}.npy"
        np.save(os.path.join(self.reader.db_dir_base, filename), y)
        print(f"y saved to {filename}")

        self.__data_aug = prev_state

    def _check_nan(self) -> NoReturn:
        """ finished, checked,

        during training, sometimes nan values are encountered,
        which ruins the whole training process
        """
        prev_state = self.__data_aug
        self.disable_data_augmentation()

        for idx, (values, labels) in self:
            if np.isnan(values).any():
                print(f"values of {self.records[idx]} have nan values")
            if np.isnan(labels).any():
                print(f"labels of {self.records[idx]} have nan values")

        self.__data_aug = prev_state
