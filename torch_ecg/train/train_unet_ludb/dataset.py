"""
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

# from .data_reader import LUDBReader as LR
from ..database_reader.database_reader.physionet_databases import LUDB as LR
from .cfg import TrainCfg

if TrainCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "LUDB",
]


class LUDB(Dataset):
    """
    """
    def __init__(self, config:ED, leads:Optional[Union[Sequence[str], str]], training:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
            can be one of "A", "B", "AB", "E", "F", or None (or '', defaults to "ABEF")
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = LR(db_dir=config.db_dir)
        self.training = training
        self.classes = self.config.classes
        self.n_classes = len(self.classes)
        self.siglen = self.config.input_len
        if leads is None:
            self.leads = self.reader.all_leads
        elif isinstance(leads, str):
            self.leads = [leads]
        else:
            self.leads = list(leads)
        self.leads = [
            self.reader.all_leads[idx] \
                for idx,l in enumerate(self.leads) if l.lower() in self.reader.all_leads_lower
        ]

        self.records = self._train_test_split(config.train_ratio)

        # self.__data_aug = self.training
        self.__data_aug = False

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        if self.config.lead is not None:
            rec_idx = index
            lead_idx = self.config.leads_ordering.index(self.config.lead)
        elif self.config.use_single_lead:
            rec_idx, lead_idx = divmod(index, 12)
        else:
            rec_idx, lead_idx = index, None
        rec = self.records[rec_idx]
        values = self.reader.load_data(
            rec, data_format='channel_first', units='mV',
        )
        if self.config.normalize_data:
            values = (values - np.mean(values)) / np.std(values)
        masks = self.reader.load_masks(
            rec, leads=self.leads, mask_format='channel_first',
            class_map=self.config.class_map,
        )
        sampfrom = randint(
            self.config.start_from,
            masks.shape[1] - self.config.config.end_to - self.siglen
        )
        sampto = sampfrom + self.siglen
        values = values[..., sampfrom:sampto]
        masks = masks[..., sampfrom:sampto]

        if self.__data_aug:
            # data augmentation for input
            raise NotImplementedError

        if lead_idx is not None:
            values = values[idx:idx+1, ...]
            masks = masks[idx:idx+1, ...]

        return values, masks


    def __len__(self) -> int:
        """
        """
        if self.config.lead is None and self.config.use_single_lead:
            return 12 * len(self.records)
        return len(self.records)


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    
    def _train_test_split(self, train_ratio:float=0.8, force_recompute:bool=False) -> List[str]:
        """ finished, checked,

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
        file_suffix = f"_siglen_{self.siglen}.json"
        train_file = os.path.join(self.reader.db_dir, f"train_ratio_{_train_ratio}{file_suffix}")
        test_file = os.path.join(self.reader.db_dir, f"test_ratio_{_test_ratio}{file_suffix}")

        if force_recompute or not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            all_records = deepcopy(self.reader.all_records)
            shuffle(all_records)
            split_idx = int(train_ratio * len(all_records))
            train_set = all_records[:split_idx]
            test_set = all_records[split_idx:]
            with open(train_file, "w") as f:
                json.dump(train_set, f, ensure_ascii=False)
            with open(test_file, "w") as f:
                json.dump(test_set, f, ensure_ascii=False)
        else:
            with open(train_file, "r") as f:
                train_set = json.load(f)
            with open(test_file, "r") as f:
                test_set = json.load(f)
        if self.training:
            records = train_set
        else:
            records = test_set
        if self.config.over_sampling > 1:
            records = records * self.config.over_sampling
            shuffle(records)
        return records


    @DeprecationWarning
    def persistence(self) -> NoReturn:
        """ NOT finished, NOT checked,

        make the dataset persistent w.r.t. the tranches and the ratios in `self.config`
        """
        prev_state = self.__data_aug
        self.disable_data_augmentation()
        if self.training:
            ratio = int(self.config.train_ratio*100)
        else:
            ratio = 100 - int(self.config.train_ratio*100)
        raise NotImplementedError

        self.__data_aug = prev_state
