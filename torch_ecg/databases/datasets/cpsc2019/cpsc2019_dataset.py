"""
data generator for feeding data into pytorch models
"""

import json
import math
from copy import deepcopy
from random import shuffle
from typing import List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from ...._preprocessors import PreprocManager
from ....cfg import CFG
from ....databases import CPSC2019 as CR
from ....utils.misc import ReprMixin
from .cpsc2019_cfg import CPSC2019TrainCfg  # noqa: F401


__all__ = [
    "CPSC2019Dataset",
]


class CPSC2019Dataset(ReprMixin, Dataset):
    """ """

    __DEBUG__ = False
    __name__ = "CPSC2019Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = False,
    ) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            configurations for the Dataset,
            ref. `CPSC2019TrainCfg`
            a simple example is:
            >>> config = deepcopy(CPSC2019TrainCfg)
            >>> config.db_dir = "some/path/to/db"
            >>> dataset = CPSC2019Dataset(config, training=True, lazy=False)
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        lazy: bool, default False,
            if True, the data will not be loaded immediately,

        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        self.training = training
        self.n_classes = 1
        self.lazy = lazy

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.siglen = self.config.input_len  # alias, for simplicity
        self.records = []
        self._train_test_split(
            train_ratio=self.config.train_ratio,
            force_recompute=False,
        )
        self.ppm = PreprocManager.from_config(self.config)

        self.fdr = FastDataReader(self.reader, self.records, self.config, self.ppm)

        self._signals = None
        self._labels = None
        if not self.lazy:
            self._load_all_data()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        if self.lazy:
            signal, label = self.fdr[index]
        else:
            signal, label = self._signals[index], self._labels[index]
        return signal, label

    def __len__(self) -> int:
        """ """
        return len(self.fdr)

    def _load_all_data(self) -> NoReturn:
        """ """
        self._signals, self._labels = [], []
        with tqdm(self.fdr, desc="loading data", unit="records") as pbar:
            for sig, lab in pbar:
                self._signals.append(sig)
                self._labels.append(lab)
        self._signals = np.array(self._signals)
        self._labels = np.array(self._labels)

    @property
    def signals(self) -> np.ndarray:
        """ """
        return self._signals

    @property
    def labels(self) -> np.ndarray:
        """ """
        return self._labels

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """

        do train test split,
        it is ensured that both the train and the test set contain all classes

        Parameters
        ----------
        train_ratio: float, default 0.8,
            ratio of the train set in the whole dataset (or the whole tranche(s))
        force_recompute: bool, default False,
            if True, force redo the train-test split,
            regardless of the existing ones stored in json files

        Returns
        -------
        records: list of str,
            list of the records split for training or validation

        """
        assert 0 < train_ratio < 100
        _train_ratio = train_ratio if train_ratio < 1 else train_ratio / 100
        split_fn = self.reader.db_dir / f"train_test_split_{_train_ratio:.2f}.json"
        if split_fn.is_file() and not force_recompute:
            split_res = json.loads(split_fn.read_text())
            if self.training:
                self.records = split_res["train"]
                shuffle(self.records)
            else:
                self.records = split_res["test"]
            return
        records = deepcopy(self.reader.all_records)
        shuffle(records)
        split_num = int(_train_ratio * len(records))
        train = sorted(records[:split_num])
        test = sorted(records[split_num:])
        split_res = {"train": train, "test": test}
        split_fn.write_text(json.dumps(split_res, ensure_ascii=False))
        if self.training:
            self.records = train
            shuffle(self.records)
        else:
            self.records = test

    def extra_repr_keys(self) -> List[str]:
        return [
            "training",
            "reader",
        ]


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: CR,
        records: Sequence[str],
        config: CFG,
        ppm: Optional[PreprocManager] = None,
    ) -> NoReturn:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.ppm = ppm

        self.siglen = self.config.input_len  # alias, for simplicity

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        rec_name = self.records[index]
        values = self.reader.load_data(rec_name, units="mV", keep_dim=False)
        rpeaks = self.reader.load_ann(rec_name, keep_dim=False)
        if self.config.get("recover_length", False):
            reduction = 1
        else:
            reduction = self.config.reduction
        labels = np.zeros((self.siglen // reduction))
        # rpeak indices to mask
        for r in rpeaks:
            if r < self.config.skip_dist or r >= self.siglen - self.config.skip_dist:
                continue
            start_idx = math.floor((r - self.config.bias_thr) / reduction)
            end_idx = math.ceil((r + self.config.bias_thr) / reduction)
            labels[start_idx:end_idx] = 1

        values = values.reshape((self.config.n_leads, self.siglen))
        labels = labels[..., np.newaxis]

        values, _ = self.ppm(values, self.config.fs)

        return values, labels

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]
