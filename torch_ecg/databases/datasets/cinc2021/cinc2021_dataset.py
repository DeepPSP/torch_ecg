"""
data generator for feeding data into pytorch models
"""

import json
import textwrap
import time
from copy import deepcopy
from pathlib import Path
from random import sample, shuffle
from typing import List, NoReturn, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

from ...._preprocessors import PreprocManager
from ....cfg import CFG
from ....databases import CINC2021 as CR
from ....utils.ecg_arrhythmia_knowledge import Standard12Leads
from ....utils.misc import ReprMixin, list_sum
from ....utils.utils_data import ensure_siglen
from ....utils.utils_signal import remove_spikes_naive
from .cinc2021_cfg import CINC2021TrainCfg  # noqa: F401


__all__ = [
    "CINC2021Dataset",
]


class CINC2021Dataset(ReprMixin, Dataset):
    """ """

    __DEBUG__ = False
    __name__ = "CINC2021Dataset"

    def __init__(
        self, config: Optional[CFG] = None, training: bool = True, lazy: bool = True
    ) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            configurations for the Dataset,
            ref. `CINC2021TrainCfg`,
            a simple example is:
            >>> config = deepcopy(CINC2021TrainCfg)
            >>> config.db_dir = "some/path/to/db"
            >>> dataset = CINC2021Dataset(config, training=True, lazy=False)
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        lazy: bool, default True,
            if True, the data will not be loaded immediately,

        """
        super().__init__()
        self.config = deepcopy(config)
        assert self.config.db_dir is not None, "db_dir must be specified"
        self.config.db_dir = Path(self.config.db_dir)
        self._TRANCHES = (
            self.config.tranche_classes.keys()
        )  # ["A", "B", "AB", "E", "F", "G",]
        self.reader = CR(db_dir=self.config.db_dir)
        self.tranches = self.config.tranches_for_training
        self.training = training
        if self.config.torch_dtype == torch.float64:
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
        self.config.all_classes = deepcopy(self.all_classes)
        self.n_classes = len(self.all_classes)
        # print(f"tranches = {self.tranches}, all_classes = {self.all_classes}")
        # print(f"class_weights = {dict_to_str(self.class_weights)}")
        cw = np.zeros((len(self.class_weights),), dtype=self.dtype)
        for idx, c in enumerate(self.all_classes):
            cw[idx] = self.class_weights[c]
        self.class_weights = torch.from_numpy(cw.astype(self.dtype)).view(
            1, self.n_classes
        )
        # validation also goes in batches, hence length has to be fixed
        self.siglen = self.config.input_len
        self.lazy = lazy

        self._indices = [Standard12Leads.index(ld) for ld in self.config.leads]

        self.records = self._train_test_split(
            self.config.train_ratio, force_recompute=False
        )
        # TODO: consider using `remove_spikes_naive` to treat these exceptional records
        self.records = [
            r
            for r in self.records
            if r not in self.reader.exceptional_records
            and (self.reader.get_data_filepath(r)).is_file()
        ]
        if self.__DEBUG__:
            self.records = sample(self.records, int(len(self.records) * 0.01))

        ppm_config = CFG(random=False)
        ppm_config.update(self.config)
        self.ppm = PreprocManager.from_config(ppm_config)
        self.ppm.rearrange(["bandpass", "normalize"])

        self._signals = np.array([], dtype=self.dtype).reshape(
            0, len(self.config.leads), self.siglen
        )
        self._labels = np.array([], dtype=self.dtype).reshape(0, self.n_classes)
        if not self.lazy:
            self._load_all_data()

    def _load_all_data(self) -> NoReturn:
        """ """
        # self.reader can not be pickled
        # with mp.Pool(processes=max(1, mp.cpu_count()-2)) as pool:
        #     self._signals, self._labels = \
        #         zip(*pool.starmap(_load_record, [(self.reader, r, self.config) for r in self.records]))

        # self._signals = np.array([]).reshape(0, len(self.config.leads), self.siglen)
        # self._labels = np.array([]).reshape(0, self.n_classes)

        fdr = FastDataReader(self.reader, self.records, self.config, self.ppm)

        # with tqdm(self.records, desc="Loading data", unit="records") as pbar:
        #     for rec in pbar:
        # s, l = self._load_one_record(rec)  # self._load_one_record is much slower than FastDataReader
        self._signals, self._labels = [], []
        with tqdm(desc="Loading data", total=len(fdr), unit="records") as pbar:
            for idx in range(len(fdr)):
                sig, lb = fdr[idx]
                # np.concatenate slows down the process severely
                # self._signals = np.concatenate((self._signals, s), axis=0)
                # self._labels = np.concatenate((self._labels, l), axis=0)
                self._signals.append(sig)
                self._labels.append(lb)
                pbar.update(1)
        self._signals = np.concatenate(self._signals, axis=0).astype(self.dtype)
        self._labels = np.concatenate(self._labels, axis=0)

    def _load_one_record(self, rec: str) -> Tuple[np.ndarray, np.ndarray]:
        """

        load a record from the database using data reader

        NOTE
        ----
        DO NOT USE THIS FUNCTION DIRECTLY for preloading data,
        use `FastDataReader` instead

        Parameters
        ----------
        rec: str,
            the record to load

        Returns
        -------
        values: np.ndarray,
            the values of the record
        labels: np.ndarray,
            the labels of the record

        """
        values = self.reader.load_resampled_data(
            rec,
            leads=self.config.leads,
            # leads=Standard12Leads,
            data_format=self.config.data_format,
            siglen=None,
        )
        for idx in range(values.shape[0]):
            values[idx] = remove_spikes_naive(values[idx])
        values, _ = self.ppm(values, self.config.fs)
        values = ensure_siglen(
            values,
            siglen=self.siglen,
            fmt=self.config.data_format,
            tolerance=self.config.sig_slice_tol,
        ).astype(self.dtype)
        if values.ndim == 2:
            values = values[np.newaxis, ...]

        labels = self.reader.get_labels(rec, scored_only=True, fmt="a", normalize=True)
        labels = (
            np.isin(self.all_classes, labels)
            .astype(self.dtype)[np.newaxis, ...]
            .repeat(values.shape[0], axis=0)
        )

        return values, labels

    def to(self, leads: Sequence[str]) -> NoReturn:
        """ """
        prev_leads = self.config.leads
        self.config.leads = leads
        self._indices = [prev_leads.index(ld) for ld in leads]
        self._signals = self._signals[:, self._indices, :]

    def emtpy(self, leads: Optional[Sequence[str]] = None) -> NoReturn:
        """ """
        if leads is None:
            leads = self.config.leads
        else:
            self.config.leads = leads
        self._signals = np.array([], dtype=self.dtype).reshape(
            0, len(leads), self.siglen
        )

    @classmethod
    def from_extern(cls, ext_ds: "CINC2021Dataset", config: CFG) -> "CINC2021Dataset":
        """ """
        new_ds = cls(config, ext_ds.training, lazy=True)
        indices = [ext_ds.config.leads.index(ld) for ld in new_ds.config.leads]
        new_ds._signals = ext_ds._signals[:, indices, :]
        new_ds._labels = ext_ds._labels.copy()
        return new_ds

    def reload_from_extern(self, ext_ds: "CINC2021Dataset") -> NoReturn:
        """ """
        indices = [ext_ds.config.leads.index(ld) for ld in self.config.leads]
        self._signals = ext_ds._signals[:, indices, :]
        self._labels = ext_ds._labels.copy()

    @property
    def signals(self) -> np.ndarray:
        """ """
        return self._signals

    @property
    def labels(self) -> np.ndarray:
        """ """
        return self._labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        return self.signals[index], self.labels[index]

    def __len__(self) -> int:
        """ """
        return len(self._signals)

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
        time.sleep(1)
        start = time.time()
        print("\nstart performing train test split...\n")
        time.sleep(1)
        _TRANCHES = list("ABEFG")
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        ns = "_ns" if len(self.config.special_classes) == 0 else ""
        file_suffix = f"_siglen_{self.siglen}{ns}.json"
        train_file = (
            self.reader.db_dir_base / f"train_ratio_{_train_ratio}{file_suffix}"
        )
        test_file = self.reader.db_dir_base / f"test_ratio_{_test_ratio}{file_suffix}"

        # TODO: use self.reader.df_stats (precomputed and stored in utils/stats.csv)
        # to accelerate the validity examinations
        if force_recompute or not all([train_file.is_file(), test_file.is_file()]):
            tranche_records = {t: [] for t in _TRANCHES}
            train_set = {t: [] for t in _TRANCHES}
            test_set = {t: [] for t in _TRANCHES}
            for t in _TRANCHES:
                with tqdm(
                    self.reader.all_records[t], total=len(self.reader.all_records[t])
                ) as bar:
                    for rec in bar:
                        if rec in self.reader.exceptional_records:
                            # skip exceptional records
                            continue
                        rec_labels = self.reader.get_labels(
                            rec, scored_only=True, fmt="a", normalize=True
                        )
                        rec_labels = [
                            c for c in rec_labels if c in self.config.tranche_classes[t]
                        ]
                        if len(rec_labels) == 0:
                            # skip records with no scored class
                            continue
                        rec_samples = self.reader.load_resampled_data(rec).shape[1]
                        # NEW in CinC2021 compared to CinC2020
                        # training input siglen raised from 4000 to 5000,
                        # hence allow tolerance in siglen now
                        if rec_samples < self.siglen - self.config.input_len_tol:
                            continue
                        tranche_records[t].append(rec)
                time.sleep(1)
                print(
                    f"tranche {t} has {len(tranche_records[t])} valid records for training"
                )
            for t in _TRANCHES:
                is_valid = False
                while not is_valid:
                    shuffle(tranche_records[t])
                    split_idx = int(len(tranche_records[t]) * train_ratio)
                    train_set[t] = tranche_records[t][:split_idx]
                    test_set[t] = tranche_records[t][split_idx:]
                    is_valid = self._check_train_test_split_validity(
                        train_set[t], test_set[t], set(self.config.tranche_classes[t])
                    )
            train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            test_file.write_text(json.dumps(test_set, ensure_ascii=False))
            print(
                textwrap.dedent(
                    f"""
                train set saved to \n\042{str(train_file)}\042
                test set saved to \n\042{str(test_file)}\042
                """
                )
            )
        else:
            train_set = json.loads(train_file.read_text())
            test_set = json.loads(test_file.read_text())

        print(f"train test split finished in {(time.time()-start)/60:.2f} minutes")

        _tranches = list(self.tranches or "ABEFG")
        if self.training == "all":
            records = list_sum([train_set[k] for k in _tranches]) + list_sum(
                [test_set[k] for k in _tranches]
            )
        elif self.training is True:
            records = list_sum([train_set[k] for k in _tranches])
        else:
            records = list_sum([test_set[k] for k in _tranches])
        return records

    def _check_train_test_split_validity(
        self, train_set: List[str], test_set: List[str], all_classes: Set[str]
    ) -> bool:
        """

        the train-test split is valid iff
        records in both `train_set` and `test` contain all classes in `all_classes`

        Parameters
        ----------
        train_set: list of str,
            list of the records in the train set
        test_set: list of str,
            list of the records in the test set
        all_classes: set of str,
            the set of all classes for training

        Returns
        -------
        is_valid: bool,
            the split is valid or not

        """
        train_classes = set(
            list_sum([self.reader.get_labels(rec, fmt="a") for rec in train_set])
        )
        train_classes.intersection_update(all_classes)
        test_classes = set(
            list_sum([self.reader.get_labels(rec, fmt="a") for rec in test_set])
        )
        test_classes.intersection_update(all_classes)
        is_valid = len(all_classes) == len(train_classes) == len(test_classes)
        print(
            textwrap.dedent(
                f"""
            all_classes:     {all_classes}
            train_classes:   {train_classes}
            test_classes:    {test_classes}
            is_valid:        {is_valid}
            """
            )
        )
        return is_valid

    def persistence(self) -> NoReturn:
        """
        make the dataset persistent w.r.t. the tranches and the ratios in `self.config`

        """
        _TRANCHES = "ABEFG"
        if self.training:
            ratio = int(self.config.train_ratio * 100)
        else:
            ratio = 100 - int(self.config.train_ratio * 100)
        fn_suffix = f"tranches_{self.tranches or _TRANCHES}_ratio_{ratio}"
        if self.config.bandpass is not None:
            bp_low = max(0, self.config.bandpass[0])
            bp_high = min(self.config.bandpass[1], self.config.fs // 2)
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
        np.save(self.reader.db_dir_base / filename, X)
        print(f"X saved to {filename}")
        filename = f"{'train' if self.training else 'test'}_y_{fn_suffix}.npy"
        np.save(self.reader.db_dir_base / filename, y)
        print(f"y saved to {filename}")

    def _check_nan(self) -> NoReturn:
        """
        during training, sometimes nan values are encountered,
        which ruins the whole training process

        """
        for idx, (values, labels) in enumerate(self):
            if np.isnan(values).any():
                print(f"values of {self.records[idx]} have nan values")
            if np.isnan(labels).any():
                print(f"labels of {self.records[idx]} have nan values")

    def extra_repr_keys(self) -> List[str]:
        return [
            "training",
            "tranches",
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
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        rec = self.records[index]
        values = self.reader.load_resampled_data(
            rec,
            leads=self.config.leads,
            # leads=Standard12Leads,
            data_format=self.config.data_format,
            siglen=None,
        )
        for idx in range(values.shape[0]):
            values[idx] = remove_spikes_naive(values[idx])
        if self.ppm:
            values, _ = self.ppm(values, self.config.fs)
        values = ensure_siglen(
            values,
            siglen=self.config.input_len,
            fmt=self.config.data_format,
            tolerance=self.config.sig_slice_tol,
        ).astype(self.dtype)
        if values.ndim == 2:
            values = values[np.newaxis, ...]

        labels = self.reader.get_labels(rec, scored_only=True, fmt="a", normalize=True)
        labels = (
            np.isin(self.config.all_classes, labels)
            .astype(self.dtype)[np.newaxis, ...]
            .repeat(values.shape[0], axis=0)
        )

        return values, labels

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]


def _load_record(reader: CR, rec: str, config: CFG) -> Tuple[np.ndarray, np.ndarray]:
    """finished, NOT checked,

    load a record from the database using data reader

    Parameters
    ----------
    reader: CR,
        the data reader
    rec: str,
        the record to load
    config: dict,
        the configuration for loading record

    Returns
    -------
    values: np.ndarray,
        the values of the record
    labels: np.ndarray,
        the labels of the record

    """
    values = reader.load_resampled_data(
        rec, leads=config.leads, data_format="channel_first", siglen=None
    )
    values = ensure_siglen(values, siglen=config.input_len, fmt="channel_first")

    labels = reader.get_labels(rec, scored_only=True, fmt="a", normalize=True)
    labels = np.isin(config.all_classes, labels).astype(int)

    if config.data_format.lower() in ["channel_last", "lead_last"]:
        values = values.T

    return values, labels
