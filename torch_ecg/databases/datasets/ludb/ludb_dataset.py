"""
"""

import json
import warnings
from copy import deepcopy
from random import randint, shuffle
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from ...._preprocessors import PreprocManager
from ....cfg import CFG
from ....databases import LUDB as LR
from ....utils.misc import ReprMixin


__all__ = [
    "LUDBDataset",
]


class LUDBDataset(ReprMixin, Dataset):
    """Data generator for feeding data into pytorch models
    using the :class:`~torch_ecg.databases.LUDB` database.

    Parameters
    ----------
    config : dict
        Configurations for the dataset, ref. `LUDBTrainCfg`.
    training : bool, default True
        If True, the training set will be loaded,
        otherwise the test (val) set will be loaded.
    lazy : bool, default True
        If True, the data will not be loaded immediately,
        instead, it will be loaded on demand.
    **reader_kwargs : dict, optional
        Keyword arguments for the database reader class.

    """

    __name__ = "LUDBDataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = False,
        **reader_kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = deepcopy(config)
        if reader_kwargs.pop("db_dir", None) is not None:
            warnings.warn(
                "`db_dir` is specified in both config and reader_kwargs", RuntimeWarning
            )
        self.reader = LR(db_dir=self.config.db_dir, **reader_kwargs)
        self.config.db_dir = self.reader.db_dir
        self.training = training
        self.dtype = self.config.np_dtype
        self.classes = self.config.classes
        self.n_classes = len(self.classes)
        self.siglen = self.config.input_len
        if self.config.leads is None:
            self.leads = self.reader.all_leads
        elif isinstance(self.config.leads, str):
            self.leads = [self.config.leads]
        else:
            self.leads = list(self.config.leads)
        self.lazy = lazy

        self.ppm = PreprocManager.from_config(self.config)
        self.records = self._train_test_split(self.config.train_ratio)
        self.fdr = _FastDataReader(self.reader, self.records, self.config)
        self.waveform_priority = ["N", "t", "p", "i"]

        self._signals = None
        self._labels = None
        if not self.lazy:
            self._load_all_data()

    def __len__(self) -> int:
        if self.config.use_single_lead:
            return len(self.leads) * len(self.records)
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.use_single_lead:
            rec_idx, lead_idx = divmod(index, len(self.leads))
        else:
            rec_idx, lead_idx = index, None
        rec = self.records[rec_idx]
        if not self.lazy:
            signals = self._signals[rec_idx]
            labels = self._labels[rec_idx]
        else:
            signals, labels = self.fdr[rec_idx]
        if lead_idx is not None:
            signals = signals[[lead_idx], ...]
            labels = labels[lead_idx, ...]
        else:
            # merge labels in all leads to one
            # TODO: map via self.waveform_priority
            labels = np.max(labels, axis=0)
        sampfrom = randint(
            self.config.start_from, signals.shape[1] - self.config.end_at - self.siglen
        )
        sampto = sampfrom + self.siglen
        signals = signals[..., sampfrom:sampto]
        labels = labels[sampfrom:sampto, ...]

        return signals, labels

    def _load_all_data(self) -> None:
        """Load all data into memory."""
        self._signals, self._labels = [], []

        with tqdm(
            self.fdr, total=len(self.fdr), dynamic_ncols=True, mininterval=1.0
        ) as bar:
            for signals, labels in bar:
                self._signals.append(signals)
                self._labels.append(labels)

        self._signals = np.array(self._signals)
        self._labels = np.array(self._labels)

    @property
    def signals(self) -> np.ndarray:
        """Cached signals, only available when `lazy=False`
        or preloading is performed manually.
        """
        return self._signals

    @property
    def labels(self) -> np.ndarray:
        """Cached labels, only available when `lazy=False`
        or preloading is performed manually.
        """
        return self._labels

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """Perform train-test split.

        Parameters
        ----------
        train_ratio : float, default 0.8
            ratio of the train set in the whole dataset.
        force_recompute : bool, default False
            If True, the train-test split will be recomputed,
            regardless of the existing ones stored in json files.

        Returns
        -------
        List[str]
            The list of the records split for training or validation.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0
        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"

        if self.reader._subsample is not None:
            force_recompute = True

        if force_recompute or not all([train_file.is_file(), test_file.is_file()]):
            all_records = deepcopy(self.reader.all_records)
            shuffle(all_records)
            split_idx = int(_train_ratio * len(all_records) / 100)
            train_set = all_records[:split_idx]
            test_set = all_records[split_idx:]
            if self.reader._subsample is None:
                train_file.write_text(json.dumps(train_set, ensure_ascii=False))
                test_file.write_text(json.dumps(test_set, ensure_ascii=False))
        else:
            train_set = json.loads(train_file.read_text())
            test_set = json.loads(test_file.read_text())
        if self.training:
            records = train_set
        else:
            records = test_set
        if self.config.over_sampling > 1:
            records = records * self.config.over_sampling
            shuffle(records)
        return records

    def extra_repr_keys(self) -> List[str]:
        return [
            "training",
            "reader",
        ]


class _FastDataReader(ReprMixin, Dataset):
    """Fast data reader.

    Parameters
    ----------
    reader : CR
        The reader to read the data.
    records : Sequence[str]
        The list of records to read.
    config : CFG
        The configuration.
    ppm : PreprocManager, optional
        The preprocessor manager.

    """

    def __init__(
        self,
        reader: LR,
        records: Sequence[str],
        config: CFG,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        self.reader = reader
        self.records = records
        self.config = config
        self.ppm = ppm
        self.dtype = self.config.np_dtype

        if self.config.leads is None:
            self.leads = self.reader.all_leads
        elif isinstance(self.config.leads, str):
            self.leads = [self.config.leads]
        else:
            self.leads = list(self.config.leads)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        rec = self.records[index]
        signals = self.reader.load_data(
            rec,
            data_format="channel_first",
            units="mV",
        ).astype(self.dtype)
        if self.ppm:
            signals, _ = self.ppm(signals, self.config.fs)
        masks = self.reader.load_masks(
            rec,
            leads=self.leads,
            mask_format="channel_first",
            class_map=self.config.class_map,
        ).astype(self.dtype)
        if self.config.loss == "CrossEntropyLoss":
            return signals, masks
        # expand masks to have n vectors, with n = n_classes
        labels = np.ones(
            (*masks.shape, len(self.config.mask_class_map)), dtype=self.dtype
        )
        for i in range(len(self.leads)):
            for key, val in self.config.mask_class_map.items():
                labels[i, ..., val] = (
                    masks[i, ...] == self.config.class_map[key]
                ).astype(self.dtype)
        return signals, labels

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]
