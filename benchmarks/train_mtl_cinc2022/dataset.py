"""
"""

import json
from random import shuffle, sample
from copy import deepcopy
from typing import Optional, List, Sequence, NoReturn, Dict

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin, list_sum
from torch_ecg.utils.utils_data import ensure_siglen, stratified_train_test_split
from torch_ecg._preprocessors import PreprocManager

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from cfg import BaseCfg, TrainCfg, ModelCfg  # noqa: F401
from inputs import (  # noqa: F401
    InputConfig,
    WaveformInput,
    SpectrogramInput,
    MelSpectrogramInput,
    MFCCInput,
    SpectralInput,
)  # noqa: F401
from data_reader import PCGDataBase, CINC2022Reader


__all__ = [
    "CinC2022Dataset",
]


class CinC2022Dataset(Dataset, ReprMixin):
    """ """

    __name__ = "CinC2022Dataset"

    def __init__(
        self, config: CFG, task: str, training: bool = True, lazy: bool = True
    ) -> NoReturn:
        """ """
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()  # task will be set in self.__set_task
        self.training = training
        self.lazy = lazy

        self.reader = CINC2022Reader(
            self.config.db_dir,
            ignore_unannotated=self.config.get("ignore_unannotated", True),
        )

        self.subjects = self._train_test_split()
        df = self.reader.df_stats[
            self.reader.df_stats["Patient ID"].isin(self.subjects)
        ]
        self.records = list_sum(
            [self.reader.subject_records[row["Patient ID"]] for _, row in df.iterrows()]
        )
        if self.config.get("entry_test_flag", False):
            self.records = sample(self.records, int(len(self.records) * 0.2))
        if self.training:
            shuffle(self.records)

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config.classification))
        seg_ppm_config = CFG(random=False)
        seg_ppm_config.update(deepcopy(self.config.segmentation))
        self.ppm = PreprocManager.from_config(ppm_config)
        self.seg_ppm = PreprocManager.from_config(seg_ppm_config)

        self.__cache = None
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        """ """
        if self.cache is None:
            self._load_all_data()
        return self.cache["waveforms"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        if self.cache is None:
            self._load_all_data()
        return {k: v[index] for k, v in self.cache.items()}

    def __set_task(self, task: str, lazy: bool) -> NoReturn:
        """ """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if (
            hasattr(self, "task")
            and self.task == task.lower()
            and self.cache is not None
            and len(self.cache["waveforms"]) > 0
        ):
            return
        self.task = task.lower()
        self.siglen = int(self.config[self.task].fs * self.config[self.task].siglen)
        self.classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        if self.task in ["classification"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        elif self.task in ["segmentation"]:
            self.fdr = FastDataReader(
                self.reader, self.records, self.config, self.task, self.seg_ppm
            )
        elif self.task in ["multi_task"]:
            self.fdr = MutiTaskFastDataReader(
                self.reader, self.records, self.config, self.task, self.ppm
            )
        else:
            raise ValueError("Illegal task")

        if self.lazy:
            return

        tmp_cache = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="records") as pbar:
            for idx in pbar:
                tmp_cache.append(self.fdr[idx])
        keys = tmp_cache[0].keys()
        self.__cache = {k: np.concatenate([v[k] for v in tmp_cache]) for k in keys}
        for k in keys:
            if self.__cache[k].ndim == 1:
                self.__cache[k] = self.__cache[k]

    def _load_all_data(self) -> NoReturn:
        """ """
        self.__set_task(self.task, lazy=False)

    def _train_test_split(
        self, train_ratio: float = 0.8, force_recompute: bool = False
    ) -> List[str]:
        """ """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
        test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        aux_train_file = (
            BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        )
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute and train_file.exists() and test_file.exists():
            if self.training:
                return json.loads(train_file.read_text())
            else:
                return json.loads(test_file.read_text())

        if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
            if self.training:
                return json.loads(aux_train_file.read_text())
            else:
                return json.loads(aux_test_file.read_text())

        df_train, df_test = stratified_train_test_split(
            self.reader.df_stats,
            [
                "Murmur",
                "Age",
                "Sex",
                "Pregnancy status",
                "Outcome",
            ],
            test_ratio=1 - train_ratio,
        )

        train_set = df_train["Patient ID"].tolist()
        test_set = df_test["Patient ID"].tolist()

        train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
        test_file.write_text(json.dumps(test_set, ensure_ascii=False))
        aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        shuffle(train_set)
        shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    @property
    def cache(self) -> List[Dict[str, np.ndarray]]:
        return self.__cache

    def extra_repr_keys(self) -> List[str]:
        """ """
        return ["task", "training"]


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: PCGDataBase,
        records: Sequence[str],
        config: CFG,
        task: str,
        ppm: Optional[PreprocManager] = None,
    ) -> NoReturn:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        rec = self.records[index]
        waveforms = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        waveforms = ensure_siglen(
            waveforms,
            siglen=self.config[self.task].input_len,
            fmt=self.config[self.task].data_format,
            tolerance=self.config[self.task].sig_slice_tol,
        ).astype(self.dtype)
        if waveforms.ndim == 2:
            waveforms = waveforms[np.newaxis, ...]

        n_segments = waveforms.shape[0]

        if self.task in ["classification"]:
            label = self.reader.load_ann(rec)
            if self.config[self.task].loss != "CrossEntropyLoss":
                label = (
                    np.isin(self.config[self.task].classes, label)
                    .astype(self.dtype)[np.newaxis, ...]
                    .repeat(n_segments, axis=0)
                )
            else:
                label = np.array(
                    [
                        self.config[self.task].class_map[label]
                        for _ in range(n_segments)
                    ],
                    dtype=int,
                )
            out = {"waveforms": waveforms, "murmur": label}
            if self.config[self.task].outcomes is not None:
                outcome = self.reader.load_outcome(rec)
                if self.config[self.task].loss["outcome"] != "CrossEntropyLoss":
                    outcome = (
                        np.isin(self.config[self.task].outcomes, outcome)
                        .astype(self.dtype)[np.newaxis, ...]
                        .repeat(n_segments, axis=0)
                    )
                else:
                    outcome = np.array(
                        [
                            self.config[self.task].outcome_map[outcome]
                            for _ in range(n_segments)
                        ],
                        dtype=int,
                    )
                out["outcome"] = outcome
            return out

        elif self.task in ["segmentation"]:
            label = self.reader.load_segmentation(
                rec,
                seg_format="binary",
                ensure_same_len=True,
                fs=self.config[self.task].fs,
            )
            label = ensure_siglen(
                label,
                siglen=self.config[self.task].input_len,
                fmt="channel_last",
                tolerance=self.config[self.task].sig_slice_tol,
            ).astype(self.dtype)
            return {"waveforms": waveforms, "segmentation": label}
        else:
            raise ValueError(f"Illegal task: {self.task}")

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]


class MutiTaskFastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        reader: PCGDataBase,
        records: Sequence[str],
        config: CFG,
        task: str = "multi_task",
        ppm: Optional[PreprocManager] = None,
    ) -> NoReturn:
        """ """
        self.reader = reader
        self.records = records
        self.config = config
        self.task = task
        assert self.task == "multi_task"
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        """ """
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """ """
        rec = self.records[index]
        waveforms = self.reader.load_data(
            rec,
            data_format=self.config[self.task].data_format,
        )
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        waveforms = ensure_siglen(
            waveforms,
            siglen=self.config[self.task].input_len,
            fmt=self.config[self.task].data_format,
            tolerance=self.config[self.task].sig_slice_tol,
        ).astype(self.dtype)
        if waveforms.ndim == 2:
            waveforms = waveforms[np.newaxis, ...]

        n_segments = waveforms.shape[0]

        label = self.reader.load_ann(rec)
        if self.config[self.task].loss["murmur"] != "CrossEntropyLoss":
            label = (
                np.isin(self.config[self.task].classes, label)
                .astype(self.dtype)[np.newaxis, ...]
                .repeat(n_segments, axis=0)
            )
        else:
            label = np.array(
                [self.config[self.task].class_map[label] for _ in range(n_segments)],
                dtype=int,
            )
        out_tensors = {
            "waveforms": waveforms,
            "murmur": label,
        }

        if self.config[self.task].outcomes is not None:
            outcome = self.reader.load_outcome(rec)
            if self.config[self.task].loss["outcome"] != "CrossEntropyLoss":
                outcome = (
                    np.isin(self.config[self.task].outcomes, outcome)
                    .astype(self.dtype)[np.newaxis, ...]
                    .repeat(n_segments, axis=0)
                )
            else:
                outcome = np.array(
                    [
                        self.config[self.task].outcome_map[outcome]
                        for _ in range(n_segments)
                    ],
                    dtype=int,
                )
            out_tensors["outcome"] = outcome

        if self.config[self.task].states is not None:
            mask = self.reader.load_segmentation(
                rec,
                seg_format="binary",
                ensure_same_len=True,
                fs=self.config[self.task].fs,
            )
            mask = ensure_siglen(
                mask,
                siglen=self.config[self.task].input_len,
                fmt="channel_last",
                tolerance=self.config[self.task].sig_slice_tol,
            ).astype(self.dtype)
            out_tensors["segmentation"] = mask

        return out_tensors

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]
