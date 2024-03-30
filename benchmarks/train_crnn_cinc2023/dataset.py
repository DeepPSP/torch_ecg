"""
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from cfg import BaseCfg, TrainCfg
from data_reader import CINC2023Reader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm

from torch_ecg._preprocessors import PreprocManager
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin, list_sum
from torch_ecg.utils.utils_data import stratified_train_test_split

__all__ = [
    "CinC2023Dataset",
]


class CinC2023Dataset(Dataset, ReprMixin):
    """Dataset for the CinC2023 Challenge.

    Parameters
    ----------
    config : CFG
        configuration for the dataset
    task : str
        task to be performed using the dataset
    training : bool, default True
        whether the dataset is for training or validation
    lazy : bool, default True
        whether to load all data into memory at initialization
    reader_kwargs : dict, optional
        keyword arguments for the data reader class

    """

    __name__ = "CinC2023Dataset"

    def __init__(
        self,
        config: CFG,
        task: str = "classification",
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(config))
        # self.task = task.lower()  # task will be set in self.__set_task
        self.training = training
        self.lazy = lazy

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        # updates reader_kwargs with the config
        for kw in ["fs", "hour_limit", "working_dir"]:
            if kw not in reader_kwargs and hasattr(self.config, kw):
                reader_kwargs[kw] = getattr(self.config, kw)

        self.reader = CINC2023Reader(db_dir=self.config.db_dir, **reader_kwargs)

        # let the data reader (re-)load the metadata dataframes
        # in which case would be read from the disk via `pd.read_csv`
        # and the string values parsed from the txt files
        # are automatically converted to the correct data types
        # e.g. "50" -> 50 or 50.0 depending on whether the column has nan values
        # and "True" -> True or "False" -> False, "nan" -> np.nan, etc.
        self.reader._ls_rec()

        ############################################################################
        # workaround for training data selection
        # part 1: select recordings from the unofficial phase
        #         whose signal quality index (SQI) is present
        # TODO: remove this workaround when the SQI computation is implemented
        ############################################################################

        self.reader._df_records = self.reader._df_records[
            self.reader._df_records.index.isin(
                self.reader._df_unofficial_phase_metadata[
                    ~self.reader._df_unofficial_phase_metadata["record"].isna()
                ].record.values
            )
        ]
        # remove those recordings whose [start_sec, end_sec]
        # does not cover those of the unofficial phase
        df_tmp = self.reader._df_unofficial_phase_metadata.copy()
        df_tmp.set_index("record", inplace=True)
        df_tmp = df_tmp.loc[self.reader._df_records.index]
        condition = (self.reader._df_records["start_sec"] <= df_tmp["start_sec"]) & (
            self.reader._df_records["end_sec"] >= df_tmp["end_sec"]
        )
        self.reader._df_records = self.reader._df_records[condition]

        self.reader._all_records = self.reader._df_records.index.tolist()
        self.reader._df_subjects[self.reader._df_subjects.index.isin(self.reader._df_records.subject)]
        self.reader._all_subjects = self.reader._df_subjects.index.tolist()
        self.reader._subject_records = {
            sbj: self.reader._df_records.loc[self.reader._df_records["subject"] == sbj].index.tolist()
            for sbj in self.reader._all_subjects
        }

        ############################################################################
        # end of workaround
        ############################################################################

        self.subjects = self._train_test_split()
        self.records = list_sum([self.reader.subject_records[sbj] for sbj in self.subjects])
        if self.training:
            DEFAULTS.RNG.shuffle(self.records)

        ############################################################################
        # workaround for training data selection
        # part 2: find the intervals with SQI computed for each recording
        # TODO: remove this workaround when the SQI computation is implemented
        ############################################################################

        self.start_indices = []
        self.end_indices = []
        # the start and end indices will eventually be passed to
        # :meth:`wfdb.rdrecord` to read the data from the disk
        with tqdm(
            self.records,
            desc="Finding intervals with SQI computed",
            unit="record",
        ) as pbar:
            for rec in pbar:
                official_phase_row = self.reader._df_records.loc[rec]
                unofficial_phase_row = self.reader._df_unofficial_phase_metadata[
                    self.reader._df_unofficial_phase_metadata.record == rec
                ].iloc[0]
                rec_fs = official_phase_row.fs
                official_start_sec = official_phase_row.start_sec
                rec_start_sec = unofficial_phase_row.start_sec
                rec_end_sec = unofficial_phase_row.end_sec
                self.start_indices.append(int(rec_fs * (rec_start_sec - official_start_sec)))
                self.end_indices.append(int(rec_fs * (rec_end_sec - official_start_sec)))

        ############################################################################
        # end of workaround
        ############################################################################

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.ppm = None
        self.__set_task(task, lazy)

    def __len__(self) -> int:
        if self.cache is None:
            # self._load_all_data()
            return len(self.fdr)
        return self.cache["waveforms"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        if self.cache is None:
            # self._load_all_data()
            return self.fdr[index]
        return {k: v[index] for k, v in self.cache.items()}

    def __set_task(self, task: str, lazy: bool) -> None:
        """Set the task and load the data."""
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if hasattr(self, "task") and self.task == task.lower() and self.cache is not None and len(self.cache["waveforms"]) > 0:
            return
        self.task = task.lower()

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config[self.task]))
        self.ppm = PreprocManager.from_config(ppm_config)

        self.siglen = self.config[self.task].siglen
        self.classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        if self.task in ["classification"]:
            self.fdr = FastDataReader(
                self.reader,
                self.records,
                self.start_indices,
                self.end_indices,
                self.config,
                self.task,
                self.ppm,
            )
        # elif self.task in ["multi_task"]:
        #     self.fdr = MutiTaskFastDataReader(
        #         self.reader, self.records, self.config, self.task, self.ppm
        #     )
        else:  # TODO: implement contrastive learning task
            raise ValueError("Illegal task")

        if self.lazy:
            return

        tmp_cache = []
        with tqdm(range(len(self.fdr)), desc="Loading data", unit="record") as pbar:
            for idx in pbar:
                tmp_cache.append(self.fdr[idx])
        keys = tmp_cache[0].keys()
        self.__cache = {
            k: np.concatenate([v[k] if v[k].shape == (1,) else v[k][np.newaxis, ...] for v in tmp_cache]) for k in keys
        }
        # for k in keys:
        #     if self.__cache[k].ndim == 1:
        #         self.__cache[k] = self.__cache[k]
        # release memory
        del tmp_cache

    def _load_all_data(self) -> None:
        """Load all data into memory."""
        self.__set_task(self.task, lazy=False)

    def _train_test_split(self, train_ratio: float = 0.8, force_recompute: bool = False) -> List[str]:
        """Train-test split the subjects."""
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        # NOTE: for CinC2023, the data folder (db_dir) is read-only
        # the workaround is writing to the model folder
        # which is set to be the working directory (working_dir)
        writable = True
        if os.access(self.reader.db_dir, os.W_OK):
            train_file = self.reader.db_dir / f"train_ratio_{_train_ratio}.json"
            test_file = self.reader.db_dir / f"test_ratio_{_test_ratio}.json"
        elif os.access(self.reader.working_dir, os.W_OK):
            train_file = self.reader.working_dir / f"train_ratio_{_train_ratio}.json"
            test_file = self.reader.working_dir / f"test_ratio_{_test_ratio}.json"
        else:
            train_file = None
            test_file = None
            writable = False

        (BaseCfg.project_dir / "utils").mkdir(exist_ok=True)
        aux_train_file = BaseCfg.project_dir / "utils" / f"train_ratio_{_train_ratio}.json"
        aux_test_file = BaseCfg.project_dir / "utils" / f"test_ratio_{_test_ratio}.json"

        if not force_recompute:
            if writable and train_file.exists() and test_file.exists():
                if self.training:
                    return json.loads(train_file.read_text())
                else:
                    return json.loads(test_file.read_text())
            elif aux_train_file.exists() and aux_test_file.exists():
                # TODO: remove this workaround after stratified_train_test_split is enhanced
                # take the intersections of the two splits with self.reader.subjects
                train_set = list(set(json.loads(aux_train_file.read_text())).intersection(self.reader.all_subjects))
                test_set = list(set(json.loads(aux_test_file.read_text())).intersection(self.reader.all_subjects))
                # and write them to the train_file and test_file if writable
                if writable:
                    train_file.write_text(json.dumps(train_set, ensure_ascii=False))
                    test_file.write_text(json.dumps(test_set, ensure_ascii=False))
                if self.training:
                    return train_set
                else:
                    return test_set

        # aux files are only used for recording the split, not for actual training
        # if not force_recompute and aux_train_file.exists() and aux_test_file.exists():
        #     if self.training:
        #         return json.loads(aux_train_file.read_text())
        #     else:
        #         return json.loads(aux_test_file.read_text())

        df = self.reader._df_subjects.copy()
        df.loc[:, "Age"] = df["Age"].fillna(df["Age"].mean()).astype(int)  # only one nan
        # to age group
        df.loc[:, "Age"] = df["Age"].apply(lambda x: str(20 * (x // 20)))
        for col in ["OHCA", "Shockable Rhythm"]:
            df.loc[:, col] = df[col].apply(lambda x: 1 if x is True else 0 if x is False else x)
            df.loc[:, col] = df[col].fillna(-1).astype(int)
            df.loc[:, col] = df[col].astype(int).astype(str)

        df_train, df_test = stratified_train_test_split(
            df,
            [
                "Age",
                "Sex",
                "OHCA",
                "Shockable Rhythm",
                "CPC",
            ],
            test_ratio=1 - train_ratio,
            reset_index=False,
        )

        train_set = df_train.index.tolist()
        test_set = df_test.index.tolist()

        if (writable and force_recompute) or not train_file.exists() or not test_file.exists():
            train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        if force_recompute or not aux_train_file.exists() or not aux_test_file.exists():
            aux_train_file.write_text(json.dumps(train_set, ensure_ascii=False))
            aux_test_file.write_text(json.dumps(test_set, ensure_ascii=False))

        DEFAULTS.RNG.shuffle(train_set)
        DEFAULTS.RNG.shuffle(test_set)

        if self.training:
            return train_set
        else:
            return test_set

    def empty_cache(self) -> None:
        """release memory"""
        self.__cache.clear()
        self.__cache = None

    def shuffle_records(self) -> None:
        """Shuffle the records."""
        # NOTE that self.start_indices and self.end_indices
        # should be shuffled along with self.records
        indices = np.arange(len(self.records))
        DEFAULTS.RNG.shuffle(indices)
        self.records = [self.records[i] for i in indices]
        self.start_indices = [self.start_indices[i] for i in indices]
        self.end_indices = [self.end_indices[i] for i in indices]

    @property
    def cache(self) -> Dict[str, np.ndarray]:
        return self.__cache

    def extra_repr_keys(self) -> List[str]:
        return ["task", "training"]


class FastDataReader(ReprMixin, Dataset):
    def __init__(
        self,
        reader: CINC2023Reader,
        records: Sequence[str],
        start_indices: Sequence[int],
        end_indices: Sequence[int],
        config: CFG,
        task: str,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        self.reader = reader
        self.records = records
        self.start_indices = start_indices
        self.end_indices = end_indices
        self.config = config
        self.task = task
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.aux_target = "outcome" if self.config[self.task].output_target == "cpc" else "cpc"
        self.aux_classes = BaseCfg.outcome if self.config[self.task].output_target == "cpc" else BaseCfg.cpc
        self.hospitals = [self.reader._df_subjects.loc[self.reader._df_records.loc[r, "subject"], "Hospital"] for r in records]
        self.hospitals = [self.config.hospitals.index(h) for h in self.hospitals]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        rec = self.records[index]
        sampfrom = DEFAULTS.RNG_randint(0, 300 * 100 - self.config[self.task].input_len)
        sampto = sampfrom + self.config[self.task].input_len
        # waveforms = self.reader.load_data(
        #     rec,
        #     data_format=self.config[self.task].data_format,
        #     sampfrom=sampfrom,
        #     sampto=sampto,
        # )[np.newaxis, ...]
        waveforms = self.reader.load_bipolar_data(
            rec,
            sampfrom=self.start_indices[index],
            sampto=self.end_indices[index],
            data_format=self.config[self.task].data_format,
            fs=self.config[self.task].fs,
        )
        # usually the data_format is "channel_first",
        # we do not distinguish the data_format here for acceleration
        waveforms = waveforms[..., sampfrom:sampto]
        waveforms = waveforms[np.newaxis, ...]
        if self.ppm:
            waveforms, _ = self.ppm(waveforms, self.reader.fs)
        ann = self.reader.load_ann(rec)
        label = ann[self.config[self.task].output_target]
        aux_label = ann[self.aux_target]
        if self.config[self.task].loss != "CrossEntropyLoss":
            label = np.isin(self.config[self.task].classes, label).astype(self.dtype)
        else:
            label = np.array([self.config[self.task].classes.index(label)])
        out_tensors = {
            "waveforms": waveforms.squeeze(0).astype(self.dtype),
            self.config[self.task].output_target: label.astype(self.dtype),
            self.aux_target: np.array([self.aux_classes.index(aux_label)]).astype(int),  # categorical
            "hospitals": np.array([self.hospitals[index]]).astype("uint8"),
        }
        return out_tensors

    def extra_repr_keys(self) -> List[str]:
        return [
            "reader",
            "ppm",
        ]


if __name__ == "__main__":
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Process CINC2023 data for training")
    parser.add_argument(
        "operations",
        nargs=argparse.ONE_OR_MORE,
        type=str,
        choices=["move_files", "move-files"],
        help="operations to perform",
    )
    parser.add_argument(
        "-d",
        "--db-dir",
        type=str,
        help="The directory to (store) the database.",
        dest="db_dir",
    )
    parser.add_argument(
        "--move-dst",
        type=str,
        default=None,
        help="The destination directory to move the files to.",
        dest="move_dst",
    )

    args = parser.parse_args()

    operations = [ops.replace("-", "_") for ops in args.operations]
    if "move_files" in operations:
        assert args.move_dst is not None, "move_dst must be specified"
        move_dst = Path(args.move_dst).expanduser().resolve()

    db_dir = Path(args.db_dir) if args.db_dir is not None else None
    train_cfg = deepcopy(TrainCfg)
    train_cfg.db_dir = db_dir
    ds = CinC2023Dataset(
        train_cfg,
        lazy=True,
    )

    if "move_files" in operations:
        with tqdm(ds.reader.all_records) as pbar:
            for rec in pbar:
                sid = ds.reader.get_subject_id(rec)
                dst = move_dst / sid
                dst.mkdir(exist_ok=True, parents=True)
                metadata_file = ds.reader.get_absolute_path(sid, extension=ds.reader.ann_ext)
                if not (dst / metadata_file.name).exists():
                    shutil.copy(metadata_file, dst)
                sig_file = ds.reader.get_absolute_path(rec, extension=ds.reader.data_ext)
                if not (dst / sig_file.name).exists():
                    shutil.copy(sig_file, dst)
                header_file = ds.reader.get_absolute_path(rec, extension=ds.reader.header_ext)
                if not (dst / header_file.name).exists():
                    shutil.copy(header_file, dst)
