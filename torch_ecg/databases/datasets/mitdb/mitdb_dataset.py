"""
"""

import json
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal as SS

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import torch
from scipy.io import loadmat, savemat
from torch.utils.data.dataset import Dataset

from ...._preprocessors import PreprocManager
from ....databases import MITDB as DR
from ....utils.misc import ReprMixin, get_record_list_recursive3, list_sum
from ....utils.utils_data import (
    ensure_siglen,
    cls_to_bin,
    generate_weight_mask,
    mask_to_intervals,
)
from ....utils.utils_signal import remove_spikes_naive
from ....cfg import CFG, DEFAULTS
from .mitdb_cfg import MITDBTrainCfg


__all__ = [
    "MITDBDataset",
]


class MITDBDataset(ReprMixin, Dataset):
    """ """

    __DEBUG__ = False
    __name__ = "MITDBDataset"

    def __init__(
        self, config: CFG, task: str, training: bool = True, lazy: bool = True
    ) -> NoReturn:
        """

        Parameters
        ----------
        config: dict,
            configurations for the Dataset,
            ref. `MITDBDataset`
            a simple example is:
            >>> config = deepcopy(MITDBDataset)
            >>> config.db_dir = "some/path/to/db"
            >>> dataset = MITDBDataset(config, training=True, lazy=False)
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        lazy: bool, default False,
            if True, the data will not be loaded immediately,

        """
        super().__init__()
        self.config = deepcopy(config)
        assert self.config.db_dir is not None, "db_dir must be specified"
        self.config.db_dir = Path(self.config.db_dir)
        self.reader = DR(db_dir=self.config.db_dir)
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.allowed_preproc = list(
            set(
                [
                    "bandpass",
                    "baseline_remove",
                    "normalize",
                ]
            ).intersection(set(self.config.keys()))
        )

        self.training = training

        self.lazy = lazy

        ppm_config = CFG(random=False)
        ppm_config.update(deepcopy(self.config))
        # ppm_config.pop("normalize")
        seg_ppm_config = CFG(random=False)
        seg_ppm_config.update(deepcopy(self.config))
        seg_ppm_config.pop("bandpass")
        self.ppm = PreprocManager.from_config(ppm_config)
        self.ppm.rearrange(["bandpass", "baseline_remove", "normalize"])
        self.seg_ppm = PreprocManager.from_config(seg_ppm_config)

        # create directories if needed
        # segments_dir for sliced segments of fixed length
        self.segments_base_dir = self.config.db_dir / "segments"
        self.segments_base_dir.mkdir(parents=True, exist_ok=True)
        self.segment_name_pattern = "S_[\\d]{3}_[\\d]{7}"
        self.segment_ext = "mat"
        # rr_dir for sequence of rr intervals of fix length
        self.rr_seq_base_dir = self.config.db_dir / "rr_seq"
        self.rr_seq_base_dir.mkdir(parents=True, exist_ok=True)
        self.rr_seq_name_pattern = "R_[\\d]{3}_[\\d]{7}"
        self.rr_seq_ext = "mat"

        self._all_data = None
        self._all_labels = None
        self._all_masks = None
        self.__set_task(task, lazy=self.lazy)

    def _load_all_data(self) -> NoReturn:
        """ """
        self.__set_task(self.task, lazy=False)

    def __set_task(self, task: str, lazy: bool = True) -> NoReturn:
        """

        Parameters
        ----------
        task: str,
            name of the task, can be one of `MITDBTrainCfg.tasks`
        lazy: bool, default True,
            if True, the data will not be loaded immediately,

        """
        assert task.lower() in MITDBTrainCfg.tasks, f"illegal task \042{task}\042"
        if (
            hasattr(self, "task")
            and self.task == task.lower()
            and self._all_data is not None
            and len(self._all_data) > 0
        ):
            return
        self.task = task.lower()
        self.all_classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)
        self.lazy = lazy

        self.seglen = self.config[task].input_len  # alias, for simplicity
        split_res = self._train_test_split(self.task)
        if self.training:
            self.records = split_res.train
        else:
            self.records = split_res.test

        if self.task in ["beat_classification"]:
            # finished, tested
            self._all_data, self._all_labels = [], []
            with tqdm(
                range(len(self.records)), desc="Loading data", unit="record"
            ) as pbar:
                for idx in pbar:
                    data = self.reader.load_data(self.records[idx])
                    data, _ = self.ppm(data, self.config.fs)
                    for beat in self.reader.load_beat_ann(self.records[idx]):
                        if beat.symbol not in self.config[self.task].classes:
                            continue
                        beat_data = ensure_siglen(
                            data[
                                :,
                                max(0, beat.index - self.config[self.task].winL) : min(
                                    data.shape[-1],
                                    beat.index + self.config[self.task].winR,
                                ),
                            ],
                            self.config[self.task].input_len,
                        )
                        self._all_data.append(beat_data)
                        self._all_labels.append(
                            self.config[self.task].class_map[beat.symbol]
                        )
                self._all_data = np.array(self._all_data)
                self._all_labels = np.array(self._all_labels)
                if self.config[self.task].loss not in ["CrossEntropyLoss"]:
                    self._all_labels = cls_to_bin(
                        self._all_labels, len(self.config[self.task].classes)
                    )
        elif self.task in [
            "qrs_detection",
            "rhythm_segmentation",
            "af_event",
        ]:
            # for qrs detection
            self.segments_dirs = CFG()
            self.__all_segments = CFG()
            self.segments_json = self.segments_base_dir / "segments.json"
            self._ls_segments()
            self.segments = list_sum([self.__all_segments[rec] for rec in self.records])
            if self.__DEBUG__:
                self.segments = DEFAULTS.RNG_sample(
                    self.segments, int(len(self.segments) * 0.01)
                ).tolist()
            if self.training:
                DEFAULTS.RNG.shuffle(self.segments)
            # preload data
            self.fdr = FastDataReader(
                self.config,
                self.task,
                self.seg_ppm,
                self.segments_dirs,
                self.segments,
                self.segment_ext,
            )
            if self.lazy:
                return
            self._all_data, self._all_labels, self._all_masks = [], [], []
            if len(self.fdr) == 0:
                warnings.warn(
                    f"No data found for task {self.task}, slice the data first."
                )
            with tqdm(range(len(self.fdr)), desc="Loading data", unit="record") as pbar:
                for idx in pbar:
                    d, l, m = self.fdr[idx]
                    self._all_data.append(d)
                    self._all_labels.append(l)
                    self._all_masks.append(m)
            self._all_data = np.array(self._all_data).astype(self.dtype)
            self._all_labels = np.array(self._all_labels).astype(self.dtype)
            if self.task == "qrs_detection":
                self._all_masks = None
            else:
                self._all_masks = np.array(self._all_masks).astype(self.dtype)
        elif self.task in [
            "rr_lstm",
        ]:
            self.rr_seq_dirs = CFG()
            self.__all_rr_seq = CFG()
            self.rr_seq_json = self.rr_seq_base_dir / "rr_seq.json"
            self._ls_rr_seq()
            self.rr_seq = list_sum([self.__all_rr_seq[rec] for rec in self.records])
            if self.__DEBUG__:
                self.rr_seq = DEFAULTS.RNG_sample(
                    self.rr_seq, int(len(self.rr_seq) * 0.01)
                ).tolist()
            if self.training:
                DEFAULTS.RNG.shuffle(self.rr_seq)
            # preload data
            self.fdr = FastDataReader(
                self.config,
                self.task,
                self.seg_ppm,
                self.rr_seq_dirs,
                self.rr_seq,
                self.rr_seq_ext,
            )
            if self.lazy:
                return
            self._all_data, self._all_labels, self._all_masks = [], [], []
            with tqdm(range(len(self.fdr)), desc="Loading data", unit="record") as pbar:
                for idx in pbar:
                    d, l, m = self.fdr[idx]
                    self._all_data.append(d)
                    self._all_labels.append(l)
                    self._all_masks.append(m)
            self._all_data = np.array(self._all_data).astype(self.dtype)
            self._all_labels = np.array(self._all_labels).astype(self.dtype)
            self._all_masks = np.array(self._all_masks).astype(self.dtype)
        else:
            raise NotImplementedError(
                f"data generator for task \042{self.task}\042 not implemented"
            )

    def reset_task(self, task: str, lazy: bool = True) -> NoReturn:
        """ """
        self.__set_task(task, lazy)

    def _ls_segments(self) -> NoReturn:
        """list all the segments"""
        for item in ["data", "ann"]:
            self.segments_dirs[item] = CFG()
            for rec in self.reader:
                self.segments_dirs[item][rec] = self.segments_base_dir / item / rec
                self.segments_dirs[item][rec].mkdir(parents=True, exist_ok=True)
        if self.segments_json.is_file():
            self.__all_segments = json.loads(self.segments_json.read_text())
            # return
        print(
            f"please allow the reader a few minutes to collect the segments from {self.segments_base_dir}..."
        )
        seg_filename_pattern = f"{self.segment_name_pattern}.{self.segment_ext}"
        self.__all_segments = CFG(
            {
                rec: get_record_list_recursive3(
                    str(self.segments_dirs.data[rec]), seg_filename_pattern
                )
                for rec in self.reader
            }
        )
        if all([len(self.__all_segments[rec]) > 0 for rec in self.reader]):
            self.segments_json.write_text(
                json.dumps(self.__all_segments, ensure_ascii=False)
            )

    def _ls_rr_seq(self) -> NoReturn:
        """list all the rr sequences"""
        for rec in self.reader:
            self.rr_seq_dirs[rec] = self.rr_seq_base_dir / rec
            self.rr_seq_dirs[rec].mkdir(parents=True, exist_ok=True)
        if self.rr_seq_json.is_file():
            self.__all_rr_seq = json.loads(self.rr_seq_json.read_text())
            # return
        print(
            f"please allow the reader a few minutes to collect the rr sequences from {self.rr_seq_base_dir}..."
        )
        rr_seq_filename_pattern = f"{self.rr_seq_name_pattern}.{self.rr_seq_ext}"
        self.__all_rr_seq = CFG(
            {
                rec: get_record_list_recursive3(
                    self.rr_seq_dirs[rec], rr_seq_filename_pattern
                )
                for rec in self.reader
            }
        )
        if all([len(self.__all_rr_seq[rec]) > 0 for rec in self.reader]):
            self.rr_seq_json.write_text(
                json.dumps(self.__all_rr_seq, ensure_ascii=False)
            )

    @property
    def all_segments(self) -> CFG:
        if self.task in [
            "qrs_detection",
            "rhythm_segmentation",
            "af_event",
        ]:
            return self.__all_segments
        else:
            return CFG()

    @property
    def all_rr_seq(self) -> CFG:
        if self.task.lower() in [
            "rr_lstm",
        ]:
            return self.__all_rr_seq
        else:
            return CFG()

    def __len__(self) -> int:
        return len(self.fdr)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        if self.task in ["beat_classification"]:
            return self._all_data[index], self._all_labels[index]
        if self.lazy:
            if self.task in ["qrs_detection"]:
                return self.fdr[index][:2]
            else:
                return self.fdr[index]
        else:
            if self.task in ["qrs_detection"]:
                return self._all_data[index], self._all_labels[index]
            else:
                return (
                    self._all_data[index],
                    self._all_labels[index],
                    self._all_masks[index],
                )

    def _get_seg_data_path(self, seg: str) -> Path:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"

        Returns
        -------
        fp: Path,
            path of the data file of the segment

        """
        rec = self._get_rec_name(seg)
        fp = self.segments_dirs.data[rec] / f"{seg}.{self.segment_ext}"
        return fp

    def _get_seg_ann_path(self, seg: str) -> Path:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"

        Returns
        -------
        fp: Path,
            path of the annotation file of the segment

        """
        rec = self._get_rec_name(seg)
        fp = self.segments_dirs.ann[rec] / f"{seg}.{self.segment_ext}"
        return fp

    def _load_seg_data(self, seg: str) -> np.ndarray:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"

        Returns
        -------
        seg_data: ndarray,
            data of the segment, of shape (2, `self.seglen`)

        """
        seg_data_fp = self._get_seg_data_path(seg)
        seg_data = loadmat(str(seg_data_fp))["ecg"]
        return seg_data

    def _load_seg_ann(self, seg: str) -> dict:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"

        Returns
        -------
        seg_ann: dict,
            annotations of the segment, including
            - rpeaks: indices of rpeaks of the segment
            - qrs_mask: mask of qrs complexes of the segment
            - rhythm_mask: mask of rhythms of the segment
            - interval: interval ([start_idx, end_idx]) in the original ECG record of the segment

        """
        seg_ann_fp = self._get_seg_ann_path(seg)
        seg_ann = {
            k: v.flatten()
            for k, v in loadmat(str(seg_ann_fp)).items()
            if not k.startswith("__")
        }
        return seg_ann

    def _load_seg_mask(
        self, seg: str, task: Optional[str] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"
        task: str, optional,
            if specified, overrides self.task,
            else if is "all", then all masks ("qrs_mask", "rhythm_mask", etc.) will be returned

        Returns
        -------
        seg_mask: np.ndarray or dict,
            mask(s) of the segment,
            of shape (self.seglen, self.n_classes)

        """
        seg_mask = {
            k: v.reshape((self.seglen, -1))
            for k, v in self._load_seg_ann(seg).items()
            if k
            in [
                "qrs_mask",
                "rhythm_mask",
            ]
        }
        _task = (task or self.task).lower()
        if _task == "all":
            return seg_mask
        if _task in [
            "qrs_detection",
        ]:
            seg_mask = seg_mask["qrs_mask"]
        elif _task in ["rhythm_segmentation", "af_event"]:
            seg_mask = seg_mask["rhythm_mask"]
        return seg_mask

    def _load_seg_seq_lab(self, seg: str, reduction: int) -> np.ndarray:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_100_0000193"
        reduction: int,
            reduction (granularity) of length of the model output,
            compared to the original signal length

        Returns
        -------
        seq_lab: np.ndarray,
            label of the sequence,
            of shape (self.seglen//reduction, self.n_classes)

        """
        seg_mask = self._load_seg_mask(seg)
        seg_len, n_classes = seg_mask.shape
        seq_lab = np.stack(
            arrays=[
                np.mean(
                    seg_mask[reduction * idx : reduction * (idx + 1)],
                    axis=0,
                    keepdims=True,
                ).astype(int)
                for idx in range(seg_len // reduction)
            ],
            axis=0,
        ).squeeze(axis=1)
        return seq_lab

    def _get_rr_seq_path(self, seq_name: str) -> Path:
        """

        Parameters
        ----------
        seq_name: str,
            name of the rr_seq, of pattern like "R_100_0000193"

        Returns
        -------
        fp: Path,
            path of the annotation file of the rr_seq

        """
        rec = self._get_rec_name(seq_name)
        fp = self.rr_seq_dirs[rec] / f"{seq_name}.{self.rr_seq_ext}"
        return fp

    def _load_rr_seq(self, seq_name: str) -> Dict[str, np.ndarray]:
        """

        Parameters
        ----------
        seq_name: str,
            name of the rr_seq, of pattern like "R_100_0000193"

        Returns
        -------
        rr_seq: dict,
            metadata of sequence of rr intervals, including
            - rr: the sequence of rr intervals, with units in seconds, of shape (self.seglen, 1)
            - label: label of the rr intervals, of shape (self.seglen, self.n_classes)
            - interval: interval of the current rr sequence in the whole rr sequence in the original record

        """
        rr_seq_path = self._get_rr_seq_path(seq_name)
        rr_seq = {
            k: v for k, v in loadmat(str(rr_seq_path)).items() if not k.startswith("__")
        }
        rr_seq["rr"] = rr_seq["rr"].reshape((self.seglen, 1))
        rr_seq["label"] = rr_seq["label"].reshape((self.seglen, self.n_classes))
        rr_seq["interval"] = rr_seq["interval"].flatten()
        return rr_seq

    def persistence(self, force_recompute: bool = False, verbose: int = 0) -> NoReturn:
        """

        make the dataset persistent w.r.t. the ratios in `self.config`

        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity

        """
        if verbose >= 1:
            print(" preprocessing data ".center("#", 110))
        self._preprocess_data(
            force_recompute=force_recompute,
            verbose=verbose,
        )
        if verbose >= 1:
            print("\n" + " slicing data into segments ".center("#", 110))
        self._slice_data(
            force_recompute=force_recompute,
            verbose=verbose,
        )
        if verbose >= 1:
            print("\n" + " generating rr sequences ".center("#", 110))
        self._slice_rr_seq(
            force_recompute=force_recompute,
            verbose=verbose,
        )

    def _get_rec_suffix(self, operations: List[str]) -> str:
        """

        Parameters
        ----------
        operations: list of str,
            names of operations to perform (or has performed),
            should be sublist of `self.allowed_preproc`

        Returns
        -------
        suffix: str,
            suffix of the filename of the preprocessed ecg signal

        """
        suffix = "-".join(sorted([item.lower() for item in operations]))
        return suffix

    def _slice_data(self, force_recompute: bool = False, verbose: int = 0) -> NoReturn:
        """

        slice all records into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`

        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity

        """
        self.__assert_task(
            [
                "qrs_detection",
                "rhythm_segmentation",
                "af_event",  # segmentation of AF events
            ]
        )
        if force_recompute:
            self._clear_cached_segments()
        with tqdm(
            enumerate(self.reader),
            total=len(self.reader),
            desc="Slicing data",
            unit="record",
        ) as pbar:
            for idx, rec in pbar:
                self._slice_one_record(
                    rec=rec,
                    force_recompute=False,
                    update_segments_json=False,
                    verbose=verbose,
                )
                # if verbose >= 1:
                #     print(f"{idx+1}/{len(self.reader)} records", end="\r")
        if force_recompute:
            self.segments_json.write_text(
                json.dump(self.__all_segments, ensure_ascii=False)
            )

    def _slice_one_record(
        self,
        rec: str,
        force_recompute: bool = False,
        update_segments_json: bool = False,
        verbose: int = 0,
    ) -> NoReturn:
        """

        slice one record into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`

        Parameters
        ----------
        rec: str,
            filename of the record
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        update_segments_json: bool, default False,
            if both `force_recompute` and `update_segments_json` are True,
            the file `self.segments_json` will be updated,
            useful when slicing not all records
        verbose: int, default 0,
            print verbosity

        """
        self.__assert_task(
            [
                "qrs_detection",
                "rhythm_segmentation",
                "af_event",  # segmentation of AF events
            ]
        )
        rec_segs = self.__all_segments[rec]
        if (not force_recompute) and len(rec_segs) > 0:
            return
        elif force_recompute:
            self._clear_cached_segments([rec])

        data, _ = self.ppm(self.reader.load_data(rec), self.config.fs)
        siglen = data.shape[1]
        rpeaks = self.reader.load_rpeak_indices(rec)
        rhythm_mask = self.reader.load_rhythm_ann(
            rec,
            rhythm_format="mask",
            rhythm_types=self.config.rhythm_segmentation.classes,
        )
        forward_len = self.seglen - self.config[self.task].overlap_len
        critical_forward_len = self.seglen - self.config[self.task].critical_overlap_len
        critical_forward_len = [critical_forward_len // 4, critical_forward_len]

        # find critical points
        critical_points = np.where(np.diff(rhythm_mask) != 0)[0]
        critical_points = [
            p
            for p in critical_points
            if critical_forward_len[1] <= p < siglen - critical_forward_len[1]
        ]

        segments = []

        # ordinary segments with constant forward_len
        print("Slicing ordinary segments with constant forward_len")
        with tqdm(
            range((siglen - self.seglen) // forward_len + 1),
            desc=f"Slicing segments for record {rec}",
            unit="segment",
        ) as pbar:
            for idx in pbar:
                start_idx = idx * forward_len
                new_seg = self.__generate_segment(
                    rec=rec,
                    data=data,
                    start_idx=start_idx,
                )
                segments.append(new_seg)
        # the tail segment
        new_seg = self.__generate_segment(
            rec=rec,
            data=data,
            end_idx=siglen,
        )
        segments.append(new_seg)

        if len(critical_points) == 0:
            # save segments
            self.__save_segments(rec, segments, update_segments_json)
            return

        # special segments around critical_points with random forward_len in critical_forward_len
        print(
            "Slicing special segments around critical_points with random forward_len in critical_forward_len"
        )
        with tqdm(
            critical_points, desc=f"Slicing segments for record {rec}", unit="segment"
        ) as pbar:
            for cp in pbar:
                start_idx = max(
                    0,
                    cp
                    - self.seglen
                    + DEFAULTS.RNG_randint(
                        critical_forward_len[0], critical_forward_len[1]
                    ),
                )
                while start_idx <= min(
                    cp - critical_forward_len[1], siglen - self.seglen
                ):
                    new_seg = self.__generate_segment(
                        rec=rec,
                        data=data,
                        start_idx=start_idx,
                    )
                    segments.append(new_seg)
                    start_idx += DEFAULTS.RNG_randint(
                        critical_forward_len[0], critical_forward_len[1]
                    )

        # save segments
        self.__save_segments(rec, segments, update_segments_json)

    def __generate_segment(
        self,
        rec: str,
        data: np.ndarray,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> CFG:
        """

        generate segment, with possible data augmentation

        Parameter
        ---------
        rec: str,
            filename of the record
        data: ndarray,
            the whole of (preprocessed) ECG record
        start_idx: int, optional,
            start index of the signal of `rec` for generating the segment
        end_idx: int, optional,
            end index of the signal of `rec` for generating the segment,
            if `start_idx` is set, `end_idx` is ignored,
            at least one of `start_idx` and `end_idx` should be set

        Returns
        -------
        new_seg: dict,
            segments (meta-)data, containing:
            - data: values of the segment, with units in mV
            - rpeaks: indices of rpeaks of the segment
            - qrs_mask: mask of qrs complexes of the segment
            - rhythm_mask: mask of rhythms of the segment
            - interval: interval ([start_idx, end_idx]) in the original ECG record of the segment

        """
        assert not all(
            [start_idx is None, end_idx is None]
        ), "at least one of `start_idx` and `end_idx` should be set"
        siglen = data.shape[1]
        # offline augmentations are done, including strech-or-compress, ...
        if self.config.stretch_compress != 0:
            sign = DEFAULTS.RNG_sample(self.config.stretch_compress_choices, 1)[0]
            if sign != 0:
                sc_ratio = self.config.stretch_compress
                sc_ratio = (
                    1 + (DEFAULTS.RNG.uniform(sc_ratio / 4, sc_ratio) * sign) / 100
                )
                sc_len = int(round(sc_ratio * self.seglen))
                if start_idx is not None:
                    end_idx = start_idx + sc_len
                else:
                    start_idx = end_idx - sc_len
                if end_idx > siglen:
                    end_idx = siglen
                    start_idx = max(0, end_idx - sc_len)
                    sc_ratio = (end_idx - start_idx) / self.seglen
                aug_seg = data[..., start_idx:end_idx]
                aug_seg = SS.resample(x=aug_seg, num=self.seglen, axis=1)
            else:
                if start_idx is not None:
                    end_idx = start_idx + self.seglen
                    if end_idx > siglen:
                        end_idx = siglen
                        start_idx = end_idx - self.seglen
                else:
                    start_idx = end_idx - self.seglen
                    if start_idx < 0:
                        start_idx = 0
                        end_idx = self.seglen
                # the segment of original signal, with no augmentation
                aug_seg = data[..., start_idx:end_idx]
                sc_ratio = 1
        else:
            if start_idx is not None:
                end_idx = start_idx + self.seglen
                if end_idx > siglen:
                    end_idx = siglen
                    start_idx = end_idx - self.seglen
            else:
                start_idx = end_idx - self.seglen
                if start_idx < 0:
                    start_idx = 0
                    end_idx = self.seglen
            aug_seg = data[..., start_idx:end_idx]
            sc_ratio = 1
        # adjust rpeaks
        seg_rpeaks = self.reader.load_rpeak_indices(
            rec=rec,
            sampfrom=start_idx,
            sampto=end_idx,
            keep_original=False,
        )
        seg_rpeaks = [
            int(round(r / sc_ratio))
            for r in seg_rpeaks
            if self.config.rpeaks_dist2border
            <= r
            < self.seglen - self.config.rpeaks_dist2border
        ]
        # generate qrs_mask from rpeaks
        seg_qrs_mask = np.zeros((self.seglen,), dtype=int)
        for r in seg_rpeaks:
            seg_qrs_mask[
                r - self.config.qrs_mask_bias : r + self.config.qrs_mask_bias
            ] = 1
        # adjust rhythm_intervals
        seg_rhythm_intervals = self.reader.load_rhythm_ann(
            rec=rec,
            sampfrom=start_idx,
            sampto=end_idx,
            rhythm_format="intervals",
            rhythm_types=self.config.rhythm_segmentation.classes,
            keep_original=False,
        )
        seg_rhythm_intervals = {
            rt: [
                [int(round(itv[0] / sc_ratio)), int(round(itv[1] / sc_ratio))]
                for itv in l_itvs
            ]
            for rt, l_itvs in seg_rhythm_intervals.items()
        }
        # generate rhythm_mask from rhythm_intervals
        seg_rhythm_mask = np.zeros((self.seglen,), dtype=int)
        for rt, l_itvs in seg_rhythm_intervals.items():
            for itv in l_itvs:
                seg_rhythm_mask[
                    itv[0] : itv[1]
                ] = self.config.rhythm_segmentation.class_map[rt]

        new_seg = CFG(
            data=aug_seg,
            rpeaks=seg_rpeaks,
            qrs_mask=seg_qrs_mask,
            rhythm_mask=seg_rhythm_mask,
            interval=[start_idx, end_idx],
        )
        return new_seg

    def __save_segments(
        self, rec: str, segments: List[CFG], update_segments_json: bool = False
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: str,
            filename of the record
        segments: list of dict,
            list of the segments (meta-)data
        update_segments_json: bool, default False,
            if True, the file `self.segments_json` will be updated

        """
        ordering = list(range(len(segments)))
        DEFAULTS.RNG.shuffle(ordering)
        for i, idx in enumerate(ordering):
            seg = segments[idx]
            filename = f"S_{rec}_{i:07d}.{self.segment_ext}"
            data_path = self.segments_dirs.data[rec] / filename
            savemat(str(data_path), {"ecg": seg.data})
            self.__all_segments[rec].append(Path(filename).with_suffix("").name)
            ann_path = self.segments_dirs.ann[rec] / filename
            savemat(
                str(ann_path),
                {
                    k: v
                    for k, v in seg.items()
                    if k
                    not in [
                        "data",
                    ]
                },
            )
        if update_segments_json:
            self.segments_json.write_text(
                json.dumps(self.__all_segments, ensure_ascii=False)
            )

    def _clear_cached_segments(self, recs: Optional[Sequence[str]] = None) -> NoReturn:
        """

        Parameters
        ----------
        recs: sequence of str, optional
            sequence of the records whose segments are to be cleared,
            defaults to all records

        """
        self.__assert_task(
            [
                "qrs_detection",
                "rhythm_segmentation",
                "af_event",  # segmentation of AF events
            ]
        )
        if recs is None:
            recs = self.reader.all_records
        for rec in recs:
            for item in [
                "data",
                "ann",
            ]:
                path = str(self.segments_dirs[item][rec])
                for f in [n for n in os.listdir(path) if n.endswith(self.segment_ext)]:
                    os.remove(os.path.join(path, f))
                    if os.path.splitext(f)[0] in self.__all_segments[rec]:
                        self.__all_segments[rec].remove(os.path.splitext(f)[0])
        self.segments = list_sum([self.__all_segments[rec] for rec in self.records])

    def _slice_rr_seq(
        self, force_recompute: bool = False, verbose: int = 0
    ) -> NoReturn:
        """

        slice sequences of rr intervals into fixed length (sub)sequences

        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity

        """
        self.__assert_task(["rr_lstm"])
        if force_recompute:
            self._clear_cached_rr_seq()
        with tqdm(
            enumerate(self.reader),
            total=len(self.reader),
            desc="Slicing rr_seq",
            unit="record",
        ) as pbar:
            for idx, rec in pbar:
                self._slice_rr_seq_one_record(
                    rec=rec,
                    force_recompute=False,
                    update_rr_seq_json=False,
                    verbose=verbose,
                )
                # if verbose >= 1:
                #     print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")
        if force_recompute:
            self.rr_seq_json.write_text(
                json.dumps(self.__all_rr_seq, ensure_ascii=False)
            )

    def _slice_rr_seq_one_record(
        self,
        rec: str,
        force_recompute: bool = False,
        update_rr_seq_json: bool = False,
        verbose: int = 0,
    ) -> NoReturn:
        """ """
        self.__assert_task(["rr_lstm"])
        rec_rr_seq = self.__all_rr_seq[rec]
        if (not force_recompute) and len(rec_rr_seq) > 0:
            return
        elif force_recompute:
            self._clear_cached_rr_seq([rec])

        forward_len = self.seglen - self.config[self.task].overlap_len
        critical_forward_len = self.seglen - self.config[self.task].critical_overlap_len
        critical_forward_len = [critical_forward_len - 2, critical_forward_len]

        rpeaks = self.reader.load_rpeak_indices(rec)
        rr = np.diff(rpeaks) / self.config.fs
        if len(rr) < self.seglen:
            return
        rhythm_mask = self.reader.load_rhythm_ann(
            rec,
            rhythm_format="mask",
            rhythm_types=self.config.rhythm_segmentation.classes,
        )
        label_seq = rhythm_mask[rpeaks][:-1]

        # find critical points
        critical_points = np.where(np.diff(label_seq) != 0)[0]
        critical_points = [
            p
            for p in critical_points
            if critical_forward_len[1] <= p < len(rr) - critical_forward_len[1]
        ]

        rr_seq = []

        # ordinary rr_seq with constant forward_len
        print("Slicing ordinary rr_seq with constant forward_len")
        with tqdm(
            range((len(rr) - self.seglen) // forward_len + 1),
            desc=f"Slicing rr_seq for record {rec}",
            unit="segment",
        ) as pbar:
            for idx in pbar:
                start_idx = idx * forward_len
                end_idx = start_idx + self.seglen
                new_rr_seq = CFG(
                    rr=rr[start_idx:end_idx],
                    label=label_seq[start_idx:end_idx],
                    interval=[start_idx, end_idx],
                )
                rr_seq.append(new_rr_seq)
        # the tail segment
        if end_idx < len(rr):
            end_idx = len(rr)
            start_idx = end_idx - self.seglen
            new_rr_seq = CFG(
                rr=rr[start_idx:end_idx],
                label=label_seq[start_idx:end_idx],
                interval=[start_idx, end_idx],
            )
            rr_seq.append(new_rr_seq)

        if len(critical_points) == 0:
            # save rr sequences
            self.__save_rr_seq(rec, rr_seq, update_rr_seq_json)
            return

        # special rr_seq around critical_points with random forward_len in critical_forward_len
        print(
            "Slicing special rr_seq around critical_points with random forward_len in critical_forward_len"
        )
        with tqdm(
            critical_points, desc=f"Slicing rr_seq for record {rec}", unit="segment"
        ) as pbar:
            for cp in pbar:
                start_idx = max(
                    0,
                    cp
                    - self.seglen
                    + DEFAULTS.RNG_randint(
                        critical_forward_len[0], critical_forward_len[1]
                    ),
                )
                while start_idx <= min(
                    cp - critical_forward_len[1], len(rr) - self.seglen
                ):
                    end_idx = start_idx + self.seglen
                    new_rr_seq = CFG(
                        rr=rr[start_idx:end_idx],
                        label=label_seq[start_idx:end_idx],
                        interval=[start_idx, end_idx],
                    )
                    rr_seq.append(new_rr_seq)
                    start_idx += DEFAULTS.RNG_randint(
                        critical_forward_len[0], critical_forward_len[1]
                    )
        # save rr sequences
        self.__save_rr_seq(rec, rr_seq, update_rr_seq_json)

    def __save_rr_seq(
        self, rec: str, rr_seq: List[CFG], update_rr_seq_json: bool = False
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: str,
            filename of the record
        rr_seq: list of dict,
            list of the rr_seq (meta-)data
        update_rr_seq_json: bool, default False,
            if True, the file `self.rr_seq_json` will be updated

        """
        ordering = list(range(len(rr_seq)))
        DEFAULTS.RNG.shuffle(ordering)
        for i, idx in enumerate(ordering):
            item = rr_seq[idx]
            filename = f"R_{rec}_{i:07d}.{self.rr_seq_ext}"
            data_path = self.rr_seq_dirs[rec] / filename
            savemat(str(data_path), item)
            self.__all_rr_seq[rec].append(Path(filename).with_suffix("").name)
        if update_rr_seq_json:
            self.rr_seq_json.write_text(
                json.dumps(self.__all_rr_seq, ensure_ascii=False)
            )

    def _clear_cached_rr_seq(self, recs: Optional[Sequence[str]] = None) -> NoReturn:
        """

        Parameters
        ----------
        recs: sequence of str, optional
            sequence of the records whose segments are to be cleared,
            defaults to all records

        """
        self.__assert_task(["rr_lstm"])
        if recs is None:
            recs = self.reader.all_records
        for rec in recs:
            path = str(self.rr_seq_dirs[rec])
            for f in [n for n in os.listdir(path) if n.endswith(self.rr_seq_ext)]:
                os.remove(os.path.join(path, f))
                if os.path.splitext(f)[0] in self.__all_rr_seq[rec]:
                    self.__all_rr_seq[rec].remove(os.path.splitext(f)[0])
        self.rr_seq = list_sum([self.__all_rr_seq[rec] for rec in self.records])

    def _get_rec_name(self, seg_or_rr: str) -> str:
        """

        Parameters
        ----------
        seg_or_rr: str,
            name of the segment or rr_seq

        Returns
        -------
        rec: str,
            name of the record that `seg` was generated from

        """
        rec = seg_or_rr.split("_")[1]
        return rec

    def _train_test_split(self, task: str) -> Dict[str, List[str]]:
        """

        do train test split,
        it is ensured that both the train and the test set contain all classes

        Parameters
        ----------
        task: str,
            task name

        Returns
        -------
        split_res: dict,
            keys are "train" and "test",
            values are list of the subjects split for training or validation

        """
        if task in [
            "beat_classification",
            "qrs_detection",
        ]:
            test_set = ["101", "102", "108", "114", "207", "223"]
            train_set = [rec for rec in self.reader if rec not in test_set]
        else:  # rhythm segmentation, af event, rr_lstm
            test_set = ["106", "114", "124", "202", "217", "232"]
            train_set = [rec for rec in self.reader if rec not in test_set]

        split_res = CFG(
            {
                "train": train_set,
                "test": test_set,
            }
        )
        return split_res

    def __assert_task(self, tasks: List[str]) -> NoReturn:
        """ """
        assert (
            self.task in tasks
        ), f"DO NOT call this method when the current task is {self.task}. Switch task using `reset_task`"

    def plot_seg(self, seg: str, ticks_granularity: int = 0) -> NoReturn:
        """

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)

        """
        raise NotImplementedError
        seg_data = self._load_seg_data(seg)
        print(f"seg_data.shape = {seg_data.shape}")
        seg_ann = self._load_seg_ann(seg)
        seg_ann["rhythm_intervals"] = mask_to_intervals(seg_ann["rhythm_mask"], vals=1)
        print(f"seg_ann = {seg_ann}")
        rec_name = self._get_rec_name(seg)
        self.reader.plot(
            rec=rec_name,  # unnecessary indeed
            data=seg_data,
            ann=seg_ann,
            ticks_granularity=ticks_granularity,
        )

    def extra_repr_keys(self) -> List[str]:
        return [
            "training",
            "task",
            "reader",
        ]


class FastDataReader(ReprMixin, Dataset):
    """ """

    def __init__(
        self,
        config: CFG,
        task: str,
        seg_ppm: PreprocManager,
        file_dirs: dict,
        files: List[str],
        file_ext: str,
    ) -> NoReturn:
        """ """
        self.config = config
        self.task = task
        self.seg_ppm = seg_ppm
        self.file_dirs = file_dirs
        self.files = files
        self.file_ext = file_ext

        self.seglen = self.config[self.task].input_len
        self.n_classes = len(self.config[task].classes)

        self._seg_keys = {
            "qrs_detection": "qrs_mask",
            "rhythm_segmentation": "rhythm_mask",
            "af_event": "rhythm_mask",  # segmentation of AF events
        }

    def __getitem__(self, index: int) -> Tuple[np.ndarray, ...]:
        """ """
        if self.task in [
            "qrs_detection",
            "rhythm_segmentation",
            "af_event",  # segmentation of AF events
        ]:
            seg_name = self.files[index]
            rec = seg_name.split("_")[1]
            seg_data_fp = self.file_dirs.data[rec] / f"{seg_name}.{self.file_ext}"
            seg_data = loadmat(str(seg_data_fp))["ecg"]
            for idx in range(seg_data.shape[0]):
                seg_data[idx] = remove_spikes_naive(seg_data[idx])
            seg_ann_fp = self.file_dirs.ann[rec] / f"{seg_name}.{self.file_ext}"
            seg_label = loadmat(str(seg_ann_fp))[self._seg_keys[self.task]].reshape(
                (self.seglen, -1)
            )
            if self.config[self.task].reduction > 1:
                reduction = self.config[self.task].reduction
                seg_len, n_classes = seg_label.shape
                seg_label = np.stack(
                    arrays=[
                        np.mean(
                            seg_data[reduction * idx : reduction * (idx + 1)],
                            axis=0,
                            keepdims=True,
                        ).astype(int)
                        for idx in range(seg_len // reduction)
                    ],
                    axis=0,
                ).squeeze(axis=1)
            seg_data, _ = self.seg_ppm(seg_data, self.config.fs)
            if self.task == [
                "rhythm_segmentation",
                "af_event",  # segmentation of AF events
            ]:
                weight_mask = generate_weight_mask(
                    target_mask=seg_label.squeeze(-1),
                    fg_weight=2,
                    fs=self.config.fs,
                    reduction=self.config[self.task].reduction,
                    radius=0.8,
                    boundary_weight=5,
                )[..., np.newaxis]
                return seg_data, seg_label, weight_mask
            return seg_data, seg_label, None
        elif self.task in [
            "rr_lstm",
        ]:
            seq_name = self.files[index]
            rec = seq_name.split("_")[1]
            rr_seq_path = self.file_dirs[rec] / f"{seq_name}.{self.file_ext}"
            rr_seq = loadmat(str(rr_seq_path))
            rr_seq["rr"] = rr_seq["rr"].reshape((self.seglen, 1))
            rr_seq["label"] = rr_seq["label"].reshape((self.seglen, self.n_classes))
            weight_mask = generate_weight_mask(
                target_mask=rr_seq["label"].squeeze(-1),
                fg_weight=2,
                fs=1 / 0.8,
                reduction=1,
                radius=2,
                boundary_weight=5,
            )[..., np.newaxis]
            return rr_seq["rr"], rr_seq["label"], weight_mask
        else:
            raise NotImplementedError(
                f"data generator for task \042{self.task}\042 not implemented"
            )

    def __len__(self) -> int:
        """ """
        return len(self.files)

    def extra_repr_keys(self) -> List[str]:
        return [
            "task",
            "reader",
            "ppm",
        ]


def _get_rec_suffix(operations: List[str]) -> str:
    """

    Parameters
    ----------
    operations: list of str,
        names of operations to perform (or has performed),

    Returns
    -------
    suffix: str,
        suffix of the filename of the preprocessed ecg signal

    """
    suffix = "-".join(sorted([item.lower() for item in operations]))
    return suffix
