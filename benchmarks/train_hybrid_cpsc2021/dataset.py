"""
data generator for feeding data into pytorch models

NOTE
----
In order to avoid potential error in the methods of slicing signals and rr intervals,
one can check using the following code

```python
from cfg import TrainCfg

ds_train = CPSC2021(TrainCfg, task="qrs_detection", training=True)
ds_val = CPSC2021(TrainCfg, task="qrs_detection", training=False)
err_list = []
for idx, seg in enumerate(ds_train.segments):
    sig, lb = ds_train[idx]
    if sig.shape != (2,6000) or lb.shape != (750, 1):
        print("\n"+f"segment {seg} has sig.shape = {sig.shape}, lb.shape = {lb.shape}"+"\n")
        err_list.append(seg)
    print(f"{idx+1}/{len(ds_train)}", end="\r")
for idx, seg in enumerate(ds_val.segments):
    sig, lb = ds_val[idx]
    if sig.shape != (2,6000) or lb.shape != (750, 1):
        print("\n"+f"segment {seg} has sig.shape = {sig.shape}, lb.shape = {lb.shape}"+"\n")
        err_list.append(seg)
    print(f"{idx+1}/{len(ds_val)}", end="\r")
for idx, seg in enumerate(err_list):
    path = ds_train._get_seg_data_path(seg)
    os.remove(path)
    path = ds_train._get_seg_ann_path(seg)
    os.remove(path)
    print(f"{idx+1}/{len(err_list)}", end="\r")
```

and similarly for the task of `rr_lstm`
"""

import os, sys
import json
import re
import time
import multiprocessing as mp
import random
from itertools import repeat
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import signal as SS
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat, savemat

from torch_ecg.databases import CPSC2021 as CR
from torch_ecg.utils.preproc import preprocess_multi_lead_signal
from torch_ecg.utils.utils_signal import normalize
from torch_ecg.utils.utils_interval import mask_to_intervals
from torch_ecg.utils.misc import (
    dict_to_str, list_sum, nildent, uniform,
    get_record_list_recursive3,
)

from cfg import (
    TrainCfg, ModelCfg,
)


if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2021",
]


_BASE_DIR = os.path.dirname(__file__)


class CPSC2021(Dataset):
    """
    1. ECGs are preprocessed and stored in one folder
    2. preprocessed ECGs are sliced with overlap to generate data and label for different tasks:
        the data files stores segments of fixed length of preprocessed ECGs,
        the annotation files contain "qrs_mask", and "af_mask"
    """
    __DEBUG__ = False
    __name__ = "CPSC2021"
    
    def __init__(self, config:ED, task:str, training:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        if ModelCfg.torch_dtype.lower() == "double":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.allowed_preproc = self.config.preproc

        self.training = training
        self.__data_aug = self.training

        # create directories if needed
        # preprocess_dir stores pre-processed signals
        self.preprocess_dir = os.path.join(config.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # segments_dir for sliced segments of fixed length
        self.segments_base_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.segments_base_dir, exist_ok=True)
        self.segment_name_pattern = "S_\d{1,3}_\d{1,2}_\d{7}"
        self.segment_ext = "mat"
        # rr_dir for sequence of rr intervals of fix length
        self.rr_seq_base_dir = os.path.join(config.db_dir, "rr_seq")
        os.makedirs(self.rr_seq_base_dir, exist_ok=True)
        self.rr_seq_name_pattern = "R_\d{1,3}_\d{1,2}_\d{7}"
        self.rr_seq_ext = "mat"

        self.__set_task(task)

    def __set_task(self, task:str) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        task: str,
            name of the task, can be one of `TrainCfg.tasks`
        """
        assert task.lower() in TrainCfg.tasks, f"illegal task \042{task}\042"
        if hasattr(self, "task") and self.task == task.lower():
            return
        self.task = task.lower()
        self.all_classes = self.config[task].classes
        self.n_classes = len(self.config[task].classes)

        self.seglen = self.config[task].input_len  # alias, for simplicity
        split_res = self._train_test_split(
            train_ratio=self.config.train_ratio,
            force_recompute=False,
        )
        if self.training:
            self.subjects = split_res.train
        else:
            self.subjects = split_res.test

        if self.task in ["qrs_detection", "main",]:
            # for qrs detection, or for the main task
            self.segments_dirs = ED()
            self.__all_segments = ED()
            self.segments_json = os.path.join(self.segments_base_dir, "segments.json")
            self._ls_segments()
            self.segments = list_sum([self.__all_segments[subject] for subject in self.subjects])
            if self.training:
                random.shuffle(self.segments)
        elif self.task in ["rr_lstm",]:
            self.rr_seq_dirs = ED()
            self.__all_rr_seq = ED()
            self.rr_seq_json = os.path.join(self.rr_seq_base_dir, "rr_seq.json")
            self._ls_rr_seq()
            self.rr_seq = list_sum([self.__all_rr_seq[subject] for subject in self.subjects])
            if self.training:
                random.shuffle(self.rr_seq)
        else:
            raise NotImplementedError(f"data generator for task \042{self.task}\042 not implemented")
        
        # more aux config on offline augmentations
        self.config.stretch_compress_choices = [-1,1] + [0] * int(2/self.config.stretch_compress_prob - 2)

    def reset_task(self, task:str) -> NoReturn:
        """ finished, checked,
        """
        self.__set_task(task)

    def _ls_segments(self) -> NoReturn:
        """ finished, checked,

        list all the segments
        """
        for item in ["data", "ann"]:
            self.segments_dirs[item] = ED()
            for s in self.reader.all_subjects:
                self.segments_dirs[item][s] = os.path.join(self.segments_base_dir, item, s)
                os.makedirs(self.segments_dirs[item][s], exist_ok=True)
        if os.path.isfile(self.segments_json):
            with open(self.segments_json, "r") as f:
                self.__all_segments = json.load(f)
            return
        print(f"please allow the reader a few minutes to collect the segments from {self.segments_base_dir}...")
        seg_filename_pattern = f"{self.segment_name_pattern}.{self.segment_ext}"
        self.__all_segments = ED({
            s: get_record_list_recursive3(self.segments_dirs.data[s], seg_filename_pattern) \
                for s in self.reader.all_subjects
        })
        if all([len(self.__all_segments[s])>0 for s in self.reader.all_subjects]):
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    def _ls_rr_seq(self) -> NoReturn:
        """ finished, checked,

        list all the rr sequences
        """
        for s in self.reader.all_subjects:
            self.rr_seq_dirs[s] = os.path.join(self.rr_seq_base_dir, s)
            os.makedirs(self.rr_seq_dirs[s], exist_ok=True)
        if os.path.isfile(self.rr_seq_json):
            with open(self.rr_seq_json, "r") as f:
                self.__all_rr_seq = json.load(f)
            return
        print(f"please allow the reader a few minutes to collect the rr sequences from {self.rr_seq_base_dir}...")
        rr_seq_filename_pattern = f"{self.rr_seq_name_pattern}.{self.rr_seq_ext}"
        self.__all_rr_seq = ED({
            s: get_record_list_recursive3(self.rr_seq_dirs[s], rr_seq_filename_pattern) \
                for s in self.reader.all_subjects
        })
        if all([len(self.__all_rr_seq[s])>0 for s in self.reader.all_subjects]):
            with open(self.rr_seq_json, "w") as f:
                json.dump(self.__all_rr_seq, f)

    @property
    def all_segments(self) -> ED:
        if self.task in ["qrs_detection", "main",]:
            return self.__all_segments
        else:
            return ED()

    @property
    def all_rr_seq(self) -> ED:
        if self.task.lower() in ["rr_lstm",]:
            return self.__all_rr_seq
        else:
            return ED()

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        if self.task in ["qrs_detection", "main",]:
            seg_name = self.segments[index]
            seg_data = self._load_seg_data(seg_name)
            if self.config[self.task].model_name == "unet":
                seg_label = self._load_seg_mask(seg_name)
            else:  # "seq_lab"
                seg_label = self._load_seg_seq_lab(seg_name, reduction=self.config[self.task].reduction)
            # augmentation
            if self.__data_aug:
                if len(self.config.flip) > 0:
                    sign = random.sample(self.config.flip, 1)[0]
                    seg_data *= sign
                if self.config.random_normalize:
                    rn_mean = np.array(uniform(
                        self.config.random_normalize_mean[0],
                        self.config.random_normalize_mean[1],
                        self.config.n_leads,
                    ))
                    rn_std = np.array(uniform(
                        self.config.random_normalize_std[0],
                        self.config.random_normalize_std[1],
                        self.config.n_leads,
                    ))
                    seg_data = normalize(
                        sig=seg_data,
                        mean=rn_mean,
                        std=rn_std,
                        per_channel=True,
                    )
                if self.config.label_smoothing > 0:
                    seg_label = (1 - self.config.label_smoothing) * seg_label \
                        + self.config.label_smoothing / (1 + self.n_classes)
            else:  # no augmentation
                if self.config.random_normalize:  # to keep consistency of data distribution
                    seg_data = normalize(
                        sig=seg_data,
                        mean=list(repeat(np.mean(self.config.random_normalize_mean), self.config.n_leads)),
                        std=list(repeat(np.mean(self.config.random_normalize_std), self.config.n_leads)),
                        per_channel=True,
                    )
            if self.task == "main":
                weight_mask = _generate_weight_mask(
                    target_mask=seg_label.squeeze(-1),
                    fg_weight=2,
                    fs=self.config.fs,
                    reduction=self.config[self.task].reduction,
                    radius=0.8,
                    boundary_weight=5,
                )[..., np.newaxis]
                return seg_data, seg_label, weight_mask
            return seg_data, seg_label
        elif self.task in ["rr_lstm",]:
            rr_seq = self._load_rr_seq(self.rr_seq[index])
            weight_mask = _generate_weight_mask(
                target_mask=rr_seq["label"].squeeze(-1), 
                fg_weight=2, fs=1/0.8, reduction=1, radius=2, boundary_weight=5
            )[..., np.newaxis]
            return rr_seq["rr"], rr_seq["label"], weight_mask
        else:
            raise NotImplementedError(f"data generator for task \042{self.task}\042 not implemented")

    def __len__(self) -> int:
        """ finished,
        """
        if self.task in ["qrs_detection", "main",]:
            return len(self.segments)
        else:  # "rr_lstm"
            return len(self.rr_seq)

    def _get_seg_data_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        fp: str,
            path of the data file of the segment
        """
        subject = seg.split("_")[1]
        fp = os.path.join(self.segments_dirs.data[subject], f"{seg}.{self.segment_ext}")
        return fp

    def _get_seg_ann_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        fp: str,
            path of the annotation file of the segment
        """
        subject = seg.split("_")[1]
        fp = os.path.join(self.segments_dirs.ann[subject], f"{seg}.{self.segment_ext}")
        return fp

    def _load_seg_data(self, seg:str) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        seg_data: ndarray,
            data of the segment, of shape (2, `self.seglen`)
        """
        seg_data_fp = self._get_seg_data_path(seg)
        seg_data = loadmat(seg_data_fp)["ecg"]
        return seg_data

    def _load_seg_ann(self, seg:str) -> dict:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"

        Returns
        -------
        seg_ann: dict,
            annotations of the segment, including
            - rpeaks: indices of rpeaks of the segment
            - qrs_mask: mask of qrs complexes of the segment
            - af_mask: mask of af episodes of the segment
            - interval: interval ([start_idx, end_idx]) in the original ECG record of the segment
        """
        seg_ann_fp = self._get_seg_ann_path(seg)
        seg_ann = {k:v.flatten() for k,v in loadmat(seg_ann_fp).items() if not k.startswith("__")}
        return seg_ann

    def _load_seg_mask(self, seg:str, task:Optional[str]=None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        task: str, optional,
            if specified, overrides self.task,
            else if is "all", then all masks ("qrs_mask", "af_mask", etc.) will be returned

        Returns
        -------
        seg_mask: np.ndarray or dict,
            mask(s) of the segment,
            of shape (self.seglen, self.n_classes)
        """
        seg_mask = {k:v.reshape((self.seglen, -1)) for k,v in self._load_seg_ann(seg).items() if k in ["qrs_mask", "af_mask",]}
        _task = (task or self.task).lower()
        if _task == "all":
            return seg_mask
        if _task in ["qrs_detection",]:
            seg_mask = seg_mask["qrs_mask"]
        elif _task in ["main",]:
            seg_mask = seg_mask["af_mask"]
        return seg_mask

    def _load_seg_seq_lab(self, seg:str, reduction:int=8) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        reduction: int, default 8,
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
                np.mean(seg_mask[reduction*idx:reduction*(idx+1)],axis=0,keepdims=True).astype(int) \
                    for idx in range(seg_len//reduction)
            ],
            axis=0,
        ).squeeze(axis=1)
        return seq_lab

    def _get_rr_seq_path(self, seq_name:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seq_name: str,
            name of the rr_seq, of pattern like "R_1_1_0000193"

        Returns
        -------
        fp: str,
            path of the annotation file of the rr_seq
        """
        subject = seq_name.split("_")[1]
        fp = os.path.join(self.rr_seq_dirs[subject], f"{seq_name}.{self.rr_seq_ext}")
        return fp

    def _load_rr_seq(self, seq_name:str) -> Dict[str, np.ndarray]:
        """ finished, checked,

        Parameters
        ----------
        seq_name: str,
            name of the rr_seq, of pattern like "R_1_1_0000193"

        Returns
        -------
        rr_seq: dict,
            metadata of sequence of rr intervals, including
            - rr: the sequence of rr intervals, with units in seconds, of shape (self.seglen, 1)
            - label: label of the rr intervals, 0 for normal, 1 for af, of shape (self.seglen, self.n_classes)
            - interval: interval of the current rr sequence in the whole rr sequence in the original record
        """
        rr_seq_path = self._get_rr_seq_path(seq_name)
        rr_seq = {k:v for k,v in loadmat(rr_seq_path).items() if not k.startswith("__")}
        rr_seq["rr"] = rr_seq["rr"].reshape((self.seglen, 1))
        rr_seq["label"] = rr_seq["label"].reshape((self.seglen, self.n_classes))
        rr_seq["interval"] = rr_seq["interval"].flatten()
        return rr_seq

    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    def enable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = True

    @property
    def use_augmentation(self) -> bool:
        return self.__data_aug

    def persistence(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

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
            self.allowed_preproc,
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

    def _preprocess_data(self, preproc:List[str], force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        preproc = self._normalize_preprocess_names(preproc, True)
        for idx, rec in enumerate(self.reader.all_records):
            self._preprocess_one_record(
                rec=rec,
                preproc=preproc,
                force_recompute=force_recompute,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")

    def _preprocess_one_record(self, rec:str, preproc:List[str], force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        rec: str,
            filename of the record
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        suffix = self._get_rec_suffix(preproc)
        save_fp = os.path.join(self.preprocess_dir, f"{rec}-{suffix}.{self.segment_ext}")
        if (not force_recompute) and os.path.isfile(save_fp):
            return
        # perform pre-process
        if "baseline" in preproc:
            bl_win = [self.config.baseline_window1, self.config.baseline_window2]
        else:
            bl_win = None
        if "bandpass" in preproc:
            band_fs = self.config.filter_band
        else:
            band_fs = None
        pps = preprocess_multi_lead_signal(
            self.reader.load_data(rec),
            fs=self.reader.fs,
            bl_win=bl_win,
            band_fs=band_fs,
            verbose=verbose,
        )
        savemat(save_fp, {"ecg": pps["filtered_ecg"]}, format="5")

    def load_preprocessed_data(self, rec:str, preproc:Optional[List[str]]=None) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            filename of the record
        preproc: list of str, optional
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`,
            defaults to `self.allowed_preproc`

        Returns
        -------
        p_sig: ndarray,
            the pre-computed processed ECG
        """
        if preproc is None:
            preproc = self.allowed_preproc
        suffix = self._get_rec_suffix(preproc)
        fp = os.path.join(self.preprocess_dir, f"{rec}-{suffix}.{self.segment_ext}")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"preprocess(es) \042{preproc}\042 not done for {rec} yet")
        p_sig = loadmat(fp)["ecg"]
        if p_sig.shape[0] != 2:
            p_sig = p_sig.T
        return p_sig

    def _normalize_preprocess_names(self, preproc:List[str], ensure_nonempty:bool) -> List[str]:
        """ finished, checked,

        to transform all preproc into lower case,
        and keep them in a specific ordering 
        
        Parameters
        ----------
        preproc: list of str,
            list of preprocesses types,
            should be sublist of `self.allowd_features`
        ensure_nonempty: bool,
            if True, when the passed `preproc` is empty,
            `self.allowed_preproc` will be returned

        Returns
        -------
        _p: list of str,
            "normalized" list of preprocess types
        """
        _p = [item.lower() for item in preproc] if preproc else []
        if ensure_nonempty:
            _p = _p or self.allowed_preproc
        # ensure ordering
        _p = [item for item in self.allowed_preproc if item in _p]
        # assert all([item in self.allowed_preproc for item in _p])
        return _p

    def _get_rec_suffix(self, operations:List[str]) -> str:
        """ finished, checked,

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

    def _slice_data(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        slice all records into segments of length `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        self.__assert_task(["qrs_detection", "main",])
        if force_recompute:
            self._clear_cached_segments()
        for idx, rec in enumerate(self.reader.all_records):
            self._slice_one_record(
                rec=rec,
                force_recompute=False,
                update_segments_json=False,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")
        # with mp.Pool(processes=max(1, mp.cpu_count()-3)) as pool:
        #     pool.starmap(
        #         func=self._slice_one_record,
        #         iterable=[(rec, force_recompute, False, verbose) for rec in self.reader.all_records]
        #     )
        if force_recompute:
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    def _slice_one_record(self, rec:str, force_recompute:bool=False, update_segments_json:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

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
        self.__assert_task(["qrs_detection", "main",])
        subject = self.reader.get_subject_id(rec)
        rec_segs = [item for item in self.__all_segments[subject] if item.startswith(rec.replace("data", "S"))]
        if (not force_recompute) and len(rec_segs) > 0:
            return
        elif force_recompute:
            self._clear_cached_segments([rec])

        # data = self.reader.load_data(rec, units="mV")
        data = self.load_preprocessed_data(rec)
        siglen = data.shape[1]
        rpeaks = self.reader.load_rpeaks(rec)
        af_mask = self.reader.load_af_episodes(rec, fmt="mask")
        forward_len = self.seglen - self.config[self.task].overlap_len
        critical_forward_len = self.seglen - self.config[self.task].critical_overlap_len
        critical_forward_len = [critical_forward_len//4, critical_forward_len]

        # skip those records that are too short
        if siglen < self.seglen:
            return

        # find critical points
        critical_points = np.where(np.diff(af_mask)!=0)[0]
        critical_points = [p for p in critical_points if critical_forward_len[1]<=p<siglen-critical_forward_len[1]]

        segments = []

        # ordinary segments with constant forward_len
        for idx in range((siglen-self.seglen)//forward_len + 1):
            start_idx = idx * forward_len
            new_seg = self.__generate_segment(
                rec=rec, data=data, start_idx=start_idx,
            )
            segments.append(new_seg)
        # the tail segment
        new_seg = self.__generate_segment(
            rec=rec, data=data, end_idx=siglen,
        )
        segments.append(new_seg)

        # special segments around critical_points with random forward_len in critical_forward_len
        for cp in critical_points:
            start_idx = max(0, cp - self.seglen + random.randint(critical_forward_len[0], critical_forward_len[1]))
            while start_idx <= min(cp - critical_forward_len[1], siglen - self.seglen):
                new_seg = self.__generate_segment(
                    rec=rec, data=data, start_idx=start_idx,
                )
                segments.append(new_seg)
                start_idx += random.randint(critical_forward_len[0], critical_forward_len[1])
        
        # return segments
        self.__save_segments(rec, segments, update_segments_json)

    def __generate_segment(self, rec:str, data:np.ndarray, start_idx:Optional[int]=None, end_idx:Optional[int]=None) -> ED:
        """ finished, checked,

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
            - af_mask: mask of af episodes of the segment
            - interval: interval ([start_idx, end_idx]) in the original ECG record of the segment
        """
        assert not all([start_idx is None, end_idx is None]), \
            "at least one of `start_idx` and `end_idx` should be set"
        siglen = data.shape[1]
        # offline augmentations are done, including strech-or-compress, ...
        if self.config.stretch_compress != 0:
            sign = random.sample(self.config.stretch_compress_choices, 1)[0]
            if sign != 0:
                sc_ratio = self.config.stretch_compress
                sc_ratio = 1 + (random.uniform(sc_ratio/4, sc_ratio) * sign) / 100
                sc_len = int(round(sc_ratio * self.seglen))
                if start_idx is not None:
                    end_idx = start_idx + sc_len
                else:
                    start_idx = end_idx - sc_len
                if end_idx > siglen:
                    end_idx = siglen
                    start_idx = max(0, end_idx - sc_len)
                    sc_ratio = (end_idx - start_idx) / self.seglen
                aug_seg = data[..., start_idx: end_idx]
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
                aug_seg = data[..., start_idx: end_idx]
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
            aug_seg = data[..., start_idx: end_idx]
            sc_ratio = 1
        # adjust rpeaks
        seg_rpeaks = self.reader.load_rpeaks(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True,
        )
        seg_rpeaks = [
            int(round(r/sc_ratio)) for r in seg_rpeaks \
                if self.config.rpeaks_dist2border <= r < self.seglen-self.config.rpeaks_dist2border
        ]
        # generate qrs_mask from rpeaks
        seg_qrs_mask = np.zeros((self.seglen,), dtype=int)
        for r in seg_rpeaks:
            seg_qrs_mask[r-self.config.qrs_mask_bias:r+self.config.qrs_mask_bias] = 1
        # adjust af_intervals
        seg_af_intervals = self.reader.load_af_episodes(
            rec=rec, sampfrom=start_idx, sampto=end_idx, zero_start=True, fmt="intervals",
        )
        seg_af_intervals = [
            [int(round(itv[0]/sc_ratio)), int(round(itv[1]/sc_ratio))] for itv in seg_af_intervals
        ]
        # generate af_mask from af_intervals
        seg_af_mask = np.zeros((self.seglen,), dtype=int)
        for itv in seg_af_intervals:
            seg_af_mask[itv[0]:itv[1]] = 1

        new_seg = ED(
            data=aug_seg,
            rpeaks=seg_rpeaks,
            qrs_mask=seg_qrs_mask,
            af_mask=seg_af_mask,
            interval=[start_idx, end_idx],
        )
        return new_seg

    def __save_segments(self, rec:str, segments:List[ED], update_segments_json:bool=False) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            filename of the record
        segments: list of dict,
            list of the segments (meta-)data
        update_segments_json: bool, default False,
            if True, the file `self.segments_json` will be updated
        """
        subject = self.reader.get_subject_id(rec)
        ordering = list(range(len(segments)))
        random.shuffle(ordering)
        for i, idx in enumerate(ordering):
            seg = segments[idx]
            filename = f"{rec}_{i:07d}.{self.segment_ext}".replace("data", "S")
            data_path = os.path.join(self.segments_dirs.data[subject], filename)
            savemat(data_path, {"ecg": seg.data})
            self.__all_segments[subject].append(os.path.splitext(filename)[0])
            ann_path = os.path.join(self.segments_dirs.ann[subject], filename)
            savemat(ann_path, {k:v for k,v in seg.items() if k not in ["data",]})
        if update_segments_json:
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    def _clear_cached_segments(self, recs:Optional[Sequence[str]]=None) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        recs: sequence of str, optional
            sequence of the records whose segments are to be cleared,
            defaults to all records
        """
        self.__assert_task(["qrs_detection", "main",])
        if recs is not None:
            for rec in recs:
                subject = self.reader.get_subject_id(rec)
                for item in ["data", "ann",]:
                    path = self.segments_dirs[item][subject]
                    for f in [n for n in os.listdir(path) if n.endswith(self.segment_ext)]:
                        if self._get_rec_name(f) == rec:
                            os.remove(os.path.join(path, f))
                            self.__all_segments[subject].remove(os.path.splitext(f)[0])
        else:
            for subject in self.reader.all_subjects:
                for item in ["data", "ann",]:
                    path = self.segments_dirs[item][subject]
                    for f in [n for n in os.listdir(path) if n.endswith(self.segment_ext)]:
                        os.remove(os.path.join(path, f))
                        self.__all_segments[subject].remove(os.path.splitext(f)[0])
        self.segments = list_sum([self.__all_segments[subject] for subject in self.subjects])

    def _slice_rr_seq(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

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
        for idx, rec in enumerate(self.reader.all_records):
            self._slice_rr_seq_one_record(
                rec=rec,
                force_recompute=False,
                update_rr_seq_json=False,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")
        if force_recompute:
            with open(self.rr_seq_json, "w") as f:
                json.dump(self.__all_rr_seq, f)

    def _slice_rr_seq_one_record(self, rec:str, force_recompute:bool=False, update_rr_seq_json:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,
        """
        self.__assert_task(["rr_lstm"])
        subject = self.reader.get_subject_id(rec)
        rec_rr_seq = [item for item in self.__all_rr_seq[subject] if item.startswith(rec.replace("data", "R"))]
        if (not force_recompute) and len(rec_rr_seq) > 0:
            return
        elif force_recompute:
            self._clear_cached_rr_seq([rec])

        forward_len = self.seglen - self.config[self.task].overlap_len
        critical_forward_len = self.seglen - self.config[self.task].critical_overlap_len
        critical_forward_len = [critical_forward_len-2, critical_forward_len]
            
        rpeaks = self.reader.load_rpeaks(rec)
        rr = np.diff(rpeaks) / self.config.fs
        if len(rr) < self.seglen:
            return
        af_mask = self.reader.load_af_episodes(rec, fmt="mask")
        label_seq = af_mask[rpeaks][:-1]

        # find critical points
        critical_points = np.where(np.diff(label_seq)!=0)[0]
        critical_points = [p for p in critical_points if critical_forward_len[1]<=p<len(rr)-critical_forward_len[1]]

        rr_seq = []

        # ordinary segments with constant forward_len
        for idx in range((len(rr)-self.seglen)//forward_len + 1):
            start_idx = idx * forward_len
            end_idx = start_idx + self.seglen
            new_rr_seq = ED(
                rr=rr[start_idx:end_idx],
                label=label_seq[start_idx:end_idx],
                interval=[start_idx,end_idx],
            )
            rr_seq.append(new_rr_seq)
        # the tail segment
        end_idx = len(rr)
        start_idx = start_idx + self.seglen
        new_rr_seq = ED(
            rr=rr[start_idx:end_idx],
            label=label_seq[start_idx:end_idx],
            interval=[start_idx,end_idx],
        )
        rr_seq.append(new_rr_seq)

        # special segments around critical_points with random forward_len in critical_forward_len
        for cp in critical_points:
            start_idx = max(0, cp - self.seglen + random.randint(critical_forward_len[0], critical_forward_len[1]))
            while start_idx <= min(cp - critical_forward_len[1], len(rr) - self.seglen):
                end_idx = start_idx + self.seglen
                new_rr_seq = ED(
                    rr=rr[start_idx:end_idx],
                    label=label_seq[start_idx:end_idx],
                    interval=[start_idx,end_idx],
                )
                rr_seq.append(new_rr_seq)
                start_idx += random.randint(critical_forward_len[0], critical_forward_len[1])
        # save rr sequences
        ordering = list(range(len(rr_seq)))
        random.shuffle(ordering)
        for i, idx in enumerate(ordering):
            item = rr_seq[idx]
            filename = f"{rec}_{i:07d}.{self.rr_seq_ext}".replace("data", "R")
            data_path = os.path.join(self.rr_seq_dirs[subject], filename)
            savemat(data_path, item)
            self.__all_rr_seq[subject].append(os.path.splitext(filename)[0])
        if update_rr_seq_json:
            with open(self.rr_seq_json, "w") as f:
                json.dump(self.__all_rr_seq, f)

    def _clear_cached_rr_seq(self, recs:Optional[Sequence[str]]=None) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        recs: sequence of str, optional
            sequence of the records whose segments are to be cleared,
            defaults to all records
        """
        self.__assert_task(["rr_lstm"])
        if recs is not None:
            for rec in recs:
                subject = self.reader.get_subject_id(rec)
                path = self.rr_seq_dirs[subject]
                for f in [n for n in os.listdir(path) if n.endswith(self.rr_seq_ext)]:
                    if self._get_rec_name(f) == rec:
                        os.remove(os.path.join(path, f))
                        self.__all_rr_seq[subject].remove(os.path.splitext(f)[0])
        else:
            for subject in self.reader.all_subjects:
                    path = self.rr_seq_dirs[subject]
                    for f in [n for n in os.listdir(path) if n.endswith(self.rr_seq_ext)]:
                        os.remove(os.path.join(path, f))
                        self.__all_rr_seq[subject].remove(os.path.splitext(f)[0])
        self.rr_seq = list_sum([self.__all_rr_seq[subject] for subject in self.subjects])

    def _get_rec_name(self, seg_or_rr:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seg_or_rr: str,
            name of the segment or rr_seq
        
        Returns
        -------
        rec: str,
            name of the record that `seg` was generated from
        """
        rec = re.sub("[RS]", "data", os.path.splitext(seg_or_rr)[0])[:-8]
        return rec

    def _train_test_split(self,
                          train_ratio:float=0.8,
                          force_recompute:bool=False) -> Dict[str, List[str]]:
        """ finished, checked,

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
        split_res: dict,
            keys are "train" and "test",
            values are list of the subjects split for training or validation
        """
        start = time.time()
        print("\nstart performing train test split...\n")
        _train_ratio = int(train_ratio*100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0

        train_file = os.path.join(self.reader.db_dir_base, f"train_ratio_{_train_ratio}.json")
        test_file = os.path.join(self.reader.db_dir_base, f"test_ratio_{_test_ratio}.json")

        if not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            train_file = os.path.join(_BASE_DIR, "utils", os.path.basename(train_file))
            test_file = os.path.join(_BASE_DIR, "utils", os.path.basename(test_file))

        if force_recompute or not all([os.path.isfile(train_file), os.path.isfile(test_file)]):
            all_subjects = set(self.reader.df_stats.subject_id.tolist())
            afp_subjects = set(self.reader.df_stats[self.reader.df_stats.label=="AFp"].subject_id.tolist())
            aff_subjects = set(self.reader.df_stats[self.reader.df_stats.label=="AFf"].subject_id.tolist()) - afp_subjects
            normal_subjects = all_subjects - afp_subjects - aff_subjects

            test_set = random.sample(afp_subjects, int(round(len(afp_subjects)*_test_ratio/100))) + \
                random.sample(aff_subjects, int(round(len(aff_subjects)*_test_ratio/100))) + \
                random.sample(normal_subjects, int(round(len(normal_subjects)*_test_ratio/100)))
            train_set = list(all_subjects - set(test_set))
            
            random.shuffle(test_set)
            random.shuffle(train_set)

            train_file_1 = os.path.join(self.reader.db_dir_base, f"train_ratio_{_train_ratio}.json")
            train_file_2 = os.path.join(_BASE_DIR, "utils", f"train_ratio_{_train_ratio}.json")
            with open(train_file_1, "w") as f1, open(train_file_2, "w") as f2:
                json.dump(train_set, f1, ensure_ascii=False)
                json.dump(train_set, f2, ensure_ascii=False)
            test_file_1 = os.path.join(self.reader.db_dir_base, f"test_ratio_{_test_ratio}.json")
            test_file_2 = os.path.join(_BASE_DIR, "utils", f"test_ratio_{_test_ratio}.json")
            with open(test_file_1, "w") as f1, open(test_file_2, "w") as f2:
                json.dump(test_set, f1, ensure_ascii=False)
                json.dump(test_set, f2, ensure_ascii=False)
            print(nildent(f"""
                train set saved to \n\042{train_file_1}\042and\n\042{train_file_2}\042
                test set saved to \n\042{test_file_1}\042and\n\042{test_file_2}\042
                """
            ))
        else:
            with open(train_file, "r") as f:
                train_set = json.load(f)
            with open(test_file, "r") as f:
                test_set = json.load(f)

        print(f"train test split finished in {(time.time()-start)/60:.2f} minutes")

        split_res = ED({
            "train": train_set,
            "test": test_set,
        })
        return split_res

    def __assert_task(self, tasks:List[str]) -> NoReturn:
        """
        """
        assert self.task in tasks, \
            f"DO NOT call this method when the current task is {self.task}. Switch task using `reset_task`"

    def plot_seg(self, seg:str, ticks_granularity:int=0) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S_1_1_0000193"
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        """
        seg_data = self._load_seg_data(seg)
        print(f"seg_data.shape = {seg_data.shape}")
        seg_ann = self._load_seg_ann(seg)
        seg_ann["af_episodes"] = mask_to_intervals(seg_ann["af_mask"], vals=1)
        print(f"seg_ann = {seg_ann}")
        rec_name = self._get_rec_name(seg)
        self.reader.plot(
            rec=rec_name,  # unnecessary indeed
            data=seg_data,
            ann=seg_ann,
            ticks_granularity=ticks_granularity,
        )


def _generate_weight_mask(target_mask:np.ndarray,
                          fg_weight:float,
                          fs:int,
                          reduction:int,
                          radius:float,
                          boundary_weight:float,
                          plot:bool=False) -> np.ndarray:
    """ finished, checked,

    generate weight mask for a binary target mask,
    accounting the foreground weight and boundary weight

    Parameters
    ----------
    target_mask: ndarray,
        the target mask, assumed to be 1d
    fg_weight: float,
        foreground weight, usually > 1
    fs: int,
        sampling frequency of the signal
    reduction: int,
        reduction ratio of the mask w.r.t. the signal
    boundary_weight: float,
        weight for the boundaries (positions where values change) of the target map
    plot: bool, default False,
        if True, target_mask and the result weight_mask will be plotted

    Returns
    -------
    weight_mask: ndarray,
        the weight mask
    """
    weight_mask = np.ones_like(target_mask, dtype=float)
    sigma = int((radius * fs) / reduction)
    weight = np.full_like(target_mask, fg_weight) - 1
    weight_mask += (target_mask > 0.5) * weight
    border = np.where(np.diff(target_mask)!=0)[0]
    for idx in border:
        # weight = np.zeros_like(target_mask, dtype=float)
        # weight[max(0, idx-sigma): (idx+sigma)] = boundary_weight
        weight = np.full_like(target_mask, boundary_weight, dtype=float)
        weight = weight * np.exp(-np.power(np.arange(len(target_mask)) - idx, 2) / sigma**2)
        weight_mask += weight
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(target_mask, label="target mask")
        ax.plot(weight_mask, label="weight mask")
        ax.set_xlabel("samples")
        ax.set_ylabel("weight")
        ax.legend(loc="best")
        plt.show()
    return weight_mask
