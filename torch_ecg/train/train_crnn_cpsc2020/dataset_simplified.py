"""
simplified data generator for feeding data into pytorch models,
with offline part all removed

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] Tan, Jen Hong, et al. "Application of stacked convolutional and long short-term memory network for accurate identification of CAD ECG signals." Computers in biology and medicine 94 (2018): 19-26.
[3] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
"""
import os, sys, re, json
from random import shuffle, randint, uniform, sample
from copy import deepcopy
from functools import reduce
from itertools import product, repeat
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import signal as SS
from scipy.io import loadmat, savemat
from easydict import EasyDict as ED
# try:
#     from tqdm.auto import tqdm
# except ModuleNotFoundError:
#     from tqdm import tqdm
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset

from torch_ecg.train.database_reader.database_reader.other_databases import CPSC2020 as CR
from .cfg import TrainCfg, ModelCfg, PreprocCfg
from .utils import (
    mask_to_intervals, list_sum,
    gen_baseline_wander,
    get_record_list_recursive3,
)

if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = np.float64
else:
    _DTYPE = np.float32


__all__ = [
    "CPSC2020",
]


class CPSC2020(Dataset):
    """

    data generator for deep learning models,

    strategy:
    ---------
    1. slice each record into short segments of length `TrainCfg.input_len`,
    and of overlap length `TrainCfg.overlap_len` around premature beats
    2. do augmentations for premature segments
    """
    __DEBUG__ = False
    __name__ = "CPSC2020"

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
        if ModelCfg.torch_dtype.lower() == "double":
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.allowed_preproc = PreprocCfg.preproc

        self.all_classes = self.config.classes
        self.n_classes = len(self.config.classes)

        self.training = training
        split_res = self.reader.train_test_split_rec(
            test_rec_num=self.config.test_rec_num
        )
        self.__data_aug = self.training

        self.seglen = self.config.input_len  # alias, for simplicity

        # create directories if needed
        # segments_dir for sliced segments
        self.segments_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)

        if self.config.model_name.lower() in ["crnn", "seq_lab"]:
            # for classification, or for sequence labeling
            self.segments_dirs = ED()
            self.__all_segments = ED()
            self.segments_json = os.path.join(self.segments_dir, "crnn_segments.json")
            self._ls_segments()

            if self.training:
                self.segments = list_sum([self.__all_segments[rec] for rec in split_res.train])
                shuffle(self.segments)
            else:
                self.segments = list_sum([self.__all_segments[rec] for rec in split_res.test])
        # elif self.config.model_name.lower() == "od":  # object detection
        #     pass
        else:
            raise NotImplementedError(f"data generator for model \042{self.config.model_name}\042 not implemented")

        if self.config.bw:
            self._n_bw_choices = len(self.config.bw_ampl_ratio)
            self._n_gn_choices = len(self.config.bw_gaussian)


    def _ls_segments(self) -> NoReturn:
        """ finished, checked,
        """
        for item in ["data", "ann"]:
            self.segments_dirs[item] = ED()
            for rec in self.reader.all_records:
                self.segments_dirs[item][rec] = os.path.join(self.segments_dir, item, rec)
                os.makedirs(self.segments_dirs[item][rec], exist_ok=True)
        if os.path.isfile(self.segments_json):
            with open(self.segments_json, "r") as f:
                self.__all_segments = json.load(f)
            return
        print(f"please allow the reader a few minutes to collect the segments from {self.segments_dir}...")
        seg_filename_pattern = f"S\d{{2}}_\d{{7}}{self.reader.rec_ext}"
        self.__all_segments = ED({
            rec: get_record_list_recursive3(self.segments_dirs.data[rec], seg_filename_pattern) \
                for rec in self.reader.all_records
        })
        if all([len(self.__all_segments[rec])>0 for rec in self.reader.all_records]):
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    @property
    def all_segments(self):
        return self.__all_segments


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, checked,
        """
        seg_name = self.segments[index]
        seg_data = self._load_seg_data(seg_name)
        if self.config.model_name.lower() == "crnn":
            seg_label = self._load_seg_label(seg_name)
        elif self.config.model_name.lower() == "seq_lab":
            seg_label = self._load_seg_seq_lab(
                seg=seg_name,
                reduction=self.config.seq_lab_reduction,
            )
        # seg_ampl = np.max(seg_data) - np.min(seg_data)
        seg_ampl = self._get_seg_ampl(seg_data)
        # spb_indices = ann["SPB_indices"]
        # pvc_indices = ann["PVC_indices"]
        if self.__data_aug:
            if self.config.bw:
                ar = self.config.bw_ampl_ratio[randint(0, self._n_bw_choices-1)]
                gm, gs = self.config.bw_gaussian[randint(0, self._n_gn_choices-1)]
                bw_ampl = ar * seg_ampl
                g_ampl = gm * seg_ampl
                bw = gen_baseline_wander(
                    siglen=self.seglen,
                    fs=self.config.fs,
                    bw_fs=self.config.bw_fs,
                    amplitude=bw_ampl,
                    amplitude_mean=gm,
                    amplitude_std=gs,
                )
                seg_data = seg_data + bw
            if len(self.config.flip) > 0:
                sign = sample(self.config.flip, 1)[0]
                seg_data *= sign
            if self.config.random_normalize:
                rn_mean = uniform(
                    self.config.random_normalize_mean[0],
                    self.config.random_normalize_mean[1],
                )
                rn_std = uniform(
                    self.config.random_normalize_std[0],
                    self.config.random_normalize_std[1],
                )
                seg_data = (seg_data-np.mean(seg_data)+rn_mean) / np.std(seg_data) * rn_std
            if self.config.label_smoothing > 0:
                seg_label = (1 - self.config.label_smoothing) * seg_label \
                    + self.config.label_smoothing / self.n_classes

        if self.__DEBUG__:
            self.reader.plot(
                rec="",  # unnecessary indeed
                data=seg_data,
                ann=self._load_seg_beat_ann(seg_name),
                ticks_granularity=2,
            )
        
        seg_data = seg_data.reshape((self.config.n_leads, self.seglen))

        return seg_data, seg_label


    def __len__(self) -> int:
        """
        """
        return len(self.segments)


    def _get_seg_ampl(self, seg_data:np.ndarray, window:int=80) -> float:
        """ finished, checked,

        get amplitude of a segment

        Parameters:
        -----------
        seg_data: ndarray,
            data of the segment
        window: int, default 80 (corr. to 200ms),
            window length of a window for computing amplitude, with units in number of sample points

        Returns:
        --------
        ampl: float,
            amplitude of `seg_data`
        """
        half_window = window // 2
        ampl = 0
        for idx in range(len(seg_data)//half_window-1):
            s = seg_data[idx*half_window: idx*half_window+window]
            ampl = max(ampl, np.max(s)-np.min(s))
        return ampl


    def _get_seg_data_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns:
        --------
        fp: str,
            path of the data file of the segment
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "data", rec, f"{seg}{self.reader.rec_ext}")
        return fp


    def _get_seg_ann_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns:
        --------
        fp: str,
            path of the annotation file of the segment
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "ann", rec, f"{seg}{self.reader.rec_ext}")
        return fp


    def _load_seg_data(self, seg:str) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns:
        --------
        seg_data: ndarray,
            data of the segment, of shape (self.seglen,)
        """
        seg_data_fp = self._get_seg_data_path(seg)
        seg_data = loadmat(seg_data_fp)["ecg"].squeeze()
        return seg_data


    def _load_seg_label(self, seg:str) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns:
        --------
        seg_label: ndarray,
            label of the segment, of shape (self.n_classes,)
        """
        seg_ann_fp = self._get_seg_ann_path(seg)
        seg_label = loadmat(seg_ann_fp)["label"].squeeze()
        return seg_label


    def _load_seg_beat_ann(self, seg:str) -> Dict[str, np.ndarray]:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns:
        --------
        seg_beat_ann: dict,
            "SPB_indices", "PVC_indices", each of ndarray values
        """
        seg_ann_fp = self._get_seg_ann_path(seg)
        seg_beat_ann = loadmat(seg_ann_fp)
        seg_beat_ann = {
            k:v.flatten() for k,v in seg_beat_ann.items() if k in ["SPB_indices", "PVC_indices"]
        }
        return seg_beat_ann


    def _load_seg_seq_lab(self, seg:str, reduction:int=8) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"
        reduction: int, default 8,
            reduction (granularity) of length of the model output,
            compared to the original signal length

        Returns:
        --------
        seq_lab: np.ndarray,
            label of the sequence,
            of shape (self.seglen//reduction, self.n_classes)
        """
        seg_beat_ann = {
            k: np.round(v/reduction).astype(int) \
                for k,v in self._load_seg_beat_ann(seg).items()
        }
        bias_thr = int(round(self.config.bias_thr / reduction))
        seq_lab = np.zeros(
            shape=(self.seglen//reduction, self.n_classes),
            dtype=_DTYPE,
        )
        for p in seg_beat_ann["SPB_indices"]:
            start_idx = max(0, p-bias_thr)
            end_idx = min(seq_lab.shape[0], p+bias_thr+1)
            seq_lab[start_idx:end_idx, self.config.classes.index("S")] = 1
        for p in seg_beat_ann["PVC_indices"]:
            start_idx = max(0, p-bias_thr)
            end_idx = min(seq_lab.shape[0], p+bias_thr+1)
            seq_lab[start_idx:end_idx, self.config.classes.index("V")] = 1

        return seq_lab


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False

    def enable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = True
