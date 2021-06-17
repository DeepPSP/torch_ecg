"""
data generator for feeding data into pytorch models

Augmentations:
--------------
    - label smoothing (label, on the fly)
    - flip (signal, on the fly)
    - (re-)normalize to random mean and std (signal, on the fly)
    - baseline wander (signal, on the fly, combination of sinusoidal noise of several different frequencies, together with an optional Gaussian noise)
    - sinusoidal noise (signal, on the fly, done in baseline wander)
    - Gaussian noise (signal, on the fly, done in baseline wander)
    - stretch and compress (signal, offline)

Issues:
-------
1. flat segments are found in the original CPSC dataset, e.g. 170*4000 to 185*4000,
which could be checked for example via:
>>> raw_data = ds.reader.load_data("A02",keep_dim=False)
>>> flat_segs = []
>>> for idx in range(len(raw_data)//ds.seglen):
>>>     seg_data = raw_data[idx*ds.seglen:(idx+1)*ds.seglen]
>>>     if ds._get_seg_ampl(seg_data) < 0.1:
>>>         flat_segs.append(idx)
>>>     print(f"{idx+1}/{len(raw_data)//ds.seglen}", end="\r")
>>> ds.reader.plot(rec="A02", sampfrom=169*ds.seglen, sampto=186*ds.seglen)
also for sliced segments via:
>>> flat_segs = {rec:[] for rec in ds.reader.all_records}
>>> valid_segs = {rec:[] for rec in ds.reader.all_records}
>>> for i, rec in enumerate(ds.reader.all_records):
>>>     for idx, seg in enumerate(ds.all_segments[rec]):
>>>         seg_data = ds._load_seg_data(seg)
>>>         if ds._get_seg_ampl(seg_data) < 0.1:
>>>             flat_segs[rec].append(seg)
>>>         else:
>>>             valid_segs[rec].append(seg)
>>>         print(f"{idx+1}/{len(ds.all_segments[rec])} @ {i+1}/{len(ds.reader.all_records)}", end="\r")
UPDATE: using `ecg_denoise` as follows:
>>> valid_segs = {rec:[] for rec in ds.all_segments.keys()}
>>> invalid_segs = []
>>> for i, (rec, l_seg) in enumerate(ds.all_segments.items()):
>>>     for idx, seg in enumerate(l_seg):
>>>         if ecg_denoise(ds._load_seg_data(seg), ds.reader.fs ,{"ampl_min":0.15}) == [[0,ds.seglen]]:
>>>             valid_segs[rec].append(seg)
>>>         else:
>>>             invalid_segs.append(seg)
>>>         print(f"{idx+1}/{len(l_seg)} @ {i+1}/{len(ds.all_segments)}", end="\r")

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
from sklearn.preprocessing import StandardScaler

from train.database_reader.database_reader.other_databases import CPSC2020 as CR
from train.train_crnn_cpsc2020 import signal_processing as SP
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

    strategy
    --------
    1. slice each record into short segments of length `TrainCfg.input_len`,
    and of overlap length `TrainCfg.overlap_len` around premature beats
    2. do augmentations for premature segments
    """
    __DEBUG__ = False
    __name__ = "CPSC2020"

    def __init__(self, config:ED, training:bool=True) -> NoReturn:
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
        # preprocess_dir stores pre-processed signals
        self.preprocess_dir = os.path.join(config.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # segments_dir for sliced segments
        self.segments_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)
        # rpeaks_dir for detected r peaks, for optional use
        self.rpeaks_dir = os.path.join(config.db_dir, "rpeaks")
        os.makedirs(self.rpeaks_dir, exist_ok=True)

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

        Parameters
        ----------
        seg_data: ndarray,
            data of the segment
        window: int, default 80 (corr. to 200ms),
            window length of a window for computing amplitude, with units in number of sample points

        Returns
        -------
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

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns
        -------
        fp: str,
            path of the data file of the segment
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "data", rec, f"{seg}{self.reader.rec_ext}")
        return fp


    def _get_seg_ann_path(self, seg:str) -> str:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns
        -------
        fp: str,
            path of the annotation file of the segment
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "ann", rec, f"{seg}{self.reader.rec_ext}")
        return fp


    def _load_seg_data(self, seg:str) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns
        -------
        seg_data: ndarray,
            data of the segment, of shape (self.seglen,)
        """
        seg_data_fp = self._get_seg_data_path(seg)
        seg_data = loadmat(seg_data_fp)["ecg"].squeeze()
        return seg_data


    def _load_seg_label(self, seg:str) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns
        -------
        seg_label: ndarray,
            label of the segment, of shape (self.n_classes,)
        """
        seg_ann_fp = self._get_seg_ann_path(seg)
        seg_label = loadmat(seg_ann_fp)["label"].squeeze()
        return seg_label


    def _load_seg_beat_ann(self, seg:str) -> Dict[str, np.ndarray]:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"

        Returns
        -------
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

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"
        reduction: int, default 8,
            reduction (granularity) of length of the model output,
            compared to the original signal length

        Returns
        -------
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
        # if force_recompute:
        #     self.__all_segments = ED({rec: [] for rec in self.reader.all_records})
        self._preprocess_data(
            self.allowed_preproc,
            force_recompute=force_recompute,
            verbose=verbose,
        )
        self._slice_data(
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
        config = deepcopy(PreprocCfg)
        config.preproc = preproc
        for idx, rec in enumerate(self.reader.all_records):
            self._preprocess_one_record(
                rec=rec,
                config=config,
                force_recompute=force_recompute,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")

    def _preprocess_one_record(self, rec:Union[int,str], config:dict, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters
        ----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        config: dict,
            configurations of preprocessing
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        # format save path
        save_fp = ED()
        rec_name = self.reader._get_rec_name(rec)
        suffix = self._get_rec_suffix(config.preproc)
        save_fp.data = os.path.join(self.preprocess_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
        save_fp.rpeaks = os.path.join(self.rpeaks_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
        if (not force_recompute) and os.path.isfile(save_fp.data) and os.path.isfile(save_fp.rpeaks):
            return
        # perform pre-process
        pps = SP.parallel_preprocess_signal(
            self.reader.load_data(rec, keep_dim=False),
            fs=self.reader.fs,
            config=config,
            verbose=verbose,
        )
        # `rpeaks_skip_dist` useless for `seq_lab_detect`, as already set internally
        # pps['rpeaks'] = pps['rpeaks'][np.where( (pps['rpeaks']>=config.rpeaks_skip_dist) & (pps['rpeaks']<len(pps['filtered_ecg'])-config.rpeaks_skip_dist) )[0]]
        # save mat, keep in accordance with original mat files
        savemat(save_fp.data, {'ecg': np.atleast_2d(pps['filtered_ecg']).T}, format='5')
        savemat(save_fp.rpeaks, {'rpeaks': np.atleast_2d(pps['rpeaks']).T}, format='5')

    def _normalize_preprocess_names(self, preproc:List[str], ensure_nonempty:bool) -> List[str]:
        """ finished, checked

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
            'normalized' list of preprocess types
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
        suffix = '-'.join(sorted([item.lower() for item in operations]))
        return suffix


    def _slice_data(self, force_recompute:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        slice all records into segments of length `self.config.input_len`, i.e. `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        verbose: int, default 0,
            print verbosity
        """
        for idx,rec in enumerate(self.reader.all_records):
            self._slice_one_record(
                rec=rec,
                force_recompute=force_recompute,
                update_segments_json=False,
                verbose=verbose,
            )
            if verbose >= 1:
                print(f"{idx+1}/{len(self.reader.all_records)} records", end="\r")
        if force_recompute:
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)

    def _slice_one_record(self, rec:Union[int,str], force_recompute:bool=False, update_segments_json:bool=False, verbose:int=0) -> NoReturn:
        """ finished, checked,

        slice one record into segments of length `self.config.input_len`, i.e. `self.seglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters
        ----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        update_segments_json: bool, default False,
            if both `force_recompute` and `update_segments_json` are True,
            the file `self.segments_json` will be updated,
            useful when slicing not all records
        verbose: int, default 0,
            print verbosity
        """
        rec_name = self.reader._get_rec_name(rec)
        save_dirs = ED()
        save_dirs.data = self.segments_dirs.data[rec_name]
        save_dirs.ann = self.segments_dirs.ann[rec_name]
        os.makedirs(save_dirs.data, exist_ok=True)
        os.makedirs(save_dirs.data, exist_ok=True)
        if (not force_recompute) and len(self.__all_segments[rec_name]) > 0:
            return
        elif force_recompute:
            self.__all_segments[rec_name] = []

        data = self.reader.load_data(rec, units="mV", keep_dim=False)
        ann = self.reader.load_ann(rec)
        border_dist = int(0.5 * self.config.fs)
        forward_len = self.seglen - self.config.overlap_len

        spb_mask = np.zeros((len(data),), dtype=int)
        pvc_mask = np.zeros((len(data),), dtype=int)
        spb_mask[ann["SPB_indices"]] = 1
        pvc_mask[ann["PVC_indices"]] = 1

        # generate initial segments with no overlap for non premature beats
        n_init_seg = len(data) // self.seglen
        segments = (data[:self.seglen*n_init_seg]).reshape((n_init_seg, self.seglen))
        labels = np.zeros((n_init_seg, self.n_classes))
        labels[..., self.config.class_map["N"]] = 1
        # leave only non premature segments
        non_premature = np.logical_or(spb_mask, pvc_mask)[:self.seglen*n_init_seg]
        non_premature = non_premature.reshape((n_init_seg, self.seglen)).sum(axis=1)
        non_premature = np.where(non_premature == 0)[0]
        segments = segments[non_premature, ...]
        labels = labels[non_premature, ...]
        beat_ann = list(repeat(
            {"SPB_indices":np.array([],dtype=int), "PVC_indices":np.array([],dtype=int)},
            len(non_premature)
        ))

        if verbose >= 1:
            print(f"\nn_init_seg = {n_init_seg}")
            print(f"segments.shape = {segments.shape}")
            print(f"finish extracting non-premature segments, totally {len(non_premature)}")
            print("start doing augmentation...")

        # do data augmentation for premature beats
        # first locate all possible premature segments
        # mask for segment start indices
        premature_mask = np.zeros((len(data),), dtype=int)
        for idx in np.concatenate((ann["SPB_indices"], ann["PVC_indices"])):
            start_idx = max(0, idx-self.seglen+border_dist)
            end_idx = max(start_idx, min(idx-border_dist, len(data)-self.seglen))
            premature_mask[start_idx: end_idx] = 1
        # intervals for allowed start of augmented segments
        premature_intervals = mask_to_intervals(premature_mask, 1)
        n_original = 0
        n_added = 0
        for itv in premature_intervals:
            start_idx = itv[0]
            n_original += 1
            while start_idx < itv[1]:
                # stretch and compress the signal
                if self.config.stretch_compress != 0:
                    sign = sample([-1,1]+[0]*4, 1)[0]
                    if sign != 0:
                        sc_ratio = self.config.stretch_compress
                        sc_ratio = 1 + (uniform(sc_ratio/4, sc_ratio) * sign) / 100
                        sc_len = int(round(sc_ratio * self.seglen))
                        end_idx = start_idx + sc_len
                        aug_seg = data[start_idx: end_idx]
                        aug_seg = SS.resample(x=aug_seg, num=self.seglen).reshape((1,-1))
                    else:
                        end_idx = start_idx + self.seglen
                        # the segment of original signal, with no augmentation
                        aug_seg = data[start_idx: end_idx]
                        sc_ratio = 1

                    seg_label = np.zeros((self.n_classes,))
                    seg_spb_inds = np.where(spb_mask[start_idx: end_idx]==1)[0]
                    seg_pvc_inds = np.where(pvc_mask[start_idx: end_idx]==1)[0]
                    seg_spb_inds = np.round(seg_spb_inds * sc_ratio).astype(int)
                    seg_pvc_inds = np.round(seg_pvc_inds * sc_ratio).astype(int)
                    seg_beat_ann = {
                        "SPB_indices": seg_spb_inds,
                        "PVC_indices": seg_pvc_inds,
                        "interval": np.array([start_idx, end_idx]),
                    }
                    if len(seg_spb_inds) > 0:
                        seg_label[self.config.class_map["S"]] = 1
                    if len(seg_pvc_inds) > 0:
                        seg_label[self.config.class_map["V"]] = 1
                    seg_label = seg_label.reshape((1,-1))
                    segments = np.append(segments, aug_seg.reshape((1,-1)), axis=0)
                    labels = np.append(labels, seg_label.copy(), axis=0)
                    beat_ann.append(seg_beat_ann.copy())
                    n_added += 1
                    if verbose >= 2:
                        print(f"{n_added} aug seg generated, start_idx at {start_idx}/{len(data)}", end="\r")

                # start_idx += forward_len  # should be randomly forwarded
                start_idx += randint(forward_len//4, forward_len)

        if verbose >= 1:
            print(f"\ngenerate {n_added} premature segments out from {n_original} in total via data augmentation")

        # randomly shuffle the data and save into separate files
        seg_inds = list(range(segments.shape[0]))
        shuffle(seg_inds)
        for i, ind in enumerate(seg_inds):
            save_fp = ED()
            seg_name = f"{rec_name.replace('A', 'S')}_{i:07d}"
            save_fp.data = os.path.join(self.segments_dirs.data[rec_name], f"{seg_name}{self.reader.rec_ext}")
            save_fp.ann = os.path.join(self.segments_dirs.ann[rec_name], f"{seg_name}{self.reader.rec_ext}")
            seg = segments[ind, ...]
            # if self._get_seg_ampl(seg) < 0.1:  # drop out flat segments
            #     continue
            if SP.ecg_denoise(seg, self.reader.fs, config={"ampl_min":0.15}) != [[0, self.seglen]]:
                continue
            savemat(save_fp.data, {"ecg": seg}, format="5")
            seg_label = labels[ind, ...]
            seg_beat_ann = beat_ann[ind]
            save_ann_dict = seg_beat_ann.copy()
            save_ann_dict.update({"label": seg_label})
            savemat(save_fp.ann, save_ann_dict, format="5")
            self.__all_segments[rec_name].append(seg_name)
            if verbose >= 2:
                print(f"saving {i+1}/{len(seg_inds)}...", end="\r")
        if update_segments_json:
            with open(self.segments_json, "w") as f:
                json.dump(self.__all_segments, f)


    def plot_seg(self, seg:str, ticks_granularity:int=0, rpeak_inds:Optional[Union[Sequence[int],np.ndarray]]=None) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        seg: str,
            name of the segment, of pattern like "S01_0000193"
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        rpeak_inds: array_like, optional,
            indices of R peaks,
        """
        seg_data = self._load_seg_data(seg)
        seg_beat_ann = self._load_seg_beat_ann(seg)
        rec_name = seg.split("_")[0].replace("S", "A")
        self.reader.plot(
            rec=rec_name,  # unnecessary indeed
            data=seg_data,
            ann=seg_beat_ann,
            ticks_granularity=ticks_granularity,
            rpeak_inds=rpeak_inds,
        )
