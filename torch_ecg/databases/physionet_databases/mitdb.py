# -*- coding: utf-8 -*-
"""
"""

from collections import defaultdict, Counter
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Union

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

from ...cfg import CFG
from ...utils.misc import get_record_list_recursive3
from ...utils.utils_interval import generalized_intervals_intersection
from ..base import (
    BeatAnn,
    PhysioNetDataBase,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
)

__all__ = [
    "MITDB",
]


class MITDB(PhysioNetDataBase):
    """NOT finished,

    MIT-BIH Arrhythmia Database

    ABOUT mitdb
    -----------
    1. contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects.
    2. recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
    3. annotations contains:
        - beat-wise or finer (e.g. annotations of flutter wave) annotations, accessed via the `symbol` attribute of an `Annotation`.
        - rhythm annotations, accessed via the `aux_note` attribute of an `Annotation`.

    NOTE
    ----

    ISSUES
    ------

    Usage
    -----
    1. Beat classification
    2. rhythm classification (segmentation)
    3. R peaks detection

    References
    ----------
    1. <a name="ref1"></a> https://physionet.org/content/mitdb/1.0.0/

    """

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        db_dir: str or Path, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="mitdb",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 360
        self.data_ext = "dat"
        self.ann_ext = "atr"

        self.beat_types_extended = list("""!"+/AEFJLNQRSV[]aefjx|~""")
        self.nonbeat_types = [
            item
            for item in self.beat_types_extended
            if item in WFDB_Non_Beat_Annotations
        ]
        self.beat_types = [
            item for item in self.beat_types_extended if item in WFDB_Beat_Annotations
        ]
        self.beat_types_map = {item: i for i, item in enumerate(self.beat_types)}
        self.beat_types_extended_map = {
            item: i for i, item in enumerate(self.beat_types_extended)
        }
        self.rhythm_types = [
            "(AB",
            "(AFIB",
            "(AFL",
            "(B",
            "(BII",
            "(IVR",
            "(N",
            "(NOD",
            "(P",
            "(PREX",
            "(SBR",
            "(SVTA",
            "(T",
            "(VFL",
            "(VT",
            "MISSB",
            "PSE",
            "TS",
        ]
        self.rhythm_types = [
            rt.lstrip("(") for rt in self.rhythm_types if rt in WFDB_Rhythm_Annotations
        ]
        self.rhythm_types_map = {rt: idx for idx, rt in enumerate(self.rhythm_types)}
        self._rhythm_ignore_index = -100

        self.all_leads = ["MLII", "V1", "V2", "V4", "V5"]

        self._ls_rec()

        self._stats = pd.DataFrame()
        self._stats_columns = ["record", "beat_num", "beat_type_num", "rhythm_len"]
        self._aggregate_stats()

    def _ls_rec(self) -> NoReturn:
        """ """
        super()._ls_rec()
        if len(self._all_records) == 0:
            self._all_records = get_record_list_recursive3(
                self.db_dir, f"^[\\d]{{3}}.{self.data_ext}$"
            )

    def _aggregate_stats(self) -> NoReturn:
        """ """
        self._stats = pd.DataFrame(columns=self._stats_columns)
        with tqdm(range(len(self)), desc="Aggregating stats", unit="records") as pbar:
            for idx in pbar:
                rec_ann = self.load_ann(idx)
                beat_type_num = {
                    k: v
                    for k, v in Counter(
                        [item.symbol for item in rec_ann["beat"]]
                    ).most_common()
                }
                beat_num = sum(beat_type_num.values())
                rhythm_len = {
                    k: sum([itv[1] - itv[0] for itv in v])
                    for k, v in rec_ann["rhythm"].items()
                }
                self._stats = self._stats.append(
                    dict(
                        record=self[idx],
                        beat_num=beat_num,
                        beat_type_num=beat_type_num,
                        rhythm_len=rhythm_len,
                    ),
                    ignore_index=True,
                )

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, List[str]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: str = "mV",
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """

        load physical (converted from digital) ECG data,
        which is more understandable for humans

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        leads: str or list of str, optional,
            the leads to load
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        data_format: str, default "channel_first",
            format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = str(self.db_dir / rec)
        if not leads:
            _leads = self._get_lead_names(rec)
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        assert set(_leads).issubset(self.all_leads)
        # p_signal in the format of "lead_last", and in units "mV"
        data = wfdb.rdrecord(
            fp,
            sampfrom=sampfrom or 0,
            sampto=sampto,
            physical=True,
            channel_names=_leads,
        ).p_signal
        if units.lower() in ["μv", "uv"]:
            data = 1000 * data
        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=0)
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        return data

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "intervals",
        rhythm_types: Optional[Sequence[str]] = None,
        beat_format: str = "beat",
        beat_types: Optional[Sequence[str]] = None,
        extended_beats: bool = False,
        keep_original: bool = False,
    ) -> dict:
        """

        load rhythm and beat annotations,
        which are stored in the `aux_note`, `symbol` attributes of corresponding annotation files.
        NOTE that qrs annotations (.qrs files) do NOT contain any rhythm annotations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        rhythm_format: str, default "intervals", case insensitive,
            format of returned annotation, can also be "mask"
        rhythm_types: list of str, optional,
            defaults to `self.rhythm_types`
            if not None, only the rhythm annotations with the specified types will be returned
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        beat_types: list of str, optional,
            defaults to `self.beat_types`
            if not None, only the beat annotations with the specified types will be returned
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict,
            the annotations of `rhythm` and `beat`, with
            `rhythm` annotatoins in the format of `intervals`, or `mask`;
            `beat` annotations in the format of `dict` or `BeatAnn`

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert rhythm_format.lower() in [
            "intervals",
            "mask",
        ], f"`rhythm_format` must be one of ['intervals', 'mask'], got {rhythm_format}"
        assert beat_format.lower() in [
            "beat",
            "dict",
        ], f"`beat_format` must be one of ['beat', 'dict'], got {beat_format}"
        fp = str(self.db_dir / rec)
        wfdb_ann = wfdb.rdann(fp, extension=self.ann_ext)
        header = wfdb.rdheader(fp)
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        sample_inds = wfdb_ann.sample
        indices = np.where((sample_inds >= sf) & (sample_inds < st))[0]

        if beat_types is None:
            beat_types = self.beat_types

        beat_ann = [
            BeatAnn(i, s)
            for i, s in zip(sample_inds[indices], np.array(wfdb_ann.symbol)[indices])
            if s in beat_types
        ]

        if rhythm_types is None:
            rhythm_types = self.rhythm_types
            rhythm_types_map = self.rhythm_types_map
        else:
            rhythm_types = [rt.lstrip("(") for rt in rhythm_types]
            rhythm_types_map = {rt: idx for idx, rt in enumerate(rhythm_types)}

        rhythm_intervals = defaultdict(list)
        start_idx, rhythm = None, None
        for ra, si in zip(wfdb_ann.aux_note, sample_inds):
            ra = ra.rstrip("\x00").lstrip("(")
            if ra in rhythm_types:
                if start_idx is not None:
                    rhythm_intervals[rhythm].append([start_idx, si])
                start_idx = si
                rhythm = ra.lstrip("(")
        if start_idx is not None:
            rhythm_intervals[rhythm].append([start_idx, si])
        rhythm_intervals = {
            k: np.array(generalized_intervals_intersection(v, [[sf, st]]))
            for k, v in rhythm_intervals.items()
        }
        if rhythm_format.lower() == "mask":
            rhythm_mask = np.full((st - sf,), self._rhythm_ignore_index, dtype=int)
            for k, v in rhythm_intervals.items():
                for itv in v:
                    rhythm_mask[itv[0] - sf : itv[1] - sf] = self.rhythm_types_map[k]

        if not keep_original:
            rhythm_intervals = {k: v - sf for k, v in rhythm_intervals.items()}
            for b in beat_ann:
                b.index -= sf

        if not extended_beats:
            beat_ann = [b for b in beat_ann if b.symbol in self.beat_types]

        if beat_format.lower() == "dict":
            beat_ann = {
                s: np.array([b.index for b in beat_ann if b.symbol == s], dtype=int)
                for s in self.beat_types_extended
            }
            beat_ann = {k: v for k, v in beat_ann.items() if len(v) > 0}

        ann = {}
        ann["beat"] = beat_ann
        if rhythm_format.lower() == "intervals":
            ann["rhythm"] = rhythm_intervals
        else:
            ann["rhythm"] = rhythm_mask
        return ann

    def load_rhythm_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "intervals",
        rhythm_types: Optional[Sequence[str]] = None,
        keep_original: bool = False,
    ) -> Union[Dict[str, list], np.ndarray]:
        """

        load rhythm annotations,
        which are stored in the `aux_note` attribute of corresponding annotation files.

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        rhythm_format: str, default "intervals", case insensitive,
            format of returned annotation, can also be "mask"
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        rhythm_types: list of str, optional,
            defaults to `self.rhythm_types`
            if not None, only the rhythm annotations with the specified types will be returned
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict or ndarray,
            the annotations in the format of intervals, or in the format of mask

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self.load_ann(
            rec,
            sampfrom,
            sampto,
            rhythm_format=rhythm_format,
            rhythm_types=rhythm_types,
            keep_original=keep_original,
        )["rhythm"]

    def load_beat_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        beat_format: str = "beat",
        beat_types: Optional[Sequence[str]] = None,
        keep_original: bool = False,
    ) -> Union[Dict[str, np.ndarray], List[BeatAnn]]:
        """

        load beat annotations,
        which are stored in the `symbol` attribute of corresponding annotation files

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        beat_types: list of str, optional,
            defaults to `self.beat_types`
            if not None, only the beat annotations with the specified types will be returned
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict or list,
            locations (indices) of the all the beat types ("A", "N", "Q", "V",)

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self.load_ann(
            rec,
            sampfrom,
            sampto,
            beat_format=beat_format,
            beat_types=beat_types,
            keep_original=keep_original,
        )["beat"]

    def load_rpeak_indices(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
    ) -> np.ndarray:
        """

        load rpeak indices, or equivalently qrs complex locations,
        which are stored in the `symbol` attribute of corresponding annotation files,
        regardless of their beat types,

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        rpeak_inds, ndarray,
            locations (indices) of the all the rpeaks (qrs complexes)

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = str(self.db_dir / rec)
        wfdb_ann = wfdb.rdann(fp, extension=self.ann_ext)
        header = wfdb.rdheader(fp)
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        rpeak_inds = wfdb_ann.sample
        indices = np.where(
            (rpeak_inds >= sf)
            & (rpeak_inds < st)
            & (np.isin(wfdb_ann.symbol, self.beat_types))
        )[0]
        rpeak_inds = rpeak_inds[indices]
        if not keep_original:
            rpeak_inds -= sf
        return rpeak_inds

    def _get_lead_names(self, rec: Union[str, int]) -> List[str]:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        list of str:
            a list of names of the leads contained in the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = str(self.db_dir / rec)
        return wfdb.rdheader(fp).sig_name

    @property
    def df_stats(self) -> pd.DataFrame:
        """ """
        if self._stats.empty:
            self._aggregate_stats()
        return self._stats

    @property
    def df_stats_expanded(self) -> pd.DataFrame:
        """ """
        df = self.df_stats.copy(deep=True)
        for bt in self.beat_types:
            df[f"beat_{bt}"] = df["beat_type_num"].apply(lambda d: d.get(bt, 0))
        for rt in self.rhythm_types:
            df[f"rhythm_{rt}"] = df["rhythm_len"].apply(lambda d: d.get(rt, 0))
        return df.drop(columns=["beat_num", "beat_type_num", "rhythm_len"])

    @property
    def df_stats_expanded_boolean(self) -> pd.DataFrame:
        """ """
        df = self.df_stats_expanded.copy(deep=True)
        for col in df.columns:
            if col == "record":
                continue
            df[col] = df[col].apply(lambda x: int(x > 0))
        return df

    @property
    def db_stats(self) -> Dict[str, Dict[str, int]]:
        """ """
        if self._stats.empty:
            self._aggregate_stats()
        rhythm_len = defaultdict(int)
        for rl_dict in self._stats["rhythm_len"]:
            for k, v in rl_dict.items():
                rhythm_len[k] += v
        beat_type_num = defaultdict(int)
        for btn_dict in self._stats["beat_type_num"]:
            for k, v in btn_dict.items():
                beat_type_num[k] += v
        return CFG(rhythm_len=dict(rhythm_len), beat_type_num=dict(beat_type_num))

    def _categorize_records(self, by: str) -> Dict[str, List[str]]:
        """

        categorize records by specific attributes

        Parameters
        ----------
        by: str,
            the attribute to categorize the records,
            can be one of "beat", "rhythm", case insensitive

        Returns
        -------
        dict of list of str:
            a dict of lists of record names,

        """
        assert by.lower() in [
            "beat",
            "rhythm",
        ], f"`by` should be one of 'beat' or 'rhythm', but got {by}"
        key = dict(beat="beat_type_num", rhythm="rhythm_len")[by.lower()]
        return CFG(
            {
                item: [
                    row["record"]
                    for _, row in self.df_stats.iterrows()
                    if item in row[key]
                ]
                for item in self.db_stats[key]
            }
        )

    @property
    def beat_types_records(self) -> Dict[str, List[str]]:
        """ """
        return self._categorize_records("beat")

    @property
    def rhythm_types_records(self) -> Dict[str, List[str]]:
        """ """
        return self._categorize_records("rhythm")

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[Dict[str, np.ndarray]] = None,
        beat_ann: Optional[Dict[str, np.ndarray]] = None,
        rpeak_inds: Optional[Union[Sequence[int], np.ndarray]] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[int, List[int]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        same_range: bool = False,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        raise NotImplementedError
