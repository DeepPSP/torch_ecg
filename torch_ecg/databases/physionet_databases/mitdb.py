# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

from ...cfg import CFG, DEFAULTS
from ...utils.misc import get_record_list_recursive3, add_docstring
from ...utils.utils_interval import generalized_intervals_intersection
from ..base import (
    BeatAnn,
    DataBaseInfo,
    PhysioNetDataBase,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
)


__all__ = [
    "MITDB",
]


_MITDB_INFO = DataBaseInfo(
    title="""
    MIT-BIH Arrhythmia Database
    """,
    about="""
    1. contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects.
    2. recordings were digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range.
    3. annotations contains:

        - beat-wise or finer (e.g. annotations of flutter wave) annotations, accessed via the `symbol` attribute of an `Annotation`.
        - rhythm annotations, accessed via the `aux_note` attribute of an `Annotation`.
    4. Webpage of the database on PhysioNet [1]_.
    """,
    usage=[
        "Beat classification",
        "Rhythm classification (segmentation)",
        "R peaks detection",
    ],
    references=[
        "https://physionet.org/content/mitdb/",
    ],
    doi=[
        "10.1109/51.932724",
        "10.13026/C2F305",
    ],
)


@add_docstring(_MITDB_INFO.format_database_docstring(), mode="prepend")
class MITDB(PhysioNetDataBase):
    """
    Parameters
    ----------
    db_dir : str or pathlib.Path, optional
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : str, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "MITDB"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="mitdb",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 360
        self.data_ext = "dat"
        self.data_pattern = "^[\\d]{3}$"
        self.data_pattern_with_ext = f"^[\\d]{{3}}\\.{self.data_ext}$"
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

        # records have different lead names
        # therefore, self.all_leads should not be set
        # otherwise, it will cause problems when loading data using `self.load_data`
        self._all_leads = ["MLII", "V1", "V2", "V4", "V5"]

        self._ls_rec()

        self._stats = pd.DataFrame()
        self._stats_columns = ["record", "beat_num", "beat_type_num", "rhythm_len"]
        self._aggregate_stats()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        subsample = self._subsample
        self._subsample = None  # so that no subsampling in super()._ls_rec()
        super()._ls_rec()
        # filters out records with names not matching `self.data_pattern`
        if len(self._df_records) > 0:
            self._df_records = self._df_records[
                self._df_records.index.str.match(self.data_pattern)
            ]
        if len(self._all_records) == 0:
            self._df_records = pd.DataFrame()
            self._df_records["path"] = get_record_list_recursive3(
                self.db_dir, self.data_pattern_with_ext, relative=False
            )
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.stem
            )
            self._df_records.set_index("record", inplace=True)
        if subsample is not None:
            size = min(
                len(self._df_records),
                max(1, int(round(subsample * len(self._df_records)))),
            )
            self.logger.debug(
                f"subsample `{size}` records from `{len(self._df_records)}`"
            )
            self._df_records = self._df_records.sample(
                n=size, random_state=DEFAULTS.SEED, replace=False
            )
        self._all_records = self._df_records.index.tolist()
        self._subsample = subsample

    def _aggregate_stats(self) -> None:
        """Aggregate statistics for all records in the database."""
        self._stats = pd.DataFrame(columns=self._stats_columns)
        if len(self) == 0:
            return
        with tqdm(
            range(len(self)),
            desc="Aggregating stats",
            unit="record",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=(self.verbose < 1),
        ) as pbar:
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
                self._stats = pd.concat(
                    [
                        self._stats,
                        pd.DataFrame(
                            [
                                [
                                    self._all_records[idx],
                                    beat_num,
                                    beat_type_num,
                                    rhythm_len,
                                ]
                            ],
                            columns=self._stats_columns,
                        ),
                    ],
                    ignore_index=True,
                )

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "intervals",
        rhythm_types: Optional[Sequence[str]] = None,
        beat_format: str = "beat",
        beat_types: Optional[Sequence[str]] = None,
        keep_original: bool = False,
    ) -> dict:
        """Load rhythm and beat annotations of the record.

        Rhythm and beat annotations are stored in the `aux_note`, `symbol`
        attributes of corresponding annotation files.
        NOTE that qrs annotations (.qrs files) do NOT contain any rhythm annotations.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the annotations to be loaded.
        sampto : int, optional
            End index of the annotations to be loaded.
        rhythm_format : {"interval", "mask"}, optional
            Format of returned annotation, by default "interval",
            case insensitive.
        rhythm_types : list of str, optional
            Defaults to `self.rhythm_types`.
            If is not None, only the rhythm annotations
            with the specified types will be returned.
        beat_format : {"beat", "dict"}, optional
            Format of returned annotation, by default "beat",
            case insensitive.
        beat_types : List[str], optional
            Beat types to be loaded, by default `self.beat_types`.
            If is not None, only the beat annotations
            with the specified types will be returned.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        ann : dict
            The annotations of ``rhythm`` and ``beat``, with
            ``rhythm`` annotatoins in the format of intervals, or mask;
            ``beat`` annotations in the format of dict or
            :class:`~torch_ecg.databases.BeatAnn`.

        """
        assert rhythm_format.lower() in [
            "intervals",
            "mask",
        ], f"`rhythm_format` must be one of ['intervals', 'mask'], got {rhythm_format}"
        assert beat_format.lower() in [
            "beat",
            "dict",
        ], f"`beat_format` must be one of ['beat', 'dict'], got {beat_format}"
        fp = str(self.get_absolute_path(rec))
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

        # if not extended_beats:
        #     beat_ann = [b for b in beat_ann if b.symbol in self.beat_types]

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
        """Load rhythm annotations of the record.

        Rhythm annotations are stored in the `aux_note` attribute
        of corresponding annotation files.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the annotations to be loaded.
        sampto : int, optional
            End index of the annotations to be loaded.
        rhythm_format : {"interval", "mask"}, optional
            Format of returned annotation, by default "interval",
            case insensitive.
        rhythm_types : list of str, optional
            Defaults to `self.rhythm_types`.
            If is not None, only the rhythm annotations
            with the specified types will be returned.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        ann : dict or numpy.ndarray
            Annotations in the format of intervals or mask.

        """
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
        """Load beat annotations of the record.

        Beat annotations are stored in the `symbol` attribute
        of corresponding annotation files.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the annotations to be loaded.
        sampto : int, optional
            End index of the annotations to be loaded.
        beat_format : {"beat", "dict"}, optional
            Format of returned annotation, by default "beat",
            case insensitive.
        beat_types : List[str], optional
            Beat types to be loaded, by default `self.beat_types`.
            If is not None, only the beat annotations
            with the specified types will be returned.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        ann : dict or list
            Locations (indices) of the all the
            beat types ("A", "N", "Q", "V",).

        """
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
        """Load rpeak indices of the record.

        Rpeak indices, or equivalently qrs complex locations,
        are stored in the `symbol` attribute of corresponding annotation files,
        regardless of their beat types.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the annotations to be loaded.
        sampto : int, optional
            End index of the annotations to be loaded.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        rpeak_inds : numpy.ndarray
            Locations (indices) of the all the rpeaks (qrs complexes).

        """
        fp = str(self.get_absolute_path(rec))
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
        """Get the names of the leads contained in the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        List[str]
            A list of names of the leads contained in the record.

        """
        return wfdb.rdheader(str(self.get_absolute_path(rec))).sig_name

    @property
    def df_stats(self) -> pd.DataFrame:
        """DataFrame of the statistics of the dataset."""
        if self._stats.empty:
            self._aggregate_stats()
        return self._stats

    @property
    def df_stats_expanded(self) -> pd.DataFrame:
        """Expanded DataFrame of the statistics of the dataset."""
        df = self.df_stats.copy(deep=True)
        for bt in self.beat_types:
            df[f"beat_{bt}"] = df["beat_type_num"].apply(lambda d: d.get(bt, 0))
        for rt in self.rhythm_types:
            df[f"rhythm_{rt}"] = df["rhythm_len"].apply(lambda d: d.get(rt, 0))
        return df.drop(columns=["beat_num", "beat_type_num", "rhythm_len"])

    @property
    def df_stats_expanded_boolean(self) -> pd.DataFrame:
        """Expanded DataFrame of the statistics of the dataset,
        with boolean values.
        """
        df = self.df_stats_expanded.copy(deep=True)
        for col in df.columns:
            if col == "record":
                continue
            df[col] = df[col].apply(lambda x: int(x > 0))
        return df

    @property
    def db_stats(self) -> Dict[str, Dict[str, int]]:
        """Dictionary of the statistics of the dataset."""
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
        """Categorize records by specific attributes.

        Parameters
        ----------
        by : {"beat", "rhythm"}
            The attribute to categorize the records,
            case insensitive.

        Returns
        -------
        dict
            A dict of lists of record names.

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
        """Dictionary of records with specific beat types."""
        return self._categorize_records("beat")

    @property
    def rhythm_types_records(self) -> Dict[str, List[str]]:
        """Dictionary of records with specific rhythm types."""
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
    ) -> None:
        """Not implemented."""
        raise NotImplementedError

    @property
    def database_info(self) -> DataBaseInfo:
        return _MITDB_INFO
