# -*- coding: utf-8 -*-
"""
"""

import json
import math
import os
import time
import warnings
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import wfdb
from scipy.signal import resample, resample_poly  # noqa: F401

from ...cfg import CFG
from ...utils.misc import (
    get_record_list_recursive3,
    list_sum,
    ms2samples,
    add_docstring,
)
from ...utils.utils_interval import generalized_intervals_intersection
from ..base import (  # noqa: F401
    DEFAULT_FIG_SIZE_PER_SEC,
    CPSCDataBase,
    PhysioNetDataBase,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
    WFDB_Rhythm_Annotations,
)

__all__ = [
    "CPSC2021",
    "compute_metrics",
]


# configurations for visualization
PlotCfg = CFG()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60


class CPSC2021(PhysioNetDataBase):
    r"""

    The 4th China Physiological Signal Challenge 2021:
    Paroxysmal Atrial Fibrillation Events Detection from Dynamic ECG Recordings

    ABOUT CPSC2021
    --------------
    1. source ECG data are recorded from 12-lead Holter or 3-lead wearable ECG monitoring devices
    2. dataset provides variable-length ECG fragments extracted from lead I and lead II of the long-term source ECG data, each sampled at 200 Hz
    3. AF event is limited to be no less than 5 heart beats
    4. training set in the 1st stage consists of 730 records, extracted from the Holter records from 12 AF patients and 42 non-AF patients (usually including other abnormal and normal rhythms); training set in the 2nd stage consists of 706 records from 37 AF patients (18 PAF patients) and 14 non-AF patients
    5. test set comprises data from the same source as the training set as well as DIFFERENT data source, which are NOT to be released at any point
    6. annotations are standardized according to PhysioBank Annotations (Ref. [2] or PhysioNetDataBase.helper), and include the beat annotations (R peak location and beat type), the rhythm annotations (rhythm change flag and rhythm type) and the diagnosis of the global rhythm
    7. classification of a record is stored in corresponding .hea file, which can be accessed via the attribute `comments` of a wfdb Record obtained using `wfdb.rdheader`, `wfdb.rdrecord`, and `wfdb.rdsamp`; beat annotations and rhythm annotations can be accessed using the attributes `symbol`, `aux_note` of a wfdb Annotation obtained using `wfdb.rdann`, corresponding indices in the signal can be accessed via the attribute `sample`
    8. challenge task:
        (1). clasification of rhythm types: non-AF rhythm (N), persistent AF rhythm (AFf) and paroxysmal AF rhythm (AFp)
        (2). locating of the onset and offset for any AF episode prediction
    9. challenge metrics:
        (1) metrics (Ur, scoring matrix) for classification:
                Prediction
                N        AFf        AFp
        N      +1        -1         -0.5
        AFf    -2        +1          0
        AFp    -1         0         +1
        (2) metric (Ue) for detecting onsets and offsets for AF events (episodes):
        +1 if the detected onset (or offset) is within ±1 beat of the annotated position, and +0.5 if within ±2 beats
        (3) final score (U):
        U = \dfrac{1}{N} \sum\limits_{i=1}^N \left( Ur_i + \dfrac{Ma_i}{\max\{Mr_i, Ma_i\}} \right)
        where N is the number of records, Ma is the number of annotated AF episodes, Mr the number of predicted AF episodes

    NOTE
    ----
    1. if an ECG record is classified as AFf, the provided onset and offset locations should be the first and last record points. If an ECG record is classified as N, the answer should be an empty list
    2. it can be inferred from the classification scoring matrix that the punishment of false negatives of AFf is very heavy, while mixing-up of AFf and AFp is not punished
    3. flag of atrial fibrillation and atrial flutter ("AFIB" and "AFL") in annotated information are seemed as the same type when scoring the method
    4. the 3 classes can coexist in ONE subject (not one record). For example, subject 61 has 6 records with label "N", 1 with label "AFp", and 2 with label "AFf"
    5. rhythm change annotations ("(AFIB", "(AFL", "(N" in the `aux_note` field or "+" in the `symbol` field of the annotation files) are inserted 0.15s ahead of or behind (onsets or offset resp.) of corresponding R peaks.
    6. some records are revised if there are heart beats of the AF episode or the pause between adjacent AF episodes less than 5. The id numbers of the revised records are summarized in the attached REVISED_RECORDS

    ISSUES
    ------
    1.

    TODO
    ----
    1.

    Usage
    -----
    1. AF (event, fine) detection

    References
    ----------
    1. <a name="ref1"></a> http://www.icbeb.org/CPSC2021
    2. <a name="ref2"></a> https://www.physionet.org/content/cpsc2021/1.0.0/
    3. <a name="ref3"></a> https://archive.physionet.org/physiobank/annotations.shtml

    """

    def __init__(
        self,
        db_dir: str,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="cpsc2021",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.db_dir_base = Path(db_dir)
        self.db_tranches = [
            "training_I",
            "training_II",
        ]
        self.db_dirs = CFG({t: None for t in self.db_tranches})

        self.fs = 200
        self.spacing = 1000 / self.fs
        self.rec_ext = "dat"
        self.ann_ext = "atr"
        self.header_ext = "hea"
        self.all_leads = ["I", "II"]
        self.rec_patterns_with_ext = f"^data_(?:\\d+)_(?:\\d+).{self.rec_ext}$"

        self._labels_f2a = {  # fullname to abbreviation
            "non atrial fibrillation": "N",
            "paroxysmal atrial fibrillation": "AFp",
            "persistent atrial fibrillation": "AFf",
        }
        self._labels_f2n = {  # fullname to number
            "non atrial fibrillation": 0,
            "paroxysmal atrial fibrillation": 2,
            "persistent atrial fibrillation": 1,
        }

        self.nb_records = CFG({"training_I": 730, "training_II": 706})
        self._all_records = CFG({t: [] for t in self.db_tranches})
        self.__all_records = None
        self.__revised_records = []
        self._all_subjects = CFG({t: [] for t in self.db_tranches})
        self.__all_subjects = None
        self._subject_records = CFG({t: [] for t in self.db_tranches})
        self._stats = pd.DataFrame()
        self._stats_columns = [
            "record",
            "tranche",
            "subject_id",
            "record_id",
            "label",
            "fs",
            "sig_len",
            "sig_len_sec",
            "revised",
        ]
        self._ls_rec()
        self._aggregate_stats()

        self._diagnoses_records_list = None
        self._ls_diagnoses_records()

        self._epsilon = 1e-7  # dealing with round(0.5) = 0, hence keeping accordance with output length of `resample_poly`

        # self.palette = {"spb": "yellow", "pvc": "red",}

    @property
    def all_records(self) -> List[str]:
        """ """
        if self.__all_records is None:
            self._ls_rec()
        return self.__all_records

    def _ls_rec(self) -> NoReturn:
        """

        list all the records and load into `self._all_records`,
        facilitating further uses

        """
        self._all_records = CFG({t: [] for t in self.db_tranches})
        self._all_subjects = CFG({t: [] for t in self.db_tranches})
        self._subject_records = CFG({t: [] for t in self.db_tranches})

        self._ls_rec_split()
        if self.__all_records is not None and len(self.__all_records) > 0:
            pass
        else:
            for rec in get_record_list_recursive3(
                self.db_dir_base, self.rec_patterns_with_ext
            ):
                rec_dir = self.db_dir_base / Path(rec).parent
                rec_name = Path(rec).name
                if int(self.get_subject_id(rec_name)) > 53:
                    tranche = self.db_tranches[1]
                else:
                    tranche = self.db_tranches[0]
                if self.db_dirs[tranche] is None:
                    self.db_dirs[tranche] = rec_dir
                elif self.db_dirs[tranche] != rec_dir:
                    raise ValueError(
                        "Records from the same tranche should be in the same directory"
                        f" (some in {str(self.db_dirs[tranche])}, and some other in {str(rec_dir)})"
                    )
                self._all_records[tranche].append(rec_name)

        for t in self.db_tranches:
            self._all_subjects[t] = sorted(
                list(set([self.get_subject_id(rec) for rec in self._all_records[t]])),
                key=lambda s: int(s),
            )
            self._subject_records[t] = CFG(
                {
                    sid: [
                        rec
                        for rec in self._all_records[t]
                        if self.get_subject_id(rec) == sid
                    ]
                    for sid in self._all_subjects[t]
                }
            )

        self._all_records_inv = {
            r: t for t, l_r in self._all_records.items() for r in l_r
        }
        self._all_subjects_inv = {
            s: t for t, l_s in self._all_subjects.items() for s in l_s
        }
        self.__all_records = sorted(list_sum(self._all_records.values()))
        self.__all_subjects = sorted(
            list_sum(self._all_subjects.values()), key=lambda s: int(s)
        )

    def _ls_rec_split(self) -> NoReturn:
        """

        list all the records assuming the records
        are split into two folders (training_I and training_II)

        """
        fn = "RECORDS"
        rev_fn = "REVISED_RECORDS"
        for t in self.db_tranches:
            dir_candidate = self.db_dir_base / t.replace("training_", "training") / t
            if dir_candidate.is_dir():
                dir_tranche = dir_candidate
            else:
                dir_tranche = self.db_dir_base / t
            if dir_tranche.is_dir():
                self.db_dirs[t] = dir_tranche

            record_list_fp = dir_tranche / fn
            if record_list_fp.is_file():
                self._all_records[t] = record_list_fp.read_text().splitlines()
            else:
                self._all_records[t] = []
            if len(self._all_records[t]) == self.nb_records[t]:
                pass
            else:
                if not dir_tranche.is_dir():
                    continue
                print("Please wait patiently to let the reader find all records...")
                start = time.time()
                self._all_records[t] = get_record_list_recursive3(
                    str(dir_tranche), self.rec_patterns_with_ext
                )
                print(f"Done in {time.time() - start:.5f} seconds!")
                record_list_fp.write_text("\n".join(self._all_records[t]))

            record_list_fp = dir_tranche / rev_fn
            if record_list_fp.is_file():
                self.__revised_records.extend(record_list_fp.read_text().splitlines())

    def _aggregate_stats(self) -> NoReturn:
        """aggregate stats on the whole dataset"""
        stats_file = "stats.csv"
        stats_file_fp = self.db_dir_base / stats_file
        if stats_file_fp.is_file():
            self._stats = pd.read_csv(stats_file_fp)

        if self._stats.empty or set(self._stats_columns) != set(self._stats.columns):
            print(
                "Please wait patiently to let the reader aggregate statistics on the whole dataset..."
            )
            start = time.time()
            self._stats = pd.DataFrame(
                self.all_records, columns=["record"]
            )  # use self.all_records to ensure it's computed
            self._stats["tranche"] = self._stats["record"].apply(
                lambda s: self._all_records_inv[s]
            )
            self._stats["subject_id"] = self._stats["record"].apply(
                lambda s: int(s.split("_")[1])
            )
            self._stats["record_id"] = self._stats["record"].apply(
                lambda s: int(s.split("_")[2])
            )
            self._stats["label"] = self._stats["record"].apply(
                lambda s: self.load_label(s)
            )
            self._stats["fs"] = self.fs
            self._stats["sig_len"] = self._stats["record"].apply(
                lambda s: wfdb.rdheader(str(self._get_path(s))).sig_len
            )
            self._stats["sig_len_sec"] = self._stats["sig_len"] / self._stats["fs"]
            self._stats["revised"] = self._stats["record"].apply(
                lambda s: 1 if s in self.__revised_records else 0
            )
            self._stats = self._stats.sort_values(
                by=["subject_id", "record_id"], ignore_index=True
            )
            self._stats = self._stats[self._stats_columns]
            self._stats.to_csv(stats_file_fp, index=False)
            print(f"Done in {time.time() - start:.5f} seconds!")
        else:
            pass  # currently no need to parse the loaded csv file
        self._stats["subject_id"] = self._stats["subject_id"].apply(lambda s: str(s))
        self.__all_records = self._stats["record"].tolist()

    @property
    def all_subjects(self) -> List[str]:
        """ """
        return self.__all_subjects

    @property
    def subject_records(self) -> CFG:
        """ """
        return self._subject_records

    @property
    def df_stats(self) -> pd.DataFrame:
        """ """
        return self._stats

    def _ls_diagnoses_records(self) -> NoReturn:
        """list all the records for all diagnoses"""
        fn = "diagnoses_records_list.json"
        dr_fp = self.db_dir_base / fn
        if dr_fp.is_file():
            self._diagnoses_records_list = json.loads(dr_fp.read_text())
        else:
            start = time.time()
            if self.df_stats.empty:
                print(
                    "Please wait several minutes patiently to let the reader list records for each diagnosis..."
                )
                self._diagnoses_records_list = {
                    d: [] for d in self._labels_f2a.values()
                }
                for rec in self.all_records:
                    lb = self.load_label(rec)
                    self._diagnoses_records_list[lb].append(rec)
                print(f"Done in {time.time() - start:.5f} seconds!")
            else:
                self._diagnoses_records_list = {
                    d: self.df_stats[self.df_stats["label"] == d]["record"].tolist()
                    for d in self._labels_f2a.values()
                }
            dr_fp.write_text(
                json.dumps(self._diagnoses_records_list, ensure_ascii=False)
            )
        self._diagnoses_records_list = CFG(self._diagnoses_records_list)

    @property
    def diagnoses_records_list(self):
        """ """
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list

    def get_subject_id(self, rec: Union[str, int]) -> str:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        sid: str,
            subject id corresponding to the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        sid = rec.split("_")[1]
        return sid

    def _get_path(self, rec: Union[str, int], ext: Optional[str] = None) -> Path:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ext: str, optional,
            file extension of the path

        Returns
        -------
        p: Path,
            path (with or without file extension) of the record
        """
        if isinstance(rec, int):
            rec = self[rec]
        if ext:
            rec += f".{ext}"
        p = self.db_dirs[self._all_records_inv[rec]] / rec
        return p

    def _validate_samp_interval(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        validate `sampfrom` and `sampto` so that they are reasonable

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded

        Returns
        -------
        (sf, st): tuple of int,
        sf: int,
            index sampling from
        st: int,
            index sampling to

        """
        if isinstance(rec, int):
            rec = self[rec]
        sf, st = (
            sampfrom or 0,
            sampto or self.df_stats[self.df_stats.record == rec].iloc[0].sig_len,
        )
        if sf >= st:
            raise ValueError("Invalid `sampfrom` and `sampto`")
        return sf, st

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, List[str]]] = None,
        data_format: str = "channel_first",
        units: str = "mV",
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
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
        data_format: str, default "channel_first",
            format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        fs: real number, optional,
            if not None, the loaded data will be resampled to this sampling frequency

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert data_format.lower() in [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
        ]
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        assert all([ld in self.all_leads for ld in _leads])

        rec_fp = self._get_path(rec)
        sf, st = self._validate_samp_interval(rec, sampfrom, sampto)
        wfdb_rec = wfdb.rdrecord(
            str(rec_fp), sampfrom=sf, sampto=st, physical=True, channel_names=_leads
        )
        data = np.asarray(wfdb_rec.p_signal.T)
        # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    def load_ann(
        self,
        rec: Union[str, int],
        field: Optional[str] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[dict, np.ndarray, List[List[int]], str]:
        """

        load annotations of the record

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        field: str, optional
            field of the annotation, can be one of "rpeaks", "af_episodes", "label", "raw", "wfdb",
            if not specified, all fields of the annotation will be returned in the form of a dict,
            if is "raw" or "wfdb", then the corresponding wfdb "Annotation" will be returned
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        kwargs: dict,
            key word arguments for functions loading rpeaks, af_episodes, and label respectively,
            including:
            fs: int, optional,
                the resampling frequency
            fmt: str,
                format of af_episodes, or format of label,
                for more details, ref. corresponding functions
            used only when field is specified,

        Returns
        -------
        ann: dict, or list, or ndarray, or str,
            annotaton of the record
        """
        if isinstance(rec, int):
            rec = self[rec]
        sf, st = self._validate_samp_interval(rec, sampfrom, sampto)
        ann = wfdb.rdann(
            str(self._get_path(rec)), extension=self.ann_ext, sampfrom=sf, sampto=st
        )
        # `load_af_episodes` should not use sampfrom, sampto
        func = {
            "rpeaks": self.load_rpeaks,
            "af_episodes": self.load_af_episodes,
            "label": self.load_label,
        }
        if field is None:
            ann = {k: f(rec, ann, sf, st) for k, f in func.items()}
            if kwargs:
                warnings.warn(
                    f"key word arguments {list(kwargs.keys())} ignored when field is not specified!"
                )
            return ann
        elif field.lower() in [
            "raw",
            "wfdb",
        ]:
            return ann

        try:
            f = func[field.lower()]
        except Exception:
            raise ValueError("invalid field")
        ann = f(rec, ann, sf, st, **kwargs)
        return ann

    def load_rpeaks(
        self,
        rec: Union[str, int],
        ann: Optional[wfdb.Annotation] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """

        load position (in terms of samples) of rpeaks

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann: Annotation, optional,
            the wfdb Annotation of the record,
            if None, corresponding annotation file will be read
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
        fs: real number, optional,
            if not None, positions of the loaded rpeaks will be ajusted according to this sampling frequency

        Returns
        -------
        rpeaks: ndarray,
            position (in terms of samples) of rpeaks of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        if ann is None:
            sf, st = self._validate_samp_interval(rec, sampfrom, sampto)
            ann = wfdb.rdann(
                str(self._get_path(rec)), extension=self.ann_ext, sampfrom=sf, sampto=st
            )
        critical_points = ann.sample
        symbols = ann.symbol
        rpeaks_valid = np.isin(symbols, list(WFDB_Beat_Annotations.keys()))
        if sampfrom and not keep_original:
            critical_points = critical_points - sampfrom
        if fs is not None and fs != self.fs:
            critical_points = np.round(
                critical_points * fs / self.fs + self._epsilon
            ).astype(int)
        rpeaks = critical_points[rpeaks_valid]
        return rpeaks

    @add_docstring(load_rpeaks.__doc__)
    def load_rpeak_indices(
        self,
        rec: Union[str, int],
        ann: Optional[wfdb.Annotation] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """
        alias of `self.load_rpeaks`
        """
        return self.load_rpeaks(rec, ann, sampfrom, sampto, keep_original, fs)

    def load_af_episodes(
        self,
        rec: Union[str, int],
        ann: Optional[wfdb.Annotation] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        fs: Optional[Real] = None,
        fmt: str = "intervals",
    ) -> Union[List[List[int]], np.ndarray]:
        """

        load the episodes of atrial fibrillation, in terms of intervals or mask

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann: Annotation, optional,
            the wfdb Annotation of the record,
            if None, corresponding annotation file will be read
        sampfrom: int, optional,
            start index of the data to be loaded,
            not used when `fmt` is "c_intervals"
        sampto: int, optional,
            end index of the data to be loaded,
            not used when `fmt` is "c_intervals"
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
            works only when `fmt` is not "c_intervals"
        fs: real number, optional,
            if not None, positions of the loaded intervals or mask will be ajusted according to this sampling frequency
        fmt: str, default "intervals",
            format of the episodes of atrial fibrillation, can be one of "intervals", "mask", "c_intervals"

        Returns
        -------
        af_episodes: list or ndarray,
            episodes of atrial fibrillation, in terms of intervals or mask

        """
        if isinstance(rec, int):
            rec = self[rec]
        header = wfdb.rdheader(str(self._get_path(rec)))
        label = self._labels_f2a[header.comments[0]]
        siglen = header.sig_len
        _ann = wfdb.rdann(str(self._get_path(rec)), extension=self.ann_ext)
        sf, st = self._validate_samp_interval(rec, sampfrom, sampto)
        aux_note = np.array(_ann.aux_note)
        critical_points = _ann.sample
        af_start_inds = np.where((aux_note == "(AFIB") | (aux_note == "(AFL"))[
            0
        ]  # ref. NOTE 3.
        af_end_inds = np.where(aux_note == "(N")[0]
        assert len(af_start_inds) == len(
            af_end_inds
        ), "unequal number of af period start indices and af period end indices"

        if fmt.lower() in [
            "c_intervals",
        ]:
            if sf > 0 or st < siglen:
                raise ValueError(
                    "when `fmt` is `c_intervals`, `sampfrom` and `sampto` should never be used!"
                )
            af_episodes = [
                [start, end] for start, end in zip(af_start_inds, af_end_inds)
            ]
            return af_episodes

        intervals = []
        for start, end in zip(af_start_inds, af_end_inds):
            itv = [critical_points[start], critical_points[end]]
            intervals.append(itv)
        intervals = generalized_intervals_intersection(intervals, [[sf, st]])

        siglen = st - sf
        if fs is not None and fs != self.fs:
            siglen = self._round(siglen * fs / self.fs)
            sf = self._round(sf * fs / self.fs)
            if label == "AFf":
                # ref. NOTE. 1 of the class docstring
                # the `ann.sample` does not always satify this point after resampling
                intervals = [[sf, siglen - 1]]
            else:
                intervals = [
                    [
                        self._round(itv[0] * fs / self.fs),
                        self._round(itv[1] * fs / self.fs),
                    ]
                    for itv in intervals
                ]

        if not keep_original:
            intervals = [[itv[0] - sf, itv[1] - sf] for itv in intervals]
            sf = 0
        af_episodes = intervals

        if fmt.lower() in [
            "mask",
        ]:
            mask = np.zeros((siglen,), dtype=int)
            for itv in intervals:
                mask[itv[0] - sf : itv[1] - sf] = 1
            af_episodes = mask

        return af_episodes

    def load_label(
        self,
        rec: Union[str, int],
        ann: Optional[wfdb.Annotation] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        fmt: str = "a",
    ) -> str:
        """

        load (classifying) label of the record,
        among the following three classes:
        "non atrial fibrillation",
        "paroxysmal atrial fibrillation",
        "persistent atrial fibrillation",

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann: Annotation, optional,
            not used, to keep in accordance with other methods
        sampfrom: int, optional,
            not used, to keep in accordance with other methods
        sampto: int, optional,
            not used, to keep in accordance with other methods
        fmt: str, default "a",
            format of the label, case in-sensitive, can be one of:
            "f", "fullname": the full name of the label
            "a", "abbr", "abbrevation": abbreviation for the label
            "n", "num", "number": class number of the label (in accordance with the settings of the offical class map)

        Returns
        -------
        label: str,
            classifying label of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        header = wfdb.rdheader(str(self._get_path(rec)))
        label = header.comments[0]
        if fmt.lower() in ["a", "abbr", "abbreviation"]:
            label = self._labels_f2a[label]
        elif fmt.lower() in ["n", "num", "number"]:
            label = self._labels_f2n[label]
        elif not fmt.lower() in ["f", "fullname"]:
            raise ValueError(f"format `{fmt}` of labels is not supported!")
        return label

    def gen_endpoint_score_mask(
        self, rec: Union[str, int], bias: dict = {1: 1, 2: 0.5}
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        generate the scoring mask for the onsets and offsets of af episodes,

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        bias: dict, default {1:1, 2:0.5},
            keys are bias (with ±) in terms of number of rpeaks
            values are corresponding scores

        Returns
        -------
        (onset_score_mask, offset_score_mask): 2-tuple of ndarray,
            scoring mask for the onset and offsets predictions of af episodes

        NOTE
        ----
        the onsets in `af_intervals` are 0.15s ahead of the corresponding R peaks,
        while the offsets in `af_intervals` are 0.15s behind the corresponding R peaks,

        """
        if isinstance(rec, int):
            rec = self[rec]
        masks = gen_endpoint_score_mask(
            siglen=self.df_stats[self.df_stats.record == rec].iloc[0].sig_len,
            critical_points=wfdb.rdann(
                str(self._get_path(rec)), extension=self.ann_ext
            ).sample,
            af_intervals=self.load_af_episodes(rec, fmt="c_intervals"),
            bias=bias,
            verbose=self.verbose,
        )
        return masks

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[Dict[str, np.ndarray]] = None,
        ticks_granularity: int = 0,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        leads: Optional[Union[str, List[str]]] = None,
        waves: Optional[Dict[str, Sequence[int]]] = None,
        **kwargs,
    ) -> NoReturn:
        """to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (labels, episodes of atrial fibrillation, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data: ndarray, optional,
            (2-lead) ECG signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            annotations for `data`,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        sampfrom: int, optional,
            start index of the data to plot
        sampto: int, optional,
            end index of the data to plot
        leads: str or list of str, optional,
            the leads to plot
        waves: dict, optional,
            indices of the wave critical points, including
            "p_onsets", "p_peaks", "p_offsets",
            "q_onsets", "q_peaks", "r_peaks", "s_peaks", "s_offsets",
            "t_onsets", "t_peaks", "t_offsets"
        kwargs: dict,

        TODO
        ----
        1. slice too long records, and plot separately for each segment
        2. plot waves using `axvspan`

        NOTE
        ----
        1. `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once
        2. raw data usually have very severe baseline drifts,
        hence the isoelectric line is not plotted

        Contributors: Jeethan, and WEN Hao

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == "all":
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        assert all([ld in self.all_leads for ld in _leads])

        if data is None:
            _data = self.load_data(
                rec,
                leads=_leads,
                data_format="channel_first",
                units="μV",
                sampfrom=sampfrom,
                sampto=sampto,
            )
        else:
            units = self._auto_infer_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(
                _leads
            ), f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        sf, st = (sampfrom or 0), (sampto or len(_data))

        if waves:
            if waves.get("p_onsets", None) and waves.get("p_offsets", None):
                p_waves = [
                    [onset, offset]
                    for onset, offset in zip(waves["p_onsets"], waves["p_offsets"])
                ]
            elif waves.get("p_peaks", None):
                p_waves = [
                    [
                        max(0, p + ms2samples(PlotCfg.p_onset, fs=self.fs)),
                        min(
                            _data.shape[1], p + ms2samples(PlotCfg.p_offset, fs=self.fs)
                        ),
                    ]
                    for p in waves["p_peaks"]
                ]
            else:
                p_waves = []
            if waves.get("q_onsets", None) and waves.get("s_offsets", None):
                qrs = [
                    [onset, offset]
                    for onset, offset in zip(waves["q_onsets"], waves["s_offsets"])
                ]
            elif waves.get("q_peaks", None) and waves.get("s_peaks", None):
                qrs = [
                    [
                        max(0, q + ms2samples(PlotCfg.q_onset, fs=self.fs)),
                        min(
                            _data.shape[1], s + ms2samples(PlotCfg.s_offset, fs=self.fs)
                        ),
                    ]
                    for q, s in zip(waves["q_peaks"], waves["s_peaks"])
                ]
            elif waves.get("r_peaks", None):
                qrs = [
                    [
                        max(0, r + ms2samples(PlotCfg.qrs_radius, fs=self.fs)),
                        min(
                            _data.shape[1],
                            r + ms2samples(PlotCfg.qrs_radius, fs=self.fs),
                        ),
                    ]
                    for r in waves["r_peaks"]
                ]
            else:
                qrs = []
            if waves.get("t_onsets", None) and waves.get("t_offsets", None):
                t_waves = [
                    [onset, offset]
                    for onset, offset in zip(waves["t_onsets"], waves["t_offsets"])
                ]
            elif waves.get("t_peaks", None):
                t_waves = [
                    [
                        max(0, t + ms2samples(PlotCfg.t_onset, fs=self.fs)),
                        min(
                            _data.shape[1], t + ms2samples(PlotCfg.t_offset, fs=self.fs)
                        ),
                    ]
                    for t in waves["t_peaks"]
                ]
            else:
                t_waves = []
        else:
            p_waves, qrs, t_waves = [], [], []
        palette = {
            "p_waves": "green",
            "qrs": "yellow",
            "t_waves": "pink",
        }
        plot_alpha = 0.4

        if ann is None or data is None:
            _ann = self.load_ann(rec, sampfrom=sampfrom, sampto=sampto)
            rpeaks = _ann["rpeaks"]
            af_episodes = _ann["af_episodes"]
            af_episodes = [[itv[0] - sf, itv[1] - sf] for itv in af_episodes]
            label = _ann["label"]
        else:
            rpeaks = ann.get("rpeaks", [])
            af_episodes = ann.get("af_episodes", [])
            label = ann.get("label", "")

        nb_leads = len(_leads)

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(_data.shape[1] / line_len)

        bias_thr = 0.07
        # winL = 0.06
        # winR = 0.08

        for idx in range(nb_lines):
            seg = _data[..., idx * line_len : (idx + 1) * line_len]
            secs = (sf + np.arange(seg.shape[1]) + idx * line_len) / self.fs
            fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * seg.shape[1] / self.fs))
            # if same_range:
            #     y_ranges = np.ones((seg.shape[0],)) * np.max(np.abs(seg)) + 100
            # else:
            #     y_ranges = np.max(np.abs(seg), axis=1) + 100
            # fig_sz_h = 6 * y_ranges / 1500
            fig_sz_h = (
                6
                * sum([seg_lead.max() - seg_lead.min() + 200 for seg_lead in seg])
                / 1500
            )
            fig, axes = plt.subplots(
                nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h))
            )
            if nb_leads == 1:
                axes = [axes]

            for ax_idx in range(nb_leads):
                axes[ax_idx].plot(
                    secs, seg[ax_idx], color="black", label=f"lead - {_leads[ax_idx]}"
                )
                # axes[ax_idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
                # NOTE that `Locator` has default `MAXTICKS` equal to 1000
                if ticks_granularity >= 1:
                    axes[ax_idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                    axes[ax_idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                    axes[ax_idx].grid(
                        which="major", linestyle="-", linewidth="0.5", color="red"
                    )
                if ticks_granularity >= 2:
                    axes[ax_idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                    axes[ax_idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                    axes[ax_idx].grid(
                        which="minor", linestyle=":", linewidth="0.5", color="black"
                    )
                # add extra info. to legend
                # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
                if label:
                    axes[ax_idx].plot([], [], " ", label=f"label - {label}")
                seg_rpeaks = [
                    r / self.fs
                    for r in rpeaks
                    if idx * line_len <= r < (idx + 1) * line_len
                ]
                for r in seg_rpeaks:
                    axes[ax_idx].axvspan(
                        max(secs[0], r - bias_thr),
                        min(secs[-1], r + bias_thr),
                        color=palette["qrs"],
                        alpha=0.3,
                    )
                seg_af_episodes = generalized_intervals_intersection(
                    af_episodes,
                    [[idx * line_len, (idx + 1) * line_len]],
                )
                seg_af_episodes = [
                    [itv[0] - idx * line_len, itv[1] - idx * line_len]
                    for itv in seg_af_episodes
                ]
                for itv_start, itv_end in seg_af_episodes:
                    axes[ax_idx].plot(
                        secs[itv_start:itv_end],
                        seg[ax_idx, itv_start:itv_end],
                        color="red",
                    )
                for w in ["p_waves", "qrs", "t_waves"]:
                    for itv in eval(w):
                        axes[ax_idx].axvspan(
                            itv[0], itv[1], color=palette[w], alpha=plot_alpha
                        )
                axes[ax_idx].legend(loc="upper left")
                axes[ax_idx].set_xlim(secs[0], secs[-1])
                # axes[ax_idx].set_ylim(-y_ranges[ax_idx], y_ranges[ax_idx])
                axes[ax_idx].set_xlabel("Time [s]")
                axes[ax_idx].set_ylabel("Voltage [μV]")
            plt.subplots_adjust(hspace=0.2)
            plt.show()

    def _round(self, n: Real) -> int:
        """
        dealing with round(0.5) = 0, hence keeping accordance with output length of `resample_poly`
        """
        return int(round(n + self._epsilon))

    @property
    def url_(self) -> str:
        """URL of the compressed database file"""
        if self._url_compressed is not None:
            return self._url_compressed
        # currently, cpsc2021 is not included in the list obtained
        # using `wfdb.get_dbs()`
        self._url_compressed = f"https://www.physionet.org/static/published-projects/cpsc2021/paroxysmal-atrial-fibrillation-events-detection-from-dynamic-ECG-recordings-the-4th-china-physiological-signal-challenge-2021-{self.version}.zip"
        return self._url_compressed


###################################################################
# copied and modified from the official scoring code
###################################################################

R = np.array(
    [[1, -1, -0.5], [-2, 1, 0], [-1, 0, 1]]
)  # scoring matrix for classification


class RefInfo:
    def __init__(self, sample_path):
        self.sample_path = sample_path
        (
            self.fs,
            self.len_sig,
            self.beat_loc,
            self.af_starts,
            self.af_ends,
            self.class_true,
        ) = self._load_ref()
        self.endpoints_true = np.dstack((self.af_starts, self.af_ends))[0, :, :]
        # self.endpoints_true = np.concatenate((self.af_starts, self.af_ends), axis=-1)

        if self.class_true == 1 or self.class_true == 2:
            (
                self.onset_score_range,
                self.offset_score_range,
            ) = self._gen_endpoint_score_range()
        else:
            self.onset_score_range, self.offset_score_range = None, None

    def _load_ref(self):
        sig, fields = wfdb.rdsamp(self.sample_path)
        ann_ref = wfdb.rdann(self.sample_path, "atr")

        fs = fields["fs"]
        length = len(sig)
        sample_descrip = fields["comments"]

        beat_loc = np.array(ann_ref.sample)  # r-peak locations
        ann_note = np.array(ann_ref.aux_note)  # rhythm change flag

        af_start_scripts = np.where((ann_note == "(AFIB") | (ann_note == "(AFL"))[0]
        af_end_scripts = np.where(ann_note == "(N")[0]

        if "non atrial fibrillation" in sample_descrip:
            class_true = 0
        elif "persistent atrial fibrillation" in sample_descrip:
            class_true = 1
        elif "paroxysmal atrial fibrillation" in sample_descrip:
            class_true = 2
        else:
            print("Error: the recording is out of range!")

            return -1

        return fs, length, beat_loc, af_start_scripts, af_end_scripts, class_true

    def _gen_endpoint_score_range(self, verbose=0):
        """ """
        onset_range = np.zeros((self.len_sig,), dtype=np.float)
        offset_range = np.zeros((self.len_sig,), dtype=np.float)
        for i, af_start in enumerate(self.af_starts):
            if self.class_true == 2:
                if max(af_start - 1, 0) == 0:
                    onset_range[: self.beat_loc[af_start + 2]] += 1
                    if verbose > 0:
                        print(
                            f"official --- onset (c_ind, score 1): 0 --- {af_start+2}"
                        )
                        print(
                            f"official --- onset (sample, score 1): 0 --- {self.beat_loc[af_start+2]}"
                        )
                elif max(af_start - 2, 0) == 0:
                    onset_range[
                        self.beat_loc[af_start - 1] : self.beat_loc[af_start + 2]
                    ] += 1
                    if verbose > 0:
                        print(
                            f"official --- onset (c_ind, score 1): {af_start-1} --- {af_start+2}"
                        )
                        print(
                            f"official --- onset (sample, score 1): {self.beat_loc[af_start-1]} --- {self.beat_loc[af_start+2]}"
                        )
                    onset_range[: self.beat_loc[af_start - 1]] += 0.5
                    if verbose > 0:
                        print(
                            f"official --- onset (c_ind, score 0.5): 0 --- {af_start-1}"
                        )
                        print(
                            f"official --- onset (sample, score 0.5): 0 --- {self.beat_loc[af_start-1]}"
                        )
                else:
                    onset_range[
                        self.beat_loc[af_start - 1] : self.beat_loc[af_start + 2]
                    ] += 1
                    if verbose > 0:
                        print(
                            f"official --- onset (c_ind, score 1): {af_start-1} --- {af_start+2}"
                        )
                        print(
                            f"official --- onset (sample, score 1): {self.beat_loc[af_start-1]} --- {self.beat_loc[af_start+2]}"
                        )
                    onset_range[
                        self.beat_loc[af_start - 2] : self.beat_loc[af_start - 1]
                    ] += 0.5
                    if verbose > 0:
                        print(
                            f"official --- onset (c_ind, score 0.5): {af_start-2} --- {af_start-1}"
                        )
                        print(
                            f"official --- onset (sample, score 0.5): {self.beat_loc[af_start-2]} --- {self.beat_loc[af_start-1]}"
                        )
                onset_range[
                    self.beat_loc[af_start + 2] : self.beat_loc[af_start + 3]
                ] += 0.5
                if verbose > 0:
                    print(
                        f"official --- onset (c_ind, score 0.5): {af_start+2} --- {af_start+3}"
                    )
                    print(
                        f"official --- onset (sample, score 0.5): {self.beat_loc[af_start+2]} --- {self.beat_loc[af_start+3]}"
                    )
            elif self.class_true == 1:
                onset_range[: self.beat_loc[af_start + 2]] += 1
                if verbose > 0:
                    print(f"official --- onset (c_ind, score 1): 0 --- {af_start+2}")
                    print(
                        f"official --- onset (sample, score 1): 0 --- {self.beat_loc[af_start+2]}"
                    )
                onset_range[
                    self.beat_loc[af_start + 2] : self.beat_loc[af_start + 3]
                ] += 0.5
                if verbose > 0:
                    print(
                        f"official --- onset (c_ind, score 0.5): {af_start+2} --- {af_start+3}"
                    )
                    print(
                        f"official --- onset (sample, score 0.5): {self.beat_loc[af_start+2]} --- {self.beat_loc[af_start+3]}"
                    )
        for i, af_end in enumerate(self.af_ends):
            if self.class_true == 2:
                if min(af_end + 1, len(self.beat_loc) - 1) == len(self.beat_loc) - 1:
                    offset_range[self.beat_loc[af_end - 2] :] += 1
                    if verbose > 0:
                        print(
                            f"official --- offset (c_ind, score 1): {af_end-2} --- -1"
                        )
                        print(
                            f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- -1"
                        )
                elif min(af_end + 2, len(self.beat_loc) - 1) == len(self.beat_loc) - 1:
                    offset_range[
                        self.beat_loc[af_end - 2] : self.beat_loc[af_end + 1]
                    ] += 1
                    if verbose > 0:
                        print(
                            f"official --- offset (c_ind, score 1): {af_end-2} --- {af_end+1}"
                        )
                        print(
                            f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- {self.beat_loc[af_end+1]}"
                        )
                    offset_range[self.beat_loc[af_end + 1] :] += 0.5
                    if verbose > 0:
                        print(
                            f"official --- offset (c_ind, score 0.5): {af_end+1} --- -1"
                        )
                        print(
                            f"official --- offset (sample, score 0.5): {self.beat_loc[af_end+1]} --- -1"
                        )
                else:
                    offset_range[
                        self.beat_loc[af_end - 2] : self.beat_loc[af_end + 1]
                    ] += 1
                    if verbose > 0:
                        print(
                            f"official --- offset (c_ind, score 1): {af_end-2} --- {af_end+1}"
                        )
                        print(
                            f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- {self.beat_loc[af_end+1]}"
                        )
                    offset_range[
                        self.beat_loc[af_end + 1] : min(
                            self.beat_loc[af_end + 2], self.len_sig - 1
                        )
                    ] += 0.5
                    if verbose > 0:
                        print(
                            f"official --- offset (c_ind, score 0.5): {af_end+1} --- -1"
                        )
                        print(
                            f"official --- offset (sample, score 0.5): {self.beat_loc[af_end+1]} --- {min(self.beat_loc[af_end+2], self.len_sig-1)}"
                        )
                offset_range[
                    self.beat_loc[af_end - 3] : self.beat_loc[af_end - 2]
                ] += 0.5
                if verbose > 0:
                    print(
                        f"official --- offset (c_ind, score 0.5): {af_end-3} --- {af_end-2}"
                    )
                    print(
                        f"official --- offset (sample, score 0.5): {self.beat_loc[af_end-3]} --- {self.beat_loc[af_end-2]}"
                    )
            elif self.class_true == 1:
                offset_range[self.beat_loc[af_end - 2] :] += 1
                if verbose > 0:
                    print(f"official --- offset (c_ind, score 1): {af_end-2} --- -1")
                    print(
                        f"official --- offset (sample, score 1): {self.beat_loc[af_end-2]} --- -1"
                    )
                offset_range[
                    self.beat_loc[af_end - 3] : self.beat_loc[af_end - 2]
                ] += 0.5
                if verbose > 0:
                    print(
                        f"official --- offset (c_ind, score 0.5): {af_end-3} --- {af_end-2}"
                    )
                    print(
                        f"official --- offset (sample, score 0.5): {self.beat_loc[af_end-3]} --- {self.beat_loc[af_end-2]}"
                    )

        return onset_range, offset_range


def load_ans(ans_file):
    endpoints_pred = []
    if ans_file.endswith(".json"):
        json_file = open(ans_file, "r")
        ans_dic = json.load(json_file)
        endpoints_pred = np.array(ans_dic["predict_endpoints"])

    elif ans_file.endswith(".mat"):
        ans_struct = sio.loadmat(ans_file)
        endpoints_pred = ans_struct["predict_endpoints"] - 1

    return endpoints_pred


def ue_calculate(endpoints_pred, endpoints_true, onset_score_range, offset_score_range):
    score = 0
    ma = len(endpoints_true)
    mr = len(endpoints_pred)

    if mr == 0:
        score = 0
    else:
        for [start, end] in endpoints_pred:
            score += onset_score_range[int(start)]
            score += offset_score_range[int(end)]

    score *= ma / max(ma, mr)

    return score


def ur_calculate(class_true, class_pred):
    score = R[int(class_true), int(class_pred)]

    return score


def score(data_path, ans_path):
    # AF burden estimation
    SCORE = []

    def is_mat_or_json(file):
        return (file.endswith(".json")) + (file.endswith(".mat"))

    ans_set = filter(is_mat_or_json, os.listdir(ans_path))
    # test_set = open(os.path.join(data_path, 'RECORDS'), 'r').read().splitlines()
    for i, ans_sample in enumerate(ans_set):
        sample_nam = ans_sample.split(".")[0]
        sample_path = os.path.join(data_path, sample_nam)

        endpoints_pred = load_ans(os.path.join(ans_path, ans_sample))
        TrueRef = RefInfo(sample_path)

        if len(endpoints_pred) == 0:
            class_pred = 0
        elif (
            len(endpoints_pred) == 1
            and np.diff(endpoints_pred)[-1] == TrueRef.len_sig - 1
        ):
            class_pred = 1
        else:
            class_pred = 2

        ur_score = ur_calculate(TrueRef.class_true, class_pred)

        if TrueRef.class_true == 1 or TrueRef.class_true == 2:
            ue_score = ue_calculate(
                endpoints_pred,
                TrueRef.endpoints_true,
                TrueRef.onset_score_range,
                TrueRef.offset_score_range,
            )
        else:
            ue_score = 0

        u = ur_score + ue_score
        SCORE.append(u)

    score_avg = np.mean(SCORE)

    return score_avg


###################################################################
# custom metric computing function
###################################################################


def compute_challenge_metric(
    class_true: int,
    class_pred: int,
    endpoints_true: Sequence[Sequence[int]],
    endpoints_pred: Sequence[Sequence[int]],
    onset_score_range: Sequence[float],
    offset_score_range: Sequence[float],
) -> float:
    """

    compute challenge metric for a single record

    Parameters
    ----------
    class_true: int,
        labelled for the record
    class_pred: int,
        predicted class for the record
    endpoints_true: sequence of intervals,
        labelled intervals of AF episodes
    endpoints_pred: sequence of intervals,
        predicted intervals of AF episodes
    onset_score_range: sequence of float,
        scoring mask for the AF onset predictions
    offset_score_range: sequence of float,
        scoring mask for the AF offset predictions

    Returns
    -------
    u: float,
        the final score for the prediction
    """
    ur_score = ur_calculate(class_true, class_pred)
    ue_score = ue_calculate(
        endpoints_pred, endpoints_true, onset_score_range, offset_score_range
    )
    u = ur_score + ue_score
    return u


def gen_endpoint_score_mask(
    siglen: int,
    critical_points: Sequence[int],
    af_intervals: Sequence[Sequence[int]],
    bias: dict = {1: 1, 2: 0.5},
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    generate the scoring mask for the onsets and offsets of af episodes,

    Parameters
    ----------
    siglen: int,
        length of the signal
    critical_points: sequence of int,
        locations (indices in the signal) of the critical points,
        including R peaks, rhythm annotations, etc,
        which are stored in the `sample` fields of an wfdb annotation file
        (corr. beat ann, rhythm ann are in the `symbol`, `aux_note` fields)
    af_intervals: sequence of intervals,
        intervals of the af episodes in terms of indices in `critical_points`
    bias: dict, default {1:1, 2:0.5},
        keys are bias (with ±) in terms of number of rpeaks
        values are corresponding scores
    verbose: int, default 0,
        log verbosity

    Returns
    -------
    (onset_score_mask, offset_score_mask): 2-tuple of ndarray,
        scoring mask for the onset and offsets predictions of af episodes

    NOTE
    ----
    1. the onsets in `af_intervals` are 0.15s ahead of the corresponding R peaks,
    while the offsets in `af_intervals` are 0.15s behind the corresponding R peaks.
    2. for records [data_39_4,data_48_4,data_68_23,data_98_5,data_101_5,data_101_7,data_101_8,data_104_25,data_104_27],
    the official `RefInfo._gen_endpoint_score_range` slightly expands the scoring intervals at heads or tails of the records,
    which strictly is incorrect as defined in the `Scoring` section of the official webpage (http://www.icbeb.org/CPSC2021)
    """
    _critical_points = list(critical_points)
    if 0 not in _critical_points:
        _critical_points.insert(0, 0)
        _af_intervals = [[itv[0] + 1, itv[1] + 1] for itv in af_intervals]
        if verbose >= 2:
            print(
                f"0 added to _critical_points, len(_critical_points): {len(_critical_points)-1} ==> {len(_critical_points)}"
            )
    else:
        _af_intervals = [[itv[0], itv[1]] for itv in af_intervals]
    # records with AFf mostly have `_critical_points` ending with `siglen-1`
    # but in some rare case ending with `siglen`
    if siglen - 1 in _critical_points:
        _critical_points[-1] = siglen
        if verbose >= 2:
            print(
                f"in _critical_points siglen-1 (={siglen-1}) changed to siglen (={siglen})"
            )
    elif siglen in _critical_points:
        pass
    else:
        _critical_points.append(siglen)
        if verbose >= 2:
            print(
                f"siglen (={siglen}) appended to _critical_points, len(_critical_points): {len(_critical_points)-1} ==> {len(_critical_points)}"
            )
    onset_score_mask, offset_score_mask = np.zeros((siglen,)), np.zeros((siglen,))
    for b, v in bias.items():
        mask_onset, mask_offset = np.zeros((siglen,)), np.zeros((siglen,))
        for itv in _af_intervals:
            onset_start = _critical_points[max(0, itv[0] - b)]
            # note that the onsets and offsets in `_af_intervals` already occupy positions in `_critical_points`
            onset_end = _critical_points[min(itv[0] + 1 + b, len(_critical_points) - 1)]
            if verbose > 0:
                print(
                    f"custom --- onset (c_ind, score {v}): {max(0, itv[0]-b)} --- {min(itv[0]+1+b, len(_critical_points)-1)}"
                )
                print(
                    f"custom --- onset (sample, score {v}): {_critical_points[max(0, itv[0]-b)]} --- {_critical_points[min(itv[0]+1+b, len(_critical_points)-1)]}"
                )
            mask_onset[onset_start:onset_end] = v
            # note that the onsets and offsets in `af_intervals` already occupy positions in `_critical_points`
            offset_start = _critical_points[max(0, itv[1] - 1 - b)]
            offset_end = _critical_points[min(itv[1] + b, len(_critical_points) - 1)]
            if verbose > 0:
                print(
                    f"custom --- offset (c_ind, score {v}): {max(0, itv[1]-1-b)} --- {min(itv[1]+b, len(_critical_points)-1)}"
                )
                print(
                    f"custom --- offset (sample, score {v}): {_critical_points[max(0, itv[1]-1-b)]} --- {_critical_points[min(itv[1]+b, len(_critical_points)-1)]}"
                )
            mask_offset[offset_start:offset_end] = v
        onset_score_mask = np.maximum(onset_score_mask, mask_onset)
        offset_score_mask = np.maximum(offset_score_mask, mask_offset)
    return onset_score_mask, offset_score_mask


# aliases
gen_endpoint_score_range = gen_endpoint_score_mask
compute_metrics = compute_challenge_metric
