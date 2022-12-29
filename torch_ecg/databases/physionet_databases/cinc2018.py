# -*- coding: utf-8 -*-
"""
You Snooze You Win - The PhysioNet Computing in Cardiology Challenge 2018
"""

from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Union, Sequence, List, Dict

import numpy as np
import pandas as pd
import scipy.signal as SS
import wfdb
from tqdm.auto import tqdm

from ...cfg import DEFAULTS
from ...utils import (
    add_docstring,
    get_record_list_recursive3,
    generalized_intervals_intersection,
)
from ..base import PhysioNetDataBase, DataBaseInfo, PSGDataBaseMixin


__all__ = [
    "CINC2018",
]


_CINC2018_INFO = DataBaseInfo(
    title="""
    You Snooze You Win - The PhysioNet Computing in Cardiology Challenge 2018
    """,
    about="""
    1. includes 1,985 subjects, partitioned into balanced training (n = 994), and test sets (n = 989)
    2. signals include
        electrocardiogram (ECG),
        electroencephalography (EEG),
        electrooculography (EOG),
        electromyography (EMG),
        electrocardiology (EKG),
        oxygen saturation (SaO2),
        etc.
    3. frequency of all signal channels is 200 Hz
    4. units of signals:
        mV for ECG, EEG, EOG, EMG, EKG
        percentage for SaO2
    5. six sleep stages were annotated in 30 second contiguous intervals:
        wakefulness,
        stage 1,
        stage 2,
        stage 3,
        rapid eye movement (REM),
        undefined
    6. annotated arousals were classified as either of the following:
        spontaneous arousals,
        respiratory effort related arousals (RERA),
        bruxisms,
        hypoventilations,
        hypopneas,
        apneas (central, obstructive and mixed),
        vocalizations,
        snores,
        periodic leg movements,
        Cheyne-Stokes breathing,
        partial airway obstructions
    """,
    usage=[
        "sleep stage",
        "sleep apnea",
    ],
    references=[
        "https://physionet.org/content/challenge-2018/1.0.0/",
    ],
    doi=[
        "10.22489/CinC.2018.049",
        "10.13026/6phb-r450",
    ],
)


@add_docstring(_CINC2018_INFO.format_database_docstring())
class CINC2018(PhysioNetDataBase, PSGDataBaseMixin):
    """ """

    __name__ = "CINC2018"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[str] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_dir: str or Path, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 1
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="challenge-2018",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 200
        self._subset = kwargs.get("subset", "training")
        self.rec_ext = "mat"
        self.ann_ext = "arousal"

        # fmt: off
        self.sleep_stage_names = ["W", "R", "N1", "N2", "N3"]
        self.arousal_types = [
            "arousal_bruxism", "arousal_noise", "arousal_plm", "arousal_rera", "arousal_snore", "arousal_spontaneous",
            "resp_centralapnea", "resp_cheynestokesbreath", "resp_hypopnea", "resp_hypoventilation",
            "resp_mixedapnea", "resp_obstructiveapnea", "resp_partialobstructive",
        ]
        # fmt: on

        self.training_rec_pattern = "^tr\\d{2}\\-\\d{4}.mat$"
        self.test_rec_pattern = "^te\\d{2}\\-\\d{4}.mat$"
        self.training_records = []
        self.test_records = []
        self._all_records = []
        self._df_records = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """ """
        self._df_records = pd.DataFrame()
        records = get_record_list_recursive3(
            self.db_dir,
            {"training": self.training_rec_pattern, "test": self.test_rec_pattern},
            relative=False,
        )
        for k in records:
            df_tmp = pd.DataFrame(sorted(records[k]), columns=["path"])
            df_tmp["subset"] = k
            self._df_records = pd.concat(
                [self._df_records, df_tmp], axis=0, ignore_index=True
            )
        self._df_records["record"] = self._df_records["path"].apply(
            lambda x: Path(x).stem
        )
        self._df_records["subject_id"] = self._df_records["record"].apply(
            self.get_subject_id
        )
        self._df_records.set_index("record", inplace=True)

        self._df_records["fs"] = None
        self._df_records["siglen"] = None
        self._df_records["available_signals"] = None
        with tqdm(
            self._df_records.iterrows(),
            total=len(self._df_records),
            mininterval=1.0,
            desc="Loading metadata",
            disable=self.verbose < 1,
        ) as pbar:
            for idx, row in pbar:
                header = wfdb.rdheader(row["path"])
                self._df_records.at[idx, "fs"] = header.fs
                self._df_records.at[idx, "siglen"] = header.sig_len
                self._df_records.at[idx, "available_signals"] = header.sig_name

        if self._subset is not None:
            self._df_records = self._df_records[
                self._df_records["subset"] == self._subset
            ]

        if self._subsample is not None:
            if self._subset is None:
                df_tmp = pd.DataFrame(columns=self._df_records.columns)
                for k in records:
                    size = int(round(self._subsample * len(records[k])))
                    if size > 0:
                        df_tmp = pd.concat(
                            [
                                df_tmp,
                                self._df_records[
                                    self._df_records["subset"] == k
                                ].sample(
                                    size, random_state=DEFAULTS.SEED, replace=False
                                ),
                            ],
                            axis=0,
                            ignore_index=True,
                        )
                if len(df_tmp) == 0:
                    size = min(
                        len(self._df_records),
                        max(1, int(round(self._subsample * len(self._df_records)))),
                    )
                    df_tmp = self._df_records.sample(
                        size, random_state=DEFAULTS.SEED, replace=False
                    )
                del self._df_records
                self._df_records = df_tmp.copy()
                del df_tmp
            else:
                size = min(
                    len(self._df_records),
                    max(1, int(round(self._subsample * len(self._df_records)))),
                )
                if size > 0:
                    self._df_records = self._df_records.sample(
                        size, random_state=DEFAULTS.SEED, replace=False
                    )

        self._all_records = self._df_records.index.tolist()
        self.training_records = self._df_records[
            self._df_records["subset"] == "training"
        ].index.tolist()
        self.test_records = self._df_records[
            self._df_records["subset"] == "test"
        ].index.tolist()

    def get_subject_id(self, rec: str) -> int:
        """
        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        pid: int,
            the `subject_id` corr. to `rec`

        """
        head = "2018"
        mid = rec[2:4]
        tail = rec[-4:]
        pid = int(head + mid + tail)
        return pid

    def set_subset(self, subset: Union[str, None]) -> None:
        """ """
        assert subset in [
            "training",
            "test",
            None,
        ], "`subset` must be in [training, test, None]"
        self._subset = subset
        self._ls_rec()

    def get_available_signals(self, rec: Union[str, int]) -> List[str]:
        """
        get the available signals of a record

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        signals: list of str,
            the available signal names of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self._df_records.at[rec, "available_signals"]

    def get_fs(self, rec: Union[str, int]) -> int:
        """
        get the sampling frequency of a record

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        fs: int,
            the sampling frequency of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self._df_records.at[rec, "fs"]

    def get_siglen(self, rec: Union[str, int]) -> int:
        """
        get the length of a record

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        siglen: int,
            the length of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self._df_records.at[rec, "siglen"]

    def load_psg_data(
        self,
        rec: Union[str, int],
        channel: Optional[Union[str, Sequence[str]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        physical: bool = True,
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """
        load PSG data of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        channel: str, optional,
            name of the channel of PSG,
            if None, then all channels will be returned
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain") which is valid only when only one `channel` is passed
        physical: bool, default True,
            if True, then the data will be converted to physical units
            otherwise, the data will be in digital units
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency

        Returns
        -------
        np.ndarray:
            PSG data of the channel `channel`

        """
        available_signals = self.get_available_signals(rec)
        chn = available_signals if channel is None else channel
        if isinstance(chn, str):
            chn = [chn]
        assert set(chn).issubset(
            set(available_signals)
        ), f"`channel` should be one of `{available_signals}`, but got `{chn}`"

        allowed_data_format = [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
            "flat",
            "plain",
        ]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"
        if len(chn) > 1:
            assert data_format.lower() in [
                "channel_first",
                "lead_first",
                "channel_last",
                "lead_last",
            ], (
                "`data_format` should be one of "
                "`['channel_first', 'lead_first', 'channel_last', 'lead_last']` "
                f"when the passed number of `channel` is larger than 1, but got `{data_format}`"
            )

        frp = str(self.get_absolute_path(rec))
        wfdb_header = wfdb.rdheader(frp)
        sampfrom = max(0, sampfrom or 0)
        sampto = min(sampto or wfdb_header.sig_len, wfdb_header.sig_len)
        wfdb_rec = wfdb.rdrecord(
            frp, sampfrom=sampfrom, sampto=sampto, channel_names=chn, physical=physical
        )

        ret_data = wfdb_rec.p_signal.T if physical else wfdb_rec.d_signal.T

        if fs is not None and fs != wfdb_header.fs:
            ret_data = SS.resample_poly(ret_data, fs, wfdb_header.fs, axis=-1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            ret_data = ret_data.T
        elif data_format.lower() in ["flat", "plain"]:
            ret_data = ret_data.flatten()

        return ret_data

    def load_data(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """
        load ECG data of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain")
        units: str or None, default "mV",
            units of the output signal, can also be "μV", with aliases of "uV", "muV";
            None for digital data, without digital-to-physical conversion
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency

        Returns
        -------
        np.ndarray:
            the ECG data loaded from `rec`, with given units and format

        """
        available_signals = self.get_available_signals(rec)
        assert (
            "ECG" in available_signals
        ), f"the record `{rec}` does not have ECG signal"
        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"
        data = self.load_psg_data(
            rec=rec,
            channel="ECG",
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            physical=units is not None,
            fs=fs,
        )
        if units.lower() in ["μv", "uv", "muv"]:
            data = 1000 * data
        return data

    @add_docstring(load_data.__doc__)
    def load_ecg_data(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """alias of `load_data`"""
        return self.load_data(
            rec=rec,
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            units=units,
            fs=fs,
        )

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
    ) -> Dict[str, Dict[str, List[List[int]]]]:
        """
        load sleep stage and arousal annotations of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the corresponding PSG data
        sampto: int, optional,
            end index of the corresponding PSG data
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        dict:
            a dictionary with keys "sleep_stages" and "arousals",
            each of which is a dictionary with keys of sleep stages and arousals,
            and values of lists of lists of start and end indices of the sleep stages and arousals

        """
        frp = str(self.get_absolute_path(rec))
        wfdb_ann = wfdb.rdann(frp, extension=self.ann_ext)

        sleep_stages = defaultdict(list)
        arousals = defaultdict(list)
        current_sleep_stage = None
        current_sleep_stage_start = None
        for aux_note, sample in zip(wfdb_ann.aux_note, wfdb_ann.sample.tolist()):
            if aux_note in self.sleep_stage_names:
                if current_sleep_stage is not None:
                    sleep_stages[current_sleep_stage].append(
                        [current_sleep_stage_start, sample]
                    )
                current_sleep_stage = aux_note
                current_sleep_stage_start = sample
            else:
                if "(" in aux_note:
                    current_arousal_start = sample
                else:
                    arousals[aux_note.strip(")")].append(
                        [current_arousal_start, sample]
                    )
        siglen = self.get_siglen(rec)
        if current_sleep_stage_start < siglen:
            sleep_stages[current_sleep_stage].append(
                [current_sleep_stage_start, siglen]
            )
        sampfrom = max(0, sampfrom or 0)
        sampto = min(sampto or siglen, siglen)
        sleep_stages = {
            k: generalized_intervals_intersection(
                v, [[sampfrom, sampto]], drop_degenerate=True
            )
            for k, v in sleep_stages.items()
        }
        sleep_stages = {k: v for k, v in sleep_stages.items() if len(v) > 0}
        arousals = {
            k: generalized_intervals_intersection(
                v, [[sampfrom, sampto]], drop_degenerate=True
            )
            for k, v in arousals.items()
        }
        arousals = {k: v for k, v in arousals.items() if len(v) > 0}
        if not keep_original:
            sleep_stages = {
                k: [[s - sampfrom, e - sampfrom] for s, e in v]
                for k, v in sleep_stages.items()
            }
            arousals = {
                k: [[s - sampfrom, e - sampfrom] for s, e in v]
                for k, v in arousals.items()
            }
        return {
            "sleep_stages": sleep_stages,
            "arousals": arousals,
        }

    def load_sleep_stages_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """
        load sleep stage annotations of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the corresponding PSG data
        sampto: int, optional,
            end index of the corresponding PSG data
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        dict:
            a dictionary with keys of sleep stages and
            values of lists of lists of start and end indices of the sleep stages

        """
        return self.load_ann(
            rec=rec,
            sampfrom=sampfrom,
            sampto=sampto,
            keep_original=keep_original,
        )["sleep_stages"]

    def load_arousals_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """
        load arousal annotations of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the corresponding PSG data
        sampto: int, optional,
            end index of the corresponding PSG data
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        dict:
            a dictionary with keys of arousals and
            values of lists of lists of start and end indices of the arousals

        """
        return self.load_ann(
            rec=rec,
            sampfrom=sampfrom,
            sampto=sampto,
            keep_original=keep_original,
        )["arousals"]

    def plot(self) -> None:
        """ """
        raise NotImplementedError

    def plot_ann(self, rec: Union[str, int]) -> tuple:
        """
        plot the sleep stage and arousal annotations of the record `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        tuple:
            a tuple of matplotlib figure and axis

        TODO
        ----
        plot arousals events

        """
        ann = self.load_ann(rec)
        sleep_stages = ann["sleep_stages"]
        arousals = ann["arousals"]
        stage_mask = self.sleep_stage_intervals_to_mask(sleep_stages)
        fig, ax = self.plot_hypnogram(stage_mask)
        # TODO: plot arousals events
        return fig, ax

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2018_INFO
