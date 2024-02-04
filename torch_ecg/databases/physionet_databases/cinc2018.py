# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.signal as SS
import wfdb
from tqdm.auto import tqdm

from ...cfg import DEFAULTS
from ...utils import add_docstring, generalized_intervals_intersection, get_record_list_recursive3
from ..base import DataBaseInfo, PhysioNetDataBase, PSGDataBaseMixin

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

    7. Webpage of the database on PhysioNet [1]_.
    """,
    usage=[
        "sleep stage",
        "sleep apnea",
    ],
    references=[
        "https://physionet.org/content/challenge-2018/",
    ],
    doi=[
        "10.22489/CinC.2018.049",
        "10.13026/6phb-r450",
    ],
)


@add_docstring(_CINC2018_INFO.format_database_docstring(), mode="prepend")
class CINC2018(PhysioNetDataBase, PSGDataBaseMixin):
    """
    Parameters
    ----------
    db_dir : `path-like`, optional
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "CINC2018"

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
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
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        self._df_records = pd.DataFrame()
        records = get_record_list_recursive3(
            self.db_dir,
            {"training": self.training_rec_pattern, "test": self.test_rec_pattern},
            relative=False,
        )
        for k in records:
            df_tmp = pd.DataFrame(sorted(records[k]), columns=["path"])
            df_tmp["subset"] = k
            self._df_records = pd.concat([self._df_records, df_tmp], axis=0, ignore_index=True)
        self._df_records["record"] = self._df_records["path"].apply(lambda x: Path(x).stem)
        self._df_records["subject_id"] = self._df_records["record"].apply(self.get_subject_id)
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
            self._df_records = self._df_records[self._df_records["subset"] == self._subset]

        if self._subsample is not None:
            if self._subset is None:
                df_tmp = pd.DataFrame(columns=self._df_records.columns)
                for k in records:
                    size = int(round(self._subsample * len(records[k])))
                    if size > 0:
                        df_tmp = pd.concat(
                            [
                                df_tmp,
                                self._df_records[self._df_records["subset"] == k].sample(
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
                    df_tmp = self._df_records.sample(size, random_state=DEFAULTS.SEED, replace=False)
                del self._df_records
                self._df_records = df_tmp.copy()
                del df_tmp
            else:
                size = min(
                    len(self._df_records),
                    max(1, int(round(self._subsample * len(self._df_records)))),
                )
                if size > 0:
                    self._df_records = self._df_records.sample(size, random_state=DEFAULTS.SEED, replace=False)

        self._all_records = self._df_records.index.tolist()
        self.training_records = self._df_records[self._df_records["subset"] == "training"].index.tolist()
        self.test_records = self._df_records[self._df_records["subset"] == "test"].index.tolist()

    def get_subject_id(self, rec: str) -> int:
        """Attach a unique subject ID for the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        int
            Subject ID associated with the record.

        """
        head = "2018"
        mid = rec[2:4]
        tail = rec[-4:]
        pid = int(head + mid + tail)
        return pid

    def set_subset(self, subset: Union[str, None]) -> None:
        """Set the subset of the database to use."""
        assert subset in [
            "training",
            "test",
            None,
        ], """`subset` must be in ``["training", "test", None]``."""
        self._subset = subset
        self._ls_rec()

    def get_available_signals(self, rec: Union[str, int]) -> List[str]:
        """Get the available signals of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        signals : List[str]
            Names of available signal of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self._df_records.at[rec, "available_signals"]

    def get_fs(self, rec: Union[str, int]) -> int:
        """Get the sampling frequency of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        fs : int
            Sampling frequency of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        return self._df_records.at[rec, "fs"]

    def get_siglen(self, rec: Union[str, int]) -> int:
        """Get the length of the signal of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        siglen : int
            Length of the signal of the record.

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
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load PSG data of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        channel : str, optional
            Nname of the channel of PSG data.
            If is None, all channels will be returned.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format: str, default "channel_first".
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain") which is valid
            only when only one `channel` is passed.
        physical : bool, default True
            If True, the data will be converted to physical units,
            otherwise, the data will be in digital units.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default False
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            PSG data corr. to the given `channel` of the record.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.

        """
        available_signals = self.get_available_signals(rec)
        chn = available_signals if channel is None else channel
        if isinstance(chn, str):
            chn = [chn]
        assert set(chn).issubset(set(available_signals)), f"`channel` should be one of `{available_signals}`, but got `{chn}`"

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
        wfdb_rec = wfdb.rdrecord(frp, sampfrom=sampfrom, sampto=sampto, channel_names=chn, physical=physical)

        ret_data = wfdb_rec.p_signal.T if physical else wfdb_rec.d_signal.T

        if fs is not None and fs != wfdb_header.fs:
            ret_data = SS.resample_poly(ret_data, fs, wfdb_header.fs, axis=-1)
            data_fs = fs
        else:
            data_fs = wfdb_header.fs

        if data_format.lower() in ["channel_last", "lead_last"]:
            ret_data = ret_data.T
        elif data_format.lower() in ["flat", "plain"]:
            ret_data = ret_data.flatten()

        if return_fs:
            return ret_data, data_fs
        return ret_data

    def load_data(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load ECG data of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        leads : str or int or Sequence[str] or Sequence[int], optional
            The leads of the ECG data to load.
            None or "all" for all leads.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain") which is valid only when `leads` is a single lead
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV").
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default False
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            The ECG data loaded from the record,
            with given `units` and `data_format`.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        """
        available_signals = self.get_available_signals(rec)
        assert "ECG" in available_signals, f"the record `{rec}` does not have ECG signal"
        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"
        data, data_fs = self.load_psg_data(
            rec=rec,
            channel="ECG",
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            physical=units is not None,
            fs=fs,
            return_fs=True,
        )
        if units.lower() in ["μv", "uv", "muv"]:
            data = 1000 * data

        if return_fs:
            return data, data_fs
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
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """alias of `load_data`"""
        return self.load_data(
            rec=rec,
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            units=units,
            fs=fs,
            return_fs=return_fs,
        )

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
    ) -> Dict[str, Dict[str, List[List[int]]]]:
        """Load sleep stage and arousal annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the corresponding PSG data.
        sampto : int, optional
            End index of the corresponding PSG data.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        dict
            A dictionary with keys "sleep_stages" and "arousals",
            each of which is a dictionary with keys of sleep stages and arousals,
            and values of lists of lists of start and
            end indices of the sleep stages and arousals.

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
                    sleep_stages[current_sleep_stage].append([current_sleep_stage_start, sample])
                current_sleep_stage = aux_note
                current_sleep_stage_start = sample
            else:
                if "(" in aux_note:
                    current_arousal_start = sample
                else:
                    arousals[aux_note.strip(")")].append([current_arousal_start, sample])
        siglen = self.get_siglen(rec)
        if current_sleep_stage_start < siglen:
            sleep_stages[current_sleep_stage].append([current_sleep_stage_start, siglen])
        sampfrom = max(0, sampfrom or 0)
        sampto = min(sampto or siglen, siglen)
        sleep_stages = {
            k: generalized_intervals_intersection(v, [[sampfrom, sampto]], drop_degenerate=True)
            for k, v in sleep_stages.items()
        }
        sleep_stages = {k: v for k, v in sleep_stages.items() if len(v) > 0}
        arousals = {
            k: generalized_intervals_intersection(v, [[sampfrom, sampto]], drop_degenerate=True) for k, v in arousals.items()
        }
        arousals = {k: v for k, v in arousals.items() if len(v) > 0}
        if not keep_original:
            sleep_stages = {k: [[s - sampfrom, e - sampfrom] for s, e in v] for k, v in sleep_stages.items()}
            arousals = {k: [[s - sampfrom, e - sampfrom] for s, e in v] for k, v in arousals.items()}
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
        """Load sleep stage annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the corresponding PSG data.
        sampto : int, optional
            End index of the corresponding PSG data.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        dict
            A dictionary with keys of sleep stages and
            values of lists of lists of start and
            end indices of the sleep stages.

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
        """Load arousal annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sampfrom : int, optional
            Start index of the corresponding PSG data.
        sampto : int, optional
            End index of the corresponding PSG data.
        keep_original : bool, default False
            If True, indices will keep the same with the annotation file,
            otherwise subtract `sampfrom` if specified.

        Returns
        -------
        dict
            A dictionary with keys of arousals and
            values of lists of lists of start and
            end indices of the arousals.

        """
        return self.load_ann(
            rec=rec,
            sampfrom=sampfrom,
            sampto=sampto,
            keep_original=keep_original,
        )["arousals"]

    def plot(self) -> None:
        """NOT implemented yet."""
        raise NotImplementedError

    def plot_ann(self, rec: Union[str, int]) -> tuple:
        """Plot the sleep stage and arousal annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.

        TODO
        ----
        Plot arousals events.

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
