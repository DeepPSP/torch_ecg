# -*- coding: utf-8 -*-

from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import wfdb

from ...cfg import DEFAULTS
from ...utils import add_docstring
from ..base import DataBaseInfo, PhysioNetDataBase

__all__ = [
    "ApneaECG",
]


_ApneaECG_INFO = DataBaseInfo(
    title="""
    Apnea-ECG Database - The PhysioNet Computing in Cardiology Challenge 2000
    """,
    about="""
    1. consist of 70 single lead ECG records, divided into a learning set of 35 records (a01 through a20, b01 through b05, and c01 through c10), and a test set of 35 records (x01 through x35)
    2. recordings vary in length from slightly less than 7 hours to nearly 10 hours (401 - 578 min) each
    3. control group (c01 through c10): records having fewer than 5 min of disorder breathing
    4. borderline group (b01 through b05): records having 10-96 min of disorder breathing
    5. apnea group (a01 through a20): records having 100 min or more of disorder breathing
    6. .dat files contain the digitized ECGs, and respiration signals, all with frequency 100 Hz
    7. .apn files are (binary) annotation files (only for the learning set), containing an annotation for each minute of each recording indicating the presence or absence of apnea at that time. labels are in the member "symbol", "N" for normal, "A" for apnea
    8. .qrs files are machine-generated (binary) annotation files, unaudited and containing errors, provided for the convenience of those who do not wish to use their own QRS detectors
    9. c05 and c06 come from the same original recording (c05 begins 80 seconds later than c06). c06 may have been a corrected version of c05
    10. eight records (a01 through a04, b01, and c01 through c03) that include respiration signals have several additional files each:

        - \\*r.dat files contains respiration signals correspondingly, with 4 channels: "Resp C", "Resp A", "Resp N", "SpO2"
        - \\*er.\\* files only contain annotations
        - annotations for the respiration signals are identical to the corresponding ECG signals

    11. Webpage of the database on PhysioNet [1]_. Paper describing the database [2]_.
    """,
    usage=[
        "Sleep apnea analysis",
    ],
    references=[
        "https://physionet.org/content/apnea-ecg/",
        "T Penzel, GB Moody, RG Mark, AL Goldberger, JH Peter. The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258",
    ],
    doi=[
        "10.1109/cic.2000.898505",
        "10.13026/C23W2R",
    ],
)


@add_docstring(_ApneaECG_INFO.format_database_docstring(), mode="prepend")
class ApneaECG(PhysioNetDataBase):
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

    __name__ = "ApneaECG"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="apnea-ecg",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 100
        self.data_ext = "dat"
        self.data_pattern = "^[abcx]\\d{2}$"
        self.ann_ext = "apn"
        self.qrs_ann_ext = "qrs"

        self._ecg_records = None
        self._rsp_records = None
        self.rsp_channels = None
        self.learning_set = None
        self.test_set = None
        self.control_group = None
        self.borderline_group = None
        self.apnea_group = None
        self._ls_rec()

        self.sleep_event_keys = [
            "event_name",
            "event_start",
            "event_end",
            "event_duration",
        ]
        self.palette = {
            "Obstructive Apnea": "yellow",
        }

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        subsample = self._subsample
        self._subsample = None  # so that no subsampling in super()._ls_rec()
        super()._ls_rec()

        if len(self.all_records) == 0:
            return

        self._rsp_records = self._df_records[self._df_records.index.str.match("^[abcx]\\d{2}r$")].index.tolist()
        ecg_records = self._df_records[self._df_records.index.str.match(self.data_pattern)].index.tolist()
        if subsample is not None:
            size = min(
                len(ecg_records),
                max(1, int(round(len(ecg_records) * subsample))),
            )
            self.logger.debug(f"subsample `{size}` records from `{len(ecg_records)}` `ecg_records`")
            self._ecg_records = sorted(DEFAULTS.RNG.choice(ecg_records, size=size, replace=False).tolist())
            self._df_records = self._df_records.loc[sorted(self._ecg_records + self._rsp_records)]
            del ecg_records
        else:
            self._ecg_records = ecg_records
        self._subsample = subsample
        self._all_records = self._df_records.index.tolist()

        self.rsp_channels = ["Resp C", "Resp A", "Resp N", "SpO2"]
        self.learning_set = [r for r in self._ecg_records if "x" not in r]
        self.test_set = [r for r in self._ecg_records if "x" in r]
        self.control_group = [r for r in self.learning_set if "c" in r]
        self.borderline_group = [r for r in self.learning_set if "b" in r]
        self.apnea_group = [r for r in self.learning_set if "a" in r]

    def __len__(self) -> int:
        return len(self.ecg_records)

    def __getitem__(self, index: int) -> str:
        return self.ecg_records[index]

    @property
    def ecg_records(self) -> List[str]:
        """The list of ECG records in the database."""
        return self._ecg_records

    @property
    def rsp_records(self) -> List[str]:
        """The list of respiration records in the database."""
        return self._rsp_records

    def get_subject_id(self, rec: Union[str, int]) -> int:
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
        if isinstance(rec, int):
            rec = self[rec]
        stoi = {"a": "1", "b": "2", "c": "3", "x": "4"}
        return int("2000" + stoi[rec[0]] + rec[1:3])

    @add_docstring(
        PhysioNetDataBase.load_data.__doc__.replace(
            "ECG data",
            "ECG data or respiration data",
        )
    )
    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        return super().load_data(rec, leads, sampfrom, sampto, data_format, units, fs, return_fs)

    @add_docstring(PhysioNetDataBase.load_data.__doc__)
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
        if isinstance(rec, int):
            rec = self[rec]
        if rec not in self.ecg_records:
            raise ValueError(f"`{rec}` is not a record of ECG signals")
        return self.load_data(
            rec,
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            units=units,
            fs=fs,
            return_fs=return_fs,
        )

    @add_docstring(
        PhysioNetDataBase.load_data.__doc__.replace(
            "ECG data",
            "respiration data",
        )
    )
    def load_rsp_data(
        self,
        rec: str,
        channels: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> np.ndarray:
        if rec not in self.rsp_records:
            raise ValueError(f"`{rec}` is not a record of RSP signals")
        data = self.load_data(
            rec,
            leads=channels,
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            units=units,
            fs=fs,
            return_fs=return_fs,
        )
        return data

    def load_ann(
        self,
        rec: Union[str, int],
        ann_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> list:
        """Load annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        ann_path : str or pathlib.Path, optional
            Path of the file which contains the annotations.
            If is None, default path will be used.

        Returns
        -------
        detailed_ann : list
            List of annotations of the form ``[idx, ann]``.

        """
        if isinstance(rec, int):
            rec = self[rec]
        file_path = str(ann_path or self.get_absolute_path(rec))
        extension = kwargs.get("extension", "apn")
        wfdb_ann = wfdb.rdann(file_path, extension=extension)
        detailed_ann = [[si // (self.fs * 60), sy] for si, sy in zip(wfdb_ann.sample.tolist(), wfdb_ann.symbol)]
        return detailed_ann

    def load_apnea_event(self, rec: Union[str, int], ann_path: Optional[str] = None) -> pd.DataFrame:
        """Load annotations of apnea events of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        ann_path : str or pathlib.Path, optional
            Path of the file which contains the annotations.
            If is None, default path will be used.

        Returns
        -------
        df_apnea_ann : pandas.DataFrame
            Apnea annotations with columns
            "event_start", "event_end", "event_name", "event_duration"
            (ref. `self.sleep_event_keys`).

        """
        if isinstance(rec, int):
            rec = self[rec]
        detailed_anno = self.load_ann(rec, ann_path, extension="apn")
        apnea = np.array([p[0] for p in detailed_anno if p[1] == "A"])

        if len(apnea) > 0:
            apnea_endpoints = [apnea[0]]
            # TODO: check if split_indices is correctly computed
            split_indices = np.where(np.diff(apnea) > 1)[0].tolist()
            for i in split_indices:
                apnea_endpoints += [apnea[i], apnea[i + 1]]
            apnea_endpoints.append(apnea[-1])

            apnea_periods = []
            for i in range(len(apnea_endpoints) // 2):
                pe = [apnea_endpoints[2 * i], apnea_endpoints[2 * i + 1]]
                apnea_periods.append(pe)
        else:
            apnea_periods = []

        if len(apnea_periods) == 0:
            return pd.DataFrame(columns=self.sleep_event_keys)

        apnea_periods = np.array([[60 * p[0], 60 * p[1]] for p in apnea_periods])  # minutes to seconds
        apnea_periods = np.array(apnea_periods, dtype=int).reshape((len(apnea_periods), 2))

        df_apnea_ann = pd.DataFrame(apnea_periods, columns=["event_start", "event_end"])
        df_apnea_ann["event_name"] = "Obstructive Apnea"
        df_apnea_ann["event_duration"] = df_apnea_ann.apply(lambda row: row["event_end"] - row["event_start"], axis=1)

        df_apnea_ann = df_apnea_ann[self.sleep_event_keys]

        return df_apnea_ann

    def plot_ann(self, rec: Union[str, int], ann_path: Optional[str] = None) -> None:
        """Plot annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        ann_path : str or pathlib.Path, optional
            Path of the file which contains the annotations.
            If is None, default path will be used.

        Returns
        -------
        None

        """
        df_apnea_ann = self.load_apnea_event(rec, ann_path)
        self._plot_ann(df_apnea_ann)

    def _plot_ann(self, df_apnea_ann: pd.DataFrame) -> None:
        """Internal function to plot annotations of the record.

        Parameters
        ----------
        df_apnea_ann : DataFrame,
            Apnea events with columns
            "event_start", "event_end", "event_name", "event_duration"
            (ref. `self.sleep_event_keys` or :meth:`load_apnea_event`).

        Returns
        -------
        None

        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        patches = {k: mpatches.Patch(color=c, label=k) for k, c in self.palette.items()}
        _, ax = plt.subplots(figsize=(20, 4))
        plot_alpha = 0.5
        for _, row in df_apnea_ann.iterrows():
            ax.axvspan(
                datetime.fromtimestamp(row["event_start"]),
                datetime.fromtimestamp(row["event_end"]),
                color=self.palette[row["event_name"]],
                alpha=plot_alpha,
            )
            ax.legend(handles=[patches[k] for k in self.palette.keys()], loc="best")  # keep ordering
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", which="both", length=0)

    @property
    def database_info(self) -> DataBaseInfo:
        return _ApneaECG_INFO
