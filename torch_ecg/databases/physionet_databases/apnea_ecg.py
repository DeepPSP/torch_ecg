# -*- coding: utf-8 -*-
"""
"""

from datetime import datetime
from pathlib import Path
from typing import Any, List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
import wfdb

from ..base import PhysioNetDataBase

__all__ = [
    "ApneaECG",
]


class ApneaECG(PhysioNetDataBase):
    """finished, to be improved,

    Apnea-ECG Database

    ABOUT apnea-ecg (CinC 2000)
    ---------------------------
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
        10.1. *r.dat files contains respiration signals correspondingly, with 4 channels: "Resp C", "Resp A", "Resp N", "SpO2"
        10.2. *er.* files only contain annotations
        10.3. annotations for the respiration signals are identical to the corresponding ECG signals

    NOTE
    ----

    ISSUES
    ------

    Usage
    -----
    1. sleep apnea

    References
    ----------
    1. <a name="ref1"></a> https://physionet.org/content/apnea-ecg/1.0.0/
    2. <a name="ref2"></a> T Penzel, GB Moody, RG Mark, AL Goldberger, JH Peter. The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258

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
            db_name="apnea-ecg",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 100
        self.data_ext = "dat"
        self.ann_ext = "apn"
        self.qrs_ann_ext = "qrs"

        self._ls_rec()

        self.ecg_records = [r for r in self._all_records if "r" not in r]
        self.rsp_records = [r for r in self._all_records if "r" in r and "er" not in r]
        self.rsp_channels = ["Resp C", "Resp A", "Resp N", "SpO2"]
        self.learning_set = [r for r in self.ecg_records if "x" not in r]
        self.test_set = [r for r in self.ecg_records if "x" in r]
        self.control_group = [r for r in self.learning_set if "c" in r]
        self.borderline_group = [r for r in self.learning_set if "b" in r]
        self.apnea_group = [r for r in self.learning_set if "a" in r]

        self.sleep_event_keys = [
            "event_name",
            "event_start",
            "event_end",
            "event_duration",
        ]
        self.palette = {
            "Obstructive Apnea": "yellow",
        }

    def _ls_rec(self, local: bool = True) -> NoReturn:
        """
        find all records (relative path without file extension),
        and save into `self._all_records` for further use

        Parameters
        ----------
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`

        """
        try:
            super()._ls_rec(local=local)
        except Exception:
            self._all_records = wfdb.get_record_list(self.db_name)

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        int,
            subject id

        """
        if isinstance(rec, int):
            rec = self[rec]
        stoi = {"a": "1", "b": "2", "c": "3", "x": "4"}
        return int("2000" + stoi[rec[0]] + rec[1:3])

    def load_data(
        self,
        rec: Union[str, int],
        lead: int = 0,
        rec_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        lead: int, default 0
            number of the lead, can be 0 or 1
        rec_path: str or Path, optional,
            path of the file which contains the ECG data,
            if not given, default path will be used

        Returns
        -------
        sig: np.ndarray,
            the ECG signal or the respiration signal

        """
        if isinstance(rec, int):
            rec = self[rec]
        file_path = str(rec_path or (self.db_dir / rec))
        self.wfdb_rec = wfdb.rdrecord(file_path)
        sig = self.wfdb_rec.p_signal
        if not rec.endswith(("r", "er")):
            sig = sig[:, 0]  # flatten ECG signal
        return sig

    def load_ecg_data(
        self, rec: Union[str, int], lead: int = 0, rec_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        lead: int, default 0
            number of the lead, can be 0 or 1
        rec_path: str or Path, optional,
            path of the file which contains the ecg data,
            if not given, default path will be used

        Returns
        -------
        sig: np.ndarray,
            the ecg signal

        """
        if isinstance(rec, int):
            rec = self[rec]
        if rec.endswith(("r", "er")):
            raise ValueError(f"{rec} is not a record of ECG signals")
        return self.load_data(rec=rec, lead=lead, rec_path=rec_path)

    def load_rsp_data(
        self,
        rec: Union[str, int],
        lead: int = 0,
        channels: Optional[Union[str, List[str], Tuple[str]]] = None,
        rec_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        lead: int, default 0
            number of the lead, can be 0 or 1
        channels: str or list of str, default None
            channels to be loaded, if None, all channels will be loaded
        rec_path: str or Path, optional,
            path of the file which contains the ecg data,
            if not given, default path will be used

        Returns
        -------
        sig: np.ndarray,
            the respiration signal

        """
        if isinstance(rec, int):
            rec = self[rec]
        if not rec.endswith(("r", "er")):
            raise ValueError(f"{rec} is not a record of RSP signals")
        sig = self.load_data(rec=rec, lead=lead, rec_path=rec_path)
        if channels is not None:
            chns = [channels] if isinstance(channels, str) else list(channels)
            if any([c not in self.rsp_channels for c in chns]):
                raise ValueError(
                    f"Invalid channel(s): {[c for c in chns if c not in self.rsp_channels]}"
                )
        else:
            chns = self.rsp_channels
        sig = {c: sig[:, self.wfdb_rec.sig_name.index(c)] for c in chns}
        return sig

    def load_ann(
        self,
        rec: Union[str, int],
        ann_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> list:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann_path: str or Path, optional,
            path of the file which contains the annotations,
            if not given, default path will be used

        Returns
        -------
        detailed_ann: list,
            annotations of the form [idx, ann]

        """
        if isinstance(rec, int):
            rec = self[rec]
        file_path = str(ann_path or (self.db_dir / rec))
        extension = kwargs.get("extension", "apn")
        self.wfdb_ann = wfdb.rdann(file_path, extension=extension)
        detailed_ann = [
            [si // (self.fs * 60), sy]
            for si, sy in zip(self.wfdb_ann.sample, self.wfdb_ann.symbol)
        ]
        return detailed_ann

    def load_apnea_event(
        self, rec: Union[str, int], ann_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann_path: str, optional,
            path of the file which contains the annotations,
            if not given, default path will be used

        Returns
        -------
        df_apnea_ann: DataFrame,
            apnea annotations with columns "event_start","event_end", "event_name", "event_duration"

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

        if len(apnea_periods) > 0:
            self.logger.info(
                f"apnea period(s) (units in minutes) of record {rec} is(are): {apnea_periods}"
            )
        else:
            self.logger.info(f"record {rec} has no apnea period")

        if len(apnea_periods) == 0:
            return pd.DataFrame(columns=self.sleep_event_keys)

        apnea_periods = np.array(
            [[60 * p[0], 60 * p[1]] for p in apnea_periods]
        )  # minutes to seconds
        apnea_periods = np.array(apnea_periods, dtype=int).reshape(
            (len(apnea_periods), 2)
        )

        df_apnea_ann = pd.DataFrame(apnea_periods, columns=["event_start", "event_end"])
        df_apnea_ann["event_name"] = "Obstructive Apnea"
        df_apnea_ann["event_duration"] = df_apnea_ann.apply(
            lambda row: row["event_end"] - row["event_start"], axis=1
        )

        df_apnea_ann = df_apnea_ann[self.sleep_event_keys]

        return df_apnea_ann

    def plot_ann(
        self, rec: Union[str, int], ann_path: Optional[str] = None
    ) -> NoReturn:
        """
        Parameters
        ----------
        rec: str or int,
            name or index of the record
        ann_path: str, optional,
            path of the file which contains the annotations,
            if not given, default path will be used

        """
        df_apnea_ann = self.load_apnea_event(rec, ann_path)
        self._plot_ann(df_apnea_ann)

    def _plot_ann(self, df_apnea_ann: pd.DataFrame) -> NoReturn:
        """
        Parameters
        ----------
        df_apnea_ann: DataFrame,
            apnea events with columns `self.sleep_event_keys`

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
            ax.legend(
                handles=[patches[k] for k in self.palette.keys()], loc="best"
            )  # keep ordering
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="y", which="both", length=0)
