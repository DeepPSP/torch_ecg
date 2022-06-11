# -*- coding: utf-8 -*-
"""
"""

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence, Dict, List, Union

import numpy as np
import pandas as pd
import h5py

from ...utils.download import http_get
from ...utils.misc import get_record_list_recursive3
from ...utils import EAK
from ..base import DEFAULT_FIG_SIZE_PER_SEC, _DataBase  # noqa: F401


__all__ = [
    "SPHDataBase",
]


class SPHDataBase(_DataBase):
    """ """

    __name__ = "SPHDataBase"

    def __init__(
        self,
        db_name: str,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str or Path, optional,
            storage path of the database,
            if not specified, `wfdb` will fetch data from the website of PhysioNet
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name=db_name,
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "h5"
        self.ann_ext = None
        self.header_ext = None

        self.fs = 500
        self.all_leads = deepcopy(EAK.Standard12Leads)

        self._version = "v1"

        self._df_code = None
        self._df_metadata = None
        self._all_records = None
        self._ls_rec()

    def _ls_rec_local(self) -> NoReturn:
        """
        find all records in `self.db_dir`

        """
        record_list_fp = self.db_dir / "RECORDS"
        self._df_records = pd.DataFrame()
        write_file = False
        if record_list_fp.is_file():
            self._df_records["record"] = [
                item
                for item in record_list_fp.read_text().splitlines()
                if len(item) > 0
            ]
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: (self.db_dir / x).resolve()
            )
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.name
            )
        if len(self._df_records) == 0:
            write_file = True
            print(
                "Please wait patiently to let the reader find all records of the database from local storage..."
            )
            start = time.time()
            record_pattern = "A[\\d]{5}\\.h5"
            self._df_records["path"] = get_record_list_recursive3(
                self.db_dir, record_pattern, relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
            print(f"Done in {time.time() - start:.3f} seconds!")
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.name
            )
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()
        if write_file:
            record_list_fp.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

        self._df_code = pd.read_csv(self.db_dir / "code.csv").astype(str)
        self._df_metadata = pd.read_csv(self.db_dir / "metadata.csv")

    def get_subject_id(self, rec: Union[str, int]) -> str:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        sid: str,
            a `subject_id` attached to the record `rec`

        """
        if isinstance(rec, int):
            rec = self[rec]
        sid = self._df_metadata.loc[self._df_metadata["ECG_ID"] == rec]["Patient_ID"][0]
        return sid

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, List[str]]] = None,
        data_format: str = "channel_first",
        units: str = "mV",
    ) -> np.ndarray:
        """
        load ECG data from h5 file

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default "channel_first",
            format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        backend: str, default "wfdb",
            the backend data reader, can also be "scipy"
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        assert data_format.lower() in [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
        ]
        if leads is None or leads == "all":
            _leads = list(range(len(self.all_leads)))
        elif isinstance(leads, str):
            _leads = [self.all_leads.index(leads)]
        else:
            _leads = [self.all_leads.index(lead) for lead in leads]

        with h5py.File(self.get_absolute_path(rec), "r") as f:
            data = f["ecg"][_leads].astype(np.float32)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    def load_ann(self, rec: Union[str, int], ann_format: str = "a") -> List[str]:
        """
        load annotation from the metadata file

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        fmt: str, default "a",
            the format of labels, one of the following (case insensitive):
            - "a", abbreviations
            - "f", full names
            - "c", AHACode

        Returns
        -------
        labels: list,
            the list of labels

        """
        if isinstance(rec, int):
            rec = self[rec]
        labels = [
            lb.strip()
            for lb in self._df_metadata[self._df_metadata["ECG_ID"] == rec]["AHA_Code"][
                0
            ].split(";")
        ]
        if ann_format.lower() == "f":
            labels = [
                self._df_code[self._df_code["Code"] == lb]["Description"][0]
                for lb in labels
            ]
        elif ann_format.lower() == "a":
            raise NotImplementedError("Abbreviations are not supported yet")
        elif ann_format.lower() == "c":
            pass
        else:
            raise ValueError(f"Unknown annotation format: {ann_format}")
        return labels

    def get_subject_info(
        self, rec_or_sid: Union[str, int], items: Optional[List[str]] = None
    ) -> dict:
        """

        read auxiliary information of a subject (a record) stored in the header files

        Parameters
        ----------
        rec_or_sid: str or int,
            record name or index of the record in `self.all_records`,
            or the subject ID
        items: list of str, optional,
            items of the subject"s information (e.g. sex, age, etc.)

        Returns
        -------
        subject_info: dict,
            information about the subject, including
            "age", "sex",

        """
        if isinstance(rec_or_sid, int):
            rec_or_sid = self[rec_or_sid]
            row = self._df_metadata[self._df_metadata["ECG_ID"] == rec_or_sid].iloc[0]
        else:
            if rec_or_sid.startswith("A"):
                row = self._df_metadata[self._df_metadata["ECG_ID"] == rec_or_sid].iloc[
                    0
                ]
            else:
                row = self._df_metadata[
                    self._df_metadata["Patient_ID"] == rec_or_sid
                ].iloc[0]
        if items is None or len(items) == 0:
            info_items = [
                "age",
                "sex",
            ]
        else:
            info_items = items
        subject_info = {item: row[item] for item in info_items}

        return subject_info

    def get_age(self, rec: Union[str, int]) -> int:
        """
        get the age of the subject of the record

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        age: int,
            the age of the subject

        """
        if isinstance(rec, int):
            rec = self[rec]
        age = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["Age"][0]
        return age

    def get_sex(self, rec: Union[str, int]) -> str:
        """
        get the sex of the subject of the record

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        sex: str,
            the sex of the subject

        """
        if isinstance(rec, int):
            rec = self[rec]
        sex = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["Sex"][0]
        return sex

    def get_siglen(self, rec: Union[str, int]) -> str:
        """
        get the length of the ECG signal of the record

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        siglen: str,
            the signal length of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        siglen = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["N"][0]
        return siglen

    @property
    def url(self) -> Dict[str, str]:
        return {
            "metadata.csv": "https://springernature.figshare.com/ndownloader/files/34793152",
            "code.csv": "https://springernature.figshare.com/ndownloader/files/32630954",
            "records.tar": "https://springernature.figshare.com/ndownloader/files/32630684",
        }

    def download(self, files: Optional[Union[str, Sequence[str]]]) -> NoReturn:
        """
        download the database from the figshare website
        """
        if files is None:
            files = self.url.keys()
        if isinstance(files, str):
            files = [files]
        assert set(files).issubset(self.url)
        for filename in files:
            url = self.url[filename]
            if not (self.db_dir / filename).is_file():
                http_get(url, self.db_dir, filename=filename)

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, List[str]]] = None,
        same_range: bool = False,
        waves: Optional[Dict[str, Sequence[int]]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        data: ndarray, optional,
            (12-lead) ECG signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
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
        `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once

        Contributors: Jeethan, and WEN Hao

        """
        raise NotImplementedError
