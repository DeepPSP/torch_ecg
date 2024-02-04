# -*- coding: utf-8 -*-

import os
import re
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import scipy.signal as SS

from ...cfg import DEFAULTS
from ...utils.download import http_get
from ...utils.misc import add_docstring
from ..base import DataBaseInfo, _DataBase

__all__ = [
    "CACHET_CADB",
]


_CACHET_CADB_INFO = DataBaseInfo(  # NOT finished yet
    title="""
    CACHET-CADB: A Contextualized Ambulatory Electrocardiography Arrhythmia Dataset
    """,
    about=r"""
    1. The database has 259 days of contextualized ECG recordings from 24 patients and 1,602 manually annotated 10 s heart-rhythm samples.
    2. The length of the ECG records in the CACHET-CADB varies from 24 h to 3 weeks.
    3. The patient's ambulatory context information (activities, movement acceleration, body position, etc.) is extracted for every 10 s interval cumulatively.
    4. nearly 11% of the ECG data in the database is found to be noisy.
    5. Webpages for downloading the database [1]_ and the short-format database [2]_, see also the GitHub repository [3]_.
    """,
    usage=[
        "ECG arrhythmia detection",
        "Self-Supervised Learning",
    ],
    references=[
        "https://data.dtu.dk/articles/dataset/CACHET-CADB/14547264",
        "https://data.dtu.dk/articles/dataset/CACHET-CADB_Short_Format/14547330",
        "https://github.com/cph-cachet/cachet-ecg-db/",
    ],
    doi=[
        "10.3389/fcvm.2022.893090",
        "10.11583/DTU.14547264",
        "10.11583/DTU.14547330",
    ],
)


@add_docstring(_CACHET_CADB_INFO.format_database_docstring(), mode="prepend")
class CACHET_CADB(_DataBase):
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
        Auxilliary key word arguments

    """

    __name__ = "CACHET_CADB"

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="CACHET-CADB",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "ecg.bin"
        self.ann_ext = "annotation.csv"
        self.context_data_ext = {
            "acc": "acc.bin",
            "angularrate": "angularrate.bin",
            "hr_live": "hr_live.bin",
            "hrvrmssd_live": "hrvrmssd_live.bin",
            "marker": "marker.csv",
            "movementacceleration_live": "movementacceleration_live.bin",
            "press": "press.bin",
        }
        self.header_ext = "unisens.xml"
        self.header_keys = [
            "adcResolution",
            "baseline",
            "contentClass",
            "dataType",
            "id",
            "lsbValue",
            "sampleRate",
            "unit",
        ]
        self.context_ann_ext = "context.xlsx"

        self.class_map = {1: "AF", 2: "NSR", 3: "Noise", 4: "Others"}
        self.body_position_map = {
            0: "unknown",
            1: "lying supine",
            2: "lying left",
            3: "lying prone",
            4: "lying right",
            5: "upright",
            6: "sitting/lying",
            7: "standing",
        }
        self.activity_map = {
            0: "unknown",
            1: "lying",
            2: "sitting/standing",
            3: "cycling",
            4: "slope up",
            5: "jogging",
            6: "slope down",
            7: "walking",
            8: "sitting/lying",
            9: "standing",
        }
        self.wear_map = {0: "wear", 1: "non-wear"}

        self.fs = 1024

        self._version = "v1"

        self._subject_pattern = "^P[\\d]{1,2}|PNSR\\-\\d$"
        self._df_metadata = None
        self._metadata_cols = [
            "age",
            "gender",
            "weight",
            "height",
            "comment",
            "duration",
            "measurementId",
            "timestampStart",
            "version",
            "sensorLocation",
            "sensorVersion",
            "sensorType",
            "personId",
        ]
        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._full_data_dir = None
        self._full_ann_dir = None
        self._short_format_file = None
        self.__short_format_data = None
        self.__short_format_ann = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.

        TODO: impelement subsampling for the long format data.
        """
        # short format file
        self._short_format_file = list(self.db_dir.rglob("*.hdf5"))
        if len(self._short_format_file) == 1:
            self._short_format_file = self._short_format_file[0]
            with h5py.File(self._short_format_file, "r") as f:
                self.__short_format_data = f["signal"][:].astype(DEFAULTS.DTYPE.NP)
                self.__short_format_ann = f["labels"][:].astype(int)
        else:
            self._short_format_file = None

        # the whole database
        candidate_folders = [f for f in list(self.db_dir.rglob("*")) if re.match(self._subject_pattern, f.name) is not None]
        self._all_subjects = sorted(set([f.name for f in candidate_folders]))
        if len(self._all_subjects) == 0:
            return  # no record found
        assert set([f.parent.name for f in candidate_folders]) == {
            "annotations",
            "signal",
        }, "invalid folder structure"
        self._full_data_dir = candidate_folders[0].parents[1] / "signal"
        self._full_ann_dir = candidate_folders[0].parents[1] / "annotations"

        self._all_records = []
        for subject in self._all_subjects:
            ann_dir = self._full_ann_dir / subject
            for file_path in ann_dir.rglob(self.ann_ext):
                segment = file_path.parents[0].name
                measurement_id = file_path.parents[1].name
                self._all_records.append(f"{subject}/{measurement_id}/{segment}")
        self._df_records = pd.DataFrame()
        self._df_records["record"] = self._all_records
        self._df_records["subject"] = self._df_records["record"].apply(lambda x: x.split("/")[0])
        self._df_records["data_path"] = self._df_records["record"].apply(lambda x: self._full_data_dir / f"{x}/{self.data_ext}")
        self._df_records["ann_path"] = self._df_records["record"].apply(lambda x: self._full_ann_dir / f"{x}/{self.ann_ext}")
        self._df_records["context_ann_path"] = self._df_records["record"].apply(
            lambda x: self._full_ann_dir / f"{x}/{self.context_ann_ext}"
        )
        for context_name, context_ext in self.context_data_ext.items():
            self._df_records[f"context_{context_name}_path"] = self._df_records["record"].apply(
                lambda x: self._full_data_dir / f"{x}/{context_ext}"
            )
        self._df_records["header_path"] = self._df_records["record"].apply(
            lambda x: self._full_data_dir / f"{x}/{self.header_ext}"
        )
        self._df_records.set_index("record", inplace=True)
        self._subject_records = {
            subject: sorted(self._df_records[self._df_records["subject"] == subject].index) for subject in self._all_subjects
        }

        # the table of metadata
        self._df_metadata = pd.DataFrame([self.get_record_metadata(rec) for rec in self._all_records])
        self._df_metadata["record"] = self._all_records
        self._df_metadata.set_index("record", inplace=True)
        self._df_metadata = self._df_metadata[self._metadata_cols]

    def get_absolute_path(self, rec: Union[str, int], extension: str = "signal-ecg") -> Path:
        """Get the absolute path of the signal folder of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        extension: str, default "signal-ecg"
            Extension of the file, can be one of
            "header", "annotation", "signal",
            "annotation-context",
            "signal-ecg", "signal-acc", "signal-angularrate",
            "signal-hr_live", "signal-hrvrmssd_live", etc.

        Returns
        -------
        pathlib.Path
            Absolute path of the file.

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert rec in self.all_records, f"invalid record name: `{rec}`"
        if extension == "annotation":
            return self._df_records.loc[rec, "ann_path"]
        elif extension == "annotation-context":
            return self._df_records.loc[rec, "context_ann_path"]
        elif extension == "signal":
            return self._df_records.loc[rec, "data_path"].parent
        elif extension == "header":
            return self._df_records.loc[rec, "header_path"]

        ext1, ext2 = extension.split("-")
        assert ext1 == "signal", f"extension `{extension}` not supported"
        assert ext2 in list(self.context_data_ext) + ["ecg"], f"extension `{extension}` not supported"
        if ext2 == "ecg":
            return self._df_records.loc[rec, "data_path"]
        else:
            return self._df_records.loc[rec, f"context_{ext2}_path"]

    def get_subject_id(self, rec: Union[str, int]) -> str:
        """Attach a unique subject ID for the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        sid : str
            Subject ID attached to the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert rec in self.all_records, f"record `{rec}` not found"
        return self._df_records.loc[rec, "subject"]

    def _rdheader(self, rec: Union[str, int], key: Optional[str] = None, raw: bool = False) -> Union[str, Dict[str, Any]]:
        """Read header file of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        key : str, optional
            Key of the header to be read, can be one of
            "acc", "angularrate", "hr_live", "hrvrmssd_live",
            "movementacceleration_live", "press", "ecg", "customAttributes".
            If is None, read all keys.
        raw : bool, default False
            If True, return the raw header file,
            otherwise, return a dict.

        Returns
        -------
        header : dict or str
            Header information, or raw header file,
            depending on the value of `raw`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        header_path = self.get_absolute_path(rec, "header")
        if raw:
            return header_path.read_text()
        tree = ET.parse(header_path)
        # return tree
        root = tree.getroot()
        header = {}
        for child in root:
            if "customAttributes" in child.tag:
                header["customAttributes"] = {k: v for k, v in root.attrib.items() if k in self._metadata_cols}
                for child2 in child:
                    if child2.attrib["key"] in self._metadata_cols:
                        header["customAttributes"][child2.attrib["key"]] = child2.attrib["value"]
            elif "signalEntry" in child.tag:
                sig_key = child.attrib["id"].split(".")[0]
                header[sig_key] = deepcopy(child.attrib)
                for child2 in child:
                    if "channel" in child2.tag:
                        if "channel" not in header[sig_key]:
                            header[sig_key]["channel"] = []
                        header[sig_key]["channel"].append(child2.attrib["name"])
        if key is not None:
            assert key in header.keys(), (
                f"key `{key}` not found in header, available keys: `{list(header.keys())}`. "
                f"Perhaps record `{rec}` does not have this type of data?"
            )
            return header[key]
        return header

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
        """Load physical (converted from digital) ECG data,
        or load digital signal directly.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`,
            or "short_format" (-1) to load data from the short format file.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain").
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV");
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
            The loaded ECG data.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        """
        if isinstance(rec, int):
            if rec == -1:
                rec = "short_format"
            else:
                rec = self[rec]
        if rec == "short_format":
            if self._short_format_file is None:
                raise ValueError("Short format file not found")
            return self.__short_format_data
        elif rec not in self.all_records:
            raise ValueError(f"Invalid record name: `{rec}`")
        data_path = self._df_records.loc[rec, "data_path"]
        header = self._rdheader(rec, key="ecg")
        data = np.fromfile(data_path, dtype=header["dataType"])[sampfrom or 0 : sampto or None]
        if units is not None:
            # digital to analog conversion using the field `lsbValue` in the header
            data = (data - int(header["baseline"])) * float(header["lsbValue"])
            data = data.astype(DEFAULTS.DTYPE.NP)
            if units.lower() in ["μv", "uv"]:
                data *= 1e3
            elif units.lower() != "mv":
                raise ValueError(f"Invalid `units`: {units}")
        if data_format in ["channel_last", "lead_last"]:
            data = data[:, np.newaxis]
        elif data_format in ["channel_first", "lead_first"]:
            data = data[np.newaxis, :]
        elif data_format not in ["flat", "plain"]:
            raise ValueError(f"Invalid `data_format`: {data_format}")

        if fs is not None and fs != self.fs:
            data = SS.resample_poly(data, fs, self.fs, axis=0).astype(data.dtype)
            data_fs = fs
        else:
            data_fs = self.fs

        if return_fs:
            return data, data_fs
        return data

    def load_context_data(
        self,
        rec: Union[str, int],
        context_name: str,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        channels: Optional[Union[str, int, List[str], List[int]]] = None,
        units: Optional[str] = None,
        fs: Optional[Real] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Load context data (e.g. accelerometer, heart rate, etc.).

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        context_name : str
            Context name, can be one of
            "acc", "angularrate", "hr_live", "hrvrmssd_live",
            "movementacceleration_live", "press", "marker".
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        channels : str or int or List[str] or List[int], optional
            Channels (names or indices) to be loaded.
            If is None, all channels will be loaded.
        units : str, optional
            Units of the output signal,
            currently can only be "default";
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the output signal.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.

        Returns
        -------
        context_data : numpy.ndarray or pandas.DataFrame
            Context data in the "channel_first" format.

        NOTE
        ----
        If the record does not have the specified context data,
        empty array or DataFrame will be returned.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if rec not in self.all_records:
            raise ValueError(f"Invalid record name: `{rec}`")
        if context_name == "ecg":
            raise ValueError("Call `load_data` to load ECG data")
        assert context_name in self.context_data_ext.keys(), f"Invalid `context_name`: `{context_name}`"
        context_data_path = self._df_records.loc[rec, f"context_{context_name}_path"]

        if context_data_path.suffix == ".csv":
            if not context_data_path.exists():
                warnings.warn(
                    f"record `{rec}` does not have context data `{context_name}`",
                    RuntimeWarning,
                )
                return pd.DataFrame()
            return pd.read_csv(context_data_path, header=None, index_col=None)

        if not context_data_path.exists():
            warnings.warn(
                f"record `{rec}` does not have context data `{context_name}`",
                RuntimeWarning,
            )
            return np.array([])

        header = self._rdheader(rec, key=context_name)
        context_data = np.fromfile(context_data_path, dtype=header["dataType"])[sampfrom or 0 : sampto or None]
        if units is not None:
            assert units.lower() in ["default", header["unit"]], (
                f"`units` should be `default` or `{header['unit']}`, but got `{units}`. "
                "Currently, units conversion is not supported."
            )
            # digital to analog conversion using the field `lsbValue` in the header
            context_data = (context_data - int(header.get("baseline", 1))) * float(header["lsbValue"])
            context_data = context_data.astype(DEFAULTS.DTYPE.NP)

        # convert to "channel_first" format
        context_data = context_data.reshape(-1, len(header["channel"])).T

        if channels is None:
            return context_data
        _input_channels = channels
        if isinstance(channels, str):
            assert (
                channels in header["channel"]
            ), f"`channels` should be a subset of `{header['channel']}`, but got `{channels}`"
            channels = [header["channel"].index(channels)]
        elif isinstance(channels, int):
            assert channels < len(
                header["channel"]
            ), f"`channels` should be less than `{len(header['channel'])}`, but got `{channels}`"
            channels = [channels]
        else:
            assert all(
                [ch in header["channel"] if isinstance(ch, str) else ch in range(len(header["channel"])) for ch in channels]
            ), f"`channels` should be a subset of `{header['channel']}`, but got `{_input_channels}`"
            _channels = [header["channel"].index(ch) if isinstance(ch, str) else ch for ch in channels]
            channels = list(dict.fromkeys(_channels))
            if len(channels) != len(_channels):
                warnings.warn(
                    f"duplicate `channels` are removed, {_input_channels} -> {channels}",
                    RuntimeWarning,
                )
        context_data = context_data[channels]

        if fs is not None and fs != header["sampleRate"]:
            context_data = SS.resample_poly(context_data, fs, header["sampleRate"], axis=1).astype(context_data.dtype)

        return context_data

    def load_ann(
        self, rec: Union[str, int], ann_format: str = "pd"
    ) -> Union[pd.DataFrame, np.ndarray, Dict[Union[int, str], np.ndarray]]:
        """Load annotation from the metadata file.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        ann_format : str, default "pd"
            Format of the annotation,
            currently only "pd" is supported.

        Returns
        -------
        ann : pandas.DataFrame or numpy.ndarray or dict
            The annotation of the record.

        """
        if isinstance(rec, int):
            if rec == -1:
                rec = "short_format"
            else:
                rec = self[rec]
        if rec == "short_format":
            if self._short_format_file is None:
                raise ValueError("Short format file not found")
            return self.__short_format_ann
        elif rec not in self.all_records:
            raise ValueError(f"Invalid record name: `{rec}`")
        try:
            ann = pd.read_csv(self._df_records.loc[rec, "ann_path"])
        except pd.errors.EmptyDataError:
            ann = pd.DataFrame(columns=["Start", "End", "Class"])
        # TODO: adjust return format according to `ann_format`
        if ann_format == "pd":
            return ann
        else:
            raise ValueError(f"`ann_format`: `{ann_format}` not supported")

    def load_context_ann(
        self, rec: Union[str, int], sheet_name: Optional[str] = None
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load context annotation.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        sheet_name : str, optional
            Sheet name of the context annotation file, can be one of
            "movisens DataAnalyzer Parameter", "movisens DataAnalyzer Results".
            If is None, all sheets will be loaded.

        Returns
        -------
        context_ann : pandas.DataFrame or dict
            Context annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if rec not in self.all_records:
            raise ValueError(f"Invalid record name: `{rec}`")
        context_ann_path = self._df_records.loc[rec, "context_ann_path"]
        context_ann = pd.read_excel(context_ann_path, engine="openpyxl", sheet_name=sheet_name)
        return context_ann

    def get_record_metadata(self, rec: Union[str, int]) -> Dict[str, str]:
        """Get metadata of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`,
            or "short_format" (-1) to load data from the short format file.

        Returns
        -------
        metadata : dict
            Metadata of the record

        """
        metadata = self._rdheader(rec, key="customAttributes")
        return metadata

    def get_subject_info(self, rec_or_sid: Union[str, int], items: Optional[List[str]] = None) -> Dict[str, str]:
        """Read auxiliary information of a subject (a record)
        stored in the header files.

        Parameters
        ----------
        rec : str or int
            Record name, or index of the record in :attr:`all_records`,
            or the subject ID.
        items : List[str], optional
            Items of the subject"s information (e.g. sex, age, etc.).

        Returns
        -------
        subject_info : dict
            Information about the subject, including
            "age", "gender", "height", "weight".

        """
        subject_info_items = ["age", "gender", "height", "weight"]
        if rec_or_sid in self.all_subjects:
            rec = self.subject_records[rec_or_sid][0]
        elif isinstance(rec_or_sid, int):
            rec = self[rec_or_sid]
        else:
            rec = rec_or_sid
        assert rec in self.all_records, f"invalid record name: `{rec}`"
        subject_info = self._rdheader(rec, key="customAttributes")
        if items is not None:
            assert set(items).issubset(list(subject_info)), f"invalid items: `{items}`"
            subject_info = {k: subject_info[k] for k in items}
        else:
            subject_info = {k: subject_info[k] for k in subject_info_items}
        return subject_info

    @property
    def all_subjects(self) -> List[str]:
        """List of all subject IDs."""
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        """Dict of subject IDs and their corresponding records."""
        return self._subject_records

    @property
    def df_metadata(self) -> pd.DataFrame:
        """The table of metadata of the records."""
        return self._df_metadata

    @property
    def url(self) -> Dict[str, str]:
        return {
            "CACHET-CADB.zip": "https://data.dtu.dk/ndownloader/files/27928830",
            "cachet-cadb_short_format_without_context.hdf5.zip": "https://data.dtu.dk/ndownloader/files/27917358",
        }

    def download(self, files: Optional[Union[str, Sequence[str]]]) -> None:
        """Download the database from the DTU website.

        Parameters
        ----------
        files : str or Sequence[str], optional
            Files to download, can be subset of
            "CACHET-CADB.zip", "cachet-cadb_short_format_without_context.hdf5.zip".
            If is None, all files will be downloaded.

        """
        warnings.warn(
            "The files are large, and the connections are unstable. "
            "One might need some downloading tools (e.g. Xunlei) to download the files.",
            RuntimeWarning,
        )
        if files is None:
            files = self.url.keys()
        if isinstance(files, str):
            files = [files]
        assert set(files).issubset(self.url), f"`files` should be a subset of `{list(self.url)}`"
        for filename in files:
            url = self.url[filename]
            if not (self.db_dir / filename).is_file():
                http_get(url, self.db_dir, filename=filename)
        self._ls_rec()

    def plot(
        self,
        rec: Union[str, int],
        **kwargs: Any,
    ) -> None:
        """Not implemented."""
        raise NotImplementedError

    @property
    def database_info(self) -> DataBaseInfo:
        return _CACHET_CADB_INFO
