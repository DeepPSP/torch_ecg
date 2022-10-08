# -*- coding: utf-8 -*-
"""
Base classes for datasets from different sources:
    PhysioNet
    NSRR
    CPSC
    Other databases

Remarks:
1. for whole-dataset visualizing: http://zzz.bwh.harvard.edu/luna/vignettes/dataplots/
2. visualizing using UMAP: http://zzz.bwh.harvard.edu/luna/vignettes/nsrr-umap/

"""

import os
import posixpath
import pprint
import re
import time
import warnings
import textwrap
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from string import punctuation
from typing import Any, List, Optional, Union, Sequence

import numpy as np
import pandas as pd
import requests
import wfdb
from pyedflib import EdfReader

from ..cfg import CFG, _DATA_CACHE
from ..utils import ecg_arrhythmia_knowledge as EAK  # noqa: F401
from ..utils.download import http_get
from ..utils.misc import ReprMixin, dict_to_str, get_record_list_recursive
from .aux_data import get_physionet_dbs


__all__ = [
    "WFDB_Beat_Annotations",
    "WFDB_Non_Beat_Annotations",
    "WFDB_Rhythm_Annotations",
    "PhysioNetDataBase",
    "NSRRDataBase",
    "CPSCDataBase",
    "DEFAULT_FIG_SIZE_PER_SEC",
    "BeatAnn",
    "DataBaseInfo",
]


WFDB_Beat_Annotations = {
    "N": "Normal beat",
    "L": "Left bundle branch block beat",
    "R": "Right bundle branch block beat",
    "B": "Bundle branch block beat (unspecified)",
    "A": "Atrial premature beat",
    "a": "Aberrated atrial premature beat",
    "J": "Nodal (junctional) premature beat",
    "S": "Supraventricular premature or ectopic beat (atrial or nodal)",
    "V": "Premature ventricular contraction",
    "r": "R-on-T premature ventricular contraction",
    "F": "Fusion of ventricular and normal beat",
    "e": "Atrial escape beat",
    "j": "Nodal (junctional) escape beat",
    "n": "Supraventricular escape beat (atrial or nodal)",
    "E": "Ventricular escape beat",
    "/": "Paced beat",
    "f": "Fusion of paced and normal beat",
    "Q": "Unclassifiable beat",
    "?": "Beat not classified during learning",
}

WFDB_Non_Beat_Annotations = {
    "[": "Start of ventricular flutter/fibrillation",
    "!": "Ventricular flutter wave",
    "]": "End of ventricular flutter/fibrillation",
    "x": "Non-conducted P-wave (blocked APC)",
    "(": "Waveform onset",
    ")": "Waveform end",
    "p": "Peak of P-wave",
    "t": "Peak of T-wave",
    "u": "Peak of U-wave",
    "`": "PQ junction",
    "'": "J-point",
    "^": "(Non-captured) pacemaker artifact",
    "|": "Isolated QRS-like artifact",
    "~": "Change in signal quality",
    "+": "Rhythm change",
    "s": "ST segment change",
    "T": "T-wave change",
    "*": "Systole",
    "D": "Diastole",
    "=": "Measurement annotation",
    '"': "Comment annotation",
    "@": "Link to external data",
}

WFDB_Rhythm_Annotations = {
    "(AB": "Atrial bigeminy",
    "(AFIB": "Atrial fibrillation",
    "(AFL": "Atrial flutter",
    "(B": "Ventricular bigeminy",
    "(BII": "2° heart block",
    "(IVR": "Idioventricular rhythm",
    "(N": "Normal sinus rhythm",
    "(NOD": "Nodal (A-V junctional) rhythm",
    "(P": "Paced rhythm",
    "(PREX": "Pre-excitation (WPW)",
    "(SBR": "Sinus bradycardia",
    "(SVTA": "Supraventricular tachyarrhythmia",
    "(T": "Ventricular trigeminy",
    "(VFL": "Ventricular flutter",
    "(VT": "Ventricular tachycardia",
}


class _DataBase(ReprMixin, ABC):
    """
    universal abstract base class for all databases
    """

    def __init__(
        self,
        db_name: str,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
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
        self.db_name = db_name
        if db_dir is None:
            db_dir = _DATA_CACHE / db_name
            warnings.warn(
                f"db_dir is not specified, "
                f"using default {db_dir} as the storage path"
            )
        self.db_dir = Path(db_dir).resolve().absolute()
        if not self.db_dir.exists():
            self.db_dir.mkdir(parents=True, exist_ok=True)
            warnings.warn(
                f"{self.db_dir} does not exist. It is now created. "
                "Please check if it is set correctly. "
                "Or if you may want to download the database into this folder, "
                "please use the `download()` method."
            )
        self.working_dir = Path(working_dir or os.getcwd())
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self.verbose = verbose
        self._df_records = pd.DataFrame()
        self._all_records = None

    @abstractmethod
    def _ls_rec(self) -> None:
        """ """
        raise NotImplementedError

    @abstractmethod
    def load_data(self, rec: Union[str, int], **kwargs) -> Any:
        """
        load data from the record `rec`
        """
        raise NotImplementedError

    @abstractmethod
    def load_ann(self, rec: Union[str, int], **kwargs) -> Any:
        """
        load annotations of the record `rec`

        NOTE that the records might have several annotation files

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def database_info(self) -> "DataBaseInfo":
        """the `DataBaseInfo` object of the database"""
        raise NotImplementedError

    def get_citation(
        self, format: Optional[str] = None, style: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        lookup: bool, default True,
            whether to lookup the citation from the DOI
        format: str, optional,
            format of the final output,
            if specified, the default format ("bib") will be overrided
        style: str, optional,
            style of the final output,
            if specified, the default style ("apa") will be overrided,
            only valid when `format` is "text"
        print_result: bool, default False,
            whether to print the final output instead of returning it

        """
        return self.database_info.get_citation(
            lookup=True, format=format, style=style, timeout=10.0, print_result=True
        )

    def _auto_infer_units(self, sig: np.ndarray, sig_type: str = "ECG") -> str:
        """
        automatically infer the units of `sig`,
        under the assumption that `sig` not being raw signal, with baseline removed

        Parameters
        ----------
        sig: ndarray,
            the signal to infer its units
        sig_type: str, default "ECG", case insensitive,
            type of the signal

        Returns
        -------
        units: str,
            units of `sig`, "μV" or "mV"

        """
        if sig_type.lower() == "ecg":
            _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
            max_val = np.max(np.abs(sig))
            if max_val > _MAX_mV:
                units = "μV"
            else:
                units = "mV"
        else:
            raise NotImplementedError(f"not implemented for {sig_type}")
        return units

    @property
    def all_records(self) -> List[str]:
        """ """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """
        get the absolute path of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        extension: str, optional,
            extension of the file

        Returns
        -------
        path: Path,
            absolute path of the file

        """
        if isinstance(rec, int):
            rec = self[rec]
        path = self._df_records.loc[rec].path
        if extension is not None:
            path = path.with_suffix(
                extension if extension.startswith(".") else f".{extension}"
            )
        return path

    @classmethod
    def get_arrhythmia_knowledge(
        cls, arrhythmias: Union[str, List[str]], **kwargs: Any
    ) -> None:
        """
        knowledge about ECG features of specific arrhythmias,

        Parameters
        ----------
        arrhythmias: str, or list of str,
            the arrhythmia(s) to check, in abbreviations or in SNOMEDCTCode

        """
        if isinstance(arrhythmias, str):
            d = [arrhythmias]
        else:
            d = arrhythmias
        for idx, item in enumerate(d):
            print(dict_to_str(eval(f"EAK.{item}")))
            if idx < len(d) - 1:
                print("*" * 110)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "db_name",
            "db_dir",
        ]

    @property
    @abstractmethod
    def url(self) -> Union[str, List[str]]:
        """
        URL(s) for downloading the database
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        number of records in the database
        """
        return len(self.all_records)

    def __getitem__(self, index: int) -> str:
        """
        get the record name by index
        """
        return self.all_records[index]


class PhysioNetDataBase(_DataBase):
    """https://www.physionet.org/"""

    def __init__(
        self,
        db_name: str,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
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
        # `self.fs` for those with single signal source, e.g. ECG,
        # for those with multiple signal sources like PSG,
        # self.fs is default to the frequency of ECG if ECG applicable
        self.fs = None
        self._all_records = None
        self._version = None
        self._url_compressed = None

        self.df_all_db_info = get_physionet_dbs()

        if self.verbose > 2:
            self.df_all_db_info = (
                pd.DataFrame(
                    wfdb.get_dbs(),
                    columns=[
                        "db_name",
                        "db_description",
                    ],
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )

    def _ls_rec(self, db_name: Optional[str] = None, local: bool = True) -> None:
        """
        find all records (relative path without file extension),
        and save into `self._all_records` for further use

        Parameters
        ----------
        db_name: str, optional,
            name of the database for using `wfdb.get_record_list`,
            if not set, `self.db_name` will be used
        local: bool, default True,
            if True, read from local storage, prior to using `wfdb.get_record_list`

        """
        if local:
            self._ls_rec_local()
            return
        try:
            self._df_records = pd.DataFrame()
            self._df_records["record"] = wfdb.get_record_list(db_name or self.db_name)
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: (self.db_dir / x).resolve()
            )
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.name
            )  # remove relative path, leaving only the record name
            self._df_records.set_index("record", inplace=True)
            self._all_records = self._df_records.index.values.tolist()
        except Exception:
            self._ls_rec_local()

    def _ls_rec_local(self) -> None:
        """
        find all records in `self.db_dir`
        """
        record_list_fp = self.db_dir / "RECORDS"
        self._df_records = pd.DataFrame()
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
        else:
            print(
                "Please wait patiently to let the reader find all records of the database from local storage..."
            )
            start = time.time()
            self._df_records["path"] = get_record_list_recursive(
                self.db_dir, self.data_ext, relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
            print(f"Done in {time.time() - start:.3f} seconds!")
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.name
            )
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        int, a `subject_id` attached to the record `rec`

        """
        raise NotImplementedError

    @property
    def database_info(self, detailed: bool = False) -> None:
        """
        print the information about the database

        Parameters
        ----------
        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally

        """
        if not detailed:
            try:
                short_description = self.df_all_db_info[
                    self.df_all_db_info["db_name"] == self.db_name
                ]["db_description"].values[0]
                print(short_description)
                return
            except Exception:
                pass
        info = "\n".join(self.__doc__.split("\n")[1:])
        print(info)

    def helper(self, items: Union[List[str], str, type(None)] = None, **kwargs) -> None:
        """to be improved,

        print corr. meanings of symbols belonging to `items`

        Parameters
        ----------
        items: str, or list of str, optional,
            the items to print,
            if not specified, then a comprehensive printing of meanings of all symbols will be performed

        References
        ----------
        [1] https://archive.physionet.org/physiobank/annotations.shtml

        """
        attrs = vars(self)
        methods = [
            func
            for func in dir(self)
            if callable(getattr(self, func))
            and not (func.startswith("__") and func.endswith("__"))
        ]

        beat_annotations = deepcopy(WFDB_Beat_Annotations)
        non_beat_annotations = deepcopy(WFDB_Non_Beat_Annotations)
        rhythm_annotations = deepcopy(WFDB_Rhythm_Annotations)

        all_annotations = [
            beat_annotations,
            non_beat_annotations,
            rhythm_annotations,
        ]

        summary_items = [
            "beat",
            "non-beat",
            "rhythm",
        ]

        if items is None:
            _items = [
                "attributes",
                "methods",
                "beat",
                "non-beat",
                "rhythm",
            ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)
        if "beat" in _items:
            print("--- helpler - beat ---")
            pp.pprint(beat_annotations)
        if "non-beat" in _items:
            print("--- helpler - non-beat ---")
            pp.pprint(non_beat_annotations)
        if "rhythm" in _items:
            print("--- helpler - rhythm ---")
            pp.pprint(rhythm_annotations)

        for k in _items:
            if k in summary_items:
                continue
            for a in all_annotations:
                if k in a.keys() or "(" + k in a.keys():
                    try:
                        print(f"{k.split('(')[1]} stands for {a[k]}")
                    except Exception:
                        print(f"{k} stands for {a['('+k]}")

    @property
    def version(self) -> str:
        if self._version is not None:
            return self._version
        try:
            self._version = wfdb.io.record.get_version(self.db_name)
        except Exception:
            warnings.warn(
                "Cannot get the version number from PhysioNet! Defaults to '1.0.0'"
            )
            self._version = "1.0.0"
        return self._version

    @property
    def webpage(self) -> str:
        return posixpath.join(
            wfdb.io.download.PN_CONTENT_URL, f"{self.db_name}/{self.version}"
        )

    @property
    def url(self) -> str:
        return posixpath.join(
            wfdb.io.download.PN_INDEX_URL, f"{self.db_name}/{self.version}"
        )

    @property
    def url_(self) -> Union[str, type(None)]:
        """URL of the compressed database file"""
        if self._url_compressed is not None:
            return self._url_compressed
        domain = "https://physionet.org/static/published-projects/"
        punct = re.sub("[\\-:]", "", punctuation)
        try:
            db_desc = self.df_all_db_info[
                self.df_all_db_info["db_name"] == self.db_name
            ].iloc[0]["db_description"]
        except IndexError:
            print(
                f"\042{self.db_name}\042 is not in the database list hosted at PhysioNet!"
            )
            return None
        db_desc = re.sub(f"[{punct}]+", "", db_desc).lower()
        db_desc = re.sub("[\\s:]+", "-", db_desc)
        url = posixpath.join(domain, f"{self.db_name}/{db_desc}-{self.version}.zip")
        if requests.head(url).headers.get("Content-Type") == "application/zip":
            self._url_compressed = url
        else:
            new_url = posixpath.join(
                wfdb.io.download.PN_INDEX_URL, f"{self.db_name}/get-zip/{self.version}"
            )
            print(f"{url} is not available, try {new_url} instead")
        return self._url_compressed

    def download(self, compressed: bool = True) -> None:
        """
        download the database from PhysioNet
        """
        if compressed:
            if self.url_ is not None:
                http_get(self.url_, self.db_dir, extract=True)
                self._ls_rec()
                return
            else:
                print(
                    "No compressed database available! Downloading the uncompressed version..."
                )
        wfdb.dl_database(
            self.db_name,
            self.db_dir,
            keep_subdirs=True,
            overwrite=False,
        )
        self._ls_rec()


class NSRRDataBase(_DataBase):
    """
    https://sleepdata.org/
    """

    def __init__(
        self,
        db_name: str,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str or Path, optional,
            storage path of the database
        working_dir: str, optional,
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
        self.fs = None
        self._all_records = None
        self.file_opened = None

        all_dbs = [
            [
                "shhs",
                "Multi-cohort study focused on sleep-disordered breathing and cardiovascular outcomes",
            ],
            ["mesa", ""],
            ["oya", ""],
            [
                "chat",
                "Multi-center randomized trial comparing early adenotonsillectomy to watchful waiting plus supportive care",
            ],
            [
                "heartbeat",
                "Multi-center Phase II randomized controlled trial that evaluates the effects of supplemental nocturnal oxygen or Positive Airway Pressure (PAP) therapy",
            ],
            # more to be added
        ]
        self.df_all_db_info = pd.DataFrame(
            {
                "db_name": [item[0] for item in all_dbs],
                "db_description": [item[1] for item in all_dbs],
            }
        )
        self.kwargs = kwargs

    def safe_edf_file_operation(
        self,
        operation: str = "close",
        full_file_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Parameters
        ----------
        operation: str, default "close",
            operation name, can be "open" and "close"
        full_file_path: str or Path, optional,
            path of the file which contains the psg data,
            if not given, default path will be used

        """
        if operation == "open":
            if self.file_opened is not None:
                self.file_opened._close()
            self.file_opened = EdfReader(str(full_file_path))
        elif operation == "close":
            if self.file_opened is not None:
                self.file_opened._close()
                self.file_opened = None
        else:
            raise ValueError("Illegal operation")

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        int, a `subject_id` attached to the record `rec`

        """
        raise NotImplementedError

    def show_rec_stats(self, rec: Union[str, int]) -> None:
        """
        print the statistics about the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        """
        raise NotImplementedError

    @property
    def database_info(self, detailed: bool = False) -> None:
        """
        print the information about the database

        detailed: bool, default False,
            if False, an short introduction of the database will be printed,
            if True, then docstring of the class will be printed additionally

        """
        if not detailed:
            # raw_info = {
            #     "What": "",
            #     "Who": "",
            #     "When": "",
            #     "Funding": ""
            # }
            raw_info = self.df_all_db_info[
                self.df_all_db_info.db_name == self.db_name.lower()
            ].db_description.values[0]
            print(raw_info)
            return
        print(self.__doc__)

    def helper(self, items: Union[List[str], str, type(None)] = None, **kwargs) -> None:
        """ """
        pp = pprint.PrettyPrinter(indent=4)

        attrs = vars(self)
        methods = [
            func
            for func in dir(self)
            if callable(getattr(self, func))
            and not (func.startswith("__") and func.endswith("__"))
        ]

        if items is None:
            _items = [
                "attributes",
                "methods",
            ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)


class CPSCDataBase(_DataBase):
    """ """

    def __init__(
        self,
        db_name: str,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        r"""
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str or Path, optional,
            storage path of the database
        working_dir: str, optional,
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

        self.fs = None
        self._all_records = None

        self.kwargs = kwargs

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """
        Attach a `subject_id` to the record, in order to facilitate further uses

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        int, a `subject_id` attached to the record `rec`

        """
        raise NotImplementedError

    def helper(self, items: Union[List[str], str, type(None)] = None, **kwargs) -> None:
        """ """
        pp = pprint.PrettyPrinter(indent=4)

        attrs = vars(self)
        methods = [
            func
            for func in dir(self)
            if callable(getattr(self, func))
            and not (func.startswith("__") and func.endswith("__"))
        ]

        if items is None:
            _items = [
                "attributes",
                "methods",
            ]
        elif isinstance(items, str):
            _items = [items]
        else:
            _items = items

        pp = pprint.PrettyPrinter(indent=4)

        if "attributes" in _items:
            print("--- helpler - attributes ---")
            pp.pprint(attrs)
        if "methods" in _items:
            print("--- helpler - methods ---")
            pp.pprint(methods)

    def download(self) -> None:
        """download the database from `self.url`"""
        if isinstance(self.url, str):
            http_get(self.url, self.db_dir, extract=True)
        else:
            for url in self.url:
                http_get(url, self.db_dir, extract=True)
        self._ls_rec()


@dataclass
class DataBaseInfo:
    """ """

    title: str
    about: Union[str, Sequence[str]]
    usage: Sequence[str]
    references: Sequence[str]
    note: Optional[Union[str, Sequence[str]]] = None
    issues: Optional[Union[str, Sequence[str]]] = None
    status: Optional[str] = None
    doi: Optional[Union[str, Sequence[str]]] = None

    from bib_lookup import BibLookup

    _bl = BibLookup(timeout=1.0, ignore_errors=False)

    def format_database_docstring(self, indent: Optional[str] = None) -> str:
        """ """
        if indent is None:
            indent = " " * 4
        title = textwrap.dedent(self.title).strip("\n ")
        if isinstance(self.about, str):
            about = "ABOUT\n-----\n" + textwrap.dedent(self.about).strip("\n ")
        else:
            about = ["ABOUT", "-----"] + [
                f"{idx+1}. {line}" for idx, line in enumerate(self.about)
            ]
            about = "\n".join(about)
        if self.note is None:
            note = "NOTE\n----"
        elif isinstance(self.note, str):
            note = "NOTE\n----\n" + textwrap.dedent(self.note).strip("\n ")
        else:
            note = ["NOTE", "----"] + [
                f"{idx+1}. {line}" for idx, line in enumerate(self.note)
            ]
            note = "\n".join(note)
        if self.issues is None:
            issues = "ISSUES\n------"
        elif isinstance(self.issues, str):
            issues = "Issues\n------\n" + textwrap.dedent(self.issues).strip("\n ")
        else:
            issues = ["Issues", "-" * 6] + [
                f"{idx+1}. {line}" for idx, line in enumerate(self.issues)
            ]
            issues = "\n".join(issues)
        references = ["References", "-" * 10] + [
            f"""{idx+1}. <a name="ref{idx+1}"></a> {line}"""
            for idx, line in enumerate(self.references)
        ]
        references = "\n".join(references)
        usage = ["Usage", "------"] + [
            f"{idx+1}. {line}" for idx, line in enumerate(self.usage)
        ]
        usage = "\n".join(usage)

        docstring = textwrap.indent(
            f"""\n{title}\n\n{about}\n\n{note}\n\n{usage}\n\n{issues}\n\n{references}\n""",
            indent,
        )

        if self.status is not None and len(self.status) > 0:
            docstring = f"{self.status}\n{docstring}"

        lookup = False
        citation = self.get_citation(lookup=lookup, print_result=False)
        if citation.startswith("@"):
            citation = textwrap.indent(
                f"""Citation\n--------\n```latex\n{citation}\n```""", indent
            )
            docstring = f"{docstring}\n{citation}\n"
        elif not lookup:
            citation = textwrap.indent(f"""Citation\n--------\n{citation}""", indent)
            docstring = f"{docstring}\n{citation}\n"

        return docstring

    def get_citation(
        self,
        lookup: bool = True,
        format: Optional[str] = None,
        style: Optional[str] = None,
        timeout: Optional[float] = None,
        print_result: bool = False,
    ) -> Union[str, type(None)]:
        """
        Parameters
        ----------
        lookup: bool, default True,
            whether to lookup the citation from the DOI
        format: str, optional,
            format of the final output,
            if specified, the default format ("bib") will be overrided
        style: str, optional,
            style of the final output,
            if specified, the default style ("apa") will be overrided,
            only valid when `format` is "text"
        timeout: float, optional,
            timeout for the lookup,
            only valid when `lookup` is True,
            if not specified, the default timeout (1.0) will be used
        print_result: bool, default False,
            whether to print the final output instead of returning it

        Returns
        -------
        str, optional,
            citation(s) of the database

        """
        self._bl.clear_cache()
        citation_cache = _DATA_CACHE / "database_citation.csv"
        if citation_cache.exists():
            df_cc = pd.read_csv(citation_cache)
        else:
            df_cc = pd.DataFrame(columns=["doi", "citation"])
            df_cc.to_csv(citation_cache, index=False)

        if self.doi is not None:
            if isinstance(self.doi, str):
                doi = [self.doi]
            else:
                doi = self.doi
            if not lookup:
                citation = "\n".join(doi)
                if print_result:
                    print(citation)
                    return
                else:
                    return citation
            if format is not None and format != self._bl.format:
                citation = ""  # no cache for format other than bibtex
            else:
                citation = "\n".join(df_cc[df_cc["doi"].isin(doi)]["citation"].tolist())
                doi = [item for item in doi if item not in df_cc["doi"].tolist()]
                if print_result:
                    print(citation)
            if len(doi) > 0:
                new_citations = []
                for item in doi:
                    try:
                        bl_res = self._bl(
                            item,
                            format=format,
                            style=style,
                            print_result=False,
                            timeout=timeout,
                        )
                        if bl_res not in self._bl.lookup_errors:
                            new_citations.append(
                                {
                                    "doi": item,
                                    "citation": str(bl_res),
                                }
                            )
                            if print_result:
                                print(bl_res)
                        elif print_result:
                            print(f"{bl_res} for {item}")
                    except Exception:
                        if print_result:
                            print(f"Failed to lookup citation for {item}")
                if format is None or format == self._bl.format:
                    # only cache bibtex format
                    new_citations = [
                        item
                        for item in new_citations
                        if item["citation"] is not None
                        and item["citation"].startswith("@")
                    ]
                    df_new = pd.DataFrame(new_citations)
                    if len(df_new) > 0:
                        df_new.to_csv(
                            citation_cache, mode="a", header=False, index=False
                        )
                else:
                    df_new = pd.DataFrame(new_citations)
                if len(df_new) > 0:
                    citation += "\n" + "\n".join(df_new["citation"].tolist())
        else:
            citation = ""

        citation = citation.strip("\n ")
        if citation == "" and self.doi is not None:
            citation = "\n".join(doi)
            warnings.warn("Lookup failed, default to the DOI(s).")
            if print_result:
                print(citation)
        if not print_result:
            return citation


DEFAULT_FIG_SIZE_PER_SEC = 4.8


@dataclass
class BeatAnn:
    """ """

    index: int
    symbol: str

    @property
    def name(self) -> str:
        if self.symbol in WFDB_Beat_Annotations:
            return WFDB_Beat_Annotations[self.symbol]
        return WFDB_Non_Beat_Annotations.get(self.symbol, self.symbol)


# configurations for visualization
_PlotCfg = CFG()
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
_PlotCfg.p_onset = -40
_PlotCfg.p_offset = 40
_PlotCfg.q_onset = -20
_PlotCfg.s_offset = 40
_PlotCfg.qrs_radius = 60
_PlotCfg.t_onset = -100
_PlotCfg.t_offset = 60
