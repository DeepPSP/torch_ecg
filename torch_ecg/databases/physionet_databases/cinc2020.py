# -*- coding: utf-8 -*-

import io
import json
import os
import posixpath
import re
import time
from copy import deepcopy
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.signal as SS
import wfdb
from scipy.io import loadmat

from ...cfg import CFG, DEFAULTS
from ...utils import EAK
from ...utils.download import _stem, http_get
from ...utils.misc import add_docstring, get_record_list_recursive3, list_sum, ms2samples
from ...utils.utils_data import ensure_siglen
from ..aux_data.cinc2020_aux_data import df_weights_abbr, dx_mapping_all, dx_mapping_scored, equiv_class_dict, load_weights
from ..base import DEFAULT_FIG_SIZE_PER_SEC, DataBaseInfo, PhysioNetDataBase, _PlotCfg

__all__ = [
    "CINC2020",
    "compute_metrics",
    "compute_all_metrics",
]


_CINC2020_INFO = DataBaseInfo(
    title="""
    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
    """,
    about="""
    0. There are 6 difference tranches of training data, listed as follows:

        A. 6,877 recordings from China Physiological Signal Challenge in 2018 (CPSC2018, see also [3]_): PhysioNetChallenge2020_Training_CPSC.tar.gz
        B. 3,453 recordings from China 12-Lead ECG Challenge Database (unused data from CPSC2018 and NOT the CPSC2018 test data): PhysioNetChallenge2020_Training_2.tar.gz
        C. 74 recordings from the St Petersburg INCART 12-lead Arrhythmia Database: PhysioNetChallenge2020_Training_StPetersburg.tar.gz
        D. 516 recordings from the PTB Diagnostic ECG Database: PhysioNetChallenge2020_Training_PTB.tar.gz
        E. 21,837 recordings from the PTB-XL electrocardiography Database: PhysioNetChallenge2020_PTB-XL.tar.gz
        F. 10,344 recordings from a Georgia 12-Lead ECG Challenge Database: PhysioNetChallenge2020_Training_E.tar.gz

       In total, 43,101 labeled recordings of 12-lead ECGs from four countries (China, Germany, Russia, and the USA) across 3 continents have been posted publicly for this Challenge, with approximately the same number hidden for testing, representing the largest public collection of 12-lead ECGs. All files can be downloaded from [7]_ or [8]_.

    1. the A tranche training data comes from CPSC2018, whose folder name is `Training_WFDB`. The B tranche training data are unused training data of CPSC2018, having folder name `Training_2`. For these 2 tranches, ref. the docstring of `database_reader.cpsc_databases.cpsc2018.CPSC2018`
    2. C, D, E tranches of training data all come from corresponding PhysioNet dataset, whose details can be found in corresponding files:

        - C: INCARTDB, ref [4]_
        - D: PTBDB, ref [5]_
        - E: PTB_XL, ref [6]_

       the C tranche has folder name `Training_StPetersburg`, the D tranche has folder name `Training_PTB`, the F tranche has folder name `WFDB`
    3. the F tranche is entirely new, posted for this Challenge, and represents a unique demographic of the Southeastern United States. It has folder name `Training_E/WFDB`.
    4. only a part of diagnosis_abbr (diseases that appear in the labels of the 6 tranches of training data) are used in the scoring function (ref. `dx_mapping_scored_cinc2020`), while others are ignored (ref. `dx_mapping_unscored_cinc2020`). The scored diagnoses were chosen based on prevalence of the diagnoses in the training data, the severity of the diagnoses, and the ability to determine the diagnoses from ECG recordings. The ignored diagnosis_abbr can be put in a a "non-class" group.
    5. the (updated) scoring function has a scoring matrix with nonzero off-diagonal elements. This scoring function reflects the clinical reality that some misdiagnoses are more harmful than others and should be scored accordingly. Moreover, it reflects the fact that confusing some classes is much less harmful than confusing other classes.

    6. sampling frequencies:

        A. (CPSC2018): 500 Hz
        B. (CPSC2018-2): 500 Hz
        C. (INCART): 257 Hz
        D. (PTB): 1000 Hz
        E. (PTB-XL): 500 Hz
        F. (Georgia): 500 Hz

    7. all data are recorded in the leads ordering of

       .. code-block:: python

            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

       using for example the following code:

         .. code-block:: python

            >>> db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
            >>> working_dir = "./working_dir"
            >>> dr = CINC2020Reader(db_dir=db_dir,working_dir=working_dir)
            >>> set_leads = []
            >>> for tranche, l_rec in dr.all_records.items():
            ...     for rec in l_rec:
            ...         ann = dr.load_ann(rec)
            ...         leads = ann["df_leads"]["lead_name"].values.tolist()
            ...     if leads not in set_leads:
            ...         set_leads.append(leads)

    8. Challenge official website [1]_. Webpage of the database on PhysioNet [2]_.
    """,
    note="""
    1. The datasets have been roughly processed to have a uniform format, hence differ from their original resource (e.g. differe in sampling frequency, sample duration, etc.)
    2. The original datasets might have richer metadata (especially those from PhysioNet), which can be fetched from corresponding reader's docstring or website of the original source
    3. Each sub-dataset might have its own organizing scheme of data, which should be carefully dealt with
    4. There are few "absolute" diagnoses in 12 lead ECGs, where large discrepancies in the interpretation of the ECG can be found even inspected by experts. There is inevitably something lost in translation, especially when you do not have the context. This doesn"t mean making an algorithm isn't important
    5. The labels are noisy, which one has to deal with in all real world data
    6. each line of the following classes are considered the same (in the scoring matrix):

        - RBBB, CRBBB (NOT including IRBBB)
        - PAC, SVPB
        - PVC, VPB

    7. unfortunately, the newly added tranches (C - F) have baseline drift and are much noisier. In contrast, CPSC data have had baseline removed and have higher SNR
    8. on Aug. 1, 2020, adc gain (including "resolution", "ADC"? in .hea files) of datasets INCART, PTB, and PTB-xl (tranches C, D, E) are corrected. After correction, (the .tar files of) the 3 datasets are all put in a "WFDB" subfolder. In order to keep the structures consistant, they are moved into "Training_StPetersburg", "Training_PTB", "WFDB" as previously. Using the following code, one can check the adc_gain and baselines of each tranche:

       .. code-block:: python

            >>> db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
            >>> working_dir = "./working_dir"
            >>> dr = CINC2020(db_dir=db_dir,working_dir=working_dir)
            >>> resolution = {tranche: set() for tranche in "ABCDEF"}
            >>> baseline = {tranche: set() for tranche in "ABCDEF"}
            >>> for tranche, l_rec in dr.all_records.items():
            ...     for rec in l_rec:
            ...         ann = dr.load_ann(rec)
            ...         resolution[tranche] = resolution[tranche].union(set(ann["df_leads"]["adc_gain"]))
            ...         baseline[tranche] = baseline[tranche].union(set(ann["df_leads"]["baseline"]))
            >>> print(resolution, baseline)
            {"A": {1000.0}, "B": {1000.0}, "C": {1000.0}, "D": {1000.0}, "E": {1000.0}, "F": {1000.0}} {"A": {0}, "B": {0}, "C": {0}, "D": {0}, "E": {0}, "F": {0}}

    9. the .mat files all contain digital signals, which has to be converted to physical values using adc gain, basesline, etc. in corresponding .hea files. :func:`wfdb.rdrecord` has already done this conversion, hence greatly simplifies the data loading process. NOTE that there"s a difference when using `wfdb.rdrecord`: data from `loadmat` are in "channel_first" format, while `wfdb.rdrecord.p_signal` produces data in the "channel_last" format
    10. there are 3 equivalent (2 classes are equivalent if the corr. value in the scoring matrix is 1): (RBBB, CRBBB), (PAC, SVPB), (PVC, VPB)
    11. in the newly (Feb., 2021) created dataset, header files of each subset were gathered into one separate compressed file. This is due to the fact that updates on the dataset are almost always done in the header files. The correct usage of ref. [8]_, after uncompressing, is replacing the header files in the folder `All_training_WFDB` by header files from the 6 folders containing all header files from the 6 subsets.
    """,
    usage=[
        "ECG arrhythmia detection",
    ],
    issues="""
    1. reading the .hea files, baselines of all records are 0, however it is not the case if one plot the signal
    2. about half of the LAD records satisfy the "2-lead" criteria, but fail for the "3-lead" criteria, which means that their axis is (-30°, 0°) which is not truely LAD
    3. (Aug. 15, 2020; resolved, and changed to 1000) tranche F, the Georgia subset, has ADC gain 4880 which might be too high. Thus obtained voltages are too low. 1000 might be a suitable (correct) value of ADC gain for this tranche just as the other tranches.
    4. "E04603" (all leads), "E06072" (chest leads, epecially V1-V3), "E06909" (lead V2), "E07675" (lead V3), "E07941" (lead V6), "E08321" (lead V6) has exceptionally large values at rpeaks, reading (`load_data`) these two records using `wfdb` would bring in `nan` values. One can check using the following code

       .. code-block:: python

            >>> rec = "E04603"
            >>> dr.plot(rec, dr.load_data(rec, backend="scipy", units="uv"))  # currently raising error

    """,
    references=[
        "https://moody-challenge.physionet.org/2020/",
        "https://physionet.org/content/challenge-2020/",
        "http://2018.icbeb.org/",
        "https://physionet.org/content/incartdb/",
        "https://physionet.org/content/ptbdb/",
        "https://physionet.org/content/ptb-xl/",
        "(deprecated) https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ECG-public/",
        "(recommended) https://storage.cloud.google.com/physionetchallenge2021-public-datasets/",
    ],
    doi=[
        "10.1088/1361-6579/abc960",
        "10.22489/cinc.2020.236",
        "10.13026/F4AB-0814",
    ],
)


@add_docstring(_CINC2020_INFO.format_database_docstring(), mode="prepend")
class CINC2020(PhysioNetDataBase):
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

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="challenge-2020",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.rec_ext = "mat"
        self.ann_ext = "hea"

        self.db_tranches = list("ABCDEF")
        self.tranche_names = CFG(
            {
                "A": "CPSC",
                "B": "CPSC-Extra",
                "C": "StPetersburg",
                "D": "PTB",
                "E": "PTB-XL",
                "F": "Georgia",
            }
        )
        self.rec_prefix = CFG(
            {
                "A": "A",
                "B": "Q",
                "C": "I",
                "D": "S",
                "E": "HR",
                "F": "E",
            }
        )

        self.fs = {
            "A": 500,
            "B": 500,
            "C": 257,
            "D": 1000,
            "E": 500,
            "F": 500,
        }
        self.spacing = {t: 1000 / f for t, f in self.fs.items()}

        self.all_leads = deepcopy(EAK.Standard12Leads)

        self.df_ecg_arrhythmia = dx_mapping_all[["Dx", "SNOMED CT Code", "Abbreviation"]]
        self.ann_items = [
            "rec_name",
            "nb_leads",
            "fs",
            "nb_samples",
            "datetime",
            "age",
            "sex",
            "diagnosis",
            "df_leads",
            "medical_prescription",
            "history",
            "symptom_or_surgery",
        ]
        self.label_trans_dict = equiv_class_dict.copy()

        # self.value_correction_factor = CFG({tranche:1 for tranche in self.db_tranches})
        # self.value_correction_factor.F = 4.88  # ref. ISSUES 3

        self.exceptional_records = [
            "E04603",
            "E06072",
            "E06909",
            "E07675",
            "E07941",
            "E08321",
        ]  # ref. ISSUES 4

        self.db_dir_base = Path(db_dir)
        self._all_records = None
        self.__all_records = None
        self._ls_rec()  # loads file system structures into `self._all_records`

        self._diagnoses_records_list = None
        self._ls_diagnoses_records()

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
        s2d = {"A": "11", "B": "12", "C": "21", "D": "31", "E": "32", "F": "41"}
        s2d = {self.rec_prefix[k]: v for k, v in s2d.items()}
        prefix = "".join(re.findall(r"[A-Z]", rec))
        n = rec.replace(prefix, "")
        sid = int(f"{s2d[prefix]}{'0'*(8-len(n))}{n}")
        return sid

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        filename = f"{self.db_name}-record_list.json"
        record_list_fp = self.db_dir / filename
        write_file = False
        self._df_records = pd.DataFrame()
        self._all_records = CFG({tranche: [] for tranche in self.db_tranches})
        if record_list_fp.is_file():
            for k, v in json.loads(record_list_fp.read_text()).items():
                if k in self.tranche_names:
                    self._all_records[k] = v
            for tranche in self.db_tranches:
                df_tmp = pd.DataFrame(self._all_records[tranche], columns=["path"])
                df_tmp["tranche"] = tranche
                self._df_records = pd.concat([self._df_records, df_tmp], ignore_index=True)
            self._df_records["path"] = self._df_records.path.apply(lambda x: Path(x))
            self._df_records["record"] = self._df_records.path.apply(lambda x: x.stem)

            self._df_records = self._df_records[
                self._df_records.path.apply(lambda x: x.with_suffix(f".{self.rec_ext}").is_file())
            ]

        if len(self._df_records) == 0 or any(len(v) == 0 for v in self._all_records.values()):
            original_len = len(self._df_records)
            self._df_records = pd.DataFrame()
            self.logger.info("Please wait patiently to let the reader find all records of all the tranches...")
            start = time.time()
            rec_patterns_with_ext = {
                tranche: f"^{self.rec_prefix[tranche]}(?:\\d+)\\.{self.rec_ext}$" for tranche in self.db_tranches
            }
            self._all_records = get_record_list_recursive3(str(self.db_dir), rec_patterns_with_ext, relative=False)
            to_save = deepcopy(self._all_records)
            for tranche in self.db_tranches:
                df_tmp = pd.DataFrame(self._all_records[tranche], columns=["path"])
                df_tmp["tranche"] = tranche
                self._df_records = pd.concat([self._df_records, df_tmp], ignore_index=True)
            self._df_records["path"] = self._df_records.path.apply(lambda x: Path(x))
            self._df_records["record"] = self._df_records.path.apply(lambda x: x.stem)

            self.logger.info(f"Done in {time.time() - start:.5f} seconds!")

            if len(self._df_records) > original_len:
                write_file = True

            if write_file:
                record_list_fp.write_text(json.dumps(to_save))

        if len(self._df_records) > 0 and self._subsample is not None:
            df_tmp = pd.DataFrame()
            for tranche in self.db_tranches:
                size = int(round(len(self._all_records[tranche]) * self._subsample))
                if size > 0:
                    df_tmp = pd.concat(
                        [
                            df_tmp,
                            self._df_records[self._df_records.tranche == tranche].sample(
                                size, random_state=DEFAULTS.SEED, replace=False
                            ),
                        ],
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
            self._all_records = CFG(
                {
                    tranche: sorted(
                        [Path(x).stem for x in self._df_records[self._df_records.tranche == tranche]["path"].values]
                    )
                    for tranche in self.db_tranches
                }
            )

        self._all_records = CFG(
            {tranche: sorted([Path(x).stem for x in self._all_records[tranche]]) for tranche in self.db_tranches}
        )
        self.__all_records = list_sum(self._all_records.values())

        self._df_records.set_index("record", inplace=True)
        self._df_records["fs"] = self._df_records.tranche.apply(lambda x: self.fs[x])

        # TODO: perhaps we can load labels and metadata of all records into `self._df_records` here

    def _ls_diagnoses_records(self, force_reload: bool = False) -> None:
        """List all the records for all diagnoses.

        Parameters
        ----------
        force_reload : bool, default False
            Whether to force reload the list of records
            for each diagnosis or not.

        Returns
        -------
        None

        """
        fn = f"{self.db_name}-diagnoses_records_list.json"
        dr_fp = self.db_dir / fn
        if dr_fp.is_file() and not force_reload:
            self._diagnoses_records_list = json.loads(dr_fp.read_text())
        else:
            self.logger.info("Please wait several minutes patiently to let the reader list records for each diagnosis...")
            start = time.time()
            self._diagnoses_records_list = {d: [] for d in df_weights_abbr.columns.values.tolist()}
            for tranche, l_rec in self._all_records.items():
                for rec in l_rec:
                    ann = self.load_ann(rec)
                    ld = ann["diagnosis_scored"]["diagnosis_abbr"]
                    for d in ld:
                        self._diagnoses_records_list[d].append(rec)
            self.logger.info(f"Done in {time.time() - start:.5f} seconds!")
            if self._subsample is None:
                dr_fp.write_text(json.dumps(self._diagnoses_records_list))
        for k, v in self._diagnoses_records_list.items():
            self._diagnoses_records_list[k] = sorted(set(v).intersection(self.__all_records))

    @property
    def diagnoses_records_list(self) -> Dict[str, List[str]]:
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list

    def _get_tranche(self, rec: Union[str, int]) -> str:
        """Get the tranche"s symbol of a record via its name.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        tranche : str
            Symbol of the tranche, ref. `self.rec_prefix`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._df_records.loc[rec, "tranche"]
        return tranche

    def get_absolute_path(self, rec: Union[str, int], extension: Optional[str] = None) -> Path:
        """Get the absolute path of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        extension : str, optional
            Extension of the file.

        Returns
        -------
        abs_fp : pathlib.Path
            Absolute path of the file.

        """
        if isinstance(rec, int):
            rec = self[rec]
        abs_fp = self._df_records.loc[rec, "path"]
        if extension is not None:
            if not extension.startswith("."):
                extension = f".{extension}"
            abs_fp = abs_fp.with_suffix(extension)
        return abs_fp

    def get_data_filepath(self, rec: Union[str, int], with_ext: bool = True) -> Path:
        """Get the absolute file path of the data fileof the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        with_ext : bool, default True
            If True, the returned file path comes with file extension,
            otherwise without file extension.

        Returns
        -------
        pathlib.Path
            Absolute file path of the data file of the record.

        """
        return self.get_absolute_path(rec, self.rec_ext if with_ext else None)

    @add_docstring(get_data_filepath.__doc__.replace("data file", "header file"))
    def get_header_filepath(self, rec: Union[str, int], with_ext: bool = True) -> Path:
        return self.get_absolute_path(rec, self.ann_ext if with_ext else None)

    @add_docstring(get_data_filepath.__doc__.replace("data file", "annotation file"))
    def get_ann_filepath(self, rec: Union[str, int], with_ext: bool = True) -> str:
        """alias for `get_header_filepath`"""
        return self.get_header_filepath(rec, with_ext=with_ext)

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        data_format: str = "channel_first",
        backend: str = "wfdb",
        units: Union[str, type(None)] = "mV",
        fs: Optional[Real] = None,
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load physical (converted from digital) ECG data,
        which is more understandable for humans;
        or load digital signal directly.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        leads : str or int or List[str] or List[int], optional
            The leads of the ECG data to be loaded.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first").
        backend : {"wfdb", "scipy"}, optional
            The backend data reader, by default "wfdb".
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
            The ECG data of the record.
        data_fs : numbers.Real, optional
            Sampling frequency of the output signal.
            Returned if `return_fs` is True.

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert data_format.lower() in [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
        ], f"Invalid data_format: `{data_format}`"

        tranche = self._get_tranche(rec)

        _leads = self._normalize_leads(leads, numeric=False)

        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            # p_signal or d_signal of "lead_last" format
            wfdb_rec = wfdb.rdrecord(
                str(rec_fp),
                physical=units is not None,
                channel_names=_leads,
                return_res=DEFAULTS.DTYPE.INT,
            )
            if units is None:
                data = wfdb_rec.d_signal.T
            else:
                data = wfdb_rec.p_signal.T
            # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)
        elif backend.lower() == "scipy":
            # loadmat of "lead_first" format
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            data = loadmat(str(rec_fp))["val"]
            if units is not None:
                header_info = self.load_ann(rec, raw=False)["df_leads"]
                baselines = header_info["baseline"].values.reshape(data.shape[0], -1)
                adc_gain = header_info["adc_gain"].values.reshape(data.shape[0], -1)
                data = np.asarray(data - baselines, dtype=DEFAULTS.DTYPE.NP) / adc_gain
            leads_ind = [self.all_leads.index(item) for item in _leads]
            data = data[leads_ind, :]
            # lead_units = np.vectorize(lambda s: s.lower())(header_info["df_leads"]["adc_units"].values)
        else:
            raise ValueError(f"backend `{backend.lower()}` not supported for loading data")

        # ref. ISSUES 3, for multiplying `value_correction_factor`
        # data = data * self.value_correction_factor[tranche]

        if units is not None and units.lower() in ["uv", "μv", "muv"]:
            data = data * 1000

        if fs is not None and fs != self.fs[tranche]:
            data = SS.resample_poly(data, fs, self.fs[tranche], axis=1).astype(data.dtype)
            data_fs = fs
        else:
            data_fs = self.fs[tranche]

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        if return_fs:
            return data, data_fs
        return data

    def load_ann(self, rec: Union[str, int], raw: bool = False, backend: str = "wfdb") -> Union[dict, str]:
        """Load annotations of the record.

        The annotations are stored in the .hea files.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        raw : bool, default False
            If True, the raw annotations without parsing will be returned.
        backend : {"wfdb", "naive"}, optional
            If is "wfdb", :func:`wfdb.rdheader`
            will be used to load the annotations.
            If is "naive", annotations will be parsed
            from the lines read from the header files.

        Returns
        -------
        ann_dict : dict or str
            The annotations with items listed in `self.ann_items`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        ann_fp = self.get_ann_filepath(rec, with_ext=True)
        header_data = Path(ann_fp).read_text().splitlines()

        if raw:
            ann_dict = "\n".join(header_data)
            return ann_dict

        if backend.lower() == "wfdb":
            ann_dict = self._load_ann_wfdb(rec, header_data)
        elif backend.lower() == "naive":
            ann_dict = self._load_ann_naive(header_data)
        else:
            raise ValueError(f"backend `{backend.lower()}` not supported for loading annotations")
        return ann_dict

    def _load_ann_wfdb(self, rec: Union[str, int], header_data: List[str]) -> dict:
        """Load annotations (header) using :func:`wfdb.rdheader`.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        header_data : List[str]
            List of lines read directly from a header file.
            This data will be used, since `datetime` is
            not well parsed by :func:`wfdb.rdheader`.

        Returns
        -------
        ann_dict : dict
            The annotations with items listed in `self.ann_items`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        header_fp = self.get_header_filepath(rec, with_ext=False)
        header_reader = wfdb.rdheader(str(header_fp))
        ann_dict = {}
        (
            ann_dict["rec_name"],
            ann_dict["nb_leads"],
            ann_dict["fs"],
            ann_dict["nb_samples"],
            ann_dict["datetime"],
            daytime,
        ) = header_data[0].split(" ")

        ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
        ann_dict["fs"] = int(ann_dict["fs"])
        ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
        ann_dict["datetime"] = datetime.strptime(" ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S")
        try:  # see NOTE. 1.
            ann_dict["age"] = int([line for line in header_reader.comments if "Age" in line][0].split(": ")[-1])
        except Exception:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [line for line in header_reader.comments if "Sex" in line][0].split(": ")[-1]
        except Exception:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [line for line in header_reader.comments if "Rx" in line][0].split(": ")[-1]
        except Exception:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [line for line in header_reader.comments if "Hx" in line][0].split(": ")[-1]
        except Exception:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [line for line in header_reader.comments if "Sx" in line][0].split(": ")[-1]
        except Exception:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = [line for line in header_reader.comments if "Dx" in line][0].split(": ")[-1].split(",")
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

        df_leads = pd.DataFrame()
        for k in [
            "file_name",
            "fmt",
            "byte_offset",
            "adc_gain",
            "units",
            "adc_res",
            "adc_zero",
            "baseline",
            "init_value",
            "checksum",
            "block_size",
            "sig_name",
        ]:
            df_leads[k] = header_reader.__dict__[k]
        df_leads = df_leads.rename(
            columns={
                "sig_name": "lead_name",
                "units": "adc_units",
                "file_name": "filename",
            }
        )
        df_leads.index = df_leads["lead_name"]
        df_leads.index.name = None
        ann_dict["df_leads"] = df_leads

        return ann_dict

    def _load_ann_naive(self, header_data: List[str]) -> dict:
        """Load annotations (header) using raw data
        read directly from a header file.

        Parameters
        ----------
        header_data : List[str]
            The list of lines read directly from a header file.

        Returns
        -------
        ann_dict : dict
            The annotations with items in `self.ann_items`.

        """
        ann_dict = {}
        (
            ann_dict["rec_name"],
            ann_dict["nb_leads"],
            ann_dict["fs"],
            ann_dict["nb_samples"],
            ann_dict["datetime"],
            daytime,
        ) = header_data[0].split(" ")

        ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
        ann_dict["fs"] = int(ann_dict["fs"])
        ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
        ann_dict["datetime"] = datetime.strptime(" ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S")
        try:  # see NOTE. 1.
            ann_dict["age"] = int([line for line in header_data if line.startswith("#Age")][0].split(": ")[-1])
        except Exception:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [line for line in header_data if line.startswith("#Sex")][0].split(": ")[-1]
        except Exception:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [line for line in header_data if line.startswith("#Rx")][0].split(": ")[-1]
        except Exception:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [line for line in header_data if line.startswith("#Hx")][0].split(": ")[-1]
        except Exception:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [line for line in header_data if line.startswith("#Sx")][0].split(": ")[-1]
        except Exception:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = [line for line in header_data if line.startswith("#Dx")][0].split(": ")[-1].split(",")
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

        ann_dict["df_leads"] = self._parse_leads(header_data[1:13])

        return ann_dict

    def _parse_diagnosis(self, l_Dx: List[str]) -> Tuple[dict, dict]:
        """Parse diagnosis from a list of strings.

        Parameters
        ----------
        l_Dx : List[str]
            Raw information of diagnosis, read from a header file.

        Returns
        -------
        diag_dict : dict
            Diagnosis, including SNOMED CT Codes,
            fullnames and abbreviations of each diagnosis.
        diag_scored_dict : dict
            The scored items in `diag_dict`.

        """
        diag_dict, diag_scored_dict = {}, {}
        try:
            diag_dict["diagnosis_code"] = [item for item in l_Dx]
            # selection = dx_mapping_all["SNOMED CT Code"].isin(diag_dict["diagnosis_code"])
            # diag_dict["diagnosis_abbr"] = dx_mapping_all[selection]["Abbreviation"].tolist()
            # diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
            diag_dict["diagnosis_abbr"] = [
                dx_mapping_all[dx_mapping_all["SNOMED CT Code"] == dc]["Abbreviation"].values[0]
                for dc in diag_dict["diagnosis_code"]
            ]
            diag_dict["diagnosis_fullname"] = [
                dx_mapping_all[dx_mapping_all["SNOMED CT Code"] == dc]["Dx"].values[0] for dc in diag_dict["diagnosis_code"]
            ]
            scored_indices = np.isin(diag_dict["diagnosis_code"], dx_mapping_scored["SNOMED CT Code"].values)
            diag_scored_dict["diagnosis_code"] = [
                item for idx, item in enumerate(diag_dict["diagnosis_code"]) if scored_indices[idx]
            ]
            diag_scored_dict["diagnosis_abbr"] = [
                item for idx, item in enumerate(diag_dict["diagnosis_abbr"]) if scored_indices[idx]
            ]
            diag_scored_dict["diagnosis_fullname"] = [
                item for idx, item in enumerate(diag_dict["diagnosis_fullname"]) if scored_indices[idx]
            ]
        except Exception:  # the old version, the Dx"s are abbreviations
            diag_dict["diagnosis_abbr"] = diag_dict["diagnosis_code"]
            selection = dx_mapping_all["Abbreviation"].isin(diag_dict["diagnosis_abbr"])
            diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
        return diag_dict, diag_scored_dict

    def _parse_leads(self, l_leads_data: List[str]) -> pd.DataFrame:
        """Parse leads information from a list of strings.

        Parameters
        ----------
        l_leads_data : List[str]
            Raw information of each lead, read from a header file.

        Returns
        -------
        df_leads : pandas.DataFrame
            Infomation of each leads in the format
            of a :class:`~pandas.DataFrame`.

        """
        df_leads = pd.read_csv(io.StringIO("\n".join(l_leads_data)), delim_whitespace=True, header=None)
        df_leads.columns = [
            "filename",
            "fmt+byte_offset",
            "adc_gain+units",
            "adc_res",
            "adc_zero",
            "init_value",
            "checksum",
            "block_size",
            "lead_name",
        ]
        df_leads["fmt"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[0])
        df_leads["byte_offset"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[1])
        df_leads["adc_gain"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[0])
        df_leads["adc_units"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[1])
        for k in [
            "byte_offset",
            "adc_gain",
            "adc_res",
            "adc_zero",
            "init_value",
            "checksum",
        ]:
            df_leads[k] = df_leads[k].apply(lambda s: int(s))
        df_leads["baseline"] = df_leads["adc_zero"]
        df_leads = df_leads[
            [
                "filename",
                "fmt",
                "byte_offset",
                "adc_gain",
                "adc_units",
                "adc_res",
                "adc_zero",
                "baseline",
                "init_value",
                "checksum",
                "block_size",
                "lead_name",
            ]
        ]
        df_leads.index = df_leads["lead_name"]
        df_leads.index.name = None
        return df_leads

    @add_docstring(load_ann.__doc__)
    def load_header(self, rec: Union[str, int], raw: bool = False) -> Union[dict, str]:
        """
        alias for `load_ann`, as annotations are also stored in header files
        """
        return self.load_ann(rec, raw)

    def get_labels(
        self,
        rec: Union[str, int],
        scored_only: bool = True,
        fmt: str = "s",
        normalize: bool = True,
    ) -> List[str]:
        """Get labels (diagnoses or arrhythmias) of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        scored_only : bool, default True
            If True, only get the labels that are scored
            in the CINC2020 official phase.
        fmt : str, default "s"
            Format of labels, one of the following (case insensitive):

                - "a", abbreviations
                - "f", full names
                - "s", SNOMED CT Code

        normalize : bool, default True
            If True, the labels will be transformed into their equavalents,
            which are defined in `utils.utils_misc.cinc2020_aux_data.py`.

        Returns
        -------
        labels : List[str]
            The list of labels of the record.

        """
        ann_dict = self.load_ann(rec)
        if scored_only:
            labels = ann_dict["diagnosis_scored"]
        else:
            labels = ann_dict["diagnosis"]
        if fmt.lower() == "a":
            labels = labels["diagnosis_abbr"]
        elif fmt.lower() == "f":
            labels = labels["diagnosis_fullname"]
        elif fmt.lower() == "s":
            labels = labels["diagnosis_code"]
        else:
            raise ValueError(f"`fmt` should be one of `a`, `f`, `s`, but got `{fmt}`")
        if normalize:
            labels = [self.label_trans_dict.get(item, item) for item in labels]
        return labels

    def get_fs(self, rec: Union[str, int]) -> Real:
        """Get the sampling frequency of the record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.

        Returns
        -------
        fs : numbers.Real
            Sampling frequency of the record.

        """
        tranche = self._get_tranche(rec)
        fs = self.fs[tranche]
        return fs

    def get_subject_info(self, rec: Union[str, int], items: Optional[List[str]] = None) -> dict:
        """Get auxiliary information of a subject
        (a record) stored in the header files.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        items : List[str], optional
            Items of the subject's information (e.g. sex, age, etc.).

        Returns
        -------
        subject_info : dict
            Information about the subject, including
            "age", "sex", "medical_prescription",
            "history", "symptom_or_surgery".

        """
        if items is None or len(items) == 0:
            info_items = [
                "age",
                "sex",
                "medical_prescription",
                "history",
                "symptom_or_surgery",
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec)
        subject_info = {item: ann_dict[item] for item in info_items}

        return subject_info

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[Dict[str, np.ndarray]] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, List[str]]] = None,
        same_range: bool = False,
        waves: Optional[Dict[str, Sequence[int]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        data : numpy.ndarray, optional
            (12-lead) ECG signal to plot,
            should be of the format "channel_first",
            and compatible with `leads`.
            If is not None, data of `rec` will not be used.
            This is useful when plotting filtered data.
        ann : dict, optional
            Annotations for `data`, with 2 items: "scored", "all".
            Ignored if `data` is None.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads : str or List[str], optional
            The leads of the ECG signal to plot.
        same_range : bool, default False
            If True, all leads are forced to have the same y range.
        waves : dict, optional
            Indices of the wave critical points, including
            "p_onsets", "p_peaks", "p_offsets",
            "q_onsets", "q_peaks", "r_peaks", "s_peaks", "s_offsets",
            "t_onsets", "t_peaks", "t_offsets".
        kwargs : dict, optional
            Additional keyword arguments to pass to :func:`matplotlib.pyplot.plot`.

        TODO
        ----
        1. Slice too long records, and plot separately for each segment.
        2. Plot waves using :func:`matplotlib.pyplot.axvspan`.

        NOTE
        ----
        `Locator` of ``plt`` has default `MAXTICKS` of 1000.
        If not modifying this number, at most 40 seconds of signal could be plotted once.

        Contributors: Jeethan, and WEN Hao

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        if tranche in "CDE":
            physionet_lightwave_suffix = CFG(
                {
                    "C": "incartdb/1.0.0",
                    "D": "ptbdb/1.0.0",
                    "E": "ptb-xl/1.0.1",
                }
            )
            url = f"https://physionet.org/lightwave/?db={physionet_lightwave_suffix[tranche]}"
            self.logger.info(f"better view: {url}")

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000

        _leads = self._normalize_leads(leads, numeric=False)
        lead_indices = [self.all_leads.index(ld) for ld in _leads]

        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[lead_indices]
        else:
            units = self._auto_infer_units(data)
            self.logger.info(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            assert _data.shape[0] == len(
                _leads
            ), f"number of leads from data of shape ({_data.shape[0]}) does not match the length ({len(_leads)}) of `leads`"

        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        if waves:
            if waves.get("p_onsets", None) and waves.get("p_offsets", None):
                p_waves = [[onset, offset] for onset, offset in zip(waves["p_onsets"], waves["p_offsets"])]
            elif waves.get("p_peaks", None):
                p_waves = [
                    [
                        max(0, p + ms2samples(_PlotCfg.p_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            p + ms2samples(_PlotCfg.p_offset, fs=self.get_fs(rec)),
                        ),
                    ]
                    for p in waves["p_peaks"]
                ]
            else:
                p_waves = []
            if waves.get("q_onsets", None) and waves.get("s_offsets", None):
                qrs = [[onset, offset] for onset, offset in zip(waves["q_onsets"], waves["s_offsets"])]
            elif waves.get("q_peaks", None) and waves.get("s_peaks", None):
                qrs = [
                    [
                        max(0, q + ms2samples(_PlotCfg.q_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            s + ms2samples(_PlotCfg.s_offset, fs=self.get_fs(rec)),
                        ),
                    ]
                    for q, s in zip(waves["q_peaks"], waves["s_peaks"])
                ]
            elif waves.get("r_peaks", None):
                qrs = [
                    [
                        max(0, r + ms2samples(_PlotCfg.qrs_radius, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            r + ms2samples(_PlotCfg.qrs_radius, fs=self.get_fs(rec)),
                        ),
                    ]
                    for r in waves["r_peaks"]
                ]
            else:
                qrs = []
            if waves.get("t_onsets", None) and waves.get("t_offsets", None):
                t_waves = [[onset, offset] for onset, offset in zip(waves["t_onsets"], waves["t_offsets"])]
            elif waves.get("t_peaks", None):
                t_waves = [
                    [
                        max(0, t + ms2samples(_PlotCfg.t_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            t + ms2samples(_PlotCfg.t_offset, fs=self.get_fs(rec)),
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
            "qrs": "red",
            "t_waves": "yellow",
        }
        plot_alpha = 0.4

        if ann is None or data is None:
            diag_scored = self.get_labels(rec, scored_only=True, fmt="a")
            diag_all = self.get_labels(rec, scored_only=False, fmt="a")
        else:
            diag_scored = ann["scored"]
            diag_all = ann["all"]

        nb_leads = len(_leads)

        seg_len = self.fs[tranche] * 25  # 25 seconds
        nb_segs = _data.shape[1] // seg_len

        t = np.arange(_data.shape[1]) / self.fs[tranche]
        duration = len(t) / self.fs[tranche]
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * np.maximum(y_ranges, 750) / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=False, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            axes[idx].plot(
                t,
                _data[idx],
                color="black",
                linewidth="2.0",
                label=f"lead - {_leads[idx]}",
            )
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.4", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.2", color="gray")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot([], [], " ", label=f"labels_s - {','.join(diag_scored)}")
            axes[idx].plot([], [], " ", label=f"labels_a - {','.join(diag_all)}")
            axes[idx].plot([], [], " ", label=f"tranche - {self.tranche_names[tranche]}")
            axes[idx].plot([], [], " ", label=f"fs - {self.fs[tranche]}")
            for w in ["p_waves", "qrs", "t_waves"]:
                for itv in eval(w):
                    axes[idx].axvspan(t[itv[0]], t[itv[1]], color=palette[w], alpha=plot_alpha)
            axes[idx].legend(loc="upper left", fontsize=14)
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(min(-600, -y_ranges[idx]), max(600, y_ranges[idx]))
            axes[idx].set_xlabel("Time [s]", fontsize=16)
            axes[idx].set_ylabel("Voltage [μV]", fontsize=16)
        plt.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()

    def get_tranche_class_distribution(self, tranches: Sequence[str], scored_only: bool = True) -> Dict[str, int]:
        """Compute class distribution in the tranches.

        Parameters
        ----------
        tranches : Sequence[str]
            Tranche symbols (A-F).
        scored_only : bool, default True
            If True, only classes that are scored in the CINC2020 official phase
            are considered for computing the distribution.

        Returns
        -------
        distribution : dict
            Distribution of classes in the tranches.
            Keys are abbrevations of the classes, and
            values are appearance of corr. classes in the tranche.

        """
        tranche_names = [self.tranche_names[t] for t in tranches]
        df = dx_mapping_scored if scored_only else dx_mapping_all
        distribution = CFG()
        for _, row in df.iterrows():
            num = (row[tranche_names].values).sum()
            if num > 0:
                distribution[row["Abbreviation"]] = num
        return distribution

    def load_resampled_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        siglen: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resample the data of `rec` to 500Hz,
        or load the resampled data in 500Hz, if the corr. data file already exists

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first").
        siglen : int, optional
            Signal length, with units in number of samples.
            If is not None, signal with length longer will be
            sliced to the length of `siglen`.
            Used for preparing/doing model training for example.

        Returns
        -------
        numpy.ndarray
            2D resampled (and perhaps sliced 3D) signal data.

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        if siglen is None:
            rec_fp = self.db_dir / f"{self.db_name}-rsmp-500Hz" / self.tranche_names[tranche] / f"{rec}_500Hz.npy"
        else:
            rec_fp = (
                self.db_dir / f"{self.db_name}-rsmp-500Hz" / self.tranche_names[tranche] / f"{rec}_500Hz_siglen_{siglen}.npy"
            )
        rec_fp.parent.mkdir(parents=True, exist_ok=True)
        if not rec_fp.is_file():
            # self.logger.info(f"corresponding file {rec_fp.name} does not exist")
            data = self.load_data(rec, data_format="channel_first", units="mV", fs=None)
            if self.fs[tranche] != 500:
                data = SS.resample_poly(data, 500, self.fs[tranche], axis=1).astype(DEFAULTS.DTYPE.NP)
            if siglen is not None and data.shape[1] >= siglen:
                # slice_start = (data.shape[1] - siglen)//2
                # slice_end = slice_start + siglen
                # data = data[..., slice_start:slice_end]
                data = ensure_siglen(data, siglen=siglen, fmt="channel_first", tolerance=0.2).astype(DEFAULTS.DTYPE.NP)
                np.save(rec_fp, data)
            elif siglen is None:
                np.save(rec_fp, data)
        else:
            # self.logger.info(f"loading from local file...")
            data = np.load(rec_fp).astype(DEFAULTS.DTYPE.NP)
        if data_format.lower() in ["channel_last", "lead_last"]:
            data = np.moveaxis(data, -1, -2)
        return data

    def load_raw_data(self, rec: Union[str, int], backend: str = "scipy") -> np.ndarray:
        """
        Load raw data from corresponding files with no further processing,
        in order to facilitate feeding data into the `run_12ECG_classifier` function

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        backend : {"scipy", "wfdb"}, optional
            The backend data reader, by default "scipy".
            Note that "scipy" provides data in the format of "lead_first",
            while "wfdb" provides data in the format of "lead_last".

        Returns
        -------
        raw_data: numpy.ndarray
            Raw data (d_signal) loaded from corresponding data file,
            without digital-to-analog conversion (DAC) and resampling.

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            wfdb_rec = wfdb.rdrecord(str(rec_fp), physical=False)
            raw_data = np.asarray(wfdb_rec.d_signal, dtype=DEFAULTS.DTYPE.NP)
        elif backend.lower() == "scipy":
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            raw_data = loadmat(str(rec_fp))["val"].astype(DEFAULTS.DTYPE.NP)
        return raw_data

    def _check_nan(self, tranches: Union[str, Sequence[str]]) -> None:
        """Check if records from `tranches` has nan values.

        Accessing data using `p_signal` of `wfdb` would produce nan values,
        if exceptionally large values are encountered.
        This function could help detect abnormal records as well.

        Parameters
        ----------
        tranches : str or Sequence[str]
            Tranches to check.

        Returns
        -------
        None

        """
        for t in tranches:
            for rec in self.all_records[t]:
                data = self.load_data(rec)
                if np.isnan(data).any():
                    self.logger.info(f"record {rec} from tranche {t} has nan values")

    @property
    def url(self) -> List[str]:
        domain = "https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ECG-public/"
        return [posixpath.join(domain, f) for f in self.data_files]

    data_files = [
        "PhysioNetChallenge2020_Training_CPSC.tar.gz",
        "PhysioNetChallenge2020_Training_2.tar.gz",
        "PhysioNetChallenge2020_Training_StPetersburg.tar.gz",
        "PhysioNetChallenge2020_Training_PTB.tar.gz",
        "PhysioNetChallenge2020_Training_PTB-XL.tar.gz",
        "PhysioNetChallenge2020_Training_E.tar.gz",
    ]

    def download(self) -> None:
        for url in self.url:
            http_get(url, self.db_dir / _stem(url), extract=True)
        self._ls_rec()

    def __len__(self) -> int:
        return len(self.__all_records)

    def __getitem__(self, index: int) -> str:
        return self.__all_records[index]

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2020_INFO


def compute_all_metrics(classes: List[str], truth: Sequence, binary_pred: Sequence, scalar_pred: Sequence) -> Tuple[float]:
    """Compute all the metrics for the challenge.

    Parameters
    ----------
    classes : List[str]
        List of all the classes, in the format of abbrevations.
    truth : array_like
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    binary_pred : array_like
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    scalar_pred : array_like
        Probability predictions, of shape ``(n_records, n_classes)``,
        with values within the interval [0, 1].

    Returns
    -------
    auroc : float
        Area under the receiver operating characteristic (ROC) curve.
    auprc : float
        Area under the precision-recall curve.
    accuracy : float
        Macro-averaged accuracy.
    f_measure : float
        Macro-averaged F1 score.
    f_beta_measure : float
        Macro-averaged F-beta score.
    g_beta_measure : float
        Macro-averaged G-beta score.
    challenge_metric : float
        Challenge metric, defined by a weight matrix.

    """
    # normal_class = "426783006"
    normal_class = "NSR"
    # equivalent_classes = [["713427006", "59118001"], ["284470004", "63593006"], ["427172004", "17338001"]]
    weights = load_weights(classes=classes)

    _truth = np.array(truth)
    _binary_pred = np.array(binary_pred)
    _scalar_pred = np.array(scalar_pred)

    print("- AUROC and AUPRC...")
    auroc, auprc = _compute_auc(_truth, _scalar_pred)

    print("- Accuracy...")
    accuracy = _compute_accuracy(_truth, _binary_pred)

    print("- F-measure...")
    f_measure = _compute_f_measure(_truth, _binary_pred)

    print("- F-beta and G-beta measures...")
    f_beta_measure, g_beta_measure = _compute_beta_measures(_truth, _binary_pred, beta=2)

    print("- Challenge metric...")
    challenge_metric = compute_challenge_metric(weights, _truth, _binary_pred, classes, normal_class)

    print("Done.")

    # Return the results.
    return (
        auroc,
        auprc,
        accuracy,
        f_measure,
        f_beta_measure,
        g_beta_measure,
        challenge_metric,
    )


def _compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute recording-wise accuracy.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.

    Returns
    -------
    accuracy : float
        Macro-averaged accuracy.

    """
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


def _compute_confusion_matrices(labels: np.ndarray, outputs: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Compute confusion matrices.

    Compute a binary confusion matrix for each class k:

          [TN_k FN_k]
          [FP_k TP_k]

    If the normalize variable is set to true, then normalize the contributions
    to the confusion matrix by the number of labels per recording.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    normalize : bool, optional
        If true, normalize the confusion matrices by the number of labels per
        recording. Default is false.

    Returns
    -------
    A : numpy.ndarray
        Confusion matrices, of shape ``(n_classes, 2, 2)``.

    """
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError("Error in computing the confusion matrix.")

    return A


def _compute_f_measure(labels: np.ndarray, outputs: np.ndarray) -> float:
    """Compute macro-averaged F1 score.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.

    Returns
    -------
    f_measure : float
        Macro-averaged F1 score.

    """
    num_recordings, num_classes = np.shape(labels)

    A = _compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure


def _compute_beta_measures(labels: np.ndarray, outputs: np.ndarray, beta: Real) -> Tuple[float, float]:
    """Compute F-beta and G-beta measures.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    beta : float
        Beta parameter.

    Returns
    -------
    f_beta_measure : float
        Macro-averaged F-beta measure.
    g_beta_measure : float
        Macro-averaged G-beta measure.

    """
    num_recordings, num_classes = np.shape(labels)

    A = _compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1 + beta**2) * tp + fp + beta**2 * fn:
            f_beta_measure[k] = float((1 + beta**2) * tp) / float((1 + beta**2) * tp + fp + beta**2 * fn)
        else:
            f_beta_measure[k] = float("nan")
        if tp + fp + beta * fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
        else:
            g_beta_measure[k] = float("nan")

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure


def _compute_auc(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float]:
    """Compute macro-averaged AUROC and macro-averaged AUPRC.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.

    Returns
    -------
    auroc : float
        Macro-averaged AUROC.
    auprc : float
        Macro-averaged AUPRC.

    """
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc


def _compute_modified_confusion_matrix(labels: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Compute a binary multi-class, multi-label confusion matrix,
    where the rows are the labels and the columns are the outputs.

    Parameters
    ----------
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.

    Returns
    -------
    A : numpy.ndarray
        Modified confusion matrix, of shape ``(n_classes, n_classes)``.

    """
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization
    return A


def compute_challenge_metric(
    weights: np.ndarray,
    labels: np.ndarray,
    outputs: np.ndarray,
    classes: List[str],
    normal_class: str,
) -> float:
    """Compute the evaluation metrics for the Challenge.

    Parameters
    ----------
    weights : numpy.ndarray
        Array of weights, of shape ``(n_classes, n_classes)``.
    labels : numpy.ndarray
        Ground truth array, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    outputs : numpy.ndarray
        Binary predictions, of shape ``(n_records, n_classes)``,
        with values 0 or 1.
    classes : List[str]
        List of class names.
    normal_class : str
        Name of the normal class.

    Returns
    -------
    score : float
        Challenge metric score.

    """
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = _compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = _compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=bool)
    inactive_outputs[:, normal_index] = 1
    A = _compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


# alias
compute_metrics = compute_challenge_metric
