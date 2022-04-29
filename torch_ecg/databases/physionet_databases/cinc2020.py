# -*- coding: utf-8 -*-
"""
"""

import io
import json
import posixpath
import re
import time
from copy import deepcopy
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import resample, resample_poly  # noqa: F401

from ...cfg import CFG
from ...utils import ecg_arrhythmia_knowledge as EAK
from ...utils.download import _stem, http_get
from ...utils.misc import (
    add_docstring,
    dict_to_str,
    get_record_list_recursive3,
    list_sum,
    ms2samples,
)
from ...utils.utils_data import ensure_siglen
from ..aux_data.cinc2020_aux_data import (
    df_weights_abbr,
    dx_mapping_all,
    dx_mapping_scored,
    equiv_class_dict,
    load_weights,
    normalize_class,
)
from ..base import DEFAULT_FIG_SIZE_PER_SEC, PhysioNetDataBase

__all__ = [
    "CINC2020",
    "compute_metrics",
    "compute_all_metrics",
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


class CINC2020(PhysioNetDataBase):
    """finished, under improving,

    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020

    ABOUT CINC2020
    --------------
    0. There are 6 difference tranches of training data, listed as follows:
        A. 6,877
        recordings from China Physiological Signal Challenge in 2018 (CPSC2018): PhysioNetChallenge2020_Training_CPSC.tar.gz in ref. [6]
        B. 3,453 recordings
        from China 12-Lead ECG Challenge Database (unused data from CPSC2018 and NOT the CPSC2018 test data): PhysioNetChallenge2020_Training_2.tar.gz in ref. [6]
        C. 74 recordings
        from the St Petersburg INCART 12-lead Arrhythmia Database: PhysioNetChallenge2020_Training_StPetersburg.tar.gz in ref. [6]
        D. 516 recordings
        from the PTB Diagnostic ECG Database: PhysioNetChallenge2020_Training_PTB.tar.gz in ref. [6]
        E. 21,837 recordings
        from the PTB-XL electrocardiography Database: PhysioNetChallenge2020_PTB-XL.tar.gz in ref. [6]
        F. 10,344 recordings
        from a Georgia 12-Lead ECG Challenge Database: PhysioNetChallenge2020_Training_E.tar.gz in ref. [6]
    In total, 43,101 labeled recordings of 12-lead ECGs from four countries (China, Germany, Russia, and the USA) across 3 continents have been posted publicly for this Challenge, with approximately the same number hidden for testing, representing the largest public collection of 12-lead ECGs

    1. the A tranche training data comes from CPSC2018, whose folder name is `Training_WFDB`. The B tranche training data are unused training data of CPSC2018, having folder name `Training_2`. For these 2 tranches, ref. the docstring of `database_reader.cpsc_databases.cpsc2018.CPSC2018`
    2. C. D. E. tranches of training data all come from corresponding PhysioNet dataset, whose details can be found in corresponding files:
        C: database_reader.physionet_databases.incartdb.INCARTDB
        D: database_reader.physionet_databases.ptbdb.PTBDB
        E: database_reader.physionet_databases.ptb_xl.PTB_XL
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
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    using for example the following code:
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

    NOTE
    ----
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
    9. the .mat files all contain digital signals, which has to be converted to physical values using adc gain, basesline, etc. in corresponding .hea files. `wfdb.rdrecord` has already done this conversion, hence greatly simplifies the data loading process.
    NOTE that there"s a difference when using `wfdb.rdrecord`: data from `loadmat` are in "channel_first" format, while `wfdb.rdrecord.p_signal` produces data in the "channel_last" format
    10. there"re 3 equivalent (2 classes are equivalent if the corr. value in the scoring matrix is 1):
        (RBBB, CRBBB), (PAC, SVPB), (PVC, VPB)
    11. in the newly (Feb., 2021) created dataset (ref. [7]), header files of each subset were gathered into one separate compressed file. This is due to the fact that updates on the dataset are almost always done in the header files. The correct usage of ref. [7], after uncompressing, is replacing the header files in the folder `All_training_WFDB` by header files from the 6 folders containing all header files from the 6 subsets.

    ISSUES
    ------
    1. reading the .hea files, baselines of all records are 0, however it is not the case if one plot the signal
    2. about half of the LAD records satisfy the "2-lead" criteria, but fail for the "3-lead" criteria, which means that their axis is (-30°, 0°) which is not truely LAD
    3. (Aug. 15, 2020; resolved, and changed to 1000) tranche F, the Georgia subset, has ADC gain 4880 which might be too high. Thus obtained voltages are too low. 1000 might be a suitable (correct) value of ADC gain for this tranche just as the other tranches.
    4. "E04603" (all leads), "E06072" (chest leads, epecially V1-V3), "E06909" (lead V2), "E07675" (lead V3), "E07941" (lead V6), "E08321" (lead V6) has exceptionally large values at rpeaks, reading (`load_data`) these two records using `wfdb` would bring in `nan` values. One can check using the following code
    >>> rec = "E04603"
    >>> dr.plot(rec, dr.load_data(rec, backend="scipy", units="uv"))  # currently raising error

    Usage
    -----
    1. ECG arrhythmia detection

    References
    ----------
    1. <a name="ref1"></a> https://physionetchallenges.github.io/2020/
    2. <a name="ref2"></a> https://physionet.org/content/challenge-2020/1.0.1/
    3. <a name="ref3"></a> http://2018.icbeb.org/#
    4. <a name="ref4"></a> https://physionet.org/content/incartdb/1.0.0/
    5. <a name="ref5"></a> https://physionet.org/content/ptbdb/1.0.0/
    6. <a name="ref6"></a> https://physionet.org/content/ptb-xl/1.0.1/
    7. <a name="ref7"></a> (deprecated) https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ECG-public/
    8. <a name="ref8"></a> (recommended) https://storage.cloud.google.com/physionetchallenge2021-public-datasets/

    """

    def __init__(
        self,
        db_dir: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        db_dir: str or Path,
            storage path of the database
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
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

        self.db_dir_base = Path(db_dir)
        self.db_dirs = CFG({tranche: "" for tranche in self.db_tranches})
        self._all_records = None
        self.__all_records = None
        self._ls_rec()  # loads file system structures into self.db_dirs and self._all_records

        self._diagnoses_records_list = None
        self._ls_diagnoses_records()

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
        self._all_leads_set = set(self.all_leads)

        self.df_ecg_arrhythmia = dx_mapping_all[
            ["Dx", "SNOMED CT Code", "Abbreviation"]
        ]
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

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        sid: int,
            the `subject_id` corr. to `rec`

        """
        if isinstance(rec, int):
            rec = self[rec]
        s2d = {"A": "11", "B": "12", "C": "21", "D": "31", "E": "32", "F": "41"}
        s2d = {self.rec_prefix[k]: v for k, v in s2d.items()}
        prefix = "".join(re.findall(r"[A-Z]", rec))
        n = rec.replace(prefix, "")
        sid = int(f"{s2d[prefix]}{'0'*(8-len(n))}{n}")
        return sid

    def _ls_rec(self) -> NoReturn:
        """
        list all the records and load into `self._all_records`,
        facilitating further uses

        """
        fn = "record_list.json"
        record_list_fp = self.db_dir_base / fn
        if record_list_fp.is_file():
            self._all_records = {
                k: v
                for k, v in json.loads(record_list_fp.read_text()).items()
                if k in self.tranche_names
            }
            for tranche in self.db_tranches:
                self.db_dirs[tranche] = (
                    self.db_dir_base / Path(self._all_records[tranche][0]).parent
                )
                self._all_records[tranche] = [
                    Path(f).name for f in self._all_records[tranche]
                ]
        else:
            print(
                "Please wait patiently to let the reader find all records of all the tranches..."
            )
            start = time.time()
            rec_patterns_with_ext = {
                tranche: f"^{self.rec_prefix[tranche]}(?:\\d+).{self.rec_ext}$"
                for tranche in self.db_tranches
            }
            self._all_records = get_record_list_recursive3(
                str(self.db_dir_base), rec_patterns_with_ext
            )
            to_save = deepcopy(self._all_records)
            for tranche in self.db_tranches:
                tmp_dirname = [Path(f).parent for f in self._all_records[tranche]]
                if len(set(tmp_dirname)) != 1:
                    if len(set(tmp_dirname)) > 1:
                        raise ValueError(
                            f"records of tranche {tranche} are stored in several folders!"
                        )
                    else:
                        raise ValueError(f"no record found for tranche {tranche}!")
                        continue
                self.db_dirs[tranche] = self.db_dir_base / tmp_dirname[0]
                self._all_records[tranche] = [
                    Path(f).parent for f in self._all_records[tranche]
                ]
            print(f"Done in {time.time() - start:.5f} seconds!")
            record_list_fp.write_text(json.dumps(to_save))
        self.__all_records = list_sum(self._all_records.values())

    def _ls_diagnoses_records(self) -> NoReturn:
        """list all the records for all diagnoses"""
        fn = "diagnoses_records_list.json"
        dr_fp = self.db_dir_base / fn
        if dr_fp.is_file():
            self._diagnoses_records_list = json.loads(dr_fp.read_text())
        else:
            print(
                "Please wait several minutes patiently to let the reader list records for each diagnosis..."
            )
            start = time.time()
            self._diagnoses_records_list = {
                d: [] for d in df_weights_abbr.columns.values.tolist()
            }
            for tranche, l_rec in self._all_records.items():
                for rec in l_rec:
                    ann = self.load_ann(rec)
                    ld = ann["diagnosis_scored"]["diagnosis_abbr"]
                    for d in ld:
                        self._diagnoses_records_list[d].append(rec)
            print(f"Done in {time.time() - start:.5f} seconds!")
            dr_fp.write_text(json.dumps(self._diagnoses_records_list))
        self._all_records = CFG(self._all_records)
        self.__all_records = list_sum(self._all_records.values())

    @property
    def diagnoses_records_list(self) -> Dict[str, List[str]]:
        """ """
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list

    def _get_tranche(self, rec: Union[str, int]) -> str:
        """

        get the tranche"s symbol (one of "A","B","C","D","E","F") of a record via its name

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        tranche, str,
            symbol of the tranche, ref. `self.rec_prefix`

        """
        if isinstance(rec, int):
            rec = self[rec]
        prefix = "".join(re.findall(r"[A-Z]", rec))
        tranche = {v: k for k, v in self.rec_prefix.items()}[prefix]
        return tranche

    def get_data_filepath(self, rec: Union[str, int], with_ext: bool = True) -> str:
        """

        get the absolute file path of the data file of `rec`

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        with_ext: bool, default True,
            if True, the returned file path comes with file extension,
            otherwise without file extension,
            which is useful for `wfdb` functions

        Returns
        -------
        fp: str,
            absolute file path of the data file of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        fp = self.db_dirs[tranche] / f"{rec}.{self.rec_ext}"
        if not with_ext:
            fp = fp.with_suffix("")
        return str(fp)

    def get_header_filepath(self, rec: Union[str, int], with_ext: bool = True) -> str:
        """

        get the absolute file path of the header file of `rec`

        Parameters
        ----------
        rec: str,
            name of the record
        with_ext: bool, default True,
            if True, the returned file path comes with file extension,
            otherwise without file extension,
            which is useful for `wfdb` functions

        Returns
        -------
        fp: str,
            absolute file path of the header file of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        fp = self.db_dirs[tranche] / f"{rec}.{self.ann_ext}"
        if not with_ext:
            fp = fp.with_suffix("")
        return str(fp)

    @add_docstring(get_header_filepath.__doc__)
    def get_ann_filepath(self, rec: Union[str, int], with_ext: bool = True) -> str:
        """
        alias for `get_header_filepath`
        """
        fp = self.get_header_filepath(rec, with_ext=with_ext)
        return fp

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[str, List[str]]] = None,
        data_format: str = "channel_first",
        backend: str = "wfdb",
        units: str = "mV",
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
        if isinstance(rec, int):
            rec = self[rec]
        assert data_format.lower() in [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
        ]
        tranche = self._get_tranche(rec)
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        # if tranche in "CD" and fs == 500:  # resample will be done at the end of the function
        #     data = self.load_resampled_data(rec)
        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            # p_signal of "lead_last" format
            wfdb_rec = wfdb.rdrecord(rec_fp, physical=True, channel_names=_leads)
            data = np.asarray(wfdb_rec.p_signal.T)
            # lead_units = np.vectorize(lambda s: s.lower())(wfdb_rec.units)
        elif backend.lower() == "scipy":
            # loadmat of "lead_first" format
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            data = loadmat(rec_fp)["val"]
            header_info = self.load_ann(rec, raw=False)["df_leads"]
            baselines = header_info["baseline"].values.reshape(data.shape[0], -1)
            adc_gain = header_info["adc_gain"].values.reshape(data.shape[0], -1)
            data = np.asarray(data - baselines) / adc_gain
            leads_ind = [self.all_leads.index(item) for item in _leads]
            data = data[leads_ind, :]
            # lead_units = np.vectorize(lambda s: s.lower())(header_info["df_leads"]["adc_units"].values)
        else:
            raise ValueError(
                f"backend `{backend.lower()}` not supported for loading data"
            )

        # ref. ISSUES 3, for multiplying `value_correction_factor`
        # data = data * self.value_correction_factor[tranche]

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if fs is not None and fs != self.fs[tranche]:
            data = resample_poly(data, fs, self.fs[tranche], axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    def load_ann(
        self, rec: Union[str, int], raw: bool = False, backend: str = "wfdb"
    ) -> Union[dict, str]:
        """

        load annotations (header) stored in the .hea files

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        raw: bool, default False,
            if True, the raw annotations without parsing will be returned
        backend: str, default "wfdb", case insensitive,
            if is "wfdb", `wfdb.rdheader` will be used to load the annotations;
            if is "naive", annotations will be parsed from the lines read from the header files

        Returns
        -------
        ann_dict, dict or str,
            the annotations with items: ref. `self.ann_items`

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
            raise ValueError(
                f"backend `{backend.lower()}` not supported for loading annotations"
            )
        return ann_dict

    def _load_ann_wfdb(self, rec: Union[str, int], header_data: List[str]) -> dict:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        header_data: list of str,
            list of lines read directly from a header file,
            complementary to data read using `wfdb.rdheader` if applicable,
            this data will be used, since `datetime` is not well parsed by `wfdb.rdheader`

        Returns
        -------
        ann_dict, dict,
            the annotations with items: ref. `self.ann_items`

        """
        if isinstance(rec, int):
            rec = self[rec]
        header_fp = self.get_header_filepath(rec, with_ext=False)
        header_reader = wfdb.rdheader(header_fp)
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
        ann_dict["datetime"] = datetime.strptime(
            " ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S"
        )
        try:  # see NOTE. 1.
            ann_dict["age"] = int(
                [line for line in header_reader.comments if "Age" in line][0].split(
                    ": "
                )[-1]
            )
        except Exception:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [
                line for line in header_reader.comments if "Sex" in line
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [
                line for line in header_reader.comments if "Rx" in line
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [
                line for line in header_reader.comments if "Hx" in line
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [
                line for line in header_reader.comments if "Sx" in line
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = (
            [line for line in header_reader.comments if "Dx" in line][0]
            .split(": ")[-1]
            .split(",")
        )
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(
            l_Dx
        )

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
        """

        load annotations (header) using raw data read directly from a header file

        Parameters
        ----------
        header_data: list of str,
            list of lines read directly from a header file

        Returns
        -------
        ann_dict, dict,
            the annotations with items: ref. `self.ann_items`

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
        ann_dict["datetime"] = datetime.strptime(
            " ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S"
        )
        try:  # see NOTE. 1.
            ann_dict["age"] = int(
                [line for line in header_data if line.startswith("#Age")][0].split(
                    ": "
                )[-1]
            )
        except Exception:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [line for line in header_data if line.startswith("#Sex")][
                0
            ].split(": ")[-1]
        except Exception:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [
                line for line in header_data if line.startswith("#Rx")
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [
                line for line in header_data if line.startswith("#Hx")
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [
                line for line in header_data if line.startswith("#Sx")
            ][0].split(": ")[-1]
        except Exception:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = (
            [line for line in header_data if line.startswith("#Dx")][0]
            .split(": ")[-1]
            .split(",")
        )
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(
            l_Dx
        )

        ann_dict["df_leads"] = self._parse_leads(header_data[1:13])

        return ann_dict

    def _parse_diagnosis(self, l_Dx: List[str]) -> Tuple[dict, dict]:
        """

        Parameters
        ----------
        l_Dx: list of str,
            raw information of diagnosis, read from a header file

        Returns
        -------
        diag_dict:, dict,
            diagnosis, including SNOMED CT Codes, fullnames and abbreviations of each diagnosis
        diag_scored_dict: dict,
            the scored items in `diag_dict`

        """
        diag_dict, diag_scored_dict = {}, {}
        try:
            diag_dict["diagnosis_code"] = [item for item in l_Dx]
            # selection = dx_mapping_all["SNOMED CT Code"].isin(diag_dict["diagnosis_code"])
            # diag_dict["diagnosis_abbr"] = dx_mapping_all[selection]["Abbreviation"].tolist()
            # diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
            diag_dict["diagnosis_abbr"] = [
                dx_mapping_all[dx_mapping_all["SNOMED CT Code"] == dc][
                    "Abbreviation"
                ].values[0]
                for dc in diag_dict["diagnosis_code"]
            ]
            diag_dict["diagnosis_fullname"] = [
                dx_mapping_all[dx_mapping_all["SNOMED CT Code"] == dc]["Dx"].values[0]
                for dc in diag_dict["diagnosis_code"]
            ]
            scored_indices = np.isin(
                diag_dict["diagnosis_code"], dx_mapping_scored["SNOMED CT Code"].values
            )
            diag_scored_dict["diagnosis_code"] = [
                item
                for idx, item in enumerate(diag_dict["diagnosis_code"])
                if scored_indices[idx]
            ]
            diag_scored_dict["diagnosis_abbr"] = [
                item
                for idx, item in enumerate(diag_dict["diagnosis_abbr"])
                if scored_indices[idx]
            ]
            diag_scored_dict["diagnosis_fullname"] = [
                item
                for idx, item in enumerate(diag_dict["diagnosis_fullname"])
                if scored_indices[idx]
            ]
        except Exception:  # the old version, the Dx"s are abbreviations
            diag_dict["diagnosis_abbr"] = diag_dict["diagnosis_code"]
            selection = dx_mapping_all["Abbreviation"].isin(diag_dict["diagnosis_abbr"])
            diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
        return diag_dict, diag_scored_dict

    def _parse_leads(self, l_leads_data: List[str]) -> pd.DataFrame:
        """

        Parameters
        ----------
        l_leads_data: list of str,
            raw information of each lead, read from a header file

        Returns
        -------
        df_leads: DataFrame,
            infomation of each leads in the format of DataFrame

        """
        df_leads = pd.read_csv(
            io.StringIO("\n".join(l_leads_data)), delim_whitespace=True, header=None
        )
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
        df_leads["byte_offset"] = df_leads["fmt+byte_offset"].apply(
            lambda s: s.split("+")[1]
        )
        df_leads["adc_gain"] = df_leads["adc_gain+units"].apply(
            lambda s: s.split("/")[0]
        )
        df_leads["adc_units"] = df_leads["adc_gain+units"].apply(
            lambda s: s.split("/")[1]
        )
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
        """

        read labels (diagnoses or arrhythmias) of a record

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        scored_only: bool, default True,
            only get the labels that are scored in the CINC2020 official phase
        fmt: str, default "a",
            the format of labels, one of the following (case insensitive):
            - "a", abbreviations
            - "f", full names
            - "s", SNOMED CT Code
        normalize: bool, default True,
            if True, the labels will be transformed into their equavalents,
            which are defined in `utils.utils_misc.cinc2020_aux_data.py`

        Returns
        -------
        labels, list,
            the list of labels

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
        """

        get the sampling frequency of a record

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        fs: real number,
            sampling frequency of the record `rec`
        """
        tranche = self._get_tranche(rec)
        fs = self.fs[tranche]
        return fs

    def get_subject_info(
        self, rec: Union[str, int], items: Optional[List[str]] = None
    ) -> dict:
        """

        read auxiliary information of a subject (a record) stored in the header files

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        items: list of str, optional,
            items of the subject's information (e.g. sex, age, etc.)

        Returns
        -------
        subject_info: dict,
            information about the subject, including
            "age", "sex", "medical_prescription", "history", "symptom_or_surgery",
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
        subject_info = [ann_dict[item] for item in info_items]

        return subject_info

    def save_challenge_predictions(
        self,
        rec: Union[str, int],
        output_dir: Union[str, Path],
        scores: List[Real],
        labels: List[int],
        classes: List[str],
    ) -> NoReturn:
        """NOT finished, NOT checked, need updating,

        TODO: update for the official phase

        Parameters
        ----------
        rec: str,
            name of the record
        output_dir: str or Path,
            directory to save the predictions
        scores: list of real,
            raw predictions
        labels: list of int,
            0 or 1, binary predictions
        classes: list of str,
            SNOMED CT Code of binary predictions

        """
        if isinstance(rec, int):
            rec = self[rec]
        new_file = f"{rec}.csv"
        output_file = Path(output_dir) / new_file

        # Include the filename as the recording number
        recording_string = f"#{rec}"
        class_string = ",".join(classes)
        label_string = ",".join(str(i) for i in labels)
        score_string = ",".join(str(i) for i in scores)

        output_file.write_text(
            "\n".join([recording_string, class_string, label_string, score_string, ""])
        )

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
    ) -> NoReturn:
        """to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data: ndarray, optional,
            (12-lead) ECG signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            annotations for `data`, with 2 items: "scored", "all",
            ignored if `data` is None
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
            print(f"better view: {url}")

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == "all":
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads
        # assert all([ld in self.all_leads for ld in _leads])
        assert set(_leads).issubset(self._all_leads_set)

        # lead_list = self.load_ann(rec)["df_leads"]["lead_name"].tolist()
        # lead_indices = [lead_list.index(ld) for ld in _leads]
        lead_indices = [self.all_leads.index(ld) for ld in _leads]
        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[
                lead_indices
            ]
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

        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        if waves:
            if waves.get("p_onsets", None) and waves.get("p_offsets", None):
                p_waves = [
                    [onset, offset]
                    for onset, offset in zip(waves["p_onsets"], waves["p_offsets"])
                ]
            elif waves.get("p_peaks", None):
                p_waves = [
                    [
                        max(0, p + ms2samples(PlotCfg.p_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            p + ms2samples(PlotCfg.p_offset, fs=self.get_fs(rec)),
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
                        max(0, q + ms2samples(PlotCfg.q_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            s + ms2samples(PlotCfg.s_offset, fs=self.get_fs(rec)),
                        ),
                    ]
                    for q, s in zip(waves["q_peaks"], waves["s_peaks"])
                ]
            elif waves.get("r_peaks", None):
                qrs = [
                    [
                        max(0, r + ms2samples(PlotCfg.qrs_radius, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            r + ms2samples(PlotCfg.qrs_radius, fs=self.get_fs(rec)),
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
                        max(0, t + ms2samples(PlotCfg.t_onset, fs=self.get_fs(rec))),
                        min(
                            _data.shape[1],
                            t + ms2samples(PlotCfg.t_offset, fs=self.get_fs(rec)),
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
            "t_waves": "pink",
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
        fig, axes = plt.subplots(
            nb_leads, 1, sharex=False, figsize=(fig_sz_w, np.sum(fig_sz_h))
        )
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
                axes[idx].grid(
                    which="major", linestyle="-", linewidth="0.4", color="red"
                )
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(
                    which="minor", linestyle=":", linewidth="0.2", color="gray"
                )
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot([], [], " ", label=f"labels_s - {','.join(diag_scored)}")
            axes[idx].plot([], [], " ", label=f"labels_a - {','.join(diag_all)}")
            axes[idx].plot(
                [], [], " ", label=f"tranche - {self.tranche_names[tranche]}"
            )
            axes[idx].plot([], [], " ", label=f"fs - {self.fs[tranche]}")
            for w in ["p_waves", "qrs", "t_waves"]:
                for itv in eval(w):
                    axes[idx].axvspan(
                        itv[0], itv[1], color=palette[w], alpha=plot_alpha
                    )
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

    def get_tranche_class_distribution(
        self, tranches: Sequence[str], scored_only: bool = True
    ) -> Dict[str, int]:
        """

        Parameters
        ----------
        tranches: sequence of str,
            tranche symbols (A-F)
        scored_only: bool, default True,
            only get class distributions that are scored in the CINC2020 official phase

        Returns
        -------
        distribution: dict,
            keys are abbrevations of the classes,
            values are appearance of corr. classes in the tranche.

        """
        tranche_names = [self.tranche_names[t] for t in tranches]
        df = dx_mapping_scored if scored_only else dx_mapping_all
        distribution = CFG()
        for _, row in df.iterrows():
            num = (row[[tranche_names]].values).sum()
            if num > 0:
                distribution[row["Abbreviation"]] = num
        return distribution

    @staticmethod
    def get_arrhythmia_knowledge(
        arrhythmias: Union[str, List[str]], **kwargs
    ) -> NoReturn:
        """

        knowledge about ECG features of specific arrhythmias,

        Parameters
        ----------
        arrhythmias: str, or list of str,
            the arrhythmia(s) to check, in abbreviations or in SNOMED CT Code

        """
        if isinstance(arrhythmias, str):
            d = [normalize_class(arrhythmias)]
        else:
            d = [normalize_class(c) for c in arrhythmias]
        # pp = pprint.PrettyPrinter(indent=4)
        # unsupported = [item for item in d if item not in dx_mapping_all["Abbreviation"]]
        unsupported = [
            item for item in d if item not in dx_mapping_scored["Abbreviation"].values
        ]
        assert (
            len(unsupported) == 0
        ), f"`{unsupported}` {'is' if len(unsupported)==1 else 'are'} not supported!"
        for idx, item in enumerate(d):
            # pp.pprint(eval(f"EAK.{item}"))
            print(dict_to_str(eval(f"EAK.{item}")))
            if idx < len(d) - 1:
                print("*" * 110)

    def load_resampled_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        siglen: Optional[int] = None,
    ) -> np.ndarray:
        """

        resample the data of `rec` to 500Hz,
        or load the resampled data in 500Hz, if the corr. data file already exists

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data_format: str, default "channel_first",
            format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        siglen: int, optional,
            signal length, units in number of samples,
            if set, signal with length longer will be sliced to the length of `siglen`
            used for example when preparing/doing model training

        Returns
        -------
        data: ndarray,
            the resampled (and perhaps sliced) signal data

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        if siglen is None:
            rec_fp = self.db_dirs[tranche] / f"{rec}_500Hz.npy"
        else:
            rec_fp = self.db_dirs[tranche] / f"{rec}_500Hz_siglen_{siglen}.npy"
        if not rec_fp.is_file():
            # print(f"corresponding file {rec_fp.name} does not exist")
            data = self.load_data(rec, data_format="channel_first", units="mV", fs=None)
            if self.fs[tranche] != 500:
                data = resample_poly(data, 500, self.fs[tranche], axis=1)
            if siglen is not None and data.shape[1] >= siglen:
                # slice_start = (data.shape[1] - siglen)//2
                # slice_end = slice_start + siglen
                # data = data[..., slice_start:slice_end]
                data = ensure_siglen(data, siglen=siglen, fmt="channel_first")
                np.save(rec_fp, data)
            elif siglen is None:
                np.save(rec_fp, data)
        else:
            # print(f"loading from local file...")
            data = np.load(rec_fp)
        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T
        return data

    def load_raw_data(self, rec: Union[str, int], backend: str = "scipy") -> np.ndarray:
        """

        load raw data from corresponding files with no further processing,
        in order to facilitate feeding data into the `run_12ECG_classifier` function

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        backend: str, default "scipy",
            the backend data reader, can also be "wfdb",
            note that "scipy" provides data in the format of "lead_first",
            while "wfdb" provides data in the format of "lead_last",

        Returns
        -------
        raw_data: ndarray,
            raw data (d_signal) loaded from corresponding data file,
            without subtracting baseline nor dividing ADC gain

        """
        if isinstance(rec, int):
            rec = self[rec]
        tranche = self._get_tranche(rec)
        if backend.lower() == "wfdb":
            rec_fp = self.get_data_filepath(rec, with_ext=False)
            wfdb_rec = wfdb.rdrecord(rec_fp, physical=False)
            raw_data = np.asarray(wfdb_rec.d_signal)
        elif backend.lower() == "scipy":
            rec_fp = self.get_data_filepath(rec, with_ext=True)
            raw_data = loadmat(rec_fp)["val"]
        return raw_data

    def _check_nan(self, tranches: Union[str, Sequence[str]]) -> NoReturn:
        """

        check if records from `tranches` has nan values

        accessing data using `p_signal` of `wfdb` would produce nan values,
        if exceptionally large values are encountered,
        this could help detect abnormal records as well

        Parameters
        ----------
        tranches: str or sequence of str,
            tranches to check

        """
        for t in tranches:
            for rec in self.all_records[t]:
                data = self.load_data(rec)
                if np.isnan(data).any():
                    print(f"record {rec} from tranche {t} has nan values")

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

    def download(self) -> NoReturn:
        """ """
        for url in self.url:
            http_get(url, self.db_dir_base / _stem(url), extract=True)

    def __len__(self) -> int:
        """
        number of records in the database

        """
        return len(self.__all_records)

    def __getitem__(self, index: int) -> str:
        """
        get the record name by index

        """
        return self.__all_records[index]


def compute_all_metrics(
    classes: List[str], truth: Sequence, binary_pred: Sequence, scalar_pred: Sequence
) -> Tuple[float]:
    """

    Parameters
    ----------
    classes: list of str,
        list of all the classes, in the format of abbrevations
    truth: sequence,
        ground truth array, of shape (n_records, n_classes), with values 0 or 1
    binary_pred: sequence,
        binary predictions, of shape (n_records, n_classes), with values 0 or 1
    scalar_pred: sequence,
        probability predictions, of shape (n_records, n_classes), with values within [0,1]

    Returns
    -------
    auroc: float,
        area under the receiver operating characteristic (ROC) curve
    auprc: float,
        area under the precision-recall curve
    accuracy: float,
        accuracy
    f_measure: float,
        f1 score
    f_beta_measure: float,
        f-beta score
    g_beta_measure: float,
        g-beta score
    challenge_metric: float,
        challenge metric, defined by a weight matrix

    """
    # normal_class = "426783006"
    normal_class = "NSR"
    # equivalent_classes = [["713427006", "59118001"], ["284470004", "63593006"], ["427172004", "17338001"]]
    weights = load_weights(classes=classes)

    _truth = np.array(truth)
    _binary_pred = np.array(binary_pred)
    _scalar_pred = np.array(scalar_pred)

    print("- AUROC and AUPRC...")
    auroc, auprc = compute_auc(_truth, _scalar_pred)

    print("- Accuracy...")
    accuracy = compute_accuracy(_truth, _binary_pred)

    print("- F-measure...")
    f_measure = compute_f_measure(_truth, _binary_pred)

    print("- F-beta and G-beta measures...")
    f_beta_measure, g_beta_measure = compute_beta_measures(_truth, _binary_pred, beta=2)

    print("- Challenge metric...")
    challenge_metric = compute_challenge_metric(
        weights, _truth, _binary_pred, classes, normal_class
    )

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


# Compute recording-wise accuracy.
def compute_accuracy(labels: np.ndarray, outputs: np.ndarray) -> float:
    """ """
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(
    labels: np.ndarray, outputs: np.ndarray, normalize: bool = False
) -> np.ndarray:
    """ """
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
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


# Compute macro F-measure.
def compute_f_measure(labels: np.ndarray, outputs: np.ndarray) -> float:
    """ """
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure


# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(
    labels: np.ndarray, outputs: np.ndarray, beta: Real
) -> Tuple[float, float]:
    """ """
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1 + beta**2) * tp + fp + beta**2 * fn:
            f_beta_measure[k] = float((1 + beta**2) * tp) / float(
                (1 + beta**2) * tp + fp + beta**2 * fn
            )
        else:
            f_beta_measure[k] = float("nan")
        if tp + fp + beta * fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta * fn)
        else:
            g_beta_measure[k] = float("nan")

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float]:
    """ """
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


# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(
    labels: np.ndarray, outputs: np.ndarray
) -> np.ndarray:
    """
    Compute a binary multi-class, multi-label confusion matrix,
    where the rows are the labels and the columns are the outputs.

    """
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(
            max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1)
        )
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization
    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(
    weights: np.ndarray,
    labels: np.ndarray,
    outputs: np.ndarray,
    classes: List[str],
    normal_class: str,
) -> float:
    """ """
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(
            correct_score - inactive_score
        )
    else:
        normalized_score = 0.0

    return normalized_score


# alias
compute_metrics = compute_challenge_metric
