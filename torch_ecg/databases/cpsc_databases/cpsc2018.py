# -*- coding: utf-8 -*-
"""
"""
import io
import os
import glob
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Tuple, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
from scipy.io import loadmat

from ..aux_data.cinc2020_aux_data import (
    dx_mapping_all, dx_mapping_scored, dx_mapping_unscored,
    normalize_class, abbr_to_snomed_ct_code,
)
from ..base import CPSCDataBase, DEFAULT_FIG_SIZE_PER_SEC


__all__ = [
    "CPSC2018",
    "compute_metrics",
]


class CPSC2018(CPSCDataBase):
    """

    The China Physiological Signal Challenge 2018:
    Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs

    ABOUT CPSC2018
    --------------
    1. training set contains 6,877 (female: 3178; male: 3699) 12 leads ECG recordings lasting from 6 s to just 60 s
    2. ECG recordings were sampled as 500 Hz
    3. the training data can be downloaded using links in Ref.[1], but the link in Ref.[2] is recommended. File structure will be assumed to follow Ref.[2]
    4. the training data are in the `channel first` format
    5. types of abnormal rhythm/morphology + normal in the training set:
            name                                    abbr.       number of records
        (0) Normal                                  N           918
        (1) Atrial fibrillation                     AF          1098
        (2) First-degree atrioventricular block     I-AVB       704
        (3) Left bundle brunch block                LBBB        207
        (4) Right bundle brunch block               RBBB        1695
        (5) Premature atrial contraction            PAC         556
        (6) Premature ventricular contraction       PVC         672
        (7) ST-segment depression                   STD         825
        (8) ST-segment elevated                     STE         202
    6. ordering of the leads in the data of all the records are
        ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    7. meanings in the .hea files: to write
    8. knowledge about the abnormal rhythms: ref. cls.get_disease_knowledge

    Update
    ------
    CINC2020 (ref. [2]) released totally 3453 unused training data of CPSC2018, whose filenames start with "Q".
    These file names are not "continuous". The last record is "Q3581"

    NOTE
    ----
    1. Ages of records A0608, A1549, A1876, A2299, A5990 are "NaN"

    ISSUES
    ------

    Usage
    -----
    1. ecg arrythmia detection

    References
    ----------
    [1] http://2018.icbeb.org/#
    [2] https://physionetchallenges.github.io/2020/
    """

    def __init__(self,
                 db_dir:str,
                 working_dir:Optional[str]=None,
                 verbose:int=2,
                 **kwargs:Any) -> NoReturn:
        """ finished, to be improved,

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
        super().__init__(db_name="CPSC2018", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)

        self.fs = 500
        self.spacing = 1000 / self.fs
        self.rec_ext = "mat"
        self.ann_ext = "hea"
        self._all_records = None
        self._ls_rec()

        self.nb_records = 6877
        self.all_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]
        self.all_diagnosis = ["N", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE",]
        self.all_diagnosis_original = sorted(["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE",])
        self.diagnosis_abbr_to_full = {
            "N": "Normal",
            "AF": "Atrial fibrillation",
            "I-AVB": "First-degree atrioventricular block",
            "LBBB": "Left bundle brunch block",
            "RBBB": "Right bundle brunch block",
            "PAC": "Premature atrial contraction",
            "PVC": "Premature ventricular contraction",
            "STD": "ST-segment depression",
            "STE": "ST-segment elevated",
        }

        self.ann_items = [
            "rec_name",
            "nb_leads",
            "fs",
            "nb_samples",
            "datetime",
            "age",
            "sex",
            "diagnosis",
            "medical_prescription",
            "history",
            "symptom_or_surgery",
            "df_leads",
        ]

    def _ls_rec(self) -> NoReturn:
        """
        """
        self._all_records = [
            os.path.splitext(os.path.basename(item))[0] \
                for item in glob.glob(os.path.join(self.db_dir, f"*.{self.rec_ext}"))
        ]

    def get_subject_id(self, rec_no:Union[int,str]) -> int:
        """ not finished,

        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record

        Returns
        -------
        pid: int,
            the `subject_id` corr. to `rec_no`
        """
        raise NotImplementedError

    def load_data(self, rec_no:Union[int,str], data_format="channel_first", units:str="mV",) -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        data_format: str, default "channel_first",
            format of the ecg data, "channels_last" or "channels_first" (original)
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        
        Returns
        -------
        data: ndarray,
            the ecg data
        """
        if isinstance(rec_no, int):
            assert rec_no in range(1, self.nb_records+1), f"rec_no should be in range(1,{self.nb_records+1})"
            rec_no = f"A{rec_no:04d}"
        rec_fp = os.path.join(self.db_dir, f"{rec_no}.{self.rec_ext}")
        data = loadmat(rec_fp)
        data = np.asarray(data["val"], dtype=np.float64)
        if data_format == "channels_last":
            data = data.T

        if units.lower() == "mv" and self._auto_infer_units(data) != "mV":
            data /= 1000
        elif units.lower() in ["uv", "μv",] and self._auto_infer_units(data) != "μV":
            data *= 1000
        
        return data

    def load_ann(self, rec_no:Union[int,str], keep_original:bool=True) -> dict:
        """ finished, checked,
        
        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        keep_original: bool, default True,
            keep the original annotations or not,
            mainly concerning "N" and "Normal" ("SNR" for the newer version)
        
        Returns
        -------
        ann_dict, dict,
            the annotations with items: ref. self.ann_items
        """
        if isinstance(rec_no, int):
            assert rec_no in range(1, self.nb_records+1), f"rec_no should be in range(1, {self.nb_records+1})"
            rec_no = f"A{rec_no:04d}"
        ann_fp = os.path.join(self.db_dir, f"{rec_no}.{self.ann_ext}")
        with open(ann_fp, "r") as f:
            header_data = f.read().splitlines()

        ann_dict = {}
        ann_dict["rec_name"], ann_dict["nb_leads"], ann_dict["fs"], ann_dict["nb_samples"], ann_dict["datetime"], daytime \
            = header_data[0].split(" ")

        ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
        ann_dict["fs"] = int(ann_dict["fs"])
        ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
        ann_dict["datetime"] = datetime.strptime(" ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S")
        try: # see NOTE. 1.
            ann_dict["age"] = int([l for l in header_data if l.startswith("#Age")][0].split(": ")[-1])
        except:
            ann_dict["age"] = np.nan
        try:
            ann_dict["sex"] = [l for l in header_data if l.startswith("#Sex")][0].split(": ")[-1]
        except:
            ann_dict["sex"] = "Unknown"
        try:
            ann_dict["medical_prescription"] = [l for l in header_data if l.startswith("#Rx")][0].split(": ")[-1]
        except:
            ann_dict["medical_prescription"] = "Unknown"
        try:
            ann_dict["history"] = [l for l in header_data if l.startswith("#Hx")][0].split(": ")[-1]
        except:
            ann_dict["history"] = "Unknown"
        try:
            ann_dict["symptom_or_surgery"] = [l for l in header_data if l.startswith("#Sx")][0].split(": ")[-1]
        except:
            ann_dict["symptom_or_surgery"] = "Unknown"

        l_Dx = [l for l in header_data if l.startswith("#Dx")][0].split(": ")[-1].split(",")
        ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

        ann_dict["df_leads"] = self._parse_leads(header_data[1:13])

        return ann_dict

    def _parse_diagnosis(self, l_Dx:List[str]) -> Tuple[dict, dict]:
        """ finished, checked,

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
            diag_dict["diagnosis_abbr"] = \
                [ dx_mapping_all[dx_mapping_all["SNOMED CT Code"]==dc]["Abbreviation"].values[0] \
                    for dc in diag_dict["diagnosis_code"] ]
            diag_dict["diagnosis_fullname"] = \
                [ dx_mapping_all[dx_mapping_all["SNOMED CT Code"]==dc]["Dx"].values[0] \
                    for dc in diag_dict["diagnosis_code"] ]
            scored_indices = np.isin(diag_dict["diagnosis_code"], dx_mapping_scored["SNOMED CT Code"].values)
            diag_scored_dict["diagnosis_code"] = \
                [ item for idx, item in enumerate(diag_dict["diagnosis_code"]) \
                    if scored_indices[idx] ]
            diag_scored_dict["diagnosis_abbr"] = \
                [ item for idx, item in enumerate(diag_dict["diagnosis_abbr"]) \
                    if scored_indices[idx] ]
            diag_scored_dict["diagnosis_fullname"] = \
                [ item for idx, item in enumerate(diag_dict["diagnosis_fullname"]) \
                    if scored_indices[idx] ]
        except:  # the old version, the Dx"s are abbreviations
            diag_dict["diagnosis_abbr"] = diag_dict["diagnosis_code"]
            selection = dx_mapping_all["Abbreviation"].isin(diag_dict["diagnosis_abbr"])
            diag_dict["diagnosis_fullname"] = dx_mapping_all[selection]["Dx"].tolist()
        # if not keep_original:
        #     for idx, d in enumerate(ann_dict["diagnosis_abbr"]):
        #         if d in ["Normal", "NSR"]:
        #             ann_dict["diagnosis_abbr"] = ["N"]
        return diag_dict, diag_scored_dict

    def _parse_leads(self, l_leads_data:List[str]) -> pd.DataFrame:
        """ finished, checked,

        Parameters
        ----------
        l_leads_data: list of str,
            raw information of each lead, read from a header file

        Returns
        -------
        df_leads: DataFrame,
            infomation of each leads in the format of DataFrame
        """
        df_leads = pd.read_csv(io.StringIO("\n".join(l_leads_data)), delim_whitespace=True, header=None)
        df_leads.columns = [
            "filename", "fmt+byte_offset", "adc_gain+units", "adc_res", "adc_zero",
            "init_value", "checksum", "block_size", "lead_name",
        ]
        df_leads["fmt"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[0])
        df_leads["byte_offset"] = df_leads["fmt+byte_offset"].apply(lambda s: s.split("+")[1])
        df_leads["adc_gain"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[0])
        df_leads["adc_units"] = df_leads["adc_gain+units"].apply(lambda s: s.split("/")[1])
        for k in ["byte_offset", "adc_gain", "adc_res", "adc_zero", "init_value", "checksum",]:
            df_leads[k] = df_leads[k].apply(lambda s: int(s))
        df_leads["baseline"] = df_leads["adc_zero"]
        df_leads = df_leads[[
            "filename", "fmt", "byte_offset", "adc_gain", "adc_units", "adc_res", "adc_zero", "baseline",
            "init_value", "checksum", "block_size", "lead_name",
        ]]
        df_leads.index = df_leads["lead_name"]
        df_leads.index.name = None
        return df_leads

    def get_labels(self, rec_no:Union[int,str], keep_original:bool=False) -> List[str]:
        """ finished, checked,
        
        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        keep_original: bool, default False,
            keep the original annotations or not,
            mainly concerning "N" and "Normal"
        
        Returns
        -------
        labels, list,
            the list of labels (abbr. diagnosis)
        """
        ann_dict = self.load_ann(rec_no, keep_original=keep_original)
        labels = ann_dict["diagnosis"]
        return labels

    def get_diagnosis(self, rec_no:Union[int,str], full_name:bool=True) -> List[str]:
        """ finished, checked,
        
        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        full_name: bool, default True,
            full name of the diagnosis or short name of it (ref. self.diagnosis_abbr_to_full)
        
        Returns
        -------
        diagonosis, list,
            the list of (full) diagnosis
        """
        diagonosis = self.get_labels(rec_no)
        if full_name:
            diagonosis = diagonosis["diagnosis_fullname"]
        else:
            diagonosis = diagonosis["diagnosis_abbr"]
        return diagonosis

    def get_subject_info(self, rec_no:Union[int,str], items:Optional[List[str]]=None) -> dict:
        """ finished, checked,

        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        items: list of str, optional,
            items of the subject information (e.g. sex, age, etc.)
        
        Returns
        -------
        subject_info, dict,
        """
        if items is None or len(items) == 0:
            info_items = [
                "age", "sex", "medical_prescription", "history", "symptom_or_surgery",
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec_no)
        subject_info = {item: ann_dict[item] for item in info_items}

        return subject_info

    def save_challenge_predictions(self, rec_no:Union[int,str], output_dir:str, scores:List[Real], labels:List[int], classes:List[str]) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        output_dir: str,
            directory to save the predictions
        scores: list of real,
            ...
        labels: list of int,
            0 or 1
        classes: list of str,
            ...
        """
        if isinstance(rec_no, str):
            rec_no = int(rec_no[1:])
        assert rec_no in range(1, self.nb_records+1), f"rec_no should be in range(1, {self.nb_records+1})"
        recording = self._all_records[rec_no]
        new_file = recording + ".csv"
        output_file = os.path.join(output_dir, new_file)

        # Include the filename as the recording number
        recording_string = f"#{recording}"
        class_string = ",".join(classes)
        label_string = ",".join(str(i) for i in labels)
        score_string = ",".join(str(i) for i in scores)

        with open(output_file, "w") as f:
            # f.write(recording_string + "\n" + class_string + "\n" + label_string + "\n" + score_string + "\n")
            f.write("\n".join([recording_string, class_string, label_string, score_string, ""]))

    def plot(self, rec_no:Union[int,str], ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        rec_no: int or str,
            number of the record, NOTE that rec_no starts from 1; or name of the record,
            int only supported for the original CPSC2018 dataset
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        kwargs: auxilliary key word arguments
        """
        if isinstance(rec_no, str):
            rec_no = int(rec_no[1:])
        assert rec_no in range(1, self.nb_records+1), f"rec_no should be in range(1, {self.nb_records+1})"
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        if leads is None or leads == "all":
            leads = self.all_leads
        assert all([l in self.all_leads for l in leads])

        lead_list = self.load_ann(rec_no)["df_leads"]["lead_name"].tolist()
        lead_indices = [lead_list.index(l) for l in leads]
        data = self.load_data(rec_no, data_format="channel_first", units="μV")[lead_indices]
        y_ranges = np.max(np.abs(data), axis=1) + 100

        diag = self.get_diagnosis(rec_no, full_name=False)

        nb_leads = len(leads)

        t = np.arange(data.shape[1]) / self.fs
        duration = len(t) / self.fs
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * y_ranges / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        for idx in range(nb_leads):
            axes[idx].plot(t, data[idx], color="black", linewidth="2.0", label=f"lead - {leads[idx]}")
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.5", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.5", color="black")
            axes[idx].plot([], [], " ", label=f"labels - {','.join(diag)}")
            axes[idx].legend(loc="upper left")
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(-y_ranges[idx], y_ranges[idx])
            axes[idx].set_xlabel("Time [s]")
            axes[idx].set_ylabel("Voltage [μV]")
        plt.subplots_adjust(hspace=0.2)
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()


def compute_metrics():
    """
    """
    raise NotImplementedError
