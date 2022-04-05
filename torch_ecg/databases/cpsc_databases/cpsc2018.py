# -*- coding: utf-8 -*-
"""
"""

import io
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..aux_data.cinc2020_aux_data import dx_mapping_all, dx_mapping_scored
from ..base import DEFAULT_FIG_SIZE_PER_SEC, CPSCDataBase

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
    1. ECG arrythmia detection

    References
    ----------
    1. <a name="ref1"></a> http://2018.icbeb.org/#
    2. <a name="ref2"></a> https://physionetchallenges.github.io/2020/

    """

    def __init__(
        self,
        db_dir: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """finished, to be improved,

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
            db_name="cpsc2018",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.fs = 500
        self.spacing = 1000 / self.fs
        self.rec_ext = "mat"
        self.ann_ext = "hea"
        self._all_records = None
        self._ls_rec()

        self.nb_records = 6877
        self.all_leads = [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ]
        self.all_diagnosis = [
            "N",
            "AF",
            "I-AVB",
            "LBBB",
            "RBBB",
            "PAC",
            "PVC",
            "STD",
            "STE",
        ]
        self.all_diagnosis_original = sorted(
            [
                "Normal",
                "AF",
                "I-AVB",
                "LBBB",
                "RBBB",
                "PAC",
                "PVC",
                "STD",
                "STE",
            ]
        )
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
        """ """
        self._all_records = [
            item.with_suffix("").name for item in self.db_dir.glob(f"*.{self.rec_ext}")
        ]

    def get_subject_id(self, rec: Union[int, str]) -> int:
        """not finished,

        Parameters
        ----------
        rec: int or str,
            name or index of the record

        Returns
        -------
        pid: int,
            the `subject_id` corr. to `rec`

        """
        raise NotImplementedError

    def load_data(
        self,
        rec: Union[int, str],
        data_format="channel_first",
        units: str = "mV",
    ) -> np.ndarray:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        data_format: str, default "channel_first",
            format of the ECG data, "channels_last" or "channels_first" (original)
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        if isinstance(rec, int):
            rec = self[rec]
        rec_fp = self.db_dir / f"{rec}.{self.rec_ext}"
        data = loadmat(str(rec_fp))
        data = np.asarray(data["val"], dtype=np.float64)
        if data_format == "channels_last":
            data = data.T

        if units.lower() == "mv" and self._auto_infer_units(data) != "mV":
            data /= 1000
        elif (
            units.lower()
            in [
                "uv",
                "μv",
            ]
            and self._auto_infer_units(data) != "μV"
        ):
            data *= 1000

        return data

    def load_ann(self, rec: Union[int, str], keep_original: bool = True) -> dict:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        keep_original: bool, default True,
            keep the original annotations or not,
            mainly concerning "N" and "Normal" ("SNR" for the newer version)

        Returns
        -------
        ann_dict, dict,
            the annotations with items: ref. self.ann_items

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann_fp = self.db_dir / f"{rec}.{self.ann_ext}"
        header_data = ann_fp.read_text().splitlines()

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
        # if not keep_original:
        #     for idx, d in enumerate(ann_dict["diagnosis_abbr"]):
        #         if d in ["Normal", "NSR"]:
        #             ann_dict["diagnosis_abbr"] = ["N"]
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

    def get_labels(
        self, rec: Union[int, str], keep_original: bool = False
    ) -> List[str]:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        keep_original: bool, default False,
            keep the original annotations or not,
            mainly concerning "N" and "Normal"

        Returns
        -------
        labels, list,
            the list of labels (abbr. diagnosis)

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann_dict = self.load_ann(rec, keep_original=keep_original)
        labels = ann_dict["diagnosis"]
        return labels

    def get_diagnosis(self, rec: Union[int, str], full_name: bool = True) -> List[str]:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        full_name: bool, default True,
            full name of the diagnosis or short name of it (ref. self.diagnosis_abbr_to_full)

        Returns
        -------
        diagonosis, list,
            the list of (full) diagnosis

        """
        if isinstance(rec, int):
            rec = self[rec]
        diagonosis = self.get_labels(rec)
        if full_name:
            diagonosis = diagonosis["diagnosis_fullname"]
        else:
            diagonosis = diagonosis["diagnosis_abbr"]
        return diagonosis

    def get_subject_info(
        self, rec: Union[int, str], items: Optional[List[str]] = None
    ) -> dict:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        items: list of str, optional,
            items of the subject information (e.g. sex, age, etc.)

        Returns
        -------
        subject_info, dict,

        """
        if isinstance(rec, int):
            rec = self[rec]
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

    def save_challenge_predictions(
        self,
        rec: Union[int, str],
        output_dir: Union[str, Path],
        scores: List[Real],
        labels: List[int],
        classes: List[str],
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        output_dir: str or Path,
            directory to save the predictions
        scores: list of real,
            ...
        labels: list of int,
            0 or 1
        classes: list of str,
            ...

        """
        if isinstance(rec, int):
            rec = self[rec]
        recording = rec
        new_file = recording + ".csv"
        output_file = Path(output_dir) / new_file

        # Include the filename as the recording number
        recording_string = f"#{recording}"
        class_string = ",".join(classes)
        label_string = ",".join(str(i) for i in labels)
        score_string = ",".join(str(i) for i in scores)

        output_file.write_text(
            "\n".join([recording_string, class_string, label_string, score_string, ""])
        )

    def plot(
        self,
        rec: Union[int, str],
        ticks_granularity: int = 0,
        leads: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: int or str,
            name or index of the record
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        kwargs: auxilliary key word arguments

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        if leads is None or leads == "all":
            leads = self.all_leads
        assert all([ld in self.all_leads for ld in leads])

        lead_list = self.load_ann(rec)["df_leads"]["lead_name"].tolist()
        lead_indices = [lead_list.index(ld) for ld in leads]
        data = self.load_data(rec, data_format="channel_first", units="μV")[
            lead_indices
        ]
        y_ranges = np.max(np.abs(data), axis=1) + 100

        diag = self.get_diagnosis(rec, full_name=False)

        nb_leads = len(leads)

        t = np.arange(data.shape[1]) / self.fs
        duration = len(t) / self.fs
        fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * duration))
        fig_sz_h = 6 * y_ranges / 1500
        fig, axes = plt.subplots(
            nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h))
        )
        for idx in range(nb_leads):
            axes[idx].plot(
                t,
                data[idx],
                color="black",
                linewidth="2.0",
                label=f"lead - {leads[idx]}",
            )
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(
                    which="major", linestyle="-", linewidth="0.5", color="red"
                )
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(
                    which="minor", linestyle=":", linewidth="0.5", color="black"
                )
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

    @property
    def url(self) -> List[str]:
        return [
            "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip",
            "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip",
            "http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip",
        ]


def compute_metrics():
    """ """
    raise NotImplementedError
