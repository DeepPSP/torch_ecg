# -*- coding: utf-8 -*-
"""
"""

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Sequence, Dict, List, Union

import numpy as np
import pandas as pd
import h5py

from ...cfg import DEFAULTS
from ...utils.download import http_get
from ...utils.misc import get_record_list_recursive3, ms2samples, add_docstring
from ...utils import EAK
from ..base import DEFAULT_FIG_SIZE_PER_SEC, _DataBase, DataBaseInfo, _PlotCfg


__all__ = [
    "SPH",
]


_SPH_INFO = DataBaseInfo(
    title="""
    Shandong Provincial Hospital Database
    """,
    about=r"""
    1. contains 25770 ECG records from 24666 patients (55.36% male and 44.64% female), with between 10 and 60 seconds
    2. sampling frequency is 500 Hz
    3. records were acquired from Shandong Provincial Hospital (SPH) between 2019/08 and 2020/08
    4. diagnostic statements of all ECG records are in full compliance with the AHA/ACC/HRS recommendations, consisting of 44 primary statements and 15 modifiers
    5. 46.04% records in the dataset contain ECG abnormalities, and 14.45% records have multiple diagnostic statements
    6. (IMPORTANT) noises caused by the power line interference, baseline wander, and muscle contraction have been removed by the machine
    7. (Label production) The ECG analysis system automatically calculate nine ECG features for reference, which include heart rate, P wave duration, P-R interval, QRS duration, QT interval, corrected QT (QTc) interval, QRS axis, the amplitude of the R wave in lead V5 (RV5), and the amplitude of the S wave in lead V1 (SV1). A cardiologist made the final diagnosis in consideration of the patient health record.
    """,
    usage=[
        "ECG arrhythmia detection",
    ],
    references=[
        "https://www.nature.com/articles/s41597-022-01403-5",
        "Liu, H., Chen, D., Chen, D. et al. A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements. Sci Data 9, 272 (2022). https://doi.org/10.1038/s41597-022-01403-5",
        "Mason, J. W., Hancock, E. W. & Gettes, L. S. Recommendations for the standardization and interpretation of the electrocardiogram. Circulation 115, 1325–1332, https://doi.org/10.1161/CIRCULATIONAHA.106.180201 (2007).",
        "https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1",
    ],
    doi=[
        "10.1038/s41597-022-01403-5",
        "10.6084/m9.figshare.c.5779802.v1",
    ],
)


@add_docstring(_SPH_INFO.format_database_docstring())
class SPH(_DataBase):
    """ """

    __name__ = "SPH"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
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
            db_name="SPH",
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

    def _ls_rec(self) -> None:
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

        if (self.db_dir / "code.csv").is_file():
            self._df_code = pd.read_csv(self.db_dir / "code.csv").astype(str)
        if (self.db_dir / "metadata.csv").is_file():
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
        sid = self._df_metadata.loc[self._df_metadata["ECG_ID"] == rec][
            "Patient_ID"
        ].iloc[0]
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
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"

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

        with h5py.File(self.get_absolute_path(rec, extension=self.data_ext), "r") as f:
            data = f["ecg"][_leads].astype(DEFAULTS.DTYPE.NP)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    def load_ann(self, rec: Union[str, int], ann_format: str = "c") -> List[str]:
        """
        load annotation from the metadata file

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        ann_format: str, default "a",
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
            for lb in self._df_metadata[self._df_metadata["ECG_ID"] == rec]["AHA_Code"]
            .iloc[0]
            .split(";")
        ]
        if ann_format.lower() == "c":
            pass  # default format
        elif ann_format.lower() == "f":
            labels = [
                self._df_code[self._df_code["Code"] == lb]["Description"].iloc[0]
                for lb in labels
            ]
        elif ann_format.lower() == "a":
            raise NotImplementedError("Abbreviations are not supported yet")
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
        subject_info = {item: row[item.capitalize()] for item in info_items}

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
        age = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["Age"].iloc[0]
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
        sex = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["Sex"].iloc[0]
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
        siglen = self._df_metadata[self._df_metadata["ECG_ID"] == rec]["N"].iloc[0]
        return siglen

    @property
    def url(self) -> Dict[str, str]:
        return {
            "metadata.csv": "https://springernature.figshare.com/ndownloader/files/34793152",
            "code.csv": "https://springernature.figshare.com/ndownloader/files/32630954",
            "records.tar": "https://springernature.figshare.com/ndownloader/files/32630684",
        }

    def download(self, files: Optional[Union[str, Sequence[str]]]) -> None:
        """
        download the database from the figshare website
        """
        if files is None:
            files = self.url.keys()
        if isinstance(files, str):
            files = [files]
        assert set(files).issubset(
            self.url
        ), f"`files` should be a subset of {list(self.url)}"
        for filename in files:
            url = self.url[filename]
            if not (self.db_dir / filename).is_file():
                http_get(url, self.db_dir, filename=filename)
        self._ls_rec()

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[Sequence[str]] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, List[str]]] = None,
        same_range: bool = False,
        waves: Optional[Dict[str, Sequence[int]]] = None,
        **kwargs: Any,
    ) -> None:
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
        ann: sequence of str, optional,
            annotations for `data`,
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

        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == "all":
            _leads = self.all_leads
        elif isinstance(leads, str):
            _leads = [leads]
        else:
            _leads = leads

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
                qrs = [
                    [onset, offset]
                    for onset, offset in zip(waves["q_onsets"], waves["s_offsets"])
                ]
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
                        max(
                            0, r + ms2samples(_PlotCfg.qrs_radius, fs=self.get_fs(rec))
                        ),
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
                t_waves = [
                    [onset, offset]
                    for onset, offset in zip(waves["t_onsets"], waves["t_offsets"])
                ]
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
            "t_waves": "pink",
        }
        plot_alpha = 0.4

        if ann is None or data is None:
            ann = self.load_ann(rec, ann_format="f")

        nb_leads = len(_leads)

        seg_len = self.fs * 25  # 25 seconds
        nb_segs = _data.shape[1] // seg_len

        t = np.arange(_data.shape[1]) / self.fs
        duration = len(t) / self.fs
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
            axes[idx].plot([], [], " ", label=f"labels - {','.join(ann)}")
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

    @property
    def database_info(self) -> DataBaseInfo:
        return _SPH_INFO
