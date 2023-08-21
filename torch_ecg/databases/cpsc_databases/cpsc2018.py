# -*- coding: utf-8 -*-

import re
import warnings
from numbers import Real
from pathlib import Path
from typing import Any, List, Optional, Union, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ...cfg import DEFAULTS
from ...utils.misc import get_record_list_recursive, add_docstring
from ...utils.download import http_get
from ..base import DEFAULT_FIG_SIZE_PER_SEC, CPSCDataBase, DataBaseInfo


__all__ = [
    "CPSC2018",
    "compute_metrics",
]


_CPSC2018_INFO = DataBaseInfo(
    title="""
    The China Physiological Signal Challenge 2018:
    Automatic identification of the rhythm/morphology abnormalities in 12-lead ECGs.
    """,
    about="""
    1. training set contains 6,877 (female: 3178; male: 3699) 12 leads ECG recordings lasting from 6 s to just 60 s.
    2. ECG recordings were sampled as 500 Hz.
    3. the training data can be downloaded using links in [1]_, but the link in [2]_ is recommended. File structure will be assumed to follow [2]_.
    4. the training data are in the ``channel first`` format.
    5. types of abnormal rhythm/morphology + normal in the training set:

        +-----+-------------------------------------+-------+-------------------+
        | No. |   name                              | abbr. | number of records |
        +=====+=====================================+=======+===================+
        | 0   | Normal                              | N     | 918               |
        +-----+-------------------------------------+-------+-------------------+
        | 1   | Atrial fibrillation                 | AF    | 1098              |
        +-----+-------------------------------------+-------+-------------------+
        | 2   | First-degree atrioventricular block | I-AVB | 704               |
        +-----+-------------------------------------+-------+-------------------+
        | 3   | Left bundle brunch block            | LBBB  | 207               |
        +-----+-------------------------------------+-------+-------------------+
        | 4   | Right bundle brunch block           | RBBB  | 1695              |
        +-----+-------------------------------------+-------+-------------------+
        | 5   | Premature atrial contraction        | PAC   | 556               |
        +-----+-------------------------------------+-------+-------------------+
        | 6   | Premature ventricular contraction   | PVC   | 672               |
        +-----+-------------------------------------+-------+-------------------+
        | 7   | ST-segment depression               | STD   | 825               |
        +-----+-------------------------------------+-------+-------------------+
        | 8   | ST-segment elevated                 | STE   | 202               |
        +-----+-------------------------------------+-------+-------------------+

    6. ordering of the leads in the data of all the records are

        .. code-block:: python

            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    7. meanings in the .hea files: **to write**
    8. knowledge about the abnormal rhythms: ref. :meth:`get_disease_knowledge`.
    9. Challenge official website [1]_, see also [2]_.
    """,
    note="""
    1. Ages of records A0608, A1549, A1876, A2299, A5990 are "NaN".
    2. CINC2020 (ref. [2]_) released totally 3453 unused training data of CPSC2018, whose filenames start with "Q".
       These file names are not "continuous". The last record is "Q3581".
    """,
    usage=[
        "ECG arrythmia detection",
    ],
    references=[
        "http://2018.icbeb.org/",
        "https://physionetchallenges.github.io/2020/",
    ],
    doi="10.1166/jmihi.2018.2442",
)


@add_docstring(_CPSC2018_INFO.format_database_docstring(), mode="prepend")
class CPSC2018(CPSCDataBase):
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

    __name__ = "CPSC2018"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
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
        self.ann_ext = "mat"  # the same file as the record

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
        self.diagnosis_num_to_abbr = {
            1: "N",
            2: "AF",
            3: "I-AVB",
            4: "LBBB",
            5: "RBBB",
            6: "PAC",
            7: "PVC",
            8: "STD",
            9: "STE",
        }

        self._all_records = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        self._df_records = pd.DataFrame()
        self._df_records["path"] = get_record_list_recursive(
            self.db_dir, self.rec_ext, relative=False
        )
        if self._subsample is not None:
            size = min(
                len(self._df_records),
                max(1, int(round(self._subsample * len(self._df_records)))),
            )
            self._df_records = self._df_records.sample(
                n=size, random_state=DEFAULTS.SEED, replace=False
            )
        self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.name)
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()

        # find and load the annotation csv file
        ann_file = list(self.db_dir.rglob("REFERENCE.csv"))
        if len(ann_file) != 1:
            warnings.warn(
                "Annotation file not found. Please call method `_download_labels`, "
                "and call method `_ls_rec` again.",
                RuntimeWarning,
            )
            for c in ["labels_n", "labels_a", "labels_f"]:
                self._df_records.at[self._df_records.index, c] = None
        else:
            df_ann = pd.read_csv(ann_file[0])
            label_mat = df_ann[df_ann.columns[1:]].values
            mask = np.isnan(label_mat)
            label_mat = [
                row[~mask[idx]].astype(int).tolist()
                for idx, row in enumerate(label_mat)
            ]
            df_ann.loc[df_ann.index, "labels_n"] = df_ann.apply(
                lambda row: label_mat[row.name], axis=1
            )
            df_ann.loc[df_ann.index, "labels_a"] = df_ann.apply(
                lambda row: [self.diagnosis_num_to_abbr[i] for i in row.labels_n],
                axis=1,
            )
            df_ann.loc[df_ann.index, "labels_f"] = df_ann.apply(
                lambda row: [self.diagnosis_abbr_to_full[i] for i in row.labels_a],
                axis=1,
            )
            df_ann = df_ann[[df_ann.columns[0], "labels_n", "labels_a", "labels_f"]]
            df_ann.columns = ["record", "labels_n", "labels_a", "labels_f"]
            df_ann.set_index("record", inplace=True)
            # merge `df_ann` and `self._df_records`
            self._df_records = self._df_records.merge(
                df_ann, how="left", left_index=True, right_index=True
            )

    def get_subject_id(self, rec: Union[int, str]) -> int:
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
        s2d = {
            "A": "11",
            "B": "12",
        }
        prefix = "".join(re.findall(r"[A-Z]", rec))
        n = rec.replace(prefix, "")
        sid = int(f"{s2d[prefix]}{'0'*(8-len(n))}{n}")
        return sid

    def load_data(
        self,
        rec: Union[int, str],
        leads: Optional[Union[str, int, Sequence[Union[str, int]]]] = None,
        data_format="channel_first",
        units: str = "mV",
        return_fs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load the ECG data of a record.

        Parameters
        ----------
        rec : int or str
            Record name or index of the record in :attr:`all_records`.
        leads : str or int or Sequence[str] or Sequence[int], optional
            The leads to load,
            None or "all" for all leads.
        data_format : str, default "channel_first"
            Format of the ECG data, "channel_last" or "channel_first" (original)
        units : str, default "mV"
            Units of the output signal, can also be "μV" (with an alias "uV"),
            case insensitive.
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
            rec = self[rec]
        rec_fp = self.get_absolute_path(rec, self.rec_ext)
        _leads = self._normalize_leads(leads, numeric=True)
        allowed_data_format = [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
        ]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"

        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"

        data = loadmat(str(rec_fp))
        data = np.asarray(data["ECG"]["data"][0, 0], dtype=DEFAULTS.DTYPE.NP)[_leads, :]
        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T
        if units.lower() == "mv" and self._auto_infer_units(data) != "mV":
            data /= 1000
        elif (
            units.lower() in ["uv", "μv", "muv"]
            and self._auto_infer_units(data) != "μV"
        ):
            data *= 1000

        if return_fs:
            return data, self.fs
        return data

    def load_ann(self, rec: Union[str, int], ann_format: str = "n") -> List[str]:
        """Load labels (diagnoses or arrhythmias) of a record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`all_records`.
        ann_format : str, default "n"
            Format of labels, one of the following (case insensitive):

                - "a", abbreviations
                - "f", full names
                - "n", numeric codes

        Returns
        -------
        labels : List[str]
            The list of labels.

        """
        if isinstance(rec, int):
            rec = self[rec]
        try:
            col = {
                "a": "labels_a",
                "f": "labels_f",
                "n": "labels_n",
            }[ann_format.lower()]
        except KeyError:
            raise ValueError(
                f"`ann_format` should be one of `['a', 'f', 'n']`, but got `{ann_format}`"
            )
        labels = self._df_records.loc[rec, col]
        return labels

    @add_docstring(load_ann.__doc__)
    def get_labels(self, rec: Union[str, int], ann_format: str = "n") -> List[str]:
        """alias of `load_ann`"""
        return self.load_ann(rec, ann_format)

    def get_subject_info(
        self, rec: Union[int, str], items: Optional[List[str]] = None
    ) -> dict:
        """Get subject information (e.g sex, age, etc.).

        Parameters
        ----------
        rec : int or str
            Record name or index of the record in :attr:`all_records`.
        items : List[str], optional
            Items of the subject information (e.g. sex, age, etc.).

        Returns
        -------
        subject_info : dict
            The subject information.

        """
        if items is None or len(items) == 0:
            info_items = ["age", "sex"]
        else:
            info_items = items

        if isinstance(rec, int):
            rec = self[rec]
        rec_fp = self.get_absolute_path(rec, self.ann_ext)
        data = loadmat(str(rec_fp))["ECG"]
        subject_info = {
            "age": data["age"][0, 0][0, 0],
            "sex": data["sex"][0, 0][0],
        }

        return subject_info

    def plot(
        self,
        rec: Union[int, str],
        ticks_granularity: int = 0,
        leads: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        """Plot the ECG data of a record.

        Parameters
        ----------
        rec : int or str
            Record name or index of the record in :attr:`all_records`.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads : str or List[str], optional
            The leads to plot
        kwargs : dict, optional
            Auxilliary key word arguments to pass to :func:`matplotlib.pyplot.subplots`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt

        leads = self._normalize_leads(leads)

        data = self.load_data(rec, data_format="channel_first", units="μV", leads=leads)
        y_ranges = np.max(np.abs(data), axis=1) + 100

        diag = self.get_labels(rec, ann_format="f")

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

    @property
    def database_info(self) -> DataBaseInfo:
        return _CPSC2018_INFO

    @property
    def webpage(self) -> str:
        return "http://2018.icbeb.org/Challenge.html"

    def _download_labels(self) -> None:
        label_url = "http://2018.icbeb.org/file/REFERENCE.csv"
        http_get(label_url, self.db_dir, extract=False)

    def download(self) -> None:
        """Download the database from :attr:`self.url`."""
        for url in self.url:
            http_get(url, self.db_dir, extract=True)
        self._download_labels()
        self._ls_rec()


def compute_metrics():
    """ """
    raise NotImplementedError
