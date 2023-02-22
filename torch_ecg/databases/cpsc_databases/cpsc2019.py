# -*- coding: utf-8 -*-

import json
from numbers import Real
from pathlib import Path
from typing import Any, Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import scipy.signal as SS
from scipy.io import loadmat

from ...cfg import DEFAULTS
from ...utils.misc import add_docstring
from ..base import DEFAULT_FIG_SIZE_PER_SEC, CPSCDataBase, DataBaseInfo


__all__ = [
    "CPSC2019",
    "compute_metrics",
]


_CPSC2019_INFO = DataBaseInfo(
    title="""
    The 2nd China Physiological Signal Challenge (CPSC 2019):
    Challenging QRS Detection and Heart Rate Estimation from Single-Lead ECG Recordings
    """,
    about="""
    1. Training data consists of 2,000 single-lead ECG recordings collected from patients with cardiovascular disease (CVD)
    2. Each of the recording last for 10 s
    3. Sampling rate = 500 Hz
    """,
    usage=[
        "ECG wave delineation",
    ],
    issues="""
    1. there're 13 records with unusual large values (> 20 mV):
        data_00098, data_00167, data_00173, data_00223, data_00224, data_00245, data_00813,
        data_00814, data_00815, data_00833, data_00841, data_00949, data_00950.

        .. code-block:: python

            >>> for rec in dr.all_records:
            >>>     data = dr.load_data(rec)
            >>>     if np.max(data) > 20:
            >>>         print(f"{rec} has max value ({np.max(data)} mV) > 20 mV")
            data_00173 has max value (32.72031811111111 mV) > 20 mV
            data_00223 has max value (32.75516713333333 mV) > 20 mV
            data_00224 has max value (32.7519272 mV) > 20 mV
            data_00245 has max value (32.75305293939394 mV) > 20 mV
            data_00813 has max value (32.75865595876289 mV) > 20 mV
            data_00814 has max value (32.75865595876289 mV) > 20 mV
            data_00815 has max value (32.75558282474227 mV) > 20 mV
            data_00833 has max value (32.76330123809524 mV) > 20 mV
            data_00841 has max value (32.727626558139534 mV) > 20 mV
            data_00949 has max value (32.75699667692308 mV) > 20 mV
            data_00950 has max value (32.769551661538465 mV) > 20 mV
    2. rpeak references (annotations) loaded from files has dtype = uint16, which would produce unexpected large positive values when subtracting values larger than it, rather than the correct negative value. This might cause confusion in computing metrics when using annotations subtracting (instead of being subtracted by) predictions.
    3. official scoring function has errors, which would falsely omit the interval between the 0-th and the 1-st ref rpeaks, thus potentially missing false positive
    """,
    references=[
        "http://2019.icbeb.org/Challenge.html",
    ],
    doi="10.1166/jmihi.2019.2800",
)


@add_docstring(_CPSC2019_INFO.format_database_docstring())
class CPSC2019(CPSCDataBase):
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

    __name__ = "CPSC2019"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="cpsc2019",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.fs = 500
        self.spacing = 1000 / self.fs

        self.rec_ext = "mat"
        self.ann_ext = "mat"

        # self.all_references = self.all_annotations
        self.rec_dir = self.db_dir / "data"
        self.ann_dir = self.db_dir / "ref"

        # aliases
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir

        self.n_records = 2000
        self._all_records = None
        self._all_annotations = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        records_fn = self.db_dir / "records.json"
        self._all_records = [f"data_{i:05d}" for i in range(1, 1 + self.n_records)]
        self._all_annotations = [f"R_{i:05d}" for i in range(1, 1 + self.n_records)]
        self._df_records = pd.DataFrame()
        self.logger.info(
            "Please allow some time for the reader to confirm "
            "the existence of corresponding data files and annotation files..."
        )
        self._df_records["record"] = [
            f"data_{i:05d}" for i in range(1, 1 + self.n_records)
        ]
        self._df_records["path"] = self._df_records["record"].apply(
            lambda x: self.data_dir / x
        )
        self._df_records["annotation"] = self._df_records["record"].apply(
            lambda x: x.replace("data", "R")
        )
        self._df_records.index = self._df_records["record"]
        self._df_records = self._df_records.drop(columns="record")
        self._all_annotations = [f"R_{i:05d}" for i in range(1, 1 + self.n_records)]
        self._all_records = [
            rec
            for rec in self._all_records
            if self.get_absolute_path(rec, self.rec_ext).is_file()
        ]
        self._all_annotations = [
            ann
            for ann in self._all_annotations
            if self.get_absolute_path(ann, self.ann_ext, ann=True).is_file()
        ]
        common = set([rec.split("_")[1] for rec in self._all_records]) & set(
            [ann.split("_")[1] for ann in self._all_annotations]
        )
        if self._subsample is not None:
            # random subsample with ratio `self._subsample`
            size = min(len(common), max(1, int(round(self._subsample * len(common)))))
            common = DEFAULTS.RNG.choice(list(common), size, replace=False)
        common = sorted(common)
        self._all_records = [f"data_{item}" for item in common]
        self._all_annotations = [f"R_{item}" for item in common]
        self._df_records = self._df_records.loc[self._all_records]
        records_json = {"rec": self._all_records, "ann": self._all_annotations}
        records_fn.write_text(json.dumps(records_json, ensure_ascii=False))

    @property
    def all_annotations(self) -> List[str]:
        return self._all_annotations

    @property
    def all_references(self) -> List[str]:
        return self._all_annotations

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """Attach a unique subject id to each record.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`self.all_records`.

        Returns
        -------
        pid : int
            The ``subject_id`` corr. to `rec`.

        """
        if isinstance(rec, int):
            rec = self[rec]
        return int(f"19{int(rec.split('_')[1]):08d}")

    def get_absolute_path(
        self,
        rec: Union[str, int],
        extension: Optional[str] = None,
        ann: bool = False,
    ) -> Path:
        """Get the absolute path of the record `rec`.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`self.all_records`.
        extension : str, optional
            Extension of the file.
            If not provided, no extension will be added.
        ann : bool, default False
            Whether to get the annotation file path or not.

        Returns
        -------
        Path
            Absolute path of the file.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if extension is not None and not extension.startswith("."):
            extension = f".{extension}"
        if ann:
            rec = rec.replace("data", "R")
            return self.ann_dir / f"{rec}{extension or ''}"
        return self.data_dir / f"{rec}{extension or ''}"

    def load_data(
        self,
        rec: Union[int, str],
        data_format: str = "channel_first",
        units: str = "mV",
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """Load the ECG data of the record `rec`.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`self.all_records`.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain").
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (with aliases "uV", "muV").
        fs : numbers.Real, optional
            If provided, the loaded data will be resampled to this frequency,
            otherwise the original sampling frequency will be used.

        Returns
        -------
        data : numpy.ndarray,
            The loaded ECG data.

        """
        fp = self.get_absolute_path(rec, self.rec_ext)
        data = loadmat(str(fp))["ecg"].astype(DEFAULTS.DTYPE.NP)
        if fs is not None and fs != self.fs:
            data = SS.resample_poly(data, fs, self.fs, axis=0).astype(data.dtype)
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        elif data_format.lower() in ["flat", "plain"]:
            data = data.flatten()
        elif data_format.lower() not in ["channel_last", "lead_last"]:
            raise ValueError(f"Invalid `data_format`: {data_format}")
        if units.lower() in ["uv", "μv", "muv"]:
            data = (1000 * data).astype(int)
        elif units.lower() != "mv":
            raise ValueError(f"Invalid `units`: {units}")
        return data

    def load_ann(self, rec: Union[int, str]) -> np.ndarray:
        """Load the annotations (indices of R peaks) of the record `rec`.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`self.all_records`.

        Returns
        -------
        ann : numpy.ndarray,
            Array of indices of R peaks.

        """
        fp = self.get_absolute_path(rec, self.ann_ext, ann=True)
        ann = loadmat(str(fp))["R_peak"].astype(int).flatten()
        return ann

    @add_docstring(load_ann.__doc__)
    def load_rpeaks(self, rec: Union[int, str]) -> np.ndarray:
        """
        alias of `self.load_ann`
        """
        return self.load_ann(rec=rec)

    @add_docstring(load_rpeaks.__doc__)
    def load_rpeak_indices(self, rec: Union[int, str]) -> np.ndarray:
        """
        alias of `self.load_rpeaks`
        """
        return self.load_rpeaks(rec=rec)

    def plot(
        self,
        rec: Union[int, str],
        data: Optional[np.ndarray] = None,
        ann: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        **kwargs: Any,
    ) -> None:
        """Plot the ECG data of the record `rec`.

        Parameters
        ----------
        rec : str or int
            Record name or index of the record in :attr:`self.all_records`.
        data : numpy.ndarray, optional
            ECG signal to plot.
            If provided, data of `rec` will not be used,
            which is useful when plotting filtered data.
        ann : numpy.ndarray, optional
            Annotations (rpeak indices) for `data`.
            Ignored if `data` is None.
        ticks_granularity : int, default 0
            Granularity to plot axis ticks, the higher the more ticks.
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt

        if data is None:
            _data = self.load_data(rec, units="μV", data_format="flat")
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        duration = len(_data) / self.fs
        secs = np.linspace(0, duration, len(_data))
        if ann is None or data is None:
            rpeak_secs = self.load_rpeaks(rec) / self.fs
        else:
            rpeak_secs = np.array(ann) / self.fs

        fig_sz_w = int(DEFAULT_FIG_SIZE_PER_SEC * duration)
        y_range = np.max(np.abs(_data))
        fig_sz_h = 6 * y_range / 1500
        fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
        ax.plot(
            secs,
            _data,
            color="black",
            linewidth="2.0",
        )
        ax.axhline(y=0, linestyle="-", linewidth="1.0", color="red")
        if ticks_granularity >= 1:
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
        if ticks_granularity >= 2:
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
            ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
        for r in rpeak_secs:
            ax.axvspan(r - 0.01, r + 0.01, color="green", alpha=0.9)
            ax.axvspan(r - 0.075, r + 0.075, color="green", alpha=0.3)
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-y_range, y_range)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [μV]")
        if kwargs.get("save_path", None):
            plt.savefig(kwargs["save_path"], dpi=200, bbox_inches="tight")
        else:
            plt.show()

    @property
    def url(self) -> str:
        return "https://www.dropbox.com/s/75nee0pqdy3f9r2/CPSC2019-train.zip?dl=1"

    @property
    def database_info(self) -> DataBaseInfo:
        return _CPSC2019_INFO

    @property
    def webpage(self) -> str:
        return "http://2019.icbeb.org/Challenge.html"


def compute_metrics(
    rpeaks_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rpeaks_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    thr: float = 0.075,
    verbose: int = 0,
) -> float:
    """Metric (scoring) function modified from the official one,
    with errors fixed.

    Parameters
    ----------
    rpeaks_truths : sequence
        Sequence of ground truths of rpeaks locations from multiple records.
    rpeaks_preds : sequence
        Predictions of ground truths of rpeaks locations for multiple records.
    fs : numbers.Real
        Sampling frequency of ECG signal.
    thr : float, default 0.075
        Threshold for a prediction to be truth positive,
        with units in seconds.
    verbose : int, default 0
        Verbosity level for printing.

    Returns
    -------
    rec_acc : float
        Accuracy of predictions.

    """
    assert len(rpeaks_truths) == len(rpeaks_preds), (
        f"number of records does not match, truth indicates {len(rpeaks_truths)}, "
        f"while pred indicates {len(rpeaks_preds)}"
    )
    n_records = len(rpeaks_truths)
    record_flags = np.ones((len(rpeaks_truths),), dtype=float)
    thr_ = thr * fs
    if verbose >= 1:
        print(f"number of records = {n_records}")
        print(f"threshold in number of sample points = {thr_}")
    for idx, (truth_arr, pred_arr) in enumerate(zip(rpeaks_truths, rpeaks_preds)):
        false_negative = 0
        false_positive = 0
        true_positive = 0
        extended_truth_arr = np.concatenate((truth_arr.astype(int), [int(9.5 * fs)]))
        for j, t_ind in enumerate(extended_truth_arr[:-1]):
            next_t_ind = extended_truth_arr[j + 1]
            loc = np.where(np.abs(pred_arr - t_ind) <= thr_)[0]
            if j == 0:
                err = np.where(
                    (pred_arr >= 0.5 * fs + thr_) & (pred_arr <= t_ind - thr_)
                )[0]
            else:
                err = np.array([], dtype=int)
            err = np.append(
                err,
                np.where((pred_arr >= t_ind + thr_) & (pred_arr <= next_t_ind - thr_))[
                    0
                ],
            )

            false_positive += len(err)
            if len(loc) >= 1:
                true_positive += 1
                false_positive += len(loc) - 1
            elif len(loc) == 0:
                false_negative += 1

        if false_negative + false_positive > 1:
            record_flags[idx] = 0
        elif false_negative == 1 and false_positive == 0:
            record_flags[idx] = 0.3
        elif false_negative == 0 and false_positive == 1:
            record_flags[idx] = 0.7

        if verbose >= 2:
            print(
                f"for the {idx}-th record,\ntrue positive = {true_positive}\n"
                f"false positive = {false_positive}\nfalse negative = {false_negative}"
            )

    rec_acc = round(np.sum(record_flags) / n_records, 4)
    print(f"QRS_acc: {rec_acc}")
    print("Scoring complete.")

    return rec_acc
