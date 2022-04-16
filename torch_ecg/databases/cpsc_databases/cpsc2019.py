# -*- coding: utf-8 -*-
"""
"""

import json
from numbers import Real
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence, Union

import numpy as np
from scipy.io import loadmat

from ...utils.download import http_get
from ...utils.misc import add_docstring
from ..base import DEFAULT_FIG_SIZE_PER_SEC, CPSCDataBase

__all__ = [
    "CPSC2019",
    "compute_metrics",
]


class CPSC2019(CPSCDataBase):
    """

    The 2nd China Physiological Signal Challenge (CPSC 2019):
    Challenging QRS Detection and Heart Rate Estimation from Single-Lead ECG Recordings

    ABOUT CPSC2019
    --------------
    1. Training data consists of 2,000 single-lead ECG recordings collected from patients with cardiovascular disease (CVD)
    2. Each of the recording last for 10 s
    3. Sampling rate = 500 Hz

    NOTE
    ----

    ISSUES
    ------
    1. there're 13 records with unusual large values (> 20 mV):
        data_00098, data_00167, data_00173, data_00223, data_00224, data_00245, data_00813,
        data_00814, data_00815, data_00833, data_00841, data_00949, data_00950
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
    2. rpeak references (annotations) loaded from files has dtype = uint16,
    which would produce unexpected large positive values when subtracting values larger than it,
    rather than the correct negative value.
    This might cause confusion in computing metrics when using annotations subtracting
    (instead of being subtracted by) predictions.
    3. official scoring function has errors,
    which would falsely omit the interval between the 0-th and the 1-st ref rpeaks,
    thus potentially missing false positive

    Usage
    -----
    1. ECG wave delineation

    References
    ----------
    1. <a name="ref1"></a> http://2019.icbeb.org/Challenge.html

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

        self.n_records = 2000
        self._all_records = [f"data_{i:05d}" for i in range(1, 1 + self.n_records)]
        self._all_annotations = [f"R_{i:05d}" for i in range(1, 1 + self.n_records)]
        self._ls_rec()

        # aliases
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir

    def _ls_rec(self) -> NoReturn:
        """ """
        records_fn = self.db_dir / "records.json"
        if records_fn.is_file():
            records_json = json.loads(records_fn.read_text())
            self._all_records = records_json["rec"]
            self._all_annotations = records_json["ann"]
            return
        print(
            "Please allow some time for the reader to confirm the existence of corresponding data files and annotation files..."
        )
        self._all_records = [
            rec
            for rec in self._all_records
            if (self.rec_dir / f"{rec}.{self.rec_ext}").is_file()
        ]
        self._all_annotations = [
            ann
            for ann in self._all_annotations
            if (self.ann_dir / f"{ann}.{self.ann_ext}").is_file()
        ]
        common = set([rec.split("_")[1] for rec in self._all_records]) & set(
            [ann.split("_")[1] for ann in self._all_annotations]
        )
        common = sorted(list(common))
        self._all_records = [f"data_{item}" for item in common]
        self._all_annotations = [f"R_{item}" for item in common]
        records_json = {"rec": self._all_records, "ann": self._all_annotations}
        records_fn.write_text(json.dumps(records_json, ensure_ascii=False))

    @property
    def all_annotations(self):
        """ """
        return self._all_annotations

    @property
    def all_references(self):
        """ """
        return self._all_annotations

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """not finished,

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        pid: int,
            the `subject_id` corr. to `rec_no`

        """
        pid = 0
        raise NotImplementedError

    def load_data(
        self, rec: Union[int, str], units: str = "mV", keep_dim: bool = True
    ) -> np.ndarray:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = self.data_dir / f"{self._get_rec_name(rec)}.{self.rec_ext}"
        data = loadmat(str(fp))["ecg"]
        if units.lower() in [
            "uv",
            "μv",
        ]:
            data = (1000 * data).astype(int)
        if not keep_dim:
            data = data.flatten()
        return data

    def load_ann(self, rec: Union[int, str], keep_dim: bool = True) -> np.ndarray:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)

        Returns
        -------
        ann: ndarray,
            array of indices of R peaks

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = self.ann_dir / f"{self._get_ann_name(rec)}.{self.ann_ext}"
        ann = loadmat(str(fp))["R_peak"].astype(int)
        if not keep_dim:
            ann = ann.flatten()
        return ann

    @add_docstring(load_ann.__doc__)
    def load_rpeaks(self, rec: Union[int, str], keep_dim: bool = True) -> np.ndarray:
        """
        alias of `self.load_ann`
        """
        return self.load_ann(rec=rec, keep_dim=keep_dim)

    @add_docstring(load_rpeaks.__doc__)
    def load_rpeak_indices(
        self, rec: Union[int, str], keep_dim: bool = True
    ) -> np.ndarray:
        """
        alias of `self.load_rpeaks`
        """
        return self.load_rpeaks(rec=rec, keep_dim=keep_dim)

    def _get_rec_name(self, rec: Union[int, str]) -> str:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        rec_name: str,
            filename of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        rec_name = rec
        return rec_name

    def _get_ann_name(self, rec: Union[int, str]) -> str:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        ann_name: str,
            filename of annotations of the record `rec`

        """
        if isinstance(rec, int):
            rec = self[rec]
        rec_name = self._get_rec_name(rec)
        ann_name = rec_name.replace("data", "R")
        return ann_name

    def plot(
        self,
        rec: Union[int, str],
        data: Optional[np.ndarray] = None,
        ann: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data: ndarray, optional,
            ECG signal to plot,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: ndarray, optional,
            annotations (rpeak indices) for `data`,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt

        if data is None:
            _data = self.load_data(rec, units="μV", keep_dim=False)
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        duration = len(_data) / self.fs
        secs = np.linspace(0, duration, len(_data))
        if ann is None or data is None:
            rpeak_secs = self.load_rpeaks(rec, keep_dim=False) / self.fs
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
        return "http://2019.icbeb.org/file/train.rar"

    def download(self) -> NoReturn:
        """download the database from self.url"""
        http_get(self.url, self.db_dir, extract=False)


def compute_metrics(
    rpeaks_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rpeaks_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    thr: float = 0.075,
    verbose: int = 0,
) -> float:
    """

    metric (scoring) function modified from the official one, with errors fixed

    Parameters
    ----------
    rpeaks_truths: sequence,
        sequence of ground truths of rpeaks locations from multiple records
    rpeaks_preds: sequence,
        predictions of ground truths of rpeaks locations for multiple records
    fs: real number,
        sampling frequency of ECG signal
    thr: float, default 0.075,
        threshold for a prediction to be truth positive,
        with units in seconds,
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    rec_acc: float,
        accuracy of predictions

    """
    assert len(rpeaks_truths) == len(
        rpeaks_preds
    ), f"number of records does not match, truth indicates {len(rpeaks_truths)}, while pred indicates {len(rpeaks_preds)}"
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
                f"for the {idx}-th record,\ntrue positive = {true_positive}\nfalse positive = {false_positive}\nfalse negative = {false_negative}"
            )

    rec_acc = round(np.sum(record_flags) / n_records, 4)
    print(f"QRS_acc: {rec_acc}")
    print("Scoring complete.")

    return rec_acc
