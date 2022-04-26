# -*- coding: utf-8 -*-
"""
"""

import math
import zipfile
import warnings
from pathlib import Path
from typing import Any, NoReturn, Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import wfdb

from ...utils.misc import get_record_list_recursive3
from ..base import DEFAULT_FIG_SIZE_PER_SEC, PhysioNetDataBase

__all__ = [
    "CINC2017",
]


class CINC2017(PhysioNetDataBase):
    """

    AF Classification from a Short Single Lead ECG Recording
    - The PhysioNet Computing in Cardiology Challenge 2017

    ABOUT CINC2017
    --------------
    1. training set contains 8,528 single lead ECG recordings lasting from 9 s to just over 60 s, and the test set contains 3,658 ECG recordings of similar lengths
    2. records are of frequency 300 Hz and have been band pass filtered
    3. data distribution:
        Type	        	                    Time length (s)
                        # recording     Mean	SD	    Max	    Median	Min
        Normal	        5154	        31.9	10.0	61.0	30	    9.0
        AF              771             31.6	12.5	60	    30	    10.0
        Other rhythm	2557	        34.1	11.8	60.9	30	    9.1
        Noisy	        46	            27.1	9.0	    60	    30	    10.2
        Total	        8528	        32.5	10.9	61.0	30	    9.0

    NOTE
    ----

    ISSUES
    ------

    Usage
    -----
    1. atrial fibrillation (AF) detection

    References
    ----------
    1. <a name="ref1"></a> https://physionet.org/content/challenge-2017/1.0.0/

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
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="challenge-2017",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 300

        self.rec_ext = "mat"
        self.ann_ext = "hea"

        self._all_records = []
        self._df_ann = pd.DataFrame()
        self._df_ann_ori = pd.DataFrame()
        self._all_ann = []
        self.data_dir = None
        self._ls_rec()

        self.d_ann_names = {
            "N": "Normal rhythm",
            "A": "AF rhythm",
            "O": "Other rhythm",
            "~": "Noisy",
        }
        self.palette = {
            "N": "green",
            "A": "red",
            "O": "yellow",
            "~": "blue",
        }

        self._url_compressed = (
            "https://physionet.org/static/published-projects/challenge-2017/"
            "af-classification-from-a-short-single-lead-ecg-recording-"
            "the-physionetcomputing-in-cardiology-challenge-2017-1.0.0.zip"
        )

    def _ls_rec(self) -> NoReturn:
        """ """
        fp = self.db_dir / "RECORDS"
        if fp.exists():
            self._all_records = fp.read_text().splitlines()
            # return
        self._all_records = get_record_list_recursive3(
            db_dir=str(self.db_dir), rec_patterns=f"A[\\d]{{5}}.{self.rec_ext}"
        )
        parent_dir = set([str(Path(item).parent) for item in self.all_records])
        if len(parent_dir) > 1:
            raise ValueError("all records should be in the same directory")
        self.data_dir = self.db_dir / parent_dir.pop()
        self._all_records = [Path(item).name for item in self.all_records]
        # fp.write_text("\n".join(self._all_records) + "\n")
        if len(self._all_records) == 0:
            warnings.warn(
                f"No record found in {self.db_dir}. "
                "Perhaps the user should call the `download` method to download the database first."
            )
            return
        self._df_ann = pd.read_csv(self.data_dir / "REFERENCE.csv", header=None)
        self._df_ann.columns = [
            "rec",
            "ann",
        ]
        self._df_ann_ori = pd.read_csv(
            self.data_dir / "REFERENCE-original.csv", header=None
        )
        self._df_ann_ori.columns = [
            "rec",
            "ann",
        ]
        # ["N", "A", "O", "~"]
        self._all_ann = list(
            set(
                self._df_ann.ann.unique().tolist()
                + self._df_ann_ori.ann.unique().tolist()
            )
        )

    def load_data(
        self,
        rec: Union[str, int],
        data_format: str = "channel_first",
        units: str = "mV",
    ) -> np.ndarray:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data_format: str, default "channel_first",
            format of the ecg data, case insensitive, can be
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (of dimension 1, without channel dimension)
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"

        Returns
        -------
        data: ndarray,
            data loaded from `rec`, with given units and format
        """
        if isinstance(rec, int):
            rec = self[rec]
        assert data_format.lower() in [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
            "flat",
        ]
        assert units.lower() in [
            "mv",
            "uv",
            "μv",
        ]
        wr = wfdb.rdrecord(str(self.data_dir / rec))
        data = wr.p_signal

        if wr.units[0].lower() == units.lower():
            pass
        elif wr.units[0].lower() in ["uv", "μv"] and units.lower() == "mv":
            data = data / 1000
        elif units.lower() in ["uv", "μv"] and wr.units[0].lower() == "mv":
            data = data * 1000

        data = data.squeeze()
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data[np.newaxis, ...]
        elif data_format.lower() in ["channel_last", "lead_last"]:
            data = data[..., np.newaxis]
        return data

    def load_ann(
        self, rec: Union[str, int], original: bool = False, ann_format: str = "a"
    ) -> str:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        original: bool, default False,
            if True, load annotations from the file `REFERENCE-original.csv`,
            otherwise from `REFERENCE.csv`
        ann_format: str, default "a",
            format of returned annotation, can be one of "a", "f",
            "a" - abbreviation
            "f" - full name

        Returns
        -------
        ann: str,
            annotation (label) of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert rec in self.all_records and ann_format.lower() in ["a", "f"]
        if original:
            df = self._df_ann_ori
        else:
            df = self._df_ann
        row = df[df.rec == rec].iloc[0]
        ann = row.ann
        if ann_format.lower() == "f":
            ann = self.d_ann_names[ann]
        return ann

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[str] = None,
        ticks_granularity: int = 0,
        rpeak_inds: Optional[Union[Sequence[int], np.ndarray]] = None,
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data: ndarray, optional,
            ecg signal to plot,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            annotations for `data`,
            "SPB_indices", "PVC_indices", each of ndarray values,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        rpeak_inds: array_like, optional,
            indices of R peaks,

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if data is None:
            _data = self.load_data(
                rec,
                units="μV",
                data_format="flat",
            )
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        if ann is None or data is None:
            ann = self.load_ann(rec, ann_format="a")
            ann_fullname = self.load_ann(rec, ann_format="f")
        else:
            ann_fullname = self.d_ann_names.get(ann, ann)
        patch = mpatches.Patch(color=self.palette.get(ann, "blue"), label=ann_fullname)

        if rpeak_inds is not None:
            rpeak_secs = np.array(rpeak_inds) / self.fs

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(len(_data) / line_len)

        for idx in range(nb_lines):
            seg = _data[idx * line_len : (idx + 1) * line_len]
            secs = (np.arange(len(seg)) + idx * line_len) / self.fs
            fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * len(seg) / self.fs))
            y_range = np.max(np.abs(seg)) + 100
            fig_sz_h = 6 * y_range / 1500
            fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
            ax.plot(secs, seg, color="black")
            ax.axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            if ticks_granularity >= 1:
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
                ax.yaxis.set_major_locator(plt.MultipleLocator(500))
                ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
            if ticks_granularity >= 2:
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
                ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
            ax.legend(handles=[patch], loc="lower left", prop={"size": 16})
            if rpeak_inds is not None:
                for r in rpeak_secs:
                    ax.axvspan(r - 0.01, r + 0.01, color="green", alpha=0.7)
            ax.set_xlim(secs[0], secs[-1])
            ax.set_ylim(-y_range, y_range)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [μV]")
            plt.show()

    def download(self, compressed: bool = False) -> NoReturn:
        """ """
        super().download(compressed=compressed)
        if compressed:
            with zipfile.ZipFile(str(self.db_dir / "training2017.zip")) as zip_ref:
                zip_ref.extractall()

    def _validation_set(self) -> List[str]:
        """
        the validation set specified at https://physionet.org/content/challenge-2017/1.0.0/
        """
        return (
            "A00001,A00002,A00003,A00004,A00005,A00006,A00007,A00008,A00009,A00010,"
            "A00011,A00012,A00013,A00014,A00015,A00016,A00017,A00018,A00019,A00020,"
            "A00021,A00022,A00023,A00024,A00025,A00026,A00027,A00028,A00029,A00030,"
            "A00031,A00032,A00033,A00034,A00035,A00036,A00037,A00038,A00039,A00040,"
            "A00041,A00042,A00043,A00044,A00045,A00046,A00047,A00048,A00049,A00050,"
            "A00051,A00052,A00053,A00054,A00055,A00056,A00057,A00058,A00059,A00060,"
            "A00061,A00062,A00063,A00064,A00065,A00066,A00067,A00068,A00069,A00070,"
            "A00071,A00072,A00073,A00074,A00075,A00076,A00077,A00078,A00079,A00080,"
            "A00081,A00082,A00083,A00084,A00085,A00086,A00087,A00088,A00089,A00090,"
            "A00091,A00092,A00093,A00094,A00095,A00096,A00097,A00098,A00099,A00100,"
            "A00101,A00102,A00103,A00104,A00105,A00106,A00107,A00108,A00109,A00110,"
            "A00111,A00112,A00113,A00114,A00115,A00116,A00117,A00118,A00119,A00120,"
            "A00121,A00122,A00123,A00124,A00125,A00126,A00127,A00128,A00129,A00130,"
            "A00131,A00132,A00133,A00134,A00135,A00136,A00137,A00138,A00139,A00140,"
            "A00141,A00142,A00143,A00144,A00145,A00146,A00147,A00148,A00149,A00150,"
            "A00151,A00152,A00153,A00154,A00155,A00156,A00157,A00158,A00159,A00160,"
            "A00161,A00162,A00163,A00164,A00165,A00166,A00167,A00168,A00169,A00170,"
            "A00171,A00172,A00173,A00174,A00175,A00176,A00177,A00178,A00179,A00180,"
            "A00181,A00182,A00183,A00184,A00185,A00186,A00187,A00188,A00189,A00190,"
            "A00191,A00192,A00193,A00194,A00195,A00196,A00197,A00198,A00199,A00200,"
            "A00201,A00202,A00203,A00204,A00205,A00206,A00207,A00208,A00209,A00210,"
            "A00211,A00212,A00213,A00214,A00215,A00216,A00217,A00218,A00219,A00220,"
            "A00221,A00222,A00223,A00224,A00225,A00226,A00227,A00228,A00229,A00230,"
            "A00231,A00232,A00233,A00234,A00235,A00236,A00237,A00238,A00239,A00240,"
            "A00241,A00242,A00244,A00245,A00247,A00248,A00249,A00253,A00267,A00271,"
            "A00301,A00321,A00375,A00395,A00397,A00405,A00422,A00432,A00438,A00439,"
            "A00441,A00456,A00465,A00473,A00486,A00509,A00519,A00520,A00524,A00542,"
            "A00551,A00585,A01006,A01070,A01246,A01299,A01521,A01567,A01707,A01727,"
            "A01772,A01833,A02168,A02372,A02772,A02785,A02833,A03549,A03738,A04086,"
            "A04137,A04170,A04186,A04216,A04282,A04452,A04522,A04701,A04735,A04805"
        ).split(",")
