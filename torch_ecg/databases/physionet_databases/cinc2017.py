# -*- coding: utf-8 -*-
"""
AF Classification from a Short Single Lead ECG Recording
-- The PhysioNet Computing in Cardiology Challenge 2017
"""

import math
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union, List

import numpy as np
import pandas as pd
import wfdb  # noqa: F401

from ...cfg import DEFAULTS
from ...utils.misc import get_record_list_recursive3, add_docstring
from ..base import DEFAULT_FIG_SIZE_PER_SEC, PhysioNetDataBase, DataBaseInfo


__all__ = [
    "CINC2017",
]


_CINC2017_INFO = DataBaseInfo(
    title="""
    AF Classification from a Short Single Lead ECG Recording
    -- The PhysioNet Computing in Cardiology Challenge 2017
    """,
    about="""
    1. training set contains 8,528 single lead ECG recordings lasting from 9 s to just over 60 s, and the test set contains 3,658 ECG recordings of similar lengths
    2. records are of frequency 300 Hz and have been band pass filtered
    3. data distribution:
        Type            # recording             Time length (s)
                                        Mean    SD      Max     Median  Min
        Normal          5154            31.9    10.0    61.0    30      9.0
        AF              771             31.6    12.5    60      30      10.0
        Other rhythm    2557            34.1    11.8    60.9    30      9.1
        Noisy           46              27.1    9.0     60      30      10.2
        Total           8528            32.5    10.9    61.0    30      9.0
    """,
    usage=[
        "Atrial fibrillation (AF) detection",
    ],
    references=[
        "https://physionet.org/content/challenge-2017/1.0.0/",
    ],
    doi=[
        "10.22489/CinC.2017.065-469",
    ],
)


@add_docstring(_CINC2017_INFO.format_database_docstring())
class CINC2017(PhysioNetDataBase):
    """ """

    __name__ = "CINC2017"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        db_dir: str or Path, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 1
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

        # self._url_compressed = (
        #     "https://physionet.org/static/published-projects/challenge-2017/"
        #     "af-classification-from-a-short-single-lead-ecg-recording-"
        #     "the-physionetcomputing-in-cardiology-challenge-2017-1.0.0.zip"
        # )
        self._url_compressed = self.get_file_download_url("training2017.zip")

    def _ls_rec(self) -> None:
        """ """
        record_list_fp = self.db_dir / "RECORDS"
        self._df_records = pd.DataFrame()
        if record_list_fp.is_file():
            self._df_records["record"] = [
                item
                for item in record_list_fp.read_text().splitlines()
                if len(item) > 0
            ]
            if len(self._df_records) > 0:
                if self._subsample is not None:
                    size = min(
                        len(self._df_records),
                        max(1, int(round(self._subsample * len(self._df_records)))),
                    )
                    self._df_records = self._df_records.sample(
                        n=size, random_state=DEFAULTS.SEED, replace=False
                    )
                self._df_records["path"] = self._df_records["record"].apply(
                    lambda x: (self.db_dir / x).resolve()
                )
                self._df_records = self._df_records[
                    self._df_records["path"].apply(lambda x: x.is_file())
                ]
                self._df_records["record"] = self._df_records["path"].apply(
                    lambda x: x.name
                )

        if len(self._df_records) == 0:
            self.logger.info(
                "Please wait patiently to let the reader find "
                "all records of the database from local storage..."
            )
            start = time.time()
            self._df_records["path"] = get_record_list_recursive3(
                db_dir=str(self.db_dir),
                rec_patterns=f"A[\\d]{{5}}\\.{self.rec_ext}",
                relative=False,
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
            self.logger.info(f"Done in {time.time() - start:.3f} seconds!")
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.name
            )
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()

        ann_file = list(self.db_dir.rglob("REFERENCE.csv"))
        if len(ann_file) > 0:
            self._df_ann = pd.read_csv(ann_file[0], header=None)
            self._df_ann.columns = ["rec", "ann"]
        else:
            self._df_ann = pd.DataFrame(columns=["rec", "ann"])
            warnings.warn(
                "Cannot find the annotation file `REFERENCE.csv`!",
                RuntimeWarning,
            )
        ann_file = list(self.db_dir.rglob("REFERENCE-original.csv"))
        if len(ann_file) > 0:
            self._df_ann_ori = pd.read_csv(ann_file[0], header=None)
            self._df_ann_ori.columns = ["rec", "ann"]
        else:
            self._df_ann_ori = pd.DataFrame(columns=["rec", "ann"])
            warnings.warn(
                "Cannot find the annotation file `REFERENCE-original.csv`!",
                RuntimeWarning,
            )
        # ["N", "A", "O", "~"]
        self._all_ann = list(
            set(
                self._df_ann.ann.unique().tolist()
                + self._df_ann_ori.ann.unique().tolist()
            )
        )

    def load_ann(
        self, rec: Union[str, int], original: bool = False, ann_format: str = "a"
    ) -> str:
        """
        load the annotation of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
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
    ) -> None:
        """
        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
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

    @property
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

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2017_INFO
