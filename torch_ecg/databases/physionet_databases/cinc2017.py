# -*- coding: utf-8 -*-
"""
"""
import os
import math
from datetime import datetime
from typing import Union, Optional, Any, List, Sequence, NoReturn
from numbers import Real

import wfdb
import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd

from ...utils.misc import (
    get_record_list_recursive,
    get_record_list_recursive3,
)
from ..base import PhysioNetDataBase, DEFAULT_FIG_SIZE_PER_SEC


__all__ = [
    "CINC2017",
]


class CINC2017(PhysioNetDataBase):
    """ finished, checked,

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
    [1] https://physionet.org/content/challenge-2017/1.0.0/
    """

    def __init__(self,
                 db_dir:str,
                 working_dir:Optional[str]=None,
                 verbose:int=2,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        db_dir: str, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(db_name="CINC2017", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.fs = 300
        
        self.rec_ext = "mat"
        self.ann_ext = "hea"

        self._all_records = []
        self._ls_rec()

        self._df_ann = pd.read_csv(os.path.join(self.db_dir, "REFERENCE.csv"), header=None)
        self._df_ann.columns = ["rec", "ann",]
        self._df_ann_ori = pd.read_csv(os.path.join(self.db_dir, "REFERENCE-original.csv"), header=None)
        self._df_ann_ori.columns = ["rec", "ann",]
        # ["N", "A", "O", "~"]
        self._all_ann = list(set(self._df_ann.ann.unique().tolist() + self._df_ann_ori.ann.unique().tolist()))
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

    def _ls_rec(self) -> NoReturn:
        """
        """
        fp = os.path.join(self.db_dir, "RECORDS")
        if os.path.isfile(fp):
            with open(fp, "r") as f:
                self._all_records = f.read().splitlines()
                return
        self._all_records = get_record_list_recursive3(
            db_dir=self.db_dir,
            rec_patterns=f"A[\d]{{5}}.{self.rec_ext}"
        )
        with open(fp, "w") as f:
            for rec in self._all_records:
                f.write(f"{rec}\n")

    def load_data(self, rec:str, data_format:str="channel_first", units:str="mV") -> np.ndarray:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            name of the record
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
        assert data_format.lower() in ["channel_first", "lead_first", "channel_last", "lead_last", "flat",]
        assert units.lower() in ["mv", "uv", "μv",]
        wr = wfdb.rdrecord(os.path.join(self.db_dir, rec))
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

    def load_ann(self, rec:str, original:bool=False, ann_format:str="a") -> str:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            name of the record
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
        assert rec in self.all_records and ann_format.lower() in ["a", "f"]
        if original:
            df = self._df_ann_ori
        else:
            df = self._df_ann
        row = df[df.rec==rec].iloc[0]
        ann = row.ann
        if ann_format.lower() == "f":
            ann = self.d_ann_names[ann]
        return ann

    def plot(self,
             rec:str,
             data:Optional[np.ndarray]=None,
             ann:Optional[str]=None,
             ticks_granularity:int=0,
             rpeak_inds:Optional[Union[Sequence[int],np.ndarray]]=None,) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            name of the record
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
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if data is None:
            _data = self.load_data(
                rec, units="μV", data_format="flat",
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
        nb_lines = math.ceil(len(_data)/line_len)

        for idx in range(nb_lines):
            seg = _data[idx*line_len: (idx+1)*line_len]
            secs = (np.arange(len(seg)) + idx*line_len) / self.fs
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
            ax.legend(
                handles=[patch],
                loc="lower left",
                prop={"size": 16}
            )
            if rpeak_inds is not None:
                for r in rpeak_secs:
                    ax.axvspan(r-0.01, r+0.01, color="green", alpha=0.7)
            ax.set_xlim(secs[0], secs[-1])
            ax.set_ylim(-y_range, y_range)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [μV]")
            plt.show()
