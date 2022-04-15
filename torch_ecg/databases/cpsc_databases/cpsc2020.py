# -*- coding: utf-8 -*-
"""
"""

import math
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.io import loadmat

from ...cfg import CFG, DEFAULTS
from ...utils.download import http_get
from ...utils.ecg_arrhythmia_knowledge import PVC, SPB  # noqa: F401
from ...utils.utils_interval import get_optimal_covering
from ..base import DEFAULT_FIG_SIZE_PER_SEC, CPSCDataBase

__all__ = [
    "CPSC2020",
    "compute_metrics",
]


class CPSC2020(CPSCDataBase):
    """

    The 3rd China Physiological Signal Challenge 2020:
    Searching for Premature Ventricular Contraction (PVC) and Supraventricular Premature Beat (SPB) from Long-term ECGs

    ABOUT CPSC2020
    --------------
    1. training data consists of 10 single-lead ECG recordings collected from arrhythmia patients, each of the recording last for about 24 hours
    2. data and annotations are stored in v5 .mat files
    3. A02, A03, A08 are patient with atrial fibrillation
    4. sampling frequency = 400 Hz
    5. Detailed information:
        -------------------------------------------------------------------------
        rec   ?AF   Length(h)   # N beats   # V beats   # S beats   # Total beats
        A01   No	25.89       109,062     0           24          109,086
        A02   Yes	22.83       98,936      4,554       0           103,490
        A03   Yes	24.70       137,249     382         0           137,631
        A04   No	24.51       77,812      19,024      3,466       100,302
        A05   No	23.57       94,614  	1	        25	        94,640
        A06   No	24.59       77,621  	0	        6	        77,627
        A07   No	23.11	    73,325  	15,150	    3,481	    91,956
        A08   Yes	25.46	    115,518 	2,793	    0	        118,311
        A09   No	25.84	    88,229  	2	        1,462	    89,693
        A10   No	23.64	    72,821	    169	        9,071	    82,061
    6. challenging factors for accurate detection of SPB and PVC:
        amplitude variation; morphological variation; noise

    NOTE
    ----
    1. the records can roughly be classified into 4 groups:
        N:  A01, A03, A05, A06
        V:  A02, A08
        S:  A09, A10
        VS: A04, A07
    2. as premature beats and atrial fibrillation can co-exists
    (via the following code, and data from CINC2020),
    the situation becomes more complicated.
    >>> from utils.scoring_aux_data import dx_cooccurrence_all
    >>> dx_cooccurrence_all.loc["AF", ["PAC","PVC","SVPB","VPB"]]
    PAC     20
    PVC     19
    SVPB     4
    VPB     20
    Name: AF, dtype: int64
    this could also be seen from this dataset, via the following code as an example:
    >>> from data_reader import CPSC2020Reader as CR
    >>> db_dir = "/media/cfs/wenhao71/data/CPSC2020/TrainingSet/"
    >>> dr = CR(db_dir)
    >>> rec = dr.all_records[1]
    >>> dr.plot(rec, sampfrom=0, sampto=4000, ticks_granularity=2)
    3. PVC and SPB can also co-exist, as illustrated via the following code (from CINC2020):
    >>> from utils.scoring_aux_data import dx_cooccurrence_all
    >>> dx_cooccurrence_all.loc[["PVC","VPB"], ["PAC","SVPB",]]
    PAC	SVPB
    PVC	14	1
    VPB	27	0
    and also from the following code:
    >>> for rec in dr.all_records:
    >>>     ann = dr.load_ann(rec)
    >>>     spb = ann["SPB_indices"]
    >>>     pvc = ann["PVC_indices"]
    >>>     if len(np.diff(spb)) > 0:
    >>>         print(f"{rec}: min dist among SPB = {np.min(np.diff(spb))}")
    >>>     if len(np.diff(pvc)) > 0:
    >>>         print(f"{rec}: min dist among PVC = {np.min(np.diff(pvc))}")
    >>>     diff = [s-p for s,p in product(spb, pvc)]
    >>>     if len(diff) > 0:
    >>>         print(f"{rec}: min dist between SPB and PVC = {np.min(np.abs(diff))}")
    A01: min dist among SPB = 630
    A02: min dist among SPB = 696
    A02: min dist among PVC = 87
    A02: min dist between SPB and PVC = 562
    A03: min dist among SPB = 7044
    A03: min dist among PVC = 151
    A03: min dist between SPB and PVC = 3750
    A04: min dist among SPB = 175
    A04: min dist among PVC = 156
    A04: min dist between SPB and PVC = 178
    A05: min dist among SPB = 182
    A05: min dist between SPB and PVC = 22320
    A06: min dist among SPB = 455158
    A07: min dist among SPB = 603
    A07: min dist among PVC = 153
    A07: min dist between SPB and PVC = 257
    A08: min dist among SPB = 2903029
    A08: min dist among PVC = 106
    A08: min dist between SPB and PVC = 350
    A09: min dist among SPB = 180
    A09: min dist among PVC = 7719290
    A09: min dist between SPB and PVC = 1271
    A10: min dist among SPB = 148
    A10: min dist among PVC = 708
    A10: min dist between SPB and PVC = 177

    ISSUES
    ------
    1. currently, using `xqrs` as qrs detector,
       a lot more (more than 1000) rpeaks would be detected for A02, A07, A08,
       which might be caused by motion artefacts (or AF?);
       a lot less (more than 1000) rpeaks would be detected for A04.
       numeric details are as follows:
       ----------------------------------------------
       rec   ?AF    # beats by xqrs     # Total beats
       A01   No     109502              109,086
       A02   Yes    119562              103,490
       A03   Yes    135912              137,631
       A04   No     92746               100,302
       A05   No     94674               94,640
       A06   No     77955               77,627
       A07   No     98390               91,956
       A08   Yes    126908              118,311
       A09   No     89972               89,693
       A10   No     83509               82,061
    2. (fixed by an official update)
    A04 has duplicate "PVC_indices" (13534856,27147621,35141190 all appear twice):
       before correction of `load_ann`:
       >>> from collections import Counter
       >>> db_dir = "/mnt/wenhao71/data/CPSC2020/TrainingSet/"
       >>> data_gen = CPSC2020Reader(db_dir=db_dir,working_dir=db_dir)
       >>> rec = 4
       >>> ann = data_gen.load_ann(rec)
       >>> Counter(ann["PVC_indices"]).most_common()[:4]
       would produce [(13534856, 2), (27147621, 2), (35141190, 2), (848, 1)]
    3. when extracting morphological features using augmented rpeaks for A04,
       `RuntimeWarning: invalid value encountered in double_scalars` would raise
       for `R_value = (R_value - y_min) / (y_max - y_min)` and
       for `y_values[n] = (y_values[n] - y_min) / (y_max - y_min)`.
       this is caused by the 13882273-th sample, which is contained in "PVC_indices",
       however, whether it is a PVC beat, or just motion artefact, is in doubt!

    TODO
    ----
    1. use SNR to filter out too noisy segments?
    2. for ML, consider more features

    Usage
    -----
    1. ECG arrhythmia (PVC, SPB) detection

    References
    ----------
    [1] http://www.icbeb.org/CPSC2020.html
    [2] https://github.com/PIA-Group/BioSPPy
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
            db_name="cpsc2020",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.fs = 400
        self.spacing = 1000 / self.fs
        self.rec_ext = "mat"
        self.ann_ext = "mat"

        self.n_records = 10
        self._all_records = None
        self._all_annotations = None
        self._ls_rec()
        self.rec_dir = self.db_dir / "data"
        self.ann_dir = self.db_dir / "ref"
        # aliases
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir

        self.subgroups = CFG(
            {
                "N": [
                    "A01",
                    "A03",
                    "A05",
                    "A06",
                ],
                "V": ["A02", "A08"],
                "S": ["A09", "A10"],
                "VS": ["A04", "A07"],
            }
        )

        self.palette = {
            "spb": "yellow",
            "pvc": "red",
        }

    def _ls_rec(self) -> NoReturn:
        """ """
        self._all_records = [f"A{i:02d}" for i in range(1, 1 + self.n_records)]
        self._all_annotations = [f"R{i:02d}" for i in range(1, 1 + self.n_records)]

    @property
    def all_annotations(self):
        """ """
        return self._all_annotations

    @property
    def all_references(self):
        """ """
        return self._all_annotations

    def get_subject_id(self, rec: Union[int, str]) -> int:
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
        self,
        rec: Union[int, str],
        units: str = "mV",
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_dim: bool = True,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        rec_name = self._get_rec_name(rec)
        rec_fp = self.data_dir / f"{rec_name}.{self.rec_ext}"
        data = loadmat(str(rec_fp))["ecg"]
        if units.lower() in ["uv", "μv"]:
            data = (1000 * data).astype(int)
        sf, st = (sampfrom or 0), (sampto or len(data))
        data = data[sf:st]
        if not keep_dim:
            data = data.flatten()
        return data

    def load_ann(
        self,
        rec: Union[int, str],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded

        Returns
        -------
        ann: dict,
            with items (ndarray) "SPB_indices" and "PVC_indices",
            which record the indices of SPBs and PVCs
        """
        if isinstance(rec, int):
            rec = self[rec]
        ann_name = self._get_ann_name(rec)
        ann_fp = self.ann_dir / f"{ann_name}.{self.ann_ext}"
        ann = loadmat(str(ann_fp))["ref"]
        sf, st = (sampfrom or 0), (sampto or np.inf)
        spb_indices = ann["S_ref"][0, 0].flatten().astype(int)
        # drop duplicates
        spb_indices = np.array(sorted(list(set(spb_indices))), dtype=int)
        spb_indices = spb_indices[np.where((spb_indices >= sf) & (spb_indices < st))[0]]
        pvc_indices = ann["V_ref"][0, 0].flatten().astype(int)
        # drop duplicates
        pvc_indices = np.array(sorted(list(set(pvc_indices))), dtype=int)
        pvc_indices = pvc_indices[np.where((pvc_indices >= sf) & (pvc_indices < st))[0]]
        ann = {
            "SPB_indices": spb_indices,
            "PVC_indices": pvc_indices,
        }
        return ann

    def _get_ann_name(self, rec: Union[int, str]) -> str:
        """

        Parameters
        ----------
        rec: str or int,
            name or index of the record

        Returns
        -------
        ann_name: str,
            filename of the annotation file

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann_name = rec.replace("A", "R")
        return ann_name

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

    def train_test_split_rec(self, test_rec_num: int = 2) -> Dict[str, List[str]]:
        """

        split the records into train set and test set

        Parameters
        ----------
        test_rec_num: int,
            number of records for the test set

        Returns
        -------
        split_res: dict,
            with items `train`, `test`, both being list of record names
        """
        if test_rec_num == 1:
            test_records = DEFAULTS.RNG_sample(self.subgroups.VS, 1).tolist()
        elif test_rec_num == 2:
            test_records = (
                DEFAULTS.RNG_sample(self.subgroups.VS, 1).tolist()
                + DEFAULTS.RNG_sample(self.subgroups.N, 1).tolist()
            )
        elif test_rec_num == 3:
            test_records = (
                DEFAULTS.RNG_sample(self.subgroups.VS, 1).tolist()
                + DEFAULTS.RNG_sample(self.subgroups.N, 2).tolist()
            )
        elif test_rec_num == 4:
            test_records = []
            for k in self.subgroups.keys():
                test_records += DEFAULTS.RNG_sample(self.subgroups[k], 1).tolist()
        else:
            raise ValueError("test data ratio too high")
        train_records = [r for r in self.all_records if r not in test_records]

        split_res = CFG(
            {
                "train": train_records,
                "test": test_records,
            }
        )

        return split_res

    def locate_premature_beats(
        self,
        rec: Union[int, str],
        premature_type: Optional[str] = None,
        window: int = 10000,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
    ) -> List[List[int]]:
        """

        locate the sample indices of premature beats in a record,
        in the form of a list of lists,
        each list contains the interval of sample indices of premature beats

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        premature_type: str, optional,
            premature beat type, can be one of "SPB", "PVC"
        window: int, default 10000,
            window length of each premature beat
        sampfrom: int, optional,
            start index of the premature beats to locate
        sampto: int, optional,
            end index of the premature beats to locate

        Returns
        -------
        premature_intervals: list,
            list of intervals of premature beats
        """
        if isinstance(rec, int):
            rec = self[rec]
        ann = self.load_ann(rec)
        if premature_type:
            premature_inds = ann[f"{premature_type.upper()}_indices"]
        else:
            premature_inds = np.append(ann["SPB_indices"], ann["PVC_indices"])
            premature_inds = np.sort(premature_inds)
        try:  # premature_inds empty?
            sf, st = (sampfrom or 0), (sampto or premature_inds[-1] + 1)
        except Exception:
            premature_intervals = []
            return premature_intervals
        premature_inds = premature_inds[(sf < premature_inds) & (premature_inds < st)]
        tot_interval = [sf, st]
        premature_intervals, _ = get_optimal_covering(
            total_interval=tot_interval,
            to_cover=premature_inds,
            min_len=window * self.fs // 1000,
            split_threshold=window * self.fs // 1000,
            traceback=False,
        )
        return premature_intervals

    def plot(
        self,
        rec: Union[int, str],
        data: Optional[np.ndarray] = None,
        ann: Optional[Dict[str, np.ndarray]] = None,
        ticks_granularity: int = 0,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rpeak_inds: Optional[Union[Sequence[int], np.ndarray]] = None,
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
        ann: dict, optional,
            annotations for `data`, covering those from annotation files,
            "SPB_indices", "PVC_indices", each of ndarray values,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        sampfrom: int, optional,
            start index of the data to plot
        sampto: int, optional,
            end index of the data to plot
        rpeak_inds: array_like, optional,
            indices of R peaks,
            if `data` is None, then indices should be the absolute indices in the record
        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        patches = {}

        if data is None:
            _data = self.load_data(
                rec, units="μV", sampfrom=sampfrom, sampto=sampto, keep_dim=False
            )
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        if ann is None or data is None:
            ann = self.load_ann(rec, sampfrom=sampfrom, sampto=sampto)
        sf, st = (sampfrom or 0), (sampto or len(_data))
        spb_indices = ann["SPB_indices"]
        pvc_indices = ann["PVC_indices"]
        spb_indices = spb_indices - sf
        pvc_indices = pvc_indices - sf

        if rpeak_inds is not None:
            if data is not None:
                rpeak_secs = np.array(rpeak_inds) / self.fs
            else:
                rpeak_secs = np.array(rpeak_inds)
                rpeak_secs = rpeak_secs[
                    np.where((rpeak_secs >= sf) & (rpeak_secs < st))[0]
                ]
                rpeak_secs = (rpeak_secs - sf) / self.fs

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(len(_data) / line_len)

        bias_thr = 0.15
        winL = 0.06
        winR = 0.08

        for idx in range(nb_lines):
            seg = _data[idx * line_len : (idx + 1) * line_len]
            secs = (np.arange(len(seg)) + idx * line_len) / self.fs
            fig_sz_w = int(round(DEFAULT_FIG_SIZE_PER_SEC * len(seg) / self.fs))
            y_range = np.max(np.abs(seg)) + 100
            fig_sz_h = 6 * y_range / 1500
            fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
            ax.plot(
                secs,
                seg,
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
            seg_spb = np.where(
                (spb_indices >= idx * line_len) & (spb_indices < (idx + 1) * line_len)
            )[0]
            # print(f"spb_indices = {spb_indices}, seg_spb = {seg_spb}")
            if len(seg_spb) > 0:
                seg_spb = spb_indices[seg_spb] / self.fs
                patches["SPB"] = mpatches.Patch(color=self.palette["spb"], label="SPB")
            seg_pvc = np.where(
                (pvc_indices >= idx * line_len) & (pvc_indices < (idx + 1) * line_len)
            )[0]
            # print(f"pvc_indices = {pvc_indices}, seg_pvc = {seg_pvc}")
            if len(seg_pvc) > 0:
                seg_pvc = pvc_indices[seg_pvc] / self.fs
                patches["PVC"] = mpatches.Patch(color=self.palette["pvc"], label="PVC")
            for t in seg_spb:
                ax.axvspan(
                    max(secs[0], t - bias_thr),
                    min(secs[-1], t + bias_thr),
                    color=self.palette["spb"],
                    alpha=0.3,
                )
                ax.axvspan(
                    max(secs[0], t - winL),
                    min(secs[-1], t + winR),
                    color=self.palette["spb"],
                    alpha=0.9,
                )
            for t in seg_pvc:
                ax.axvspan(
                    max(secs[0], t - bias_thr),
                    min(secs[-1], t + bias_thr),
                    color=self.palette["pvc"],
                    alpha=0.3,
                )
                ax.axvspan(
                    max(secs[0], t - winL),
                    min(secs[-1], t + winR),
                    color=self.palette["pvc"],
                    alpha=0.9,
                )
            if len(patches) > 0:
                ax.legend(
                    handles=[v for _, v in patches.items()],
                    loc="lower left",
                    prop={"size": 16},
                )
            if rpeak_inds is not None:
                seg_rpeak_secs = rpeak_secs[
                    np.where((rpeak_secs >= secs[0]) & (rpeak_secs < secs[-1]))[0]
                ]
                for r in seg_rpeak_secs:
                    ax.axvspan(r - 0.01, r + 0.01, color="green", alpha=0.7)
            ax.set_xlim(secs[0], secs[-1])
            ax.set_ylim(-y_range, y_range)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Voltage [μV]")
            plt.show()


def _ann_to_beat_ann_epoch_v1(
    rpeaks: np.ndarray, ann: Dict[str, np.ndarray], bias_thr: Real
) -> dict:
    """

    the naive method to label beat types using annotations provided by the dataset

    Parameters
    ----------
    rpeaks: ndarray,
        rpeaks for forming beats
    ann: dict,
        with items (ndarray) "SPB_indices" and "PVC_indices",
        which record the indices of SPBs and PVCs
    bias_thr: real number,
        tolerance for using annotations (PVC, SPB indices provided by the dataset),
        to label the type of beats given by `rpeaks`

    Returns
    -------
    retval: dict, with the following items
        - ann_matched: dict of ndarray,
            indices of annotations ("SPB_indices" and "PVC_indices")
            that match some beat from `rpeaks`.
            for v1, this term is always the same as `ann`, hence useless
        - beat_ann: ndarray,
            label for each beat from `rpeaks`

    """
    beat_ann = np.array(["N" for _ in range(len(rpeaks))])
    for idx, r in enumerate(rpeaks):
        if any([abs(r - p) < bias_thr for p in ann["SPB_indices"]]):
            beat_ann[idx] = "S"
        elif any([abs(r - p) < bias_thr for p in ann["PVC_indices"]]):
            beat_ann[idx] = "V"
    ann_matched = ann.copy()
    retval = dict(ann_matched=ann_matched, beat_ann=beat_ann)
    return retval


def _ann_to_beat_ann_epoch_v2(
    rpeaks: np.ndarray, ann: Dict[str, np.ndarray], bias_thr: Real
) -> dict:
    """has flaws, deprecated,

    similar to `_ann_to_beat_ann_epoch_v1`, but records those matched annotations,
    for further post-process, adding those beats that are in annotation,
    but not detected by the signal preprocessing algorithms (qrs detection)

    however, the comparison process (the block inside the outer `for` loop)
    is not quite correct

    Parameters
    ----------
    rpeaks: ndarray,
        rpeaks for forming beats
    ann: dict,
        with items (ndarray) "SPB_indices" and "PVC_indices",
        which record the indices of SPBs and PVCs
    bias_thr: real number,
        tolerance for using annotations (PVC, SPB indices provided by the dataset),
        to label the type of beats given by `rpeaks`

    Returns
    -------
    retval: dict, with the following items
        - ann_matched: dict of ndarray,
            indices of annotations ("SPB_indices" and "PVC_indices")
            that match some beat from `rpeaks`
        - beat_ann: ndarray,
            label for each beat from `rpeaks`

    """
    beat_ann = np.array(["N" for _ in range(len(rpeaks))], dtype="<U1")
    # used to add back those beat that is not detected via proprocess algorithm
    _ann = {k: v.astype(int).tolist() for k, v in ann.items()}
    for idx_r, r in enumerate(rpeaks):
        found = False
        for idx_a, a in enumerate(_ann["SPB_indices"]):
            if abs(r - a) < bias_thr:
                found = True
                beat_ann[idx_r] = "S"
                del _ann["SPB_indices"][idx_a]
                break
        if found:
            continue
        for idx_a, a in enumerate(_ann["PVC_indices"]):
            if abs(r - a) < bias_thr:
                found = True
                beat_ann[idx_r] = "V"
                del _ann["PVC_indices"][idx_a]
                break
    ann_matched = {
        k: np.array([a for a in v if a not in _ann[k]], dtype=int)
        for k, v in ann.items()
    }
    retval = dict(ann_matched=ann_matched, beat_ann=beat_ann)
    return retval
    # _ann["SPB_indices"] = [a for a in _ann["SPB_indices"] if prev_r<a<next_r]
    # _ann["PVC_indices"] = [a for a in _ann["PVC_indices"] if prev_r<a<next_r]
    # augmented_rpeaks = np.concatenate((rpeaks, np.array(_ann["SPB_indices"]), np.array(_ann["PVC_indices"])))
    # beat_ann = np.concatenate((beat_ann, np.array(["S" for _ in _ann["SPB_indices"]], dtype="<U1"), np.array(["V" for _ in _ann["PVC_indices"]], dtype="<U1")))
    # sorted_indices = np.argsort(augmented_rpeaks)
    # augmented_rpeaks = augmented_rpeaks[sorted_indices].astype(int)
    # beat_ann = beat_ann[sorted_indices].astype("<U1")

    # retval = dict(augmented_rpeaks=augmented_rpeaks, beat_ann=beat_ann)
    # return retval


def _ann_to_beat_ann_epoch_v3(
    rpeaks: np.ndarray, ann: Dict[str, np.ndarray], bias_thr: Real
) -> dict:
    """

    similar to `_ann_to_beat_ann_epoch_v2`, but more reasonable

    Parameters
    ----------
    rpeaks: ndarray,
        rpeaks for forming beats
    ann: dict,
        with items (ndarray) "SPB_indices" and "PVC_indices",
        which record the indices of SPBs and PVCs
    bias_thr: real number,
        tolerance for using annotations (PVC, SPB indices provided by the dataset),
        to label the type of beats given by `rpeaks`

    Returns
    -------
    retval: dict, with the following items
        - ann_matched: dict of ndarray,
            indices of annotations ("SPB_indices" and "PVC_indices")
            that match some beat from `rpeaks`
        - beat_ann: ndarray,
            label for each beat from `rpeaks`

    """
    beat_ann = np.array(["N" for _ in range(len(rpeaks))], dtype="<U1")
    ann_matched = {k: [] for k, v in ann.items()}
    for idx_r, r in enumerate(rpeaks):
        dist_to_spb = np.abs(r - ann["SPB_indices"])
        dist_to_pvc = np.abs(r - ann["PVC_indices"])
        if len(dist_to_spb) == 0:
            dist_to_spb = np.array([np.inf])
        if len(dist_to_pvc) == 0:
            dist_to_pvc = np.array([np.inf])
        argmin = np.argmin([np.min(dist_to_spb), np.min(dist_to_pvc), bias_thr])
        if argmin == 2:
            pass
        elif argmin == 1:
            beat_ann[idx_r] = "V"
            ann_matched["PVC_indices"].append(
                ann["PVC_indices"][np.argmin(dist_to_pvc)]
            )
        elif argmin == 0:
            beat_ann[idx_r] = "S"
            ann_matched["SPB_indices"].append(
                ann["SPB_indices"][np.argmin(dist_to_spb)]
            )
    ann_matched = {k: np.array(v) for k, v in ann_matched.items()}
    retval = dict(ann_matched=ann_matched, beat_ann=beat_ann)
    return retval

    @property
    def url(self) -> str:
        return (
            "https://opensz.oss-cn-beijing.aliyuncs.com/ICBEB2020/file/TrainingSet.zip"
        )

    def download(self) -> NoReturn:
        """download the database from self.url"""
        http_get(self.url, self.db_dir, extract=True)


def compute_metrics(
    sbp_true: List[np.ndarray],
    pvc_true: List[np.ndarray],
    sbp_pred: List[np.ndarray],
    pvc_pred: List[np.ndarray],
    verbose: int = 0,
) -> Union[Tuple[int], dict]:
    """

    Score Function for all (test) records

    Parameters
    ----------
    sbp_true, pvc_true, sbp_pred, pvc_pred: list of ndarray,
    verbose: int

    Returns
    -------
    retval: tuple or dict,
        tuple of (negative) scores for each ectopic beat type (SBP, PVC), or
        dict of more scoring details, including
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type

    """
    BaseCfg = CFG()
    BaseCfg.fs = 400
    BaseCfg.bias_thr = 0.15 * BaseCfg.fs
    s_score = np.zeros(
        [
            len(sbp_true),
        ],
        dtype=int,
    )
    v_score = np.zeros(
        [
            len(sbp_true),
        ],
        dtype=int,
    )
    # Scoring
    for i, (s_ref, v_ref, s_pos, v_pos) in enumerate(
        zip(sbp_true, pvc_true, sbp_pred, pvc_pred)
    ):
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        # SBP
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos - ans) <= BaseCfg.bias_thr)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
                    s_fp += len(s_pos_cand) - 1
        # PVC
        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos - ans) <= BaseCfg.bias_thr)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
                    v_fp += len(v_pos_cand) - 1
        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)

        if verbose >= 1:
            print(f"for the {i}-th record")
            print(f"s_tp = {s_tp}, s_fp = {s_fp}, s_fn = {s_fn}")
            print(f"v_tp = {v_tp}, v_fp = {v_fp}, s_fn = {v_fn}")
            print(f"s_score[{i}] = {s_score[i]}, v_score[{i}] = {v_score[i]}")

    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    if verbose >= 1:
        retval = CFG(
            total_loss=-(Score1 + Score2),
            class_loss={"S": -Score1, "V": -Score2},
            true_positive={"S": s_tp, "V": v_tp},
            false_positive={"S": s_fp, "V": v_fp},
            false_negative={"S": s_fn, "V": v_fn},
        )
    else:
        retval = Score1, Score2

    return retval
