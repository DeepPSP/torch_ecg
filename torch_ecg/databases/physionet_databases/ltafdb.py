# -*- coding: utf-8 -*-
"""
"""

import json
import math
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Union

import numpy as np
import wfdb
from scipy.signal import resample_poly

from ...cfg import CFG
from ...utils.utils_interval import generalized_intervals_intersection
from ..base import DEFAULT_FIG_SIZE_PER_SEC, BeatAnn, PhysioNetDataBase

__all__ = [
    "LTAFDB",
]


class LTAFDB(PhysioNetDataBase):
    """

    Long Term AF Database

    ABOUT ltafdb
    ------------
    1. contains 84 long-term ECG recordings of subjects with paroxysmal or sustained atrial fibrillation
    2. each record contains two simultaneously recorded ECG signals digitized at 128 Hz
    3. records have duration 24 - 25 hours
    4. qrs annotations (.qrs files) were produced by an automated QRS detector, in which detected beats (including occasional ventricular ectopic beats) are labelled "N", detected artifacts are labelled "|", and AF terminations are labelled "T" (inserted manually)
    5. atr annotations (.atr files) were obtained by manual review of the output of an automated ECG analysis system; in these annotation files, all detected beats are labelled by type ('"', "+", "A", "N", "Q", "V"), and rhythm changes ("\x01 Aux", "(AB", "(AFIB", "(B", "(IVR", "(N", "(SBR", "(SVTA", "(T", "(VT", "M", "MB", "MISSB", "PSE") are also annotated

    NOTE
    ----
    1. both channels of the signals have name "ECG"
    2. the automatically generated qrs annotations (.qrs files) contains NO rhythm annotations
    3. `aux_note` of .atr files of all but one ("64") record start with valid rhythms, all but one end with "" ("30" ends with "\x01 Aux")
    4. for more statistics on the whole database, see ref. [3]

    ISSUES
    ------

    Usage
    -----
    1. AF detection
    2. (3 or 4) beat type classification
    3. rhythm classification

    References
    ----------
    1. <a name="ref1"></a> https://physionet.org/content/ltafdb/1.0.0/
    2. <a name="ref2"></a> Petrutiu S, Sahakian AV, Swiryn S. Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans. Europace 9:466-470 (2007).
    3. <a name="ref3"></a> https://physionet.org/files/ltafdb/1.0.0/tables.shtml

    """

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        db_dir: str or Path, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        from matplotlib.pyplot import cm

        super().__init__(
            db_name="ltafdb",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 128
        self.data_ext = "dat"
        self.auto_ann_ext = "qrs"
        self.manual_ann_ext = "atr"
        self.all_leads = [0, 1]
        self._ls_rec()

        self.rhythm_types = [
            "(N",
            "(AB",
            "(AFIB",
            "(B",
            "(IVR",
            "(SBR",
            "(SVTA",
            "(T",
            "(VT",
            "NOISE",  # additional, since head of each record are noisy
        ]  # others include "\x01 Aux", "M", "MB", "MISSB", "PSE"
        self.rhythm_types_map = CFG(
            {k.replace("(", ""): idx for idx, k in enumerate(self.rhythm_types)}
        )
        self.palette = kwargs.get("palette", None)
        if self.palette is None:
            n_colors = len(
                [k for k in self.rhythm_types_map.keys() if k not in ["N", "NOISE"]]
            )
            colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))
            self.palette = CFG()
            for k in self.rhythm_types_map.keys():
                if k in ["N", "NOISE"]:
                    continue
                self.palette[k] = next(colors)

        self.beat_types = [
            "A",
            "N",
            "Q",
            "V",
            # '"', "+", are not beat types
        ]
        self.palette["qrs"] = "green"

    def get_subject_id(self, rec: str) -> int:
        """NOT finished,

        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        sid: int,
            the `get_subject_id` corr. to `rec`

        """
        raise NotImplementedError

    def load_data(
        self,
        rec: Union[str, int],
        leads: Optional[Union[int, List[int]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: str = "mV",
        fs: Optional[Real] = None,
    ) -> np.ndarray:
        """

        load physical (converted from digital) ECG data,
        which is more understandable for humans

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        leads: int or list of int, optional,
            the lead number(s) to load
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        data_format: str, default "channel_first",
            format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency

        Returns
        -------
        data: ndarray,
            the ECG data

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = self.db_dir / rec
        if not leads:
            _leads = self.all_leads
        elif isinstance(leads, int):
            _leads = [leads]
        else:
            _leads = leads
        assert set(_leads).issubset(self.all_leads)
        # p_signal in the format of "lead_last", and in units "mV"
        data = wfdb.rdrecord(
            str(fp),
            sampfrom=sampfrom or 0,
            sampto=sampto,
            physical=True,
            channels=_leads,
        ).p_signal
        if units.lower() in ["μv", "uv"]:
            data = 1000 * data
        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=0)
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data.T
        return data

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "interval",
        beat_format: str = "beat",
        keep_original: bool = False,
    ) -> dict:
        """

        load rhythm and beat annotations,
        which are stored in the `aux_note`, `symbol` attributes of corresponding annotation files.
        NOTE that qrs annotations (.qrs files) do NOT contain any rhythm annotations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        rhythm_format: str, default "interval", case insensitive,
            format of returned annotation, can also be "mask"
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict,
            the annotations of `rhythm` and `beat`, with
            `rhythm` annotatoins in the format of `intervals`, or `mask`;
            `beat` annotations in the format of `dict` or `BeatAnn`

        NOTE that at head and tail of the record, segments named "NOISE" are added

        """
        if isinstance(rec, int):
            rec = self[rec]
        ann = {
            "beat": self.load_beat_ann(
                rec,
                sampfrom,
                sampto,
                beat_format,
                keep_original,
            ),
            "rhythm": self.load_rhythm_ann(
                rec,
                sampfrom,
                sampto,
                rhythm_format,
                keep_original,
            ),
        }
        return ann

    def load_rhythm_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "interval",
        keep_original: bool = False,
    ) -> Union[Dict[str, list], np.ndarray]:
        """

        load rhythm annotations,
        which are stored in the `aux_note` attribute of corresponding annotation files.
        NOTE that qrs annotations (.qrs files) do NOT contain any rhythm annotations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        rhythm_format: str, default "interval", case insensitive,
            format of returned annotation, can also be "mask"
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict or ndarray,
            the annotations in the format of intervals, or in the format of mask

        NOTE that at head and tail of the record, segments named "NOISE" are added

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert rhythm_format.lower() in [
            "interval",
            "mask",
        ], f"rhythm_format must be 'interval' or 'mask', got {rhythm_format}"
        fp = self.db_dir / rec
        header = wfdb.rdheader(str(fp))
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        simplified_fp = self.db_dir / f"{rec}_ann.json"
        if simplified_fp.is_file():
            ann = CFG(json.loads(simplified_fp.read_text()))
        else:
            wfdb_ann = wfdb.rdann(str(fp), extension=self.manual_ann_ext)

            ann = CFG({k: [] for k in self.rhythm_types_map.keys()})
            critical_points = wfdb_ann.sample.tolist()
            aux_note = wfdb_ann.aux_note
            start = 0
            current_rhythm = "NOISE"
            for idx, rhythm in zip(critical_points, aux_note):
                if rhythm not in self.rhythm_types:
                    continue
                ann[current_rhythm].append([start, idx])
                current_rhythm = rhythm.replace("(", "")
                start = idx
            # all but one end with "" ("30" ends with "\x01 Aux")
            # i.e. none ends with (start of) valid rhythm
            ann[current_rhythm].append([start, critical_points[-1]])
            ann["NOISE"].append([critical_points[-1], sig_len])

            simplified_fp.write_text(json.dumps(ann, ensure_ascii=False))

        ann = CFG(
            {
                k: generalized_intervals_intersection(l_itv, [[sf, st]])
                for k, l_itv in ann.items()
            }
        )
        ann = CFG({k: l_itv for k, l_itv in ann.items() if len(l_itv) > 0})
        if rhythm_format.lower() == "mask":
            tmp = deepcopy(ann)
            ann = np.full(
                shape=(st - sf,), fill_value=self.rhythm_types_map.N, dtype=int
            )
            for rhythm, l_itv in tmp.items():
                for itv in l_itv:
                    ann[itv[0] - sf : itv[1] - sf] = self.rhythm_types_map[rhythm]
        elif not keep_original:
            for k, l_itv in ann.items():
                ann[k] = [[itv[0] - sf, itv[1] - sf] for itv in l_itv]

        return ann

    def load_beat_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        beat_format: str = "beat",
        keep_original: bool = False,
    ) -> Union[Dict[str, np.ndarray], List[BeatAnn]]:
        """

        load beat annotations,
        which are stored in the `symbol` attribute of corresponding annotation files

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        ann, dict or list,
            locations (indices) of the all the beat types ("A", "N", "Q", "V",)

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert beat_format.lower() in [
            "beat",
            "dict",
        ], f"beat_format must be 'beat' or 'dict', got {beat_format}"
        fp = self.db_dir / rec
        header = wfdb.rdheader(str(fp))
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        wfdb_ann = wfdb.rdann(
            str(fp),
            extension=self.manual_ann_ext,
            sampfrom=sf,
            sampto=sampto,
        )
        ann = CFG({k: [] for k in self.beat_types})
        for idx, bt in zip(wfdb_ann.sample, wfdb_ann.symbol):
            if bt not in self.beat_types:
                continue
            ann[bt].append(idx)
        if not keep_original and sampfrom is not None:
            ann = CFG({k: np.array(v, dtype=int) - sf for k, v in ann.items()})
        else:
            ann = CFG({k: np.array(v, dtype=int) for k, v in ann.items()})

        if beat_format.lower() == "beat":
            ann = [BeatAnn(i, s) for s, l in ann.items() for i in l]
        return ann

    def load_rpeak_indices(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        use_manual: bool = True,
        keep_original: bool = False,
    ) -> np.ndarray:
        """

        load rpeak indices, or equivalently qrs complex locations,
        which are stored in the `symbol` attribute of corresponding annotation files,
        regardless of their beat types,

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        use_manual: bool, default True,
            use manually annotated beat annotations (qrs),
            instead of those generated by algorithms
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified

        Returns
        -------
        rpeak_inds, ndarray,
            locations (indices) of the all the rpeaks (qrs complexes)

        """
        if isinstance(rec, int):
            rec = self[rec]
        fp = self.db_dir / rec
        if use_manual:
            ext = self.manual_ann_ext
        else:
            ext = self.auto_ann_ext
        wfdb_ann = wfdb.rdann(
            str(fp),
            extension=ext,
            sampfrom=sampfrom or 0,
            sampto=sampto,
        )
        rpeak_inds = wfdb_ann.sample[np.isin(wfdb_ann.symbol, self.beat_types)]
        if not keep_original and sampfrom is not None:
            rpeak_inds = rpeak_inds - sampfrom
        return rpeak_inds

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ann: Optional[Dict[str, np.ndarray]] = None,
        beat_ann: Optional[Dict[str, np.ndarray]] = None,
        rpeak_inds: Optional[Union[Sequence[int], np.ndarray]] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[int, List[int]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        same_range: bool = False,
        **kwargs: Any,
    ) -> NoReturn:
        """

        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str or int,
            name or index of the record
        data: ndarray, optional,
            (2-lead) ECG signal to plot,
            should be of the format "channel_first", and compatible with `leads`
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            rhythm annotations for `data`, covering those from annotation files,
            in the form of {k: l_itv, ...},
            where `k` in `self.rhythm_types_map.keys()`,
            and `l_itv` in the form of [[a,b], ...],
            ignored if `data` is None
        beat_ann: dict, optional,
            beat annotations for `data`, covering those from annotation files,
            in the form of {k: l_inds, ...},
            where `k` in `self.beat_types`, and `l_inds` array of indices,
            ignored if `data` is None
        rpeak_inds: array_like, optional,
            indices of R peaks, covering those from annotation files,
            if `data` is None, then indices should be the absolute indices in the record,
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: int or list of int, optional,
            the lead number(s) to plot
        sampfrom: int, optional,
            start index of the data to plot
        sampto: int, optional,
            end index of the data to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
        kwargs: dict,
            additional arguments to be passed to `matplotlib.pyplot.plot`, etc.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if "plt" not in dir():
            import matplotlib.pyplot as plt

            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == "all":
            _leads = self.all_leads
        elif isinstance(leads, int):
            _leads = [leads]
        else:
            _leads = leads
        assert all([ld in self.all_leads for ld in _leads])

        lead_indices = [self.all_leads.index(ld) for ld in _leads]
        if data is None:
            _data = self.load_data(
                rec,
                leads=_leads,
                sampfrom=sampfrom,
                sampto=sampto,
                data_format="channel_first",
                units="μV",
            )
        else:
            units = self._auto_infer_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
            _leads = list(range(_data.shape[0]))
        if ann is None and data is None:
            _ann = self.load_ann(
                rec,
                sampfrom=sampfrom,
                sampto=sampto,
                ann_format="interval",
                keep_original=False,
            )
        else:
            _ann = ann or CFG({k: [] for k in self.rhythm_types_map.keys()})
        # indices to time
        _ann = {
            k: [[itv[0] / self.fs, itv[1] / self.fs] for itv in l_itv]
            for k, l_itv in _ann.items()
        }
        if rpeak_inds is None and data is None:
            _rpeak = self.load_rpeak_indices(
                rec,
                sampfrom=sampfrom,
                sampto=sampto,
                use_manual=True,
                keep_original=False,
            )
            _rpeak = _rpeak / self.fs  # indices to time
        else:
            _rpeak = np.array(rpeak_inds or []) / self.fs  # indices to time
        if beat_ann is None and data is None:
            _beat_ann = self.load_beat_ann(
                rec, sampfrom=sampfrom, sampto=sampto, keep_original=False
            )
        else:
            _beat_ann = beat_ann or CFG({k: [] for k in self.beat_types})
        _beat_ann = {  # indices to time
            k: [i / self.fs for i in l_inds] for k, l_inds in _beat_ann.items()
        }

        ann_plot_alpha = 0.2
        rpeaks_plot_alpha = 0.8

        nb_leads = len(_leads)

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(_data.shape[1] / line_len)

        for seg_idx in range(nb_lines):
            seg_data = _data[..., seg_idx * line_len : (seg_idx + 1) * line_len]
            secs = (np.arange(seg_data.shape[1]) + seg_idx * line_len) / self.fs
            seg_ann = {
                k: generalized_intervals_intersection(l_itv, [[secs[0], secs[-1]]])
                for k, l_itv in _ann.items()
            }
            seg_rpeaks = _rpeak[np.where((_rpeak >= secs[0]) & (_rpeak < secs[-1]))[0]]
            seg_beat_ann = {
                k: [i for i in l_inds if secs[0] <= i <= secs[-1]]
                for k, l_inds in _beat_ann.items()
            }
            fig_sz_w = int(
                round(DEFAULT_FIG_SIZE_PER_SEC * seg_data.shape[1] / self.fs)
            )
            if same_range:
                y_ranges = (
                    np.ones((seg_data.shape[0],)) * np.max(np.abs(seg_data)) + 100
                )
            else:
                y_ranges = np.max(np.abs(seg_data), axis=1) + 100
            fig_sz_h = 6 * y_ranges / 1500
            fig, axes = plt.subplots(
                nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h))
            )
            if nb_leads == 1:
                axes = [axes]
            for idx in range(nb_leads):
                axes[idx].plot(
                    secs, seg_data[idx], color="black", label=f"lead - {_leads[idx]}"
                )
                axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
                # NOTE that `Locator` has default `MAXTICKS` equal to 1000
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
                for k, l_itv in seg_ann.items():
                    if k in ["N", "NOISE"]:
                        continue
                    for itv in l_itv:
                        axes[idx].axvspan(
                            itv[0],
                            itv[1],
                            color=self.palette[k],
                            alpha=ann_plot_alpha,
                            label=k,
                        )
                for ri in seg_rpeaks:
                    axes[idx].axvspan(
                        ri - 0.01,
                        ri + 0.01,
                        color=self.palette["qrs"],
                        alpha=rpeaks_plot_alpha,
                    )
                for k, l_t in seg_beat_ann.items():
                    for t in l_t:
                        x_pos = t + 0.05 if t + 0.05 < secs[-1] else t - 0.15
                        axes[idx].text(
                            x_pos, 0.65 * y_ranges[idx], k, color="black", fontsize=16
                        )
                axes[idx].legend(loc="upper left")
                axes[idx].set_xlim(secs[0], secs[-1])
                axes[idx].set_ylim(-y_ranges[idx], y_ranges[idx])
                axes[idx].set_xlabel("Time [s]")
                axes[idx].set_ylabel("Voltage [μV]")
            plt.subplots_adjust(hspace=0.2)
            plt.show()
