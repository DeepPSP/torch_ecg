# -*- coding: utf-8 -*-
"""
QT Database
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union  # noqa: F401

import numpy as np
import pandas as pd  # noqa: F401
import wfdb

from ...cfg import CFG
from ...utils.misc import add_docstring
from ...utils.utils_data import ECGWaveForm
from ..base import (
    PhysioNetDataBase,
    DataBaseInfo,
    BeatAnn,
    WFDB_Beat_Annotations,
    WFDB_Non_Beat_Annotations,
)


__all__ = [
    "QTDB",
]


_QTDB_INFO = DataBaseInfo(
    title="""
    QT Database
    """,
    about="""
    1. The QT Database includes ECGs which were chosen to represent a wide variety of QRS and ST-T morphologies
    2. Recordings were chosen chosen from the MIT-BIH Arrhythmia Database (MITDB), the European Society of Cardiology ST-T Database (EDB), and several other ECG databases collected at Boston's Beth Israel Deaconess Medical Center (MIT-BIH ST Change Database, MIT-BIH Supraventricular Arrhythmia Database, MIT-BIH Normal Sinus Rhythm Database, MIT-BIH Long-Term ECG Database, ``sudden death'' patients from BIH)
    2. Contains 105 fifteen-minute two-lead ECG recordings
    3. Contains onset, peak, and end markers for P, QRS, T, and (where present) U waves of from 30 to 50 selected beats in each recording
    4. Annotation file table:
        | Suffix | Meaning |
        | ------ | ------- |
        | .atr   | reference beat annotations from original database (not available for the 24 sudden death records) |
        | .man:  | reference beat annotations for selected beats only |
        | .q1c:  | manually determined waveform boundary measurements for selected beats (annotator 1 only -- second pass) |
        | .q2c:  | manually determined waveform boundary measurements for selected beats (annotator 2 only -- second pass; available for only 11 records) |
        | .qt1:  | manually determined waveform boundary measurements for selected beats (annotator 1 only -- first pass) |
        | .qt2:  | manually determined waveform boundary measurements for selected beats (annotator 2 only -- first pass; available for only 11 records) |
        | .pu:   | automatically determined waveform boundary measurements for all beats (based on both signals) |
        | .pu0:  | automatically determined waveform boundary measurements for all beats (based on signal 0 only) |
        | .pu1:  | automatically determined waveform boundary measurements for all beats (based on signal 1 only) |
    5. A part of the recordings have rhythm annotations, ST change (elevation or depression) annotations, all of which have .atr annotation files. These annotations are provided in the `aux_note` attribute of the annotation object.
    6. In the first pass manual wave delineation annotation files (.qt1, .qt2 files), fiducial points were marked by a "|" symbol, along with beat annotations (one of "A", "B", "N", "Q", "V") inherited from corresponding .man files.
    7. In the second pass manual wave delineation annotation files (.q1c, .q2c files), the final manual annotations are recorded, with the regular annotation symbols "(" ,")", "t", "p", and "u", and with annotations inherited from the .qt1, .qt2 files.
    8. The .pu0, .pu1 files contain the automatic waveform onsets and ends in signals 0 and 1 respectively, as detected using the differentiated threshold method by ecgpuwave. In the num fields of the pu* annotations, ecgpuwave classifies the T waves as normal (0), inverted (1), only upwards (2), only downwards (3), biphasic negative-positive (4), or biphasic positive-negative (5). Waveform onset ``('' and offset ``)'' annotations specify the waveform type in their num fields (0 for a P-wave, 1 for a QRS complex, 2 for a T wave, or 3 for a U-wave).
    """,
    usage=[
        "ECG wave delineation",
        "ST segment",
    ],
    references=[
        "https://physionet.org/content/qtdb/1.0.0/",
        "Laguna P, Mark RG, Goldberger AL, Moody GB. A Database for Evaluation of Algorithms for Measurement of QT and Other Waveform Intervals in the ECG. Computers in Cardiology 24:673-676 (1997).",
    ],
    issues="""
    1. According to the paper of the database, there should be .ari files containing QRS annotations obtained automatically by ARISTOTLE, which however are not available in the database.
    2. A large proportion of the wave delineation annotations lack onset indices (the T waves and U waves).
    """,
    doi=[
        "10.1109/cic.1997.648140",
        "10.13026/C24K53",
    ],
)


@add_docstring(_QTDB_INFO.format_database_docstring())
class QTDB(PhysioNetDataBase):
    """ """

    __name__ = "QTDB"

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
            storage path of the database
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="qtdb",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = 250
        self.spacing = 1000 / self.fs
        self.data_ext = "dat"

        # fmt: off
        self.all_extensions = ["atr", "man", "q1c", "q2c", "qt1", "qt2", "pu", "pu0", "pu1"]
        """
        1. .atr:    reference beat annotations from original database (not available in all cases)
        2. .man:    reference beat annotations for selected beats only
        3. .q1c:    manually determined waveform boundary measurements for selected beats (annotator 1 only -- second pass)
        4. .q2c:    manually determined waveform boundary measurements for selected beats (annotator 2 only -- second pass; available for only 11 records)
        5. .q1t:    manually determined waveform boundary measurements for selected beats (annotator 1 only -- first pass)
        6. .q2t:    manually determined waveform boundary measurements for selected beats (annotator 2 only -- first pass; available for only 11 records)
        7. .pu:     automatically determined waveform boundary measurements for all beats (based on both signals)
        8. .pu0:    automatically determined waveform boundary measurements for all beats (based on signal 0 only)
        9. .pu1:    automatically determined waveform boundary measurements for all beats (based on signal 1 only)
        """

        # records have different lead names
        # therefore, self.all_leads should not be set
        # otherwise, it will cause problems when loading data using `self.load_data`
        self._all_leads = [
            "CC5", "CM2", "CM4", "CM5", "D3", "D4", "ECG1", "ECG2", "ML5", "MLII",
            "V1", "V1-V2", "V2", "V2-V3", "V3", "V4", "V4-V5", "V5", "mod.V1",
        ]
        self.all_annotations = ["(", ")", "N", "t", "p"]
        # fmt: on

        self.beat_types_extended = list("""~"+/AFJNQRSTVaefjs|""")
        self.nonbeat_types = [
            item
            for item in self.beat_types_extended
            if item in WFDB_Non_Beat_Annotations
        ]
        self.beat_types = [
            item for item in self.beat_types_extended if item in WFDB_Beat_Annotations
        ]
        self.beat_types_map = {item: i for i, item in enumerate(self.beat_types)}
        self.beat_types_extended_map = {
            item: i for i, item in enumerate(self.beat_types_extended)
        }

        self.class_map = CFG(p=1, N=2, t=3, i=0)  # an extra isoelectric

        self._ls_rec()

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """
        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        sid: int,
            the `subject_id` corr. to `rec`

        """
        raise NotImplementedError

    def get_lead_names(self, rec: Union[str, int]) -> List[str]:
        """
        get the lead names of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`

        Returns
        -------
        list of str,
            list of the lead names

        """
        return wfdb.rdheader(str(self.get_absolute_path(rec))).sig_name

    def load_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        ignore_beat_types: bool = True,
        extension: str = "q1c",
    ) -> List[ECGWaveForm]:
        """
        load the wave delineation in the form of list of `ECGWaveForm`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
        ignore_beat_types: bool, default True,
            if True, ignore the beat types (all converted to "N") in the annotation file
        extension: str, default "q1c",
            the extension of the wave delineation file

        Returns
        -------
        wave_list: list of `ECGWaveForm`,
            the list of wave delineation in the form of `ECGWaveForm`

        """
        assert extension in [
            "q1c",
            "q2c",
            "pu1",
            "pu2",
        ], "extension should be one of `q1c`, `q2c`, `pu1`, `pu2`"
        fp = str(self.get_absolute_path(rec))
        wfdb_ann = wfdb.rdann(fp, extension=extension)
        header = wfdb.rdheader(fp)
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        subtraction = 0 if keep_original else sf

        wave_list = []
        current_onset = None
        current_wave_name = None
        current_wave_peak = None
        for idx, symbol in zip(wfdb_ann.sample, wfdb_ann.symbol):
            if idx < sf:
                continue
            if idx >= st:
                break
            if symbol == "(":
                current_onset = idx
            elif symbol == ")":
                wave_list.append(
                    ECGWaveForm(
                        onset=(current_onset or np.nan) - subtraction,
                        offset=idx - subtraction,
                        name=current_wave_name,
                        peak=current_wave_peak,
                        duration=(idx - current_onset) / header.fs
                        if current_onset is not None
                        else np.nan,
                    )
                )
                current_onset = None
                current_wave_name = None
                current_wave_peak = None
            else:
                if ignore_beat_types and symbol not in ["p", "t", "u"]:
                    symbol = "N"
                current_wave_name = symbol
                current_wave_peak = idx

        return wave_list

    @add_docstring(load_ann.__doc__)
    def load_wave_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        ignore_beat_types: bool = True,
        extension: str = "q1c",
    ) -> np.ndarray:
        """alias of self.load_ann"""
        return self.load_ann(
            rec,
            sampfrom=sampfrom,
            sampto=sampto,
            keep_original=keep_original,
            ignore_beat_types=ignore_beat_types,
            extension=extension,
        )

    def load_wave_masks(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        mask_format: str = "channel_first",
        class_map: Optional[Dict[str, int]] = None,
        extension: str = "q1c",
    ) -> np.ndarray:
        """
        load the wave delineation in the form of list of `ECGWaveForm`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        mask_format: str, default "channel_first",
            format of the mask,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        class_map: dict, optional,
            custom class map,
            if not set, `self.class_map` will be used
        extension: str, default "q1c",
            the extension of the wave delineation file

        Returns
        -------
        masks: ndarray,
            the masks corresponding to the wave delineation annotations of `rec`

        """
        raise NotImplementedError(
            "A large proportion of the wave delineation annotations lack onset indices. "
            "Has to find a rule to give default onset index for the missing ones."
        )

    def load_rhythm_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        rhythm_format: str = "intervals",
        rhythm_types: Optional[Sequence[str]] = None,
        keep_original: bool = False,
        extension: str = "atr",
    ) -> Union[Dict[str, list], np.ndarray]:
        """
        load rhythm annotations,
        which are stored in the `aux_note` attribute of corresponding annotation files.

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        rhythm_format: str, default "intervals", case insensitive,
            format of returned annotation, can also be "mask"
        rhythm_types: list of str, optional,
            defaults to `self.rhythm_types`
            if not None, only the rhythm annotations with the specified types will be returned
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
        extension: str, default "atr",
            the extension of the annotation file,
            has to be "atr", since "man" files has no rhythm annotation

        Returns
        -------
        ann, dict or ndarray,
            the annotations in the format of intervals, or in the format of mask

        """
        raise NotImplementedError(
            "Only a small part of the recordings have rhythm annotations, "
            "hence not implemented yet"
        )

    def load_beat_ann(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        beat_format: str = "beat",
        beat_types: Optional[Sequence[str]] = None,
        keep_original: bool = False,
        extension: str = "atr",
    ) -> Union[Dict[str, np.ndarray], List[BeatAnn]]:
        """
        load beat annotations,
        which are stored in the `symbol` attribute of corresponding annotation files

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        beat_format: str, default "beat", case insensitive,
            format of returned annotation, can also be "dict"
        beat_types: list of str, optional,
            defaults to `self.beat_types`
            if not None, only the beat annotations with the specified types will be returned
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
        extension: str, default "atr",
            the extension of the annotation file, can also be "man"

        Returns
        -------
        beat_ann: dict or list,
            locations (indices) of the all the beat types

        """
        assert beat_format.lower() in [
            "beat",
            "dict",
        ], f"`beat_format` must be one of ['beat', 'dict'], but got `{beat_format}`"
        fp = str(self.get_absolute_path(rec))
        wfdb_ann = wfdb.rdann(fp, extension=extension)
        header = wfdb.rdheader(fp)
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"
        subs = 0 if keep_original else sf

        sample_inds = wfdb_ann.sample
        indices = np.where((sample_inds >= sf) & (sample_inds < st))[0]

        if beat_types is None:
            beat_types = self.beat_types

        beat_ann = [
            BeatAnn(i - subs, s)
            for i, s in zip(sample_inds[indices], np.array(wfdb_ann.symbol)[indices])
            if s in beat_types
        ]

        if beat_format.lower() == "dict":
            beat_ann = {
                s: np.array([b.index for b in beat_ann if b.symbol == s], dtype=int)
                for s in self.beat_types_extended
            }
            beat_ann = {k: v for k, v in beat_ann.items() if len(v) > 0}

        return beat_ann

    def load_rpeak_indices(
        self,
        rec: Union[str, int],
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        keep_original: bool = False,
        extension: str = "atr",
    ) -> np.ndarray:
        """
        load rpeak indices, or equivalently qrs complex locations,
        which are stored in the `symbol` attribute of corresponding annotation files,
        regardless of their beat types,

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        sampfrom: int, optional,
            start index of the annotations to be loaded
        sampto: int, optional,
            end index of the annotations to be loaded
        keep_original: bool, default False,
            if True, indices will keep the same with the annotation file
            otherwise subtract `sampfrom` if specified
        extension: str, default "atr",
            the extension of the annotation file, can also be "man"

        Returns
        -------
        rpeak_inds: ndarray,
            locations (indices) of the all the rpeaks (qrs complexes)

        """
        assert extension in [
            "atr",
            "man",
        ], f"`extension` must be one of ['atr', 'man'], but got `{extension}`"
        if isinstance(rec, int):
            rec = self[rec]
        rec_fp = self.get_absolute_path(rec)
        if not rec_fp.with_suffix(f".{extension}").exists():
            another_extension = "man" if extension == "atr" else "atr"
            raise FileNotFoundError(
                f"annotation file `{rec_fp.name}` does not exist, "
                f"try setting `extension = \042{another_extension}\042`"
            )
        wfdb_ann = wfdb.rdann(str(rec_fp), extension=extension)
        header = wfdb.rdheader(str(rec_fp))
        sig_len = header.sig_len
        sf = sampfrom or 0
        st = sampto or sig_len
        assert st > sf, "`sampto` should be greater than `sampfrom`!"

        rpeak_inds = wfdb_ann.sample
        indices = np.where(
            (rpeak_inds >= sf)
            & (rpeak_inds < st)
            & (np.isin(wfdb_ann.symbol, self.beat_types))
        )[0]
        rpeak_inds = rpeak_inds[indices]
        if not keep_original:
            rpeak_inds -= sf
        return rpeak_inds

    def plot(
        self,
        rec: Union[str, int],
        data: Optional[np.ndarray] = None,
        ticks_granularity: int = 0,
        leads: Optional[Union[str, int, List[str], List[int]]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        same_range: bool = False,
        waves: Optional[ECGWaveForm] = None,
        beat_ann: Optional[Dict[str, np.ndarray]] = None,
        rpeak_inds: Optional[Union[Sequence[int], np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """
        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        data: ndarray, optional,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or int or list of str or list of int, optional,
            the leads to plot
        sampfrom: int, optional,
            start index of the record to plot
        sampto: int, optional,
            end index of the record to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
        waves: ECGWaveForm, optional,
            the waves (p waves, t waves, qrs complexes, etc.)
        beat_ann: dict, optional,
            the beat annotations
        rpeak_inds: ndarray or list of int, optional,
            the rpeak indices
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
        # _leads = self._normalize_leads(leads, standard_ordering=True, lower_cases=False)

        raise NotImplementedError

    @property
    def database_info(self) -> DataBaseInfo:
        return _QTDB_INFO
