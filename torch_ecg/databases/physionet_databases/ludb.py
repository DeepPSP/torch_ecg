# -*- coding: utf-8 -*-
"""
"""
import os
import json
from datetime import datetime
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import wfdb

from ...cfg import CFG
from ..base import PhysioNetDataBase, ECGWaveForm


__all__ = [
    "LUDB",
]


class LUDB(PhysioNetDataBase):
    """ finished, to be improved,

    Lobachevsky University Electrocardiography Database

    ABOUT ludb
    ----------
    1. consist of 200 10-second conventional 12-lead (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6) ECG signal records, with sampling frequency 500 Hz
    2. boundaries of P, T waves and QRS complexes were manually annotated by cardiologists, and with the corresponding diagnosis
    3. annotated are 16797 P waves, 21966 QRS complexes, 19666 T waves (in total, 58429 annotated waves)
    4. distributions of data:
        4.1. rhythm distribution:
            Rhythms	                        Number of ECGs
            Sinus rhythm	                143
            Sinus tachycardia	            4
            Sinus bradycardia	            25
            Sinus arrhythmia	            8
            Irregular sinus rhythm	        2
            Abnormal rhythm	                19
        4.2. electrical axis distribution:
            Heart electric axis	            Number of ECGs
            Normal	                        75
            Left axis deviation (LAD)	    66
            Vertical	                    26
            Horizontal	                    20
            Right axis deviation (RAD)	    3
            Undetermined	                10
        4.3. distribution of records with conduction abnomalities (totally 79):
            Conduction abnormalities	                        Number of ECGs
            Sinoatrial blockade, undetermined	                1
            I degree AV block	                                10
            III degree AV-block	                                5
            Incomplete right bundle branch block	            29
            Incomplete left bundle branch block	                6
            Left anterior hemiblock	                            16
            Complete right bundle branch block	                4
            Complete left bundle branch block	                4
            Non-specific intravintricular conduction delay	    4
        4.4. distribution of records with extrasystoles (totally 35):
            Extrasystoles	                                                    Number of ECGs
            Atrial extrasystole, undetermined	                                2
            Atrial extrasystole, low atrial	                                    1
            Atrial extrasystole, left atrial	                                2
            Atrial extrasystole, SA-nodal extrasystole	                        3
            Atrial extrasystole, type: single PAC	                            4
            Atrial extrasystole, type: bigemini	                                1
            Atrial extrasystole, type: quadrigemini	                            1
            Atrial extrasystole, type: allorhythmic pattern	                    1
            Ventricular extrasystole, morphology: polymorphic	                2
            Ventricular extrasystole, localisation: RVOT, anterior wall	        3
            Ventricular extrasystole, localisation: RVOT, antero-septal part	1
            Ventricular extrasystole, localisation: IVS, middle part	        1
            Ventricular extrasystole, localisation: LVOT, LVS	                2
            Ventricular extrasystole, localisation: LV, undefined	            1
            Ventricular extrasystole, type: single PVC	                        6
            Ventricular extrasystole, type: intercalary PVC	                    2
            Ventricular extrasystole, type: couplet	                            2
        4.5. distribution of records with hypertrophies (totally 253):
            Hypertrophies	                    Number of ECGs
            Right atrial hypertrophy	        1
            Left atrial hypertrophy	            102
            Right atrial overload	            17
            Left atrial overload	            11
            Left ventricular hypertrophy	    108
            Right ventricular hypertrophy	    3
            Left ventricular overload	        11
        4.6. distribution of records of pacing rhythms (totally 12):
            Cardiac pacing	                Number of ECGs
            UNIpolar atrial pacing	        1
            UNIpolar ventricular pacing	    6
            BIpolar ventricular pacing	    2
            Biventricular pacing	        1
            P-synchrony	                    2
        4.7. distribution of records with ischemia (totally 141):
            Ischemia	                                            Number of ECGs
            STEMI: anterior wall	                                8
            STEMI: lateral wall	                                    7
            STEMI: septal	                                        8
            STEMI: inferior wall	                                1
            STEMI: apical	                                        5
            Ischemia: anterior wall	                                5
            Ischemia: lateral wall	                                8
            Ischemia: septal	                                    4
            Ischemia: inferior wall	                                10
            Ischemia: posterior wall	                            2
            Ischemia: apical	                                    6
            Scar formation: lateral wall	                        3
            Scar formation: septal	                                9
            Scar formation: inferior wall	                        3
            Scar formation: posterior wall	                        6
            Scar formation: apical	                                5
            Undefined ischemia/scar/supp.NSTEMI: anterior wall	    12
            Undefined ischemia/scar/supp.NSTEMI: lateral wall	    16
            Undefined ischemia/scar/supp.NSTEMI: septal	            5
            Undefined ischemia/scar/supp.NSTEMI: inferior wall	    3
            Undefined ischemia/scar/supp.NSTEMI: posterior wall	    4
            Undefined ischemia/scar/supp.NSTEMI: apical	            11
        4.8. distribution of records with non-specific repolarization abnormalities (totally 85):
            Non-specific repolarization abnormalities	    Number of ECGs
            Anterior wall	                                18
            Lateral wall	                                13
            Septal	                                        15
            Inferior wall	                                19
            Posterior wall	                                9
            Apical	                                        11
        4.9. there are also 9 records with early repolarization syndrome
        there might well be records with multiple conditions.
    5. ludb.csv stores information about the subjects (gender, age, rhythm type, direction of the electrical axis of the heart, the presence of a cardiac pacemaker, etc.)
    

    NOTE
    ----

    ISSUES
    ------
    1. (version 1.0.0, fixed in version 1.0.1) ADC gain might be wrong, either `units` should be μV, or `adc_gain` should be 1000 times larger

    Usage
    -----
    1. ECG wave delineation
    2. ECG arrhythmia classification

    References
    ----------
    [1] https://physionet.org/content/ludb/1.0.1/
    [2] Kalyakulina, A., Yusipov, I., Moskalenko, V., Nikolskiy, A., Kozlov, A., Kosonogov, K., Zolotykh, N., & Ivanchenko, M. (2020). Lobachevsky University Electrocardiography Database (version 1.0.0).
    """

    def __init__(self,
                 db_dir:str,
                 working_dir:Optional[str]=None,
                 verbose:int=2,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,
        
        Parameters
        ----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(db_name="ludb", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        print("Version 1.0.0 has bugs, make sure that version 1.0.1 or higher is used")
        self.fs = 500
        self.spacing = 1000 / self.fs
        self.data_ext = "dat"
        self.all_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]
        self.all_leads_lower = [l.lower() for l in self.all_leads]
        # Version 1.0.0 has different beat_ann_ext: [f"atr_{item}" for item in self.all_leads_lower]
        self.beat_ann_ext = [f"{item}" for item in self.all_leads_lower]

        self._all_symbols = ["(", ")", "p", "N", "t"]
        """
        this can be obtained using the following code:
        >>> data_gen = LUDB(db_dir="/home/wenhao71/data/PhysioNet/ludb/1.0.0/")
        >>> all_symbols = set()
        >>> for rec in data_gen.all_records:
        ...     for ext in data_gen.beat_ann_ext:
        ...         ann = wfdb.rdann(data_gen._get_path(rec), extension=ext)
        ...         all_symbols.update(ann.symbol)
        """
        self._symbol_to_wavename = CFG(N="qrs", p="pwave", t="twave")
        self._wavename_to_symbol = CFG({v:k for k,v in self._symbol_to_wavename.items()})
        self.class_map = CFG(
            p=1, N=2, t=3, i=0  # an extra isoelectric
        )

        if os.path.isfile(os.path.join(self.db_dir, "ludb.csv")):
            # newly added in version 1.0.1
            self._df_subject_info = pd.read_csv(os.path.join(self.db_dir, "ludb.csv"))
            self._df_subject_info.ID = self._df_subject_info.ID.apply(str)
            self._df_subject_info.Sex = self._df_subject_info.Sex.apply(lambda s: s.strip())
            self._df_subject_info.Age = self._df_subject_info.Age.apply(lambda s: s.strip())
        else:
            self._df_subject_info = pd.DataFrame(columns=[
                "ID", "Sex", "Age", "Rhythms", "Electric axis of the heart",
                "Conduction abnormalities", "Extrasystolies", "Hypertrophies",
                "Cardiac pacing", "Ischemia", "Non-specific repolarization abnormalities",
                "Other states",
            ])

        self._ls_rec()

    def get_subject_id(self, rec:str) -> int:
        """

        """
        raise NotImplementedError

    def _get_path(self, rec:str) -> str:
        """ finished, checked,

        get the path of a record, without file extension

        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        rec_fp: str,
            path of the record, without file extension
        """
        rec_fp = os.path.join(self.db_dir, "data", rec)
        return rec_fp

    def load_data(self,
                  rec:str,
                  leads:Optional[Union[str, List[str]]]=None,
                  data_format="channel_first",
                  units:str="mV",
                  fs:Optional[Real]=None) -> np.ndarray:
        """ finished, checked,

        load physical (converted from digital) ecg data,
        which is more understandable for humans

        Parameters
        ----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default "channel_first",
            format of the ecg data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first", original)
        units: str, default "mV",
            units of the output signal, can also be "μV", with an alias of "uV"
        fs: real number, optional,
            if not None, the loaded data will be resampled to this frequency
        
        Returns
        -------
        data: ndarray,
            the ecg data
        """
        assert data_format.lower() in ["channel_first", "lead_first", "channel_last", "lead_last"]
        _leads = self._normalize_leads(leads, standard_ordering=True, lower_cases=True)

        rec_fp = self._get_path(rec)
        wfdb_rec = wfdb.rdrecord(rec_fp, physical=True, channel_names=_leads)
        # p_signal of "lead_last" format
        # ref. ISSUES 1. (fixed in version 1.0.1)
        # data = np.asarray(wfdb_rec.p_signal.T / 1000, dtype=np.float64)
        data = np.asarray(wfdb_rec.p_signal.T, dtype=np.float64)

        if units.lower() in ["uv", "μv"]:
            data = data * 1000

        if fs is not None and fs != self.fs:
            data = resample_poly(data, fs, self.fs, axis=1)

        if data_format.lower() in ["channel_last", "lead_last"]:
            data = data.T

        return data

    def load_ann(self, rec:str, leads:Optional[Sequence[str]]=None, metadata:bool=False) -> dict:
        """ finished, checked,

        load the wave delineation, along with metadata if specified

        Parameters
        ----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        metadata: bool, default False,
            if True, load metadata from corresponding head file

        Returns
        -------
        ann_dict: dict,
        """
        ann_dict = CFG()
        rec_fp = self._get_path(rec)

        # wave delineation annotations
        _leads = self._normalize_leads(leads, standard_ordering=True, lower_cases=False)
        _ann_ext = [f"{l.lower()}" for l in _leads]  # for Version 1.0.0, it is f"{l.lower()}"
        ann_dict["waves"] = CFG({l:[] for l in _leads})
        for l, e in zip(_leads, _ann_ext):
            ann = wfdb.rdann(rec_fp, extension=e)
            df_lead_ann = pd.DataFrame()
            symbols = np.array(ann.symbol)
            peak_inds = np.where(np.isin(symbols, ["p", "N", "t"]))[0]
            df_lead_ann["peak"] = ann.sample[peak_inds]
            df_lead_ann["onset"] = np.nan
            df_lead_ann["offset"] = np.nan
            for i, row in df_lead_ann.iterrows():
                peak_idx = peak_inds[i]
                if peak_idx == 0:
                    df_lead_ann.loc[i, "onset"] = row["peak"]
                    if symbols[peak_idx+1] == ")":
                        df_lead_ann.loc[i, "offset"] = ann.sample[peak_idx+1]
                    else:
                        df_lead_ann.loc[i, "offset"] = row["peak"]
                elif peak_idx == len(symbols) - 1:
                    df_lead_ann.loc[i, "offset"] = row["peak"]
                    if symbols[peak_idx-1] == "(":
                        df_lead_ann.loc[i, "onset"] = ann.sample[peak_idx-1]
                    else:
                        df_lead_ann.loc[i, "onset"] = row["peak"]
                else:
                    if symbols[peak_idx-1] == "(":
                        df_lead_ann.loc[i, "onset"] = ann.sample[peak_idx-1]
                    else:
                        df_lead_ann.loc[i, "onset"] = row["peak"]
                    if symbols[peak_idx+1] == ")":
                        df_lead_ann.loc[i, "offset"] = ann.sample[peak_idx+1]
                    else:
                        df_lead_ann.loc[i, "offset"] = row["peak"]
            # df_lead_ann["onset"] = ann.sample[np.where(symbols=="(")[0]]
            # df_lead_ann["offset"] = ann.sample[np.where(symbols==")")[0]]

            df_lead_ann["duration"] = (df_lead_ann["offset"] - df_lead_ann["onset"]) * self.spacing
            
            df_lead_ann.index = symbols[peak_inds]

            for c in ["peak", "onset", "offset"]:
                df_lead_ann[c] = df_lead_ann[c].values.astype(int)
            
            for _, row in df_lead_ann.iterrows():
                w = ECGWaveForm(
                    name=self._symbol_to_wavename[row.name],
                    onset=int(row.onset),
                    offset=int(row.offset),
                    peak=int(row.peak),
                    duration=row.duration,
                )
                ann_dict["waves"][l].append(w)

        if metadata:
            header_dict = self._load_header(rec)
            ann_dict.update(header_dict)
        
        return ann_dict

    def load_diagnoses(self, rec:str) -> List[str]:
        """ finished, checked,

        load diagnoses of the `rec`

        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        diagnoses: list of str,
        """
        diagnoses = self._load_header(rec)["diagnoses"]
        return diagnoses

    def load_masks(self,
                   rec:str,
                   leads:Optional[Sequence[str]]=None,
                   mask_format:str="channel_first",
                   class_map:Optional[Dict[str, int]]=None,) -> np.ndarray:
        """ finished, checked,

        load the wave delineation in the form of masks

        Parameters
        ----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        mask_format: str, default "channel_first",
            format of the mask,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        class_map: dict, optional,
            custom class map,
            if not set, `self.class_map` will be used

        Returns
        -------
        masks: ndarray,
            the masks corresponding to the wave delineation annotations of `rec`
        """
        _class_map = CFG(class_map) if class_map is not None else self.class_map
        _leads = self._normalize_leads(leads, standard_ordering=True, lower_cases=True)
        data = self.load_data(rec, leads=_leads, data_format="channel_first")
        masks = np.full_like(data, fill_value=_class_map.i, dtype=int)
        waves = self.load_ann(rec, leads=_leads, metadata=False)["waves"]
        for idx, (l, l_w) in enumerate(waves.items()):
            for w in l_w:
                masks[idx, w.onset: w.offset] = _class_map[self._wavename_to_symbol[w.name]]
        if mask_format.lower() not in ["channel_first", "lead_first",]:
            masks = masks.T
        return masks

    def from_masks(self,
                   masks:np.ndarray,
                   mask_format:str="channel_first",
                   leads:Optional[Sequence[str]]=None,
                   class_map:Optional[Dict[str, int]]=None,
                   fs:Optional[Real]=None) -> Dict[str, List[ECGWaveForm]]:
        """ finished, checked,

        convert masks into lists of waveforms

        Parameters
        ----------
        masks: ndarray,
            wave delineation in the form of masks,
            of shape (n_leads, seq_len), or (seq_len,)
        mask_format: str, default "channel_first",
            format of the mask, used only when `masks.ndim = 2`
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first")
        leads: str or list of str, optional,
            the leads to load
        class_map: dict, optional,
            custom class map,
            if not set, `self.class_map` will be used
        fs: real number, optional,
            sampling frequency of the signal corresponding to the `masks`,
            if is None, `self.fs` will be used, to compute `duration` of the ecg waveforms

        Returns
        -------
        waves: dict,
            each item value is a list containing the `ECGWaveForm`s corr. to the lead (item key)
        """
        if masks.ndim == 1:
            _masks = masks[np.newaxis,...]
        elif masks.ndim == 2:
            if mask_format.lower() not in ["channel_first", "lead_first",]:
                _masks = masks.T
            else:
                _masks = masks.copy()
        else:
            raise ValueError(f"masks should be of dim 1 or 2, but got a {masks.ndim}d array")

        if leads is not None:
            _leads = self._normalize_leads(leads, standard_ordering=False, lower_cases=False)
        else:
            _leads = [f"lead_{idx+1}" for idx in range(_masks.shape[0])]
        assert len(_leads) == _masks.shape[0]

        _class_map = CFG(class_map) if class_map is not None else self.class_map

        _fs = fs if fs is not None else self.fs

        waves = CFG({lead_name:[] for lead_name in _leads})
        for channel_idx, lead_name in enumerate(_leads):
            current_mask = _masks[channel_idx,...]
            for wave_symbol, wave_number in _class_map.items():
                if wave_symbol not in ["p", "N", "t",]:
                    continue
                wave_name = self._symbol_to_wavename[wave_symbol]
                current_wave_inds = np.where(current_mask==wave_number)[0]
                if len(current_wave_inds) == 0:
                    continue
                np.where(np.diff(current_wave_inds)>1)
                split_inds = np.where(np.diff(current_wave_inds)>1)[0].tolist()
                split_inds = sorted(split_inds+[i+1 for i in split_inds])
                split_inds = [0] + split_inds + [len(current_wave_inds)-1]
                for i in range(len(split_inds)//2):
                    itv_start = current_wave_inds[split_inds[2*i]]
                    itv_end = current_wave_inds[split_inds[2*i+1]]+1
                    w = ECGWaveForm(
                        name=wave_name,
                        onset=itv_start,
                        offset=itv_end,
                        peak=np.nan,
                        duration=1000*(itv_end-itv_start)/_fs,  # ms
                    )
                    waves[lead_name].append(w)
            waves[lead_name].sort(key=lambda w: w.onset)
        return waves

    def _load_header(self, rec:str) -> dict:
        """ finished, checked,

        load header data into a dict

        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        header_dict: dict,
        """
        header_dict = CFG({})
        rec_fp = self._get_path(rec)
        header_reader = wfdb.rdheader(rec_fp)
        header_dict["units"] = header_reader.units
        header_dict["baseline"] = header_reader.baseline
        header_dict["adc_gain"] = header_reader.adc_gain
        header_dict["record_fmt"] = header_reader.fmt
        try:
            header_dict["age"] = int([l for l in header_reader.comments if "<age>" in l][0].split(": ")[-1])
        except:
            header_dict["age"] = np.nan
        try:
            header_dict["sex"] = [l for l in header_reader.comments if "<sex>" in l][0].split(": ")[-1]
        except:
            header_dict["sex"] = ""
        d_start = [idx for idx, l in enumerate(header_reader.comments) if "<diagnoses>" in l][0] + 1
        header_dict["diagnoses"] = header_reader.comments[d_start:]
        return header_dict

    def _normalize_leads(self,
                         leads:Optional[Sequence[str]]=None,
                         standard_ordering:bool=True,
                         lower_cases:bool=False,) -> List[str]:
        """ finished, checked,

        Parameters
        ----------
        leads: str or list of str, optional,
            the (names of) leads to normalize
        starndard_ordering: bool, default True,
            if True, the ordering will be re-aranged to be accordance with `self.all_leads`
        lower_cases: bool, default False,
            if True, all names of the leads will be in lower cases
        """
        if leads is None:
            _leads = self.all_leads_lower
        elif isinstance(leads, str):
            _leads = [leads.lower()]
        else:
            _leads = [l.lower() for l in leads]

        if standard_ordering:
            _leads = [l for l in self.all_leads_lower if l in _leads]
        
        if not lower_cases:
            _lead_indices = [idx for idx, l in enumerate(self.all_leads_lower) if l in _leads]
            _leads = [self.all_leads[idx] for idx in _lead_indices]
        
        return _leads

    def load_subject_info(self, rec:str, fields:Optional[Union[str,Sequence[str]]]=None) -> Union[dict,str]:
        """ finished, checked,

        Parameters
        ----------
        rec: str,
            name of the record
        fields: str or sequence of str, optional,
            field(s) of the subject info of record `rec`,
            if not specified, all fields of the subject info will be returned

        Returns
        -------
        info: dict or str,
            subject info of the given fields of the record
        """
        row = self._df_subject_info[self._df_subject_info.ID==rec]
        if row.empty:
            return {}
        row = row.iloc[0]
        info = row.to_dict()
        if fields is not None:
            assert fields in self._df_subject_info.columns or \
                set(fields).issubset(set(self._df_subject_info.columns)), \
                    "No such field(s)!"
            if isinstance(fields, str):
                info = info[fields]
            else:
                info = {k:v for k,v in info.items() if k in fields}
        return info


    def plot(self,
             rec:str,
             data:Optional[np.ndarray]=None,
             ticks_granularity:int=0,
             leads:Optional[Union[str, List[str]]]=None,
             same_range:bool=False,
             waves:Optional[ECGWaveForm]=None,
             **kwargs:Any,) -> NoReturn:
        """ finished, checked, to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (fs, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters
        ----------
        rec: str,
            name of the record
        data: ndarray, optional,
            12-lead ecg signal to plot,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        leads: str or list of str, optional,
            the leads to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
        waves: ECGWaveForm, optional,
            the waves (p waves, t waves, qrs complexes)
        kwargs: dict,

        TODO:
        -----
        1. slice too long records, and plot separately for each segment
        2. plot waves using `axvspan`

        NOTE
        ----
        `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once

        Contributors: Jeethan, and WEN Hao
        """
        if "plt" not in dir():
            import matplotlib.pyplot as plt
            plt.MultipleLocator.MAXTICKS = 3000
        _leads = self._normalize_leads(leads, standard_ordering=True, lower_cases=False)

        # lead_list = self.load_ann(rec)["df_leads"]["lead_name"].tolist()
        # _lead_indices = [lead_list.index(l) for l in leads]
        _lead_indices = [self.all_leads.index(l) for l in _leads]
        if data is None:
            _data = self.load_data(rec, data_format="channel_first", units="μV")[_lead_indices]
        else:
            units = self._auto_infer_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == "mv":
                _data = 1000 * data
            else:
                _data = data
        
        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        if data is None and waves is None:
            waves = self.load_ann(rec, leads=_leads)["waves"]

        if waves is not None:
            pwaves = {l:[] for l in _leads}
            qrs = {l:[] for l in _leads}
            twaves = {l:[] for l in _leads}
            for l, l_w in waves.items():
                for w in l_w:
                    itv = [w.onset, w.offset]
                    if w.name == self._symbol_to_wavename["p"]:
                        pwaves[l].append(itv)
                    elif w.name == self._symbol_to_wavename["N"]:
                        qrs[l].append(itv)
                    elif w.name == self._symbol_to_wavename["t"]:
                        twaves[l].append(itv)
        
        palette = {"pwaves": "green", "qrs": "red", "twaves": "yellow",}
        plot_alpha = 0.4

        diagnoses = self.load_diagnoses(rec)

        nb_leads = len(_leads)

        seg_len = self.fs * 25  # 25 seconds
        nb_segs = _data.shape[1] // seg_len

        t = np.arange(_data.shape[1]) / self.fs
        duration = len(t) / self.fs
        fig_sz_w = int(round(4.8 * duration))
        fig_sz_h = 6 * y_ranges / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        if nb_leads == 1:
            axes = [axes]
        for idx in range(nb_leads):
            lead_name = self.all_leads[_lead_indices[idx]]
            axes[idx].plot(t, _data[idx], color="black", linewidth="2.0", label=f"lead - {lead_name}")
            axes[idx].axhline(y=0, linestyle="-", linewidth="1.0", color="red")
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which="major", linestyle="-", linewidth="0.5", color="red")
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which="minor", linestyle=":", linewidth="0.5", color="black")
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            for d in diagnoses:
                axes[idx].plot([], [], " ", label=d)
            for w in ["pwaves", "qrs", "twaves"]:
                for itv in eval(f"{w}['{lead_name}']"):
                    axes[idx].axvspan(
                        itv[0]/self.fs, itv[1]/self.fs,
                        color=palette[w], alpha=plot_alpha,
                    )
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
