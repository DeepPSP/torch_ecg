# -*- coding: utf-8 -*-
"""
docstring, to write
"""

import warnings
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Union

import numpy as np
import pandas as pd
import xmltodict as xtd
from pyedflib import EdfReader  # noqa: F401

from ...utils.utils_interval import intervals_union
from ..base import NSRRDataBase

__all__ = [
    "SHHS",
]


class SHHS(NSRRDataBase):
    """

    Sleep Heart Health Study

    ABOUT shhs
    ----------
    ***ABOUT the dataset:
    1. shhs1 (Visit 1):
        1.1. the baseline clinic visit and polysomnogram performed between November 1, 1995 and January 31, 1998
        1.2. in all, 6,441 men and women aged 40 years and older were enrolled
        1.3. 5,804 rows, down from the original 6,441 due to data sharing rules on certain cohorts and subjects
    2. shhs-interim-followup (Interim Follow-up):
        2.1. an interim clinic visit or phone call 2-3 years after baseline (shhs1)
        2.2. 5,804 rows, despite some subjects not having complete data, all original subjects are present in the dataset
    3. shhs2 (Visit 2):
        3.1. the follow-up clinic visit and polysomnogram performed between January 2001 and June 2003
        3.2. during this exam cycle 3, a second polysomnogram was obtained in 3,295 of the participants
        3.3. 4,080 rows, not all cohorts and subjects took part
    4. shhs-cvd (CVD Outcomes):
        4.1. the tracking of adjudicated heart health outcomes (e.g. stroke, heart attack) between baseline (shhs1) and 2008-2011 (varies by parent cohort)
        4.2. 5,802 rows, outcomes data were not provided on all subjects
    5. shhs-cvd-events (CVD Outcome Events):
        5.1. event-level details for the tracking of heart health outcomes (shhs-cvd)
        5.2. 4,839 rows, representing individual events

    6. ECG was sampled at 125 Hz in shhs1 and 250/256 Hz in shhs2
    7. `annotations-events-nsrr` and `annotations-events-profusion`: annotation files both contain xml files, the former processed in the EDF Editor and Translator tool, the latter exported from Compumedics Profusion
    8. about 10% of the records have HRV (including sleep stages and sleep events) annotations

    ***DATA Analysis Tips:
    1. Respiratory Disturbance Index (RDI):
        1.1. A number of RDI variables exist in the data set. These variables are highly skewed.
        1.2. log-transformation is recommended, among which the following transformation performed best, at least in some subsets:
            NEWVA = log(OLDVAR + 0.1)
    2. Obstructive Apnea Index (OAI):
        2.1. There is one OAI index in the data set. It reflects obstructive events associated with a 4% desaturation or arousal. Nearly 30% of the cohort has a zero value for this variable
        2.2. Dichotomization is suggested (e.g. >=3 or >=4 events per hour indicates positive)
    3. Central Apnea Index (CAI):
        3.1. Several variables describe central breathing events, with different thresholds for desaturation and requirement/non-requirement of arousals. ˜58% of the cohort have zero values
        3.2. Dichotomization is suggested (e.g. >=3 or >=4 events per hour indicates positive)
    4. Sleep Stages:
        4.1. Stage 1 and stage 3-4 are not normally distributed, but stage 2 and REM sleep are.
        4.2. To use these data as continuous dependent variables, stages 1 and 3-4 must be transformed. The following formula is suggested:
            –log(-log(val/100+0.001))
    5. Sleep time below 90% O2:
        5.1. Percent of total sleep time with oxygen levels below 75%, 80%, 85% and 90% were recorded
        5.2. Dichotomization is suggested (e.g. >5% and >10% of sleep time with oxygen levels below a specific O2 level indicates positive)

    More: [1]

    ***ABOUT signals: (ref. [10])
    1. C3/A2 and C4/A1 EEGs, sampled at 125 Hz
    2. right and left electrooculograms (EOGs), sampled at 50 Hz
    3. a bipolar submental electromyogram (EMG), sampled at 125 Hz
    4. thoracic and abdominal excursions (THOR and ABDO), recorded by inductive plethysmography bands and sampled at 10 Hz
    5. "AIRFLOW" detected by a nasal-oral thermocouple, sampled at 10 Hz
    6. finger-tip pulse oximetry sampled at 1 Hz
    7. ECG from a bipolar lead, sampled at 125 Hz for most SHHS-1 studies and 250 (and 256?) Hz for SHHS-2 studies
    8. Heart rate (PR) derived from the ECG and sampled at 1 Hz
    9. body position (using a mercury gauge sensor)
    10. ambient light (on/off, by a light sensor secured to the recording garment)

    ***ABOUT annotations (NOT including "nsrrid","visitnumber","pptid" etc.):
    1. hrv annotations: (in csv files, ref. [2])
        Start__sec_ --- 5 minute window start time
        NN_RR	    --- Ratio of consecutive normal sinus beats (NN) over all cardiac inter-beat (RR) intervals
        AVNN	    --- Mean of all normal sinus to normal sinus interbeat intervals (NN)
        IHR	        --- Instantaneous heart rate
        SDNN	    --- Standard deviation of all normal sinus to normal sinus interbeat (NN) intervals
        SDANN	    --- Standard deviation of the averages of normal sinus to normal sinus interbeat (NN) intervals in all 5-minute segments
        SDNNIDX	    --- Mean of the standard deviations of normal sinus to normal sinus interbeat (NN) intervals in all 5-minute segments
        rMSSD	    --- Square root of the mean of the squares of difference between adjacent normal sinus to normal sinus interbeat (NN) intervals
        pNN10	    --- Percentage of differences between adjacent normal sinus to normal sinus interbeat (NN) intervals that are >10 ms
        pNN20	    --- Percentage of differences between adjacent normal sinus to normal sinus interbeat (NN) intervals that are >20 ms
        pNN30	    --- Percentage of differences between adjacent normal sinus to normal sinus interbeat (NN) intervals that are >30 ms
        pNN40	    --- Percentage of differences between adjacent normal sinus to normal sinus interbeat (NN) intervals that are >40 ms
        pNN50	    --- Percentage of differences between adjacent normal sinus to normal sinus interbeat (NN) intervals that are >50 ms
        tot_pwr	    --- Total normal sinus to normal sinus interbeat (NN) interval spectral power up to 0.4 Hz
        ULF	        --- Ultra-low frequency power, the normal sinus to normal sinus interbeat (NN) interval spectral power between 0 and 0.003 Hz
        VLF	        --- Very low frequency power, the normal sinus to normal sinus interbeat (NN) interval spectral power between 0.003 and 0.04 Hz
        LF	        --- Low frequency power, the normal sinus to normal sinus interbeat (NN) interval spectral power between 0.04 and 0.15 Hz
        HF	        --- High frequency power, the normal sinus to normal sinus interbeat (NN) interval spectral power between 0.15 and 0.4 Hz
        LF_HF	    --- The ratio of low to high frequency power
        LF_n	    --- Low frequency power (normalized)
        HF_n	    --- High frequency power (normalized)
    2. wave delineation annotations: (in csv files, NOTE: see "CAUTION" by the end of this part, ref. [2])
        RPoint	    --- Sample Number indicating R Point (peak of QRS)
        Start	    --- Sample Number indicating start of beat
        End	        --- Sample Number indicating end of beat
        STLevel1    --- Level of ECG 1 in Raw data ( 65536 peak to peak rawdata = 10mV peak to peak)
        STSlope1    --- Slope of ECG 1 stored as int and to convert to a double divide raw value by 1000.0
        STLevel2    --- Level of ECG 2 in Raw data ( 65536 peak to peak rawdata = 10mV peak to peak)
        STSlope2    --- Slope of ECG 2 stored as int and to convert to a double divide raw value by 1000.0
        Manual      --- (True / False) True if record was manually inserted
        Type        --- Type of beat (0 = Artifact / 1 = Normal Sinus Beat / 2 = VE / 3 = SVE)
        Class       --- no longer used
        PPoint      --- Sample Number indicating peak of the P wave (-1 if no P wave detected)
        PStart      --- Sample Number indicating start of the P wave
        PEnd        --- Sample Number indicating end of the P wave
        TPoint      --- Sample Number indicating peak of the T wave (-1 if no T wave detected)
        TStart      --- Sample Number indicating start of the T wave
        TEnd        --- Sample Number indicating end of the T wave
        TemplateID  --- The ID of the template to which this beat has been assigned (-1 if not assigned to a template)
        nsrrid      --- nsrrid of this record
        samplingrate--- frequency of the ECG signal of this record
        seconds     --- Number of seconds from beginning of recording to R-point (Rpoint / sampling rate)
        epoch       --- Epoch (30 second) number
        rpointadj   --- R Point adjusted sample number (RPoint * (samplingrate/256))
    CAUTION: all the above sampling numbers except for rpointadj assume 256 Hz, while the rpointadj column has been added to provide an adjusted sample number based on the actual sampling rate.
    3. event annotations: (in xml files)
        TODO
    4. event_profusion annotations: (in xml files)
        TODO

    ***DEFINITION of concepts in sleep study:
    1. Arousal: (ref. [3],[4])
        1.1. interruptions of sleep lasting 3 to 15 seconds
        1.2. can occur spontaneously or as a result of sleep-disordered breathing or other sleep disorders
        1.3. sends you back to a lighter stage of sleep
        1.4. if the arousal last more than 15 seconds, it becomes an awakening
        1.5. the higher the arousal index (occurrences per hour), the more tired you are likely to feel, though people vary in their tolerance of sleep disruptions
    2. Central Sleep Apnea (CSA): (ref. [3],[5],[6])
        2.1. breathing repeatedly stops and starts during sleep
        2.2. occurs because your brain (central nervous system) doesn't send proper signals to the muscles that control your breathing, which is point that differs from obstructive sleep apnea
        2.3. may occur as a result of other conditions, such as heart failure, stroke, high altitude, etc.
    3. Obstructive Sleep Apnea (OSA): (ref. [3],[7])
        3.1. occurs when throat muscles intermittently relax and block upper airway during sleep
        3.2. a noticeable sign of obstructive sleep apnea is snoring
    4. Complex (Mixed) Sleep Apnea: (ref. [3])
        4.1. combination of both CSA and OSA
        4.2. exact mechanism of the loss of central respiratory drive during sleep in OSA is unknown but is most likely related to incorrect settings of the CPAP (Continuous Positive Airway Pressure) treatment and other medical conditions the person has
    5. Hypopnea:
        overly shallow breathing or an abnormally low respiratory rate. Hypopnea is defined by some to be less severe than apnea (the complete cessation of breathing)
    6. Apnea Hypopnea Index (AHI): to write
        6.1. used to indicate the severity of OSA
        6.2. number of apneas or hypopneas recorded during the study per hour of sleep
        6.3. based on the AHI, the severity of OSA is classified as follows
            - none/minimal: AHI < 5 per hour
            - mild: AHI ≥ 5, but < 15 per hour
            - moderate: AHI ≥ 15, but < 30 per hour
            - severe: AHI ≥ 30 per hour
    7. Oxygen Desaturation:
        7.1. used to indicate the severity of OSA
        7.2. reductions in blood oxygen levels (desaturation)
        7.3. at sea level, a normal blood oxygen level (saturation) is usually 96 - 97%
        7.4. (no generally accepted classifications for severity of oxygen desaturation)
            - mild: >= 90%
            - moderate: 80% - 89%
            - severe: < 80%

    NOTE
    ----

    ISSUES
    ------
    1. `Start__sec_` might not be the start time, but rather the end time, of the 5 minute windows in some records
    2. the current version "0.15.0" removed EEG spectral summary variables

    Usage
    -----
    1. sleep stage
    2. sleep apnea

    References
    ----------
    [1] https://sleepdata.org/datasets/shhs/pages/
    [2] https://sleepdata.org/datasets/shhs/pages/13-hrv-analysis.md
    [3] https://en.wikipedia.org/wiki/Sleep_apnea
    [4] https://www.sleepapnea.org/treat/getting-sleep-apnea-diagnosis/sleep-study-details/
    [5] https://www.mayoclinic.org/diseases-conditions/central-sleep-apnea/symptoms-causes/syc-20352109
    [6] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2287191/
    [7] https://www.mayoclinic.org/diseases-conditions/obstructive-sleep-apnea/symptoms-causes/syc-20352090
    [8] https://en.wikipedia.org/wiki/Hypopnea
    [9] http://healthysleep.med.harvard.edu/sleep-apnea/diagnosing-osa/understanding-results
    [10] https://sleepdata.org/datasets/shhs/pages/full-description.md
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
        working_dir: str or Path, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments
        """
        super().__init__(
            db_name="SHHS",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.current_version = kwargs.get("current_version", "0.15.0")

        self.psg_data_path = None
        self.ann_path = None
        self.hrv_ann_path = None
        self.eeg_ann_path = None
        self.wave_deli_path = None
        self.event_ann_path = None
        self.event_profusion_ann_path = None
        self.form_paths()

        self.fs = None
        self.file_opened = None

        # stats
        try:
            self.rec_with_hrv_ann = [
                f"shhs{int(row['visitnumber'])}-{int(row['nsrrid'])}"
                for _, row in self.load_hrv_summary_ann().iterrows()
            ]
        except Exception:
            self.rec_with_hrv_ann = []

        self.all_signals = [
            "SaO2",
            "H.R.",
            "EEG(sec)",
            "ECG",
            "EMG",
            "EOG(L)",
            "EOG(R)",
            "EEG",
            "SOUND",
            "AIRFLOW",
            "THOR RES",
            "ABDO RES",
            "POSITION",
            "LIGHT",
            "NEW AIR",
            "OX stat",
        ]

        # annotations regarding sleep analysis
        self.hrv_ann_summary_keys = [
            "nsrrid",
            "visitnumber",
            "NN_RR",
            "AVNN",
            "IHR",
            "SDNN",
            "SDANN",
            "SDNNIDX",
            "rMSSD",
            "pNN10",
            "pNN20",
            "pNN30",
            "pNN40",
            "pNN50",
            "tot_pwr",
            "ULF",
            "VLF",
            "LF",
            "HF",
            "LF_HF",
            "LF_n",
            "HF_n",
        ]
        self.hrv_ann_detailed_keys = [
            "nsrrid",
            "visitnumber",
            "Start__sec_",
            "ihr",
            "NN_RR",
            "AVNN",
            "SDNN",
            "rMSSD",
            "PNN10",
            "PNN20",
            "PNN30",
            "PNN40",
            "PNN50",
            "TOT_PWR",
            "VLF",
            "LF",
            "LF_n",
            "HF",
            "HF_n",
            "LF_HF",
            "sleepstage01",
            "sleepstage02",
            "sleepstage03",
            "sleepstage04",
            "sleepstage05",
            "sleepstage06",
            "sleepstage07",
            "sleepstage08",
            "sleepstage09",
            "sleepstage10",
            "event01start",
            "event01end",
            "event02start",
            "event02end",
            "event03start",
            "event03end",
            "event04start",
            "event04end",
            "event05start",
            "event05end",
            "event06start",
            "event06end",
            "event07start",
            "event07end",
            "event08start",
            "event08end",
            "event09start",
            "event09end",
            "event10start",
            "event10end",
            "event11start",
            "event11end",
            "event12start",
            "event12end",
            "event13start",
            "event13end",
            "event14start",
            "event14end",
            "event15start",
            "event15end",
            "event16start",
            "event16end",
            "event17start",
            "event17end",
            "event18start",
            "event18end",
            "hasrespevent",
        ]
        self.hrv_ann_epoch_len_sec = 300  # 5min
        self.sleep_ann_keys_from_hrv = [
            "Start__sec_",
            "sleepstage01",
            "sleepstage02",
            "sleepstage03",
            "sleepstage04",
            "sleepstage05",
            "sleepstage06",
            "sleepstage07",
            "sleepstage08",
            "sleepstage09",
            "sleepstage10",
            "event01start",
            "event01end",
            "event02start",
            "event02end",
            "event03start",
            "event03end",
            "event04start",
            "event04end",
            "event05start",
            "event05end",
            "event06start",
            "event06end",
            "event07start",
            "event07end",
            "event08start",
            "event08end",
            "event09start",
            "event09end",
            "event10start",
            "event10end",
            "event11start",
            "event11end",
            "event12start",
            "event12end",
            "event13start",
            "event13end",
            "event14start",
            "event14end",
            "event15start",
            "event15end",
            "event16start",
            "event16end",
            "event17start",
            "event17end",
            "event18start",
            "event18end",
            "hasrespevent",
        ]
        self.sleep_stage_ann_keys_from_hrv = [
            "Start__sec_",
            "sleepstage01",
            "sleepstage02",
            "sleepstage03",
            "sleepstage04",
            "sleepstage05",
            "sleepstage06",
            "sleepstage07",
            "sleepstage08",
            "sleepstage09",
            "sleepstage10",
        ]
        self.sleep_event_ann_keys_from_hrv = [
            "Start__sec_",
            "event01start",
            "event01end",
            "event02start",
            "event02end",
            "event03start",
            "event03end",
            "event04start",
            "event04end",
            "event05start",
            "event05end",
            "event06start",
            "event06end",
            "event07start",
            "event07end",
            "event08start",
            "event08end",
            "event09start",
            "event09end",
            "event10start",
            "event10end",
            "event11start",
            "event11end",
            "event12start",
            "event12end",
            "event13start",
            "event13end",
            "event14start",
            "event14end",
            "event15start",
            "event15end",
            "event16start",
            "event16end",
            "event17start",
            "event17end",
            "event18start",
            "event18end",
            "hasrespevent",
        ]

        # annotations from events-nsrr and events-profusion folders
        self.event_keys = [
            "EventType",
            "EventConcept",
            "Start",
            "Duration",
            "SignalLocation",
            "SpO2Nadir",
            "SpO2Baseline",
        ]
        # NOTE: the union of names from shhs1-200001 to shhs1-200399
        # NOT a full search
        self.short_event_types_from_event = [
            "Respiratory",
            "Stages",
            "Arousals",
        ]
        self.long_event_types_from_event = [
            "Respiratory|Respiratory",
            "Stages|Stages",
            "Arousals|Arousals",
        ]
        # NOTE: the union of names from shhs1-200001 to shhs1-200399
        # NOT a full search
        # NOT including sleep stages
        self.short_event_names_from_event = [
            "Central Apnea",
            "Obstructive Apnea",
            "Mixed Apnea",
            "Hypopnea",
            "SpO2 artifact",
            "SpO2 desaturation",
            "Arousal ()",
            "Arousal (Standard)",
            "Arousal (STANDARD)",
            "Arousal (CHESHIRE)",
            "Arousal (ASDA)",
            "Unsure",
        ]
        self.long_event_names_from_event = [
            "Central apnea|Central Apnea",
            "Obstructive apnea|Obstructive Apnea",
            "Mixed apnea|Mixed Apnea",
            "Hypopnea|Hypopnea",
            "SpO2 artifact|SpO2 artifact",
            "SpO2 desaturation|SpO2 desaturation",
            "Arousal|Arousal ()",
            "Arousal|Arousal (Standard)",
            "Arousal|Arousal (STANDARD)",
            "Arousal resulting from Chin EMG|Arousal (CHESHIRE)",
            "ASDA arousal|Arousal (ASDA)",
            "Unsure|Unsure",
        ]
        self.event_profusion_keys = [
            "Name",
            "Start",
            "Duration",
            "Input",
            "LowestSpO2",
            "Desaturation",
        ]
        # NOTE: currently the union of names from shhs1-200001 to shhs1-200099,
        # NOT a full search
        self.event_names_from_event_profusion = [
            "Central Apnea",
            "Obstructive Apnea",
            "Mixed Apnea",
            "Hypopnea",
            "SpO2 artifact",
            "SpO2 desaturation",
            "Arousal ()",
            "Arousal (ASDA)",
            "Unsure",
        ]

        self.apnea_types = [
            "Central Apnea",
            "Obstructive Apnea",
            "Mixed Apnea",
            "Hypopnea",
        ]

        # annotations regarding wave delineation
        self.wave_deli_keys = [
            "RPoint",
            "Start",
            "End",
            "STLevel1",
            "STSlope1",
            "STLevel2",
            "STSlope2",
            "Manual",
            "Type",
            "PPoint",
            "PStart",
            "PEnd",
            "TPoint",
            "TStart",
            "TEnd",
            "TemplateID",
            "nsrrid",
            "samplingrate",
            "seconds",
            "epoch",
            "rpointadj",
        ]
        self.wave_deli_samp_num_keys = [
            "RPoint",
            "Start",
            "End",
            "PPoint",
            "PStart",
            "PEnd",
            "TPoint",
            "TStart",
            "TEnd",
        ]

        # TODO: other annotation files: EEG

        # self-defined items
        self.sleep_stage_keys = ["start_sec", "sleep_stage"]
        self.sleep_event_keys = [
            "event_name",
            "event_start",
            "event_end",
            "event_duration",
        ]
        self.sleep_epoch_len_sec = 30
        self.ann_sleep_stages = [0, 1, 2, 3, 4, 5, 9]
        """
        0	--- Wake
        1	--- sleep stage 1
        2	--- sleep stage 2
        3	--- sleep stage 3/4
        4	--- sleep stage 3/4
        5	--- REM stage
        9	--- Movement/Wake or Unscored?
        """
        self.sleep_stage_protocol = kwargs.get("sleep_stage_protocol", "aasm")
        self.all_sleep_stage_names = ["W", "R", "N1", "N2", "N3", "N4"]
        self.sleep_stage_name_value_mapping = {
            "W": 0,
            "R": 1,
            "N1": 2,
            "N2": 3,
            "N3": 4,
            "N4": 5,
        }
        self.sleep_stage_names = []
        self.update_sleep_stage_names()
        self._to_simplified_states = {9: 0, 0: 0, 5: 1, 1: 2, 2: 2, 3: 3, 4: 3}
        """ 9 to nan?
        0   --- awake
        1   --- REM
        2   --- N1 (NREM1/2), shallow sleep
        3   --- N2 (NREM3/4), deep sleep
        """
        self._to_aasm_states = {9: 0, 0: 0, 5: 1, 1: 2, 2: 3, 3: 4, 4: 4}
        """ 9 to nan?
        0   --- awake
        1   --- REM
        2   --- N1 (NREM1)
        3   --- N2 (NREM2)
        4   --- N3 (NREM3/4)
        """
        self._to_shhs_states = {9: 0, 0: 0, 5: 1, 1: 2, 2: 3, 3: 4, 4: 5}

        # for plotting
        self.palette = {
            "W": "orange",
            "R": "yellow",
            "N1": "green",
            "N2": "cyan",
            "N3": "blue",
            "N4": "purple",
            "Central Apnea": "red",
            "Obstructive Apnea": "yellow",
            "Mixed Apnea": "cyan",
            "Hypopnea": "purple",
        }  # TODO: add more

    def form_paths(self) -> NoReturn:
        """finished,"""
        self.psg_data_path = self.db_dir / "polysomnography" / "edfs"
        self.ann_path = self.db_dir / "datasets"
        self.hrv_ann_path = self.ann_path / "hrv-analysis"
        self.eeg_ann_path = self.ann_path / "eeg-spectral-analysis"
        self.wave_deli_path = self.db_dir / "polysomnography" / "annotations-rpoints"
        self.event_ann_path = (
            self.db_dir / "polysomnography" / "annotations-events-nsrr"
        )
        self.event_profusion_ann_path = (
            self.db_dir / "polysomnography" / "annotations-events-profusion"
        )

    def update_sleep_stage_names(self) -> NoReturn:
        """finished,"""
        if self.sleep_stage_protocol == "aasm":
            nb_stages = 5
        elif self.sleep_stage_protocol == "simplified":
            nb_stages = 4
        elif self.sleep_stage_protocol == "shhs":
            nb_stages = 6
        else:
            raise ValueError(f"No stage protocol named {self.sleep_stage_protocol}")

        self.sleep_stage_names = self.all_sleep_stage_names[:nb_stages]

    def get_subject_id(self, rec: str) -> int:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"

        Returns
        -------
        pid, int, `subject_id` derived from `rec`
        """
        head_shhs1, head_shhs2v3, head_shhs2v4 = "30000", "30001", "30002"
        dataset_no, no = rec.split("-")
        dataset_no = dataset_no[-1]
        if dataset_no == "2":
            raise ValueError(
                "SHHS2 has two different sampling frequencies, currently could not be distinguished using only `rec`"
            )
        pid = int(head_shhs1 + dataset_no + no)
        return pid

    def get_visit_number(self, rec: str) -> int:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"

        Returns
        -------
        int, visit number extracted from `rec`
        """
        return int(rec.split("-")[0][-1])

    def get_nsrrid(self, rec: str) -> int:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"

        Returns
        -------
        int, nsrrid extracted from `rec`
        """
        return int(rec.split("-")[1])

    def get_fs(self, rec: str, sig: str = "ECG", rec_path: Optional[str] = None) -> int:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        sig: str, default "ECG",
            signal name
        rec_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used

        Returns
        -------
        fs, int,
            the sampling frequency of the signal `sig` of the record `rec`
        """
        frp = self.match_full_rec_path(rec, rec_path)
        self.safe_edf_file_operation("open", frp)
        chn_num = self.file_opened.getSignalLabels().index(self.match_channel(sig))
        fs = self.file_opened.getSampleFrequency(chn_num)
        self.safe_edf_file_operation("close")
        return fs

    def get_chn_num(
        self, rec: str, sig: str = "ECG", rec_path: Optional[str] = None
    ) -> int:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        sig: str, default "ECG",
            signal name
        rec_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used

        Returns
        -------
        chn_num, int,
            the number of channel of the signal `sig` of the record `rec`
        """
        frp = self.match_full_rec_path(rec, rec_path)
        self.safe_edf_file_operation("open", frp)
        chn_num = self.file_opened.getSignalLabels().index(self.match_channel(sig))
        self.safe_edf_file_operation("close")
        return chn_num

    def match_channel(self, channel: str) -> str:
        """finished,

        Parameters
        ----------
        channel: str,
            channel name

        Returns
        -------
        str, the standard channel name in SHHS
        """
        for sig in self.all_signals:
            if sig.lower() == channel.lower():
                return sig
        raise ValueError(f"No channel named {channel}")

    def match_full_rec_path(
        self,
        rec: str,
        rec_path: Optional[Union[str, Path]] = None,
        rec_type: str = "psg",
    ) -> Path:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rec_path: str or Path, optional,
            path of the file which contains the desired data,
            if not given, default path will be used
        rec_type: str, default "psg",
            record type, data or annotations

        Returns
        -------
        rp: Path,
        """
        extension = {
            "psg": ".edf",
            "wave_delineation": "-rpoint.csv",
            "event": "-nsrr.xml",
            "event_profusion": "-profusion.xml",
        }
        folder_or_file = {
            "psg": self.psg_data_path,
            "hrv_summary": self.hrv_ann_path
            / f"shhs{self.get_visit_number(rec)}-hrv-summary-{self.current_version}.csv",
            "hrv_5min": self.hrv_ann_path
            / f"shhs{self.get_visit_number(rec)}-hrv-5min-{self.current_version}.csv",
            "eeg_band_summary": self.eeg_ann_path
            / f"shhs{self.get_visit_number(rec)}-eeg-band-summary-dataset-{self.current_version}.csv",
            "eeg_spectral_summary": self.eeg_ann_path
            / f"shhs{self.get_visit_number(rec)}-eeg-spectral-summary-dataset-{self.current_version}.csv",
            "wave_delineation": self.wave_deli_path,
            "event": self.event_ann_path,
            "event_profusion": self.event_profusion_ann_path,
        }

        if rec_path is not None:
            rp = Path(rec_path)
        elif rec_type.split("_")[0] in ["hrv", "eeg"]:
            rp = folder_or_file[rec_type]
        else:
            rp = Path(str(folder_or_file[rec_type]) + rec.split("-")[0]) / (
                rec + extension[rec_type]
            )

        return rp

    def database_stats(self) -> NoReturn:
        """ """
        raise NotImplementedError

    def database_info(self, detailed: bool = False) -> NoReturn:
        """finished,

        print information about the database

        Parameters
        ----------
        detailed: bool, default False,
            if False, "What","Who","When","Funding" will be printed,
            if True, then docstring of the class will be printed additionally
        """
        raw_info = {
            "What": "Multi-cohort study focused on sleep-disordered breathing and cardiovascular outcomes",
            "Who": "5804 adults aged 40 and older",
            "When": "Two exam cycles, 1995-1998 and 2001-2003. Cardiovascular disease outcomes were tracked until 2010",
            "Funding": "National Heart, Lung, and Blood Institute",
        }

        print(raw_info)

        if detailed:
            print(self.__doc__)

    def show_rec_stats(self, rec: str, rec_path: Optional[str] = None) -> NoReturn:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rec_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used

        """
        frp = self.match_full_rec_path(rec, rec_path, rec_type="psg")
        self.safe_edf_file_operation("open", frp)
        for chn, lb in enumerate(self.file_opened.getSignalLabels()):
            print("SignalLabel:", lb)
            print("Prefilter:", self.file_opened.getPrefilter(chn))
            print("Transducer:", self.file_opened.getTransducer(chn))
            print("PhysicalDimension:", self.file_opened.getPhysicalDimension(chn))
            print("SampleFrequency:", self.file_opened.getSampleFrequency(chn))
            print("*" * 40)
        self.safe_edf_file_operation("close")

    def load_data(self, rec: str) -> NoReturn:
        """ """
        raise ValueError("Please load specific data, for example, psg, ecg, eeg, etc.")

    def load_ann(self, rec: str) -> NoReturn:
        """ """
        raise ValueError(
            "Please load specific annotations, for example, event annotations, etc."
        )

    def load_psg_data(
        self, rec: str, channel: str = "all", rec_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        channel: str, default "all",
            name of the channel of PSG,
            if is "all", then all channels will be returned
        rec_path: str, optional,
            path of the file which contains the psg data,
            if not given, default path will be used

        Returns
        -------
        dict, psg data
        """
        chn = self.match_channel(channel) if channel.lower() != "all" else "all"
        frp = self.match_full_rec_path(rec, rec_path, rec_type="psg")
        self.safe_edf_file_operation("open", frp)

        data_dict = {
            k: self.file_opened.readSignal(idx)
            for idx, k in enumerate(self.file_opened.getSignalLabels())
        }

        self.safe_edf_file_operation("close")

        if chn == "all":
            return data_dict
        else:
            return {chn: data_dict[chn]}

    def load_ecg_data(self, rec: str, rec_path: Optional[str] = None) -> np.ndarray:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rec_path: str, optional,
            path of the file which contains the ecg data,
            if not given, default path will be used

        Returns
        -------

        """
        return self.load_psg_data(rec=rec, channel="ecg", rec_path=rec_path)[
            self.match_channel("ecg")
        ]

    def load_event_ann(
        self, rec: str, event_ann_path: Optional[str] = None, simplify: bool = False
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        event_ann_path: str, optional,
            path of the file which contains the events-nsrr annotations,
            if not given, default path will be used

        Returns
        -------
        df_events: DataFrame,
        """
        file_path = self.match_full_rec_path(rec, event_ann_path, rec_type="event")
        doc = xtd.parse(file_path.read_text())
        df_events = pd.DataFrame(
            doc["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"][1:]
        )
        if simplify:
            df_events["EventType"] = df_events["EventType"].apply(
                lambda s: s.split("|")[1]
            )
            df_events["EventConcept"] = df_events["EventConcept"].apply(
                lambda s: s.split("|")[1]
            )
        for c in ["Start", "Duration", "SpO2Nadir", "SpO2Baseline"]:
            df_events[c] = df_events[c].apply(self.str_to_real_number)

        return df_events

    def load_event_profusion_ann(
        self, rec: str, event_profusion_ann_path: Optional[str] = None
    ) -> dict:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        event_profusion_ann_path: str, optional,
            path of the file which contains the events-profusion annotations,
            if not given, default path will be used

        Returns
        -------
        dict, with items "sleep_stage_list", "df_events"

        TODO:
            merge "sleep_stage_list" and "df_events" into one DataFrame
        """
        file_path = self.match_full_rec_path(
            rec, event_profusion_ann_path, rec_type="event_profusion"
        )
        doc = xtd.parse(file_path.read_text())
        sleep_stage_list = [
            int(ss) for ss in doc["CMPStudyConfig"]["SleepStages"]["SleepStage"]
        ]
        df_events = pd.DataFrame(doc["CMPStudyConfig"]["ScoredEvents"]["ScoredEvent"])
        for c in ["Start", "Duration", "LowestSpO2", "Desaturation"]:
            df_events[c] = df_events[c].apply(self.str_to_real_number)
        ret = {"sleep_stage_list": sleep_stage_list, "df_events": df_events}

        return ret

    def load_hrv_summary_ann(
        self, rec: Optional[str] = None, hrv_ann_path: Optional[str] = None
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str, optional,
            record name, typically in the form "shhs1-200001"
        hrv_ann_path: str, optional,
            path of the summary HRV annotation file,
            if not given, default path will be used

        Returns
        -------
        df_hrv_ann, DataFrame,
            if `rec` is not None, df_hrv_ann is the summary HRV annotations of `rec`;
            if `rec` is None, df_hrv_ann is the summary HRV annotations of all records
            that had HRV annotations (about 10% of all the records in SHHS)
        """
        if rec is None:
            file_path = self.match_full_rec_path(
                "shhs1-200001", hrv_ann_path, rec_type="hrv_summary"
            )
            df_hrv_ann = pd.read_csv(file_path, engine="python")
            file_path = self.match_full_rec_path(
                "shhs2-200001", hrv_ann_path, rec_type="hrv_summary"
            )
            df_hrv_ann = pd.concat(
                [df_hrv_ann, pd.read_csv(file_path, engine="python")]
            )
            return df_hrv_ann
        file_path = self.match_full_rec_path(rec, hrv_ann_path, rec_type="hrv_summary")

        df_hrv_ann = pd.read_csv(file_path, engine="python")
        if rec is None:
            return df_hrv_ann

        df_hrv_ann = df_hrv_ann[
            df_hrv_ann["nsrrid"] == self.get_nsrrid(rec)
        ].reset_index(drop=True)
        return df_hrv_ann

    def load_hrv_detailed_ann(
        self, rec: str, hrv_ann_path: Optional[str] = None
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        hrv_ann_path: str, optional,
            path of the detailed HRV annotation file,
            if not given, default path will be used

        Returns
        -------
        df_hrv_ann, DataFrame,
            detailed HRV annotations of `rec`
        """
        file_path = self.match_full_rec_path(rec, hrv_ann_path, rec_type="hrv_5min")

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Record {rec} has no HRV annotation (including sleep annotaions). "
                f"Or the annotation file has not been downloaded yet. Or the path {file_path} is not correct. Please check!"
            )

        self.logger.info(
            f"HRV annotations of record {rec} will be loaded from the file\n{str(file_path)}"
        )

        df_hrv_ann = pd.read_csv(file_path, engine="python")
        df_hrv_ann = df_hrv_ann[
            df_hrv_ann["nsrrid"] == self.get_nsrrid(rec)
        ].reset_index(drop=True)

        self.logger.info(
            f"Record {rec} has {len(df_hrv_ann)} HRV annotations, with {len(self.hrv_ann_detailed_keys)} column(s)"
        )

        return df_hrv_ann

    def load_sleep_ann(
        self, rec: str, source: str, sleep_ann_path: Optional[str] = None
    ) -> Union[pd.DataFrame, dict]:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        source: str, can be "hrv", "event", "event_profusion",
            source of the annotations
        sleep_ann_path: str, optional,
            path of the file which contains the sleep annotations,
            if not given, default path will be used

        Returns
        -------
        df_sleep_ann, DataFrame or dict,
            all annotations on sleep of `rec`
        """
        if source.lower() == "hrv":
            df_hrv_ann = self.load_hrv_detailed_ann(
                rec=rec, hrv_ann_path=sleep_ann_path
            )
            df_sleep_ann = df_hrv_ann[self.sleep_ann_keys_from_hrv].reset_index(
                drop=True
            )
            self.logger.info(
                f"record {rec} has {len(df_sleep_ann)} sleep annotations from corresponding hrv annotation file, with {len(self.sleep_ann_keys_from_hrv)} column(s)"
            )
        elif source.lower() == "event":
            df_event_ann = self.load_event_ann(
                rec, event_ann_path=sleep_ann_path, simplify=False
            )
            _cols = ["EventType", "EventConcept", "Start", "Duration", "SignalLocation"]
            df_sleep_ann = df_event_ann[_cols]
            self.logger.info(
                f"record {rec} has {len(df_sleep_ann)} sleep annotations from corresponding event-nsrr annotation file, with {len(_cols)} column(s)"
            )
        elif source.lower() == "event_profusion":
            df_event_ann = self.load_event_profusion_ann(rec)
            # temporarily finished
            # latter to make imporvements
            df_sleep_ann = df_event_ann
            self.logger.info(
                f"record {rec} has {len(df_sleep_ann['df_events'])} sleep event annotations from corresponding event-profusion annotation file, with {len(df_sleep_ann['df_events'].columns)} column(s)"
            )
        return df_sleep_ann

    def load_sleep_stage_ann(
        self,
        rec: str,
        source: str,
        sleep_stage_ann_path: Optional[str] = None,
        sleep_stage_protocol: str = "aasm",
        with_stage_names: bool = True,
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        source: str, can be "hrv", "event", "event_profusion",
            source of the annotations
        sleep_stage_ann_path: str, optional,
            path of the file which contains the sleep stage annotations,
            if not given, default path will be used
        sleep_stage_protocol: str, default "aasm",
            the protocol to classify sleep stages. currently can be "aasm", "simplified", "shhs"
            the only difference lies in the number of different stages of the NREM periods
        with_stage_names: bool, default True,
            as the argument name implies

        Returns
        -------
        df_sleep_stage_ann, DataFrame,
            all annotations on sleep stage of `rec`
        """
        self.sleep_stage_protocol = sleep_stage_protocol
        self.update_sleep_stage_names()

        df_sleep_ann = self.load_sleep_ann(
            rec=rec, source=source, sleep_ann_path=sleep_stage_ann_path
        )

        df_sleep_stage_ann = pd.DataFrame(columns=self.sleep_stage_keys)
        if source.lower() == "hrv":
            df_tmp = df_sleep_ann[self.sleep_stage_ann_keys_from_hrv].reset_index(
                drop=True
            )
            for _, row in df_tmp.iterrows():
                start_sec = row["Start__sec_"]
                l_start_sec = np.arange(
                    start_sec,
                    start_sec + self.hrv_ann_epoch_len_sec,
                    self.sleep_epoch_len_sec,
                )
                l_sleep_stage = np.array(
                    [
                        row[self.sleep_stage_ann_keys_from_hrv[i]]
                        for i in range(
                            1,
                            1 + self.hrv_ann_epoch_len_sec // self.sleep_epoch_len_sec,
                        )
                    ]
                )
                df_to_concat = pd.DataFrame(
                    {"start_sec": l_start_sec, "sleep_stage": l_sleep_stage}
                )
                df_sleep_stage_ann = pd.concat(
                    [df_sleep_stage_ann, df_to_concat], axis=0, ignore_index=True
                )
        elif source.lower() == "event":
            df_tmp = df_sleep_ann[df_sleep_ann["EventType"] == "Stages|Stages"][
                ["EventConcept", "Start", "Duration"]
            ].reset_index(drop=True)
            df_tmp["EventConcept"] = df_tmp["EventConcept"].apply(
                lambda s: int(s.split("|")[1])
            )
            for _, row in df_tmp.iterrows():
                start_sec = int(row["Start"])
                duration = int(row["Duration"])
                l_start_sec = np.arange(
                    start_sec, start_sec + duration, self.sleep_epoch_len_sec
                )
                l_sleep_stage = np.full(
                    shape=len(l_start_sec), fill_value=int(row["EventConcept"])
                )
                df_to_concat = pd.DataFrame(
                    {"start_sec": l_start_sec, "sleep_stage": l_sleep_stage}
                )
                df_sleep_stage_ann = pd.concat(
                    [df_sleep_stage_ann, df_to_concat], axis=0, ignore_index=True
                )
        elif source.lower() == "event_profusion":
            df_sleep_stage_ann = pd.DataFrame(
                {
                    "start_sec": 30 * np.arange(len(df_sleep_ann["sleep_stage_list"])),
                    "sleep_stage": df_sleep_ann["sleep_stage_list"],
                }
            )

        df_sleep_stage_ann = df_sleep_stage_ann[self.sleep_stage_keys]

        if self.sleep_stage_protocol == "aasm":
            df_sleep_stage_ann["sleep_stage"] = df_sleep_stage_ann["sleep_stage"].apply(
                lambda a: self._to_aasm_states[a]
            )
        elif self.sleep_stage_protocol == "simplified":
            df_sleep_stage_ann["sleep_stage"] = df_sleep_stage_ann["sleep_stage"].apply(
                lambda a: self._to_simplified_states[a]
            )
        elif self.sleep_stage_protocol == "shhs":
            df_sleep_stage_ann["sleep_stage"] = df_sleep_stage_ann["sleep_stage"].apply(
                lambda a: self._to_shhs_states[a]
            )

        if with_stage_names:
            df_sleep_stage_ann["sleep_stage_name"] = df_sleep_stage_ann[
                "sleep_stage"
            ].apply(lambda a: self.sleep_stage_names[a])

        if source.lower() != "event_profusion":
            self.logger.info(
                f"record {rec} has {len(df_tmp)} raw (epoch_len = 5min) sleep stage annotations, with {len(self.sleep_stage_ann_keys_from_hrv)} column(s)"
            )
            self.logger.info(
                f"after being transformed (epoch_len = 30sec), record {rec} has {len(df_sleep_stage_ann)} sleep stage annotations, with {len(self.sleep_stage_keys)} column(s)"
            )

        return df_sleep_stage_ann

    def load_sleep_event_ann(
        self,
        rec: str,
        source: str,
        event_types: Optional[List[str]] = None,
        sleep_event_ann_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        source: str, can be "hrv", "event", "event_profusion",
            source of the annotations
        event_types: list of str (cases ignored), optional,
            "Respiratory" (including "Apnea", "SpO2"), "Arousal",
            "Apnea" (including "CSA", "OSA", "MSA", "Hypopnea"), "SpO2",
            "CSA", "OSA", "MSA", "Hypopnea",
            used only when `source` = "event" or "event_profusion"
        sleep_event_ann_path: str, optional,
            path of the file which contains the sleep event annotations,
            if not given, default path will be used

        Returns
        -------
        df_sleep_event_ann, DataFrame,
            all annotations on sleep events of `rec`
        """
        df_sleep_ann = self.load_sleep_ann(
            rec=rec, source=source, sleep_ann_path=sleep_event_ann_path
        )

        df_sleep_event_ann = pd.DataFrame(columns=self.sleep_event_keys)

        _et = []
        if source.lower() != "hrv":
            if event_types is None or len(event_types) == 0:
                raise ValueError(
                    f"When `source` is \042{source}\042, please specify legal `event_types`!"
                )
            else:
                _et = [s.lower() for s in event_types]

        self.logger.info(f"for record {rec}, _et (event_types) = {_et}")

        if source.lower() == "hrv":
            df_sleep_ann = df_sleep_ann[self.sleep_event_ann_keys_from_hrv].reset_index(
                drop=True
            )
            df_sleep_event_ann = pd.DataFrame(columns=self.sleep_event_keys[1:3])
            for _, row in df_sleep_ann.iterrows():
                if row["hasrespevent"] == 0:
                    continue
                l_events = row[self.sleep_event_ann_keys_from_hrv[1:-1]].values.reshape(
                    (len(self.sleep_event_ann_keys_from_hrv) // 2 - 1, 2)
                )
                l_events = l_events[~np.isnan(l_events[:, 0])]
                df_to_concat = pd.DataFrame(
                    l_events, columns=self.sleep_event_keys[1:3]
                )
                df_sleep_event_ann = pd.concat(
                    [df_sleep_event_ann, df_to_concat], axis=0, ignore_index=True
                )
            df_sleep_event_ann["event_name"] = None
            df_sleep_event_ann["event_duration"] = df_sleep_event_ann.apply(
                lambda row: row["event_end"] - row["event_start"], axis=1
            )
            df_sleep_event_ann = df_sleep_event_ann[self.sleep_event_keys]

            self.logger.info(
                f"record {rec} has {len(df_sleep_ann)} raw (epoch_len = 5min) sleep event annotations from hrv, with {len(self.sleep_event_ann_keys_from_hrv)} column(s)"
            )
            self.logger.info(
                f"after being transformed, record {rec} has {len(df_sleep_event_ann)} sleep event(s)"
            )
        elif source.lower() == "event":
            _cols = set()
            if "respiratory" in _et:
                _cols = _cols | set(self.long_event_names_from_event[:6])
            if "arousal" in _et:
                _cols = _cols | set(self.long_event_names_from_event[6:11])
            if "apnea" in _et:
                _cols = _cols | set(self.long_event_names_from_event[:4])
            if "spo2" in _et:
                _cols = _cols | set(self.long_event_names_from_event[4:6])
            if "csa" in _et:
                _cols = _cols | set(self.long_event_names_from_event[0:1])
            if "osa" in _et:
                _cols = _cols | set(self.long_event_names_from_event[1:2])
            if "msa" in _et:
                _cols = _cols | set(self.long_event_names_from_event[2:3])
            if "hypopnea" in _et:
                _cols = _cols | set(self.long_event_names_from_event[3:4])
            _cols = list(_cols)

            print(f"for record {rec}, _cols = {_cols}")

            df_sleep_event_ann = df_sleep_ann[
                df_sleep_ann["EventConcept"].isin(_cols)
            ].reset_index(drop=True)
            df_sleep_event_ann = df_sleep_event_ann.rename(
                {
                    "EventConcept": "event_name",
                    "Start": "event_start",
                    "Duration": "event_duration",
                },
                axis=1,
            )
            df_sleep_event_ann["event_name"] = df_sleep_event_ann["event_name"].apply(
                lambda s: s.split("|")[1]
            )
            df_sleep_event_ann["event_end"] = df_sleep_event_ann.apply(
                lambda row: row["event_start"] + row["event_duration"], axis=1
            )
            df_sleep_event_ann = df_sleep_event_ann[self.sleep_event_keys]
        elif source.lower() == "event_profusion":
            df_sleep_ann = df_sleep_ann["df_events"]
            _cols = set()
            if "respiratory" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[:6])
            if "arousal" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[6:8])
            if "apnea" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[:4])
            if "spo2" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[4:6])
            if "csa" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[0:1])
            if "osa" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[1:2])
            if "msa" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[2:3])
            if "hypopnea" in _et:
                _cols = _cols | set(self.event_names_from_event_profusion[3:4])
            _cols = list(_cols)

            print(f"for record {rec}, _cols = {_cols}")

            df_sleep_event_ann = df_sleep_ann[
                df_sleep_ann["Name"].isin(_cols)
            ].reset_index(drop=True)
            df_sleep_event_ann = df_sleep_event_ann.rename(
                {
                    "Name": "event_name",
                    "Start": "event_start",
                    "Duration": "event_duration",
                },
                axis=1,
            )
            df_sleep_event_ann["event_end"] = df_sleep_event_ann.apply(
                lambda row: row["event_start"] + row["event_duration"], axis=1
            )
            df_sleep_event_ann = df_sleep_event_ann[self.sleep_event_keys]

        return df_sleep_event_ann

    def load_apnea_ann(
        self,
        rec: str,
        source: str,
        apnea_types: Optional[List[str]] = None,
        apnea_ann_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        source: str, can be "event", "event_profusion",
            source of the annotations
        apnea_types: list of str (cases ignored), optional,
            "CSA", "OSA", "MSA", "Hypopnea",
            if is None, then all types of apnea will be loaded
        apnea_ann_path: str, optional,
            path of the file which contains the apnea event annotations,
            if not given, default path will be used

        Returns
        -------
        df_apnea_ann, DataFrame,
            all annotations on apnea events of `rec`
        """
        event_types = ["apnea"] if apnea_types is None else apnea_types
        if source not in ["event", "event_profusion"]:
            raise ValueError(f"source {source} contains no apnea annotations")
        df_apnea_ann = self.load_sleep_event_ann(
            rec=rec,
            source=source,
            event_types=event_types,
            sleep_event_ann_path=apnea_ann_path,
        )
        return df_apnea_ann

    def load_wave_delineation(
        self,
        rec: str,
        wave_deli_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        wave_deli_path: str, optional,
            path of the file which contains wave delineation annotations,
            if not given, default path will be used

        Returns
        -------
        df_wave_delineation, DataFrame,
            all annotations on wave delineations of `rec`

        NOTE: see the part describing wave delineation annotations of the docstring of the class, or call `self.database_info(detailed=True)`
        """
        file_path = self.match_full_rec_path(
            rec, wave_deli_path, rec_type="wave_delineation"
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                f"The annotation file of wave delineation of record {rec} has not been downloaded yet. "
                f"Or the path {str(file_path)} is not correct. Please check!"
            )

        df_wave_delineation = pd.read_csv(file_path, engine="python")
        df_wave_delineation = df_wave_delineation[self.wave_deli_keys].reset_index(
            drop=True
        )
        return df_wave_delineation

    def load_rpeak_ann(
        self,
        rec: str,
        rpeak_ann_path: Optional[str] = None,
        exclude_artifacts: bool = True,
        exclude_abnormal_beats: bool = True,
        to_ts: bool = False,
    ) -> np.ndarray:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rpeak_ann_path: str, optional,
            annotation file path,
            if not given, default path will be used
        exclude_artifacts: bool, default True,
            exlcude those beats (R peaks) that are labelled artifact or not
        exclude_abnormal_beats: bool, default True,
            exlcude those beats (R peaks) that are labelled abnormal ("VE" and "SVE") or not
        to_ts: bool, default False,

        Returns
        -------

        """
        info_items = ["Type", "rpointadj", "samplingrate"]
        df_rpeaks_with_type_info = self.load_wave_delineation(rec, rpeak_ann_path)[
            info_items
        ]
        exclude_beat_types = []
        # 0 = Artifact, 1 = Normal Sinus Beat, 2 = VE, 3 = SVE
        if exclude_artifacts:
            exclude_beat_types.append(0)
        if exclude_abnormal_beats:
            exclude_beat_types += [2, 3]

        ret = df_rpeaks_with_type_info[
            ~df_rpeaks_with_type_info["Type"].isin(exclude_beat_types)
        ]["rpointadj"].values

        if to_ts:
            fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
            ret = ret * 1000 / fs

        return (np.round(ret)).astype(int)

    def load_rr_ann(self, rec: str, rpeak_ann_path: Optional[str] = None) -> np.ndarray:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rpeak_ann_path: str, optional,
            annotation file path,
            if not given, default path will be used

        Returns
        -------
        rr: ndarray,
            array of rr intervals
        """
        rpeaks_ts = self.load_rpeak_ann(
            rec=rec,
            rpeak_ann_path=rpeak_ann_path,
            exclude_artifacts=True,
            exclude_abnormal_beats=True,
            to_ts=True,
        )
        rr = np.diff(rpeaks_ts)
        rr = np.column_stack((rpeaks_ts[:-1], rr))
        return rr

    def load_nn_ann(self, rec: str, rpeak_ann_path: Optional[str] = None) -> np.ndarray:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        rpeak_ann_path: str, optional,
            annotation file path,
            if not given, default path will be used

        Returns
        -------
        nn: ndarray,
            array of nn intervals
        """
        info_items = ["Type", "rpointadj", "samplingrate"]
        df_rpeaks_with_type_info = self.load_wave_delineation(rec, rpeak_ann_path)[
            info_items
        ]
        fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
        rpeaks_ts = (
            np.round(df_rpeaks_with_type_info["rpointadj"] * 1000 / fs)
        ).astype(int)
        rr = np.diff(rpeaks_ts)
        rr = np.column_stack((rpeaks_ts[:-1], rr))

        normal_sinus_rpeak_indices = np.where(
            df_rpeaks_with_type_info["Type"].values == 1
        )[
            0
        ]  # 1 = Normal Sinus Beat
        keep_indices = np.where(np.diff(normal_sinus_rpeak_indices) == 1)[0].tolist()
        nn = rr[normal_sinus_rpeak_indices[keep_indices]]
        return nn

    def locate_artifacts(
        self, rec: str, wave_deli_path: Optional[str] = None
    ) -> np.ndarray:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        wave_deli_path: str, optional,
            annotation file path,
            if not given, default path will be used

        Returns
        -------
        ndarray,
            indices of artifacts
        """
        df_rpeaks_with_type_info = self.load_wave_delineation(rec, wave_deli_path)[
            ["Type", "rpointadj"]
        ]

        return (
            np.round(
                df_rpeaks_with_type_info[df_rpeaks_with_type_info["Type"] == 0][
                    "rpointadj"
                ].values
            )
        ).astype(int)

    def locate_abnormal_beats(
        self,
        rec: str,
        wave_deli_path: Optional[str] = None,
        abnormal_type: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        wave_deli_path: str, optional,
            annotation file path,
            if not given, default path will be used
        abnormal_type: str, optional,
            type of abnormal beat type to locate, can be "VE", "SVE",
            if not given, both "VE" and "SVE" will be located

        Returns
        -------
        dict
        """
        if abnormal_type is not None and abnormal_type not in ["VE", "SVE"]:
            raise ValueError(
                f"No abnormal type of {abnormal_type} in wave delineation annotation files"
            )

        df_rpeaks_with_type_info = self.load_wave_delineation(rec, wave_deli_path)[
            ["Type", "rpointadj"]
        ]

        # 2 = VE, 3 = SVE
        ve = (
            np.round(
                df_rpeaks_with_type_info[df_rpeaks_with_type_info["Type"] == 2][
                    "rpointadj"
                ].values
            )
        ).astype(int)
        sve = (
            np.round(
                df_rpeaks_with_type_info[df_rpeaks_with_type_info["Type"] == 3][
                    "rpointadj"
                ].values
            )
        ).astype(int)

        abnormal_rpeaks = {"VE": ve, "SVE": sve}

        if abnormal_type is None:
            return abnormal_rpeaks
        elif abnormal_type in ["VE", "SVE"]:
            return {abnormal_type: abnormal_rpeaks[abnormal_type]}

    def load_eeg_band_ann(
        self, rec: str, eeg_band_ann_path: Optional[str] = None
    ) -> pd.DataFrame:
        """not finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        eeg_band_ann_path: str, optional,
            annotation file path,
            if not given, default path will be used

        Returns
        -------
        to write,
        """
        if self.current_version >= "0.15.0":
            print("EEG spectral summary variables are removed in this version")
        else:
            raise NotImplementedError

    def load_eeg_spectral_ann(
        self, rec: str, eeg_spectral_ann_path: Optional[str] = None
    ) -> pd.DataFrame:
        """not finished,

        Parameters
        ----------
        rec: str,
            record name, typically in the form "shhs1-200001"
        eeg_spectral_ann_path: str, optional,
            annotation file path,
            if not given, default path will be used

        Returns
        -------
        to write,
        """
        if self.current_version >= "0.15.0":
            print("EEG spectral summary variables are removed in this version")
        else:
            raise NotImplementedError

    # TODO: add more functions for annotation reading
    # TODO: add plotting functions

    def plot_ann(
        self,
        rec: str,
        stage_source: Optional[str] = None,
        stage_kw: dict = {},
        event_source: Optional[str] = None,
        event_kw: dict = {},
    ) -> NoReturn:
        """

        Parameters
        ----------
        rec, str,
            record name, typically in the form "shhs1-200001"
        stage_source: str, optional,
            source of the sleep stage annotations,
            can be one of "hrv", "event", "event_profusion",
            if is None, then annotations of sleep stages of `rec` won"t be plotted
        stage_kw: dict, default {},
            arguments to the function `self.load_sleep_stage_ann`
        event_source: str, optional,
            source of the sleep event annotations,
            can be one of "hrv", "event", "event_profusion",
            if is None, then annotations of sleep events of `rec` won"t be plotted
        event_kw: dict, default {},
            arguments to the function `self.load_sleep_event_ann`
        """
        if all([stage_source is None, event_source is None]):
            raise ValueError("No input data!")

        if stage_source is not None:
            df_sleep_stage = self.load_sleep_stage_ann(
                rec, source=stage_source, **stage_kw
            )
        else:
            df_sleep_stage = None
        if event_source is not None:
            df_sleep_event = self.load_sleep_event_ann(
                rec, source=event_source, **event_kw
            )
        else:
            df_sleep_event = None

        self._plot_ann(df_sleep_stage=df_sleep_stage, df_sleep_event=df_sleep_event)

    def _plot_ann(
        self,
        df_sleep_stage: Optional[pd.DataFrame] = None,
        df_sleep_event: Optional[pd.DataFrame] = None,
    ) -> NoReturn:
        """not finished,

        Parameters
        ----------
        df_sleep_stage: DataFrame, optional,
            sleep stage annotations
        df_sleep_event: DataFrame, optional,
            sleep event annotations
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        check = [df_sleep_stage is None, df_sleep_event is None]
        nb_axes = len(check) - np.sum(check)

        if nb_axes == 0:
            raise ValueError("No input data!")

        patches = {k: mpatches.Patch(color=c, label=k) for k, c in self.palette.items()}

        if df_sleep_stage is not None:
            sleep_stages = {}
            for k in self.sleep_stage_names:
                sleep_stages[k] = intervals_union(
                    interval_list=[
                        [sec, sec + self.sleep_epoch_len_sec]
                        for sec in df_sleep_stage[
                            df_sleep_stage["sleep_stage"]
                            == self.sleep_stage_name_value_mapping[k]
                        ]["start_sec"].values
                    ],
                    join_book_endeds=True,
                )

        if df_sleep_event is not None:
            current_legal_events = [
                "Central Apnea",
                "Obstructive Apnea",
                "Mixed Apnea",
                "Hypopnea",
            ]
            if len(current_legal_events) != len(
                set(current_legal_events) | set(df_sleep_event["event_name"])
            ):
                raise ValueError(
                    "Plotting of some type of events in `df_sleep_event` has not been implemented yet!"
                )

        _, axes = plt.subplots(nb_axes, 1, figsize=(20, 4 * nb_axes), sharex=True)
        plt.subplots_adjust(hspace=0)
        plot_alpha = 0.5

        ax_stages, ax_events = None, None
        if nb_axes == 1 and df_sleep_stage is not None:
            ax_stages = axes
            ax_stages.set_title("Sleep Stages", fontsize=24)
            ax_stages.set_xlabel("Time", fontsize=16)
            # ax_stages.set_ylabel("Stages", fontsize=16)
        elif nb_axes == 1 and df_sleep_event is not None:
            ax_events = axes
            ax_events.set_title("Sleep Events", fontsize=24)
            ax_events.set_xlabel("Time", fontsize=16)
            # ax_events.set_ylabel("Events", fontsize=16)
        else:
            ax_stages, ax_events = axes
            ax_stages.set_title("Sleep Stages and Events", fontsize=24)
            ax_events.set_xlabel("Time", fontsize=16)

        if ax_stages is not None:
            for k, v in sleep_stages.items():
                for itv in v:
                    ax_stages.axvspan(
                        datetime.fromtimestamp(itv[0]),
                        datetime.fromtimestamp(itv[1]),
                        color=self.palette[k],
                        alpha=plot_alpha,
                    )
            ax_stages.legend(
                handles=[
                    patches[k]
                    for k in self.all_sleep_stage_names
                    if k in sleep_stages.keys()
                ],
                loc="best",
            )  # keep ordering
            plt.setp(ax_stages.get_yticklabels(), visible=False)
            ax_stages.tick_params(axis="y", which="both", length=0)

        if ax_events is not None:
            for _, row in df_sleep_event.iterrows():
                ax_events.axvspan(
                    datetime.fromtimestamp(row["event_start"]),
                    datetime.fromtimestamp(row["event_end"]),
                    color=self.palette[row["event_name"]],
                    alpha=plot_alpha,
                )
            ax_events.legend(
                handles=[
                    patches[k]
                    for k in current_legal_events
                    if k in set(df_sleep_event["event_name"])
                ],
                loc="best",
            )  # keep ordering
            plt.setp(ax_events.get_yticklabels(), visible=False)
            ax_events.tick_params(axis="y", which="both", length=0)

    def str_to_real_number(self, s: Union[str, Real]) -> Real:
        """finished,

        some columns in the annotations might incorrectly been converted from real number to string, using `xmltodict`.

        Parameters
        ----------
        s: str or real number (NaN)
        """
        if isinstance(s, str):
            if "." in s:
                return float(s)
            else:
                return int(s)
        else:  # NaN case
            return s

    @property
    def url(self) -> str:
        warnings.warn(
            "one has to apply for a token and uses `nsrr` to download the data"
        )
        return ""
