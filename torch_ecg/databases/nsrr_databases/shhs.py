# -*- coding: utf-8 -*-

import itertools
import re
import warnings
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import scipy.signal as SS
import xmltodict as xtd
from tqdm.auto import tqdm

from ...cfg import DEFAULTS
from ...utils.misc import add_docstring
from ...utils.utils_interval import intervals_union
from ..base import NSRRDataBase, DataBaseInfo, PSGDataBaseMixin


__all__ = [
    "SHHS",
]


_SHHS_INFO = DataBaseInfo(
    title="""
    Sleep Heart Health Study
    """,
    about=r"""
    **ABOUT the dataset** (Main webpage [1]_):

    1. shhs1 (Visit 1):

        - the baseline clinic visit and polysomnogram performed between November 1, 1995 and January 31, 1998
        - in all, 6,441 men and women aged 40 years and older were enrolled
        - 5,804 rows, down from the original 6,441 due to data sharing rules on certain cohorts and subjects

    2. shhs-interim-followup (Interim Follow-up):

        - an interim clinic visit or phone call 2-3 years after baseline (shhs1)
        - 5,804 rows, despite some subjects not having complete data, all original subjects are present in the dataset

    3. shhs2 (Visit 2):

        - the follow-up clinic visit and polysomnogram performed between January 2001 and June 2003
        - during this exam cycle 3, a second polysomnogram was obtained in 3,295 of the participants
        - 4,080 rows, not all cohorts and subjects took part

    4. shhs-cvd (CVD Outcomes):

        - the tracking of adjudicated heart health outcomes (e.g. stroke, heart attack) between baseline (shhs1) and 2008-2011 (varies by parent cohort)
        - 5,802 rows, outcomes data were not provided on all subjects

    5. shhs-cvd-events (CVD Outcome Events):

        - event-level details for the tracking of heart health outcomes (shhs-cvd)
        - 4,839 rows, representing individual events

    6. ECG was sampled at 125 Hz in shhs1 and 250/256 Hz in shhs2
    7. `annotations-events-nsrr` and `annotations-events-profusion`:
       annotation files both contain xml files, the former processed in the EDF Editor and Translator tool,
       the latter exported from Compumedics Profusion
    8. about 10% of the records have HRV (including sleep stages and sleep events) annotations

    **DATA Analysis Tips**:

    1. Respiratory Disturbance Index (RDI):

        - A number of RDI variables exist in the data set. These variables are highly skewed.
        - log-transformation is recommended, among which the following transformation performed best, at least in some subsets:

          .. math::

            NEWVA = log(OLDVAR + 0.1)

    2. Obstructive Apnea Index (OAI):

        - There is one OAI index in the data set. It reflects obstructive events associated with a 4% desaturation or arousal. Nearly 30% of the cohort has a zero value for this variable
        - Dichotomization is suggested (e.g. >=3 or >=4 events per hour indicates positive)

    3. Central Apnea Index (CAI):

        - Several variables describe central breathing events, with different thresholds for desaturation and requirement/non-requirement of arousals. ~58% of the cohort have zero values
        - Dichotomization is suggested (e.g. >=3 or >=4 events per hour indicates positive)

    4. Sleep Stages:

        - Stage 1 and stage 3-4 are not normally distributed, but stage 2 and REM sleep are.
        - To use these data as continuous dependent variables, stages 1 and 3-4 must be transformed. The following formula is suggested:

          .. math::

            –log(-log(val/100+0.001))

    5. Sleep time below 90% O2:

        - Percent of total sleep time with oxygen levels below 75%, 80%, 85% and 90% were recorded
        - Dichotomization is suggested (e.g. >5% and >10% of sleep time with oxygen levels below a specific O2 level indicates positive)

    **ABOUT signals**: (ref. [9]_)

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

    **ABOUT annotations** (NOT including "nsrrid", "visitnumber", "pptid" etc.):

    1. hrv annotations: (in csv files, ref. [2]_)

        +-------------+------------------------------------------------------------------+
        | Start__sec_ | 5 minute window start time                                       |
        +-------------+------------------------------------------------------------------+
        | NN_RR       | Ratio of consecutive normal sinus beats (NN)                     |
        |             | over all cardiac inter-beat (RR) intervals (NN/RR)               |
        +-------------+------------------------------------------------------------------+
        | AVNN        | Mean of all normal sinus to normal sinus interbeat               |
        |             | intervals (NN)                                                   |
        +-------------+------------------------------------------------------------------+
        | IHR         | Instantaneous heart rate                                         |
        +-------------+------------------------------------------------------------------+
        | SDNN        | Standard deviation of all normal sinus                           |
        |             | to normal sinus interbeat (NN) intervals                         |
        +-------------+------------------------------------------------------------------+
        | SDANN       | Standard deviation of the averages of normal sinus to normal     |
        |             | sinus interbeat (NN) intervals in all 5-minute segments          |
        +-------------+------------------------------------------------------------------+
        | SDNNIDX     | Mean of the standard deviations of normal sinus to normal        |
        |             | sinus interbeat (NN) intervals in all 5-minute segments          |
        +-------------+------------------------------------------------------------------+
        | rMSSD       | Square root of the mean of the squares of difference between     |
        |             | adjacent normal sinus to normal sinus interbeat (NN) intervals   |
        +-------------+------------------------------------------------------------------+
        | pNN10       | Percentage of differences between adjacent normal sinus to       |
        |             | normal sinus interbeat (NN) intervals that are >10 ms            |
        +-------------+------------------------------------------------------------------+
        | pNN20       | Percentage of differences between adjacent normal sinus to       |
        |             | normal sinus interbeat (NN) intervals that are >20 ms            |
        +-------------+------------------------------------------------------------------+
        | pNN30       | Percentage of differences between adjacent normal sinus to       |
        |             | normal sinus interbeat (NN) intervals that are >30 ms            |
        +-------------+------------------------------------------------------------------+
        | pNN40       | Percentage of differences between adjacent normal sinus to       |
        |             | normal sinus interbeat (NN) intervals that are >40 ms            |
        +-------------+------------------------------------------------------------------+
        | pNN50       | Percentage of differences between adjacent normal sinus to       |
        |             | normal sinus interbeat (NN) intervals that are >50 ms            |
        +-------------+------------------------------------------------------------------+
        | tot_pwr     | Total normal sinus to normal sinus interbeat (NN) interval       |
        |             | spectral power up to 0.4 Hz                                      |
        +-------------+------------------------------------------------------------------+
        | ULF         | Ultra-low frequency power, the normal sinus to normal sinus      |
        |             | interbeat (NN) interval spectral power between 0 and 0.003 Hz    |
        +-------------+------------------------------------------------------------------+
        | VLF         | Very low frequency power, the normal sinus to normal sinus       |
        |             | interbeat (NN) interval spectral power between 0.003 and 0.04 Hz |
        +-------------+------------------------------------------------------------------+
        | LF          | Low frequency power, the normal sinus to normal sinus interbeat  |
        |             | (NN) interval spectral power between 0.04 and 0.15 Hz            |
        +-------------+------------------------------------------------------------------+
        | HF          | High frequency power, the normal sinus to normal sinus interbeat |
        |             | (NN) interval spectral power between 0.15 and 0.4 Hz             |
        +-------------+------------------------------------------------------------------+
        | LF_HF       | The ratio of low to high frequency                               |
        +-------------+------------------------------------------------------------------+
        | LF_n        | Low frequency power (normalized)                                 |
        +-------------+------------------------------------------------------------------+
        | HF_n        | High frequency power (normalized)                                |
        +-------------+------------------------------------------------------------------+

    2. wave delineation annotations: (in csv files, NOTE: see "CAUTION" by the end of this part, ref. [2]_)

        +--------------+------------------------------------------------------------------------------------------------+
        | RPoint       | Sample Number indicating R Point (peak of QRS)                                                 |
        +--------------+------------------------------------------------------------------------------------------------+
        | Start        | Sample Number indicating start of beat                                                         |
        +--------------+------------------------------------------------------------------------------------------------+
        | End          | Sample Number indicating end of beat                                                           |
        +--------------+------------------------------------------------------------------------------------------------+
        | STLevel1     | Level of ECG 1 in Raw data ( 65536 peak to peak rawdata = 10mV peak to peak)                   |
        +--------------+------------------------------------------------------------------------------------------------+
        | STSlope1     | Slope of ECG 1 stored as int and to convert to a double divide raw value by 1000.0             |
        +--------------+------------------------------------------------------------------------------------------------+
        | STLevel2     | Level of ECG 2 in Raw data ( 65536 peak to peak rawdata = 10mV peak to peak)                   |
        +--------------+------------------------------------------------------------------------------------------------+
        | STSlope2     | Slope of ECG 2 stored as int and to convert to a double divide raw value by 1000.0             |
        +--------------+------------------------------------------------------------------------------------------------+
        | Manual       | (True / False) True if record was manually inserted                                            |
        +--------------+------------------------------------------------------------------------------------------------+
        | Type         | Type of beat (0 = Artifact / 1 = Normal Sinus Beat / 2 = VE / 3 = SVE)                         |
        +--------------+------------------------------------------------------------------------------------------------+
        | Class        | no longer used                                                                                 |
        +--------------+------------------------------------------------------------------------------------------------+
        | PPoint       | Sample Number indicating peak of the P wave (-1 if no P wave detected)                         |
        +--------------+------------------------------------------------------------------------------------------------+
        | PStart       | Sample Number indicating start of the P wave                                                   |
        +--------------+------------------------------------------------------------------------------------------------+
        | PEnd         | Sample Number indicating end of the P wave                                                     |
        +--------------+------------------------------------------------------------------------------------------------+
        | TPoint       | Sample Number indicating peak of the T wave (-1 if no T wave detected)                         |
        +--------------+------------------------------------------------------------------------------------------------+
        | TStart       | Sample Number indicating start of the T wave                                                   |
        +--------------+------------------------------------------------------------------------------------------------+
        | TEnd         | Sample Number indicating end of the T wave                                                     |
        +--------------+------------------------------------------------------------------------------------------------+
        | TemplateID   | The ID of the template to which this beat has been assigned (-1 if not assigned to a template) |
        +--------------+------------------------------------------------------------------------------------------------+
        | nsrrid       | nsrrid of this record                                                                          |
        +--------------+------------------------------------------------------------------------------------------------+
        | samplingrate | frequency of the ECG signal of this record                                                     |
        +--------------+------------------------------------------------------------------------------------------------+
        | seconds      | Number of seconds from beginning of recording to R-point (Rpoint / sampling rate)              |
        +--------------+------------------------------------------------------------------------------------------------+
        | epoch        | Epoch (30 second) number                                                                       |
        +--------------+------------------------------------------------------------------------------------------------+
        | rpointadj    | R Point adjusted sample number (RPoint * (samplingrate/256))                                   |
        +--------------+------------------------------------------------------------------------------------------------+

      CAUTION that all the above sampling numbers except for rpointadj assume 256 Hz,
      while the rpointadj column has been added to provide an adjusted sample number based on the actual sampling rate.

    3. event annotations: (in xml files)
       TODO
    4. event_profusion annotations: (in xml files)
       TODO

    **DEFINITION of concepts in sleep study** (mainly apnea and arousal, ref. [8]_ for corresponding knowledge):

    1. Arousal: (ref. [3]_, [4]_)

        - interruptions of sleep lasting 3 to 15 seconds
        - can occur spontaneously or as a result of sleep-disordered breathing or other sleep disorders
        - sends you back to a lighter stage of sleep
        - if the arousal last more than 15 seconds, it becomes an awakening
        - the higher the arousal index (occurrences per hour), the more tired you are likely to feel, though people vary in their tolerance of sleep disruptions

    2. Central Sleep Apnea (CSA): (ref. [3]_, [5]_, [6]_)

        - breathing repeatedly stops and starts during sleep
        - occurs because your brain (central nervous system) doesn't send proper signals to the muscles that control your breathing, which is point that differs from obstructive sleep apnea
        - may occur as a result of other conditions, such as heart failure, stroke, high altitude, etc.

    3. Obstructive Sleep Apnea (OSA): (ref. [3]_, [7]_)

        - occurs when throat muscles intermittently relax and block upper airway during sleep
        - a noticeable sign of obstructive sleep apnea is snoring

    4. Complex (Mixed) Sleep Apnea: (ref. [3]_)

        - combination of both CSA and OSA
        - exact mechanism of the loss of central respiratory drive during sleep in OSA is unknown but is most likely related to incorrect settings of the CPAP (Continuous Positive Airway Pressure) treatment and other medical conditions the person has

    5. Hypopnea:
       overly shallow breathing or an abnormally low respiratory rate. Hypopnea is defined by some to be less severe than apnea (the complete cessation of breathing)
    6. Apnea Hypopnea Index (AHI): to write

        - used to indicate the severity of OSA
        - number of apneas or hypopneas recorded during the study per hour of sleep
        - based on the AHI, the severity of OSA is classified as follows

            - none/minimal: AHI < 5 per hour
            - mild: AHI ≥ 5, but < 15 per hour
            - moderate: AHI ≥ 15, but < 30 per hour
            - severe: AHI ≥ 30 per hour

    7. Oxygen Desaturation:

        - used to indicate the severity of OSA
        - reductions in blood oxygen levels (desaturation)
        - at sea level, a normal blood oxygen level (saturation) is usually 96 - 97%
        - (no generally accepted classifications for severity of oxygen desaturation)

            - mild: >= 90%
            - moderate: 80% - 89%
            - severe: < 80%

    """,
    usage=[
        "Sleep stage",
        "Sleep apnea",
    ],
    issues="""
    1. `Start__sec_` might not be the start time, but rather the end time, of the 5 minute windows in some records
    2. the current version "0.15.0" removed EEG spectral summary variables
    """,
    references=[
        "https://sleepdata.org/datasets/shhs/pages/",
        "https://sleepdata.org/datasets/shhs/pages/13-hrv-analysis.md",
        "https://en.wikipedia.org/wiki/Sleep_apnea",
        "https://www.sleepapnea.org/treat/getting-sleep-apnea-diagnosis/sleep-study-details/",
        "https://www.mayoclinic.org/diseases-conditions/central-sleep-apnea/symptoms-causes/syc-20352109",
        "Eckert DJ, Jordan AS, Merchia P, Malhotra A. Central sleep apnea: Pathophysiology and treatment. Chest. 2007 Feb;131(2):595-607. doi: 10.1378/chest.06.2287. PMID: 17296668; PMCID: PMC2287191.",
        "https://www.mayoclinic.org/diseases-conditions/obstructive-sleep-apnea/symptoms-causes/syc-20352090",
        "https://en.wikipedia.org/wiki/Hypopnea",
        # "http://healthysleep.med.harvard.edu/sleep-apnea/diagnosing-osa/understanding-results",  # broken link
        "https://sleepdata.org/datasets/shhs/pages/full-description.md",
    ],
    doi=[
        "10.1093/jamia/ocy064",
    ],  # PMID: 9493915 not added
)


@add_docstring(_SHHS_INFO.format_database_docstring(), mode="prepend")
class SHHS(NSRRDataBase, PSGDataBaseMixin):
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
        Auxilliary key word arguments

    """

    __name__ = "SHHS"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="SHHS",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )

        self.__create_constants(**kwargs)

        # `current_version` will be when calling `_ls_rec`
        self.current_version = kwargs.get("current_version", "0.19.0")
        self.version_pattern = "\\d+\\.\\d+\\.\\d+"

        self.rec_name_pattern = "^shhs[12]\\-\\d{6}$"

        self.psg_data_path = None
        self.ann_path = None
        self.hrv_ann_path = None
        self.eeg_ann_path = None
        self.wave_deli_path = None
        self.event_ann_path = None
        self.event_profusion_ann_path = None
        self.form_paths()

        self._df_records = pd.DataFrame()
        self._all_records = []
        self.rec_with_hrv_summary_ann = []
        self.rec_with_hrv_detailed_ann = []
        self.rec_with_event_ann = []
        self.rec_with_event_profusion_ann = []
        self.rec_with_rpeaks_ann = []
        self._tables = {}
        self._ls_rec()

        self.fs = None
        self.file_opened = None

    def form_paths(self) -> None:
        """Form paths to the database files."""
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

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in some private attributes.
        """
        self.logger.info("Finding `edf` records....")
        self._df_records = pd.DataFrame()
        self._df_records["path"] = sorted(self.db_dir.rglob("*.edf"))

        if self._subsample is not None:
            size = min(
                len(self._df_records),
                max(1, int(round(self._subsample * len(self._df_records)))),
            )
            self._df_records = self._df_records.sample(
                n=size, random_state=DEFAULTS.SEED, replace=False
            )

        # if self._df_records is non-empty, call `form_paths` again if necessary
        # typically path for a record is like:
        # self.db_dir / "polysomnography/edfs/shhs1/shhs1-200001.edf"
        if (
            len(self._df_records) > 0
            and self._df_records.iloc[0]["path"].parents[3] != self.db_dir
        ):
            self.db_dir = self._df_records.iloc[0]["path"].parents[3]
            self.form_paths()

        # get other columns
        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records["tranche"] = self._df_records["record"].apply(
            lambda x: x.split("-")[0]
        )
        self._df_records["visitnumber"] = self._df_records["record"].apply(
            lambda x: int(x.split("-")[0][4:])
        )
        self._df_records["nsrrid"] = self._df_records["record"].apply(
            lambda x: int(x.split("-")[1])
        )

        # auxiliary and annotation files
        if not self._df_records.empty:
            for key in self.extension:
                self._df_records[key] = self._df_records.apply(
                    lambda row: self.folder_or_file[key]
                    / row["tranche"]
                    / (row["record"] + self.extension[key]),
                    axis=1,
                )

        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.tolist()

        # update `current_version`
        if self.ann_path.is_dir():
            for file in self.ann_path.iterdir():
                if (
                    file.is_file()
                    and len(re.findall(self.version_pattern, file.name)) > 0
                ):
                    self.current_version = re.findall(self.version_pattern, file.name)[
                        0
                    ]
                    break

            self.logger.info("Loading tables....")
            # gather tables in self.ann_path and in self.hrv_ann_path
            for file in itertools.chain(
                self.ann_path.glob("*.csv"), self.hrv_ann_path.glob("*.csv")
            ):
                if not file.suffix == ".csv":
                    continue
                table_name = file.stem.replace(f"-{self.current_version}", "")
                try:
                    self._tables[table_name] = pd.read_csv(file, low_memory=False)
                except UnicodeDecodeError:
                    self._tables[table_name] = pd.read_csv(
                        file, low_memory=False, encoding="latin-1"
                    )

        self.logger.info("Finding records with HRV annotations....")
        # find records with hrv annotations
        self.rec_with_hrv_summary_ann = []
        for table_name in ["shhs1-hrv-summary", "shhs2-hrv-summary"]:
            if table_name in self._tables:
                self.rec_with_hrv_summary_ann.extend(
                    [
                        f"shhs{int(row['visitnumber'])}-{int(row['nsrrid'])}"
                        for _, row in self._tables[table_name].iterrows()
                    ]
                )
        self.rec_with_hrv_summary_ann = sorted(list(set(self.rec_with_hrv_summary_ann)))
        self.rec_with_hrv_detailed_ann = []
        for table_name in ["shhs1-hrv-5min", "shhs2-hrv-5min"]:
            if table_name in self._tables:
                self.rec_with_hrv_detailed_ann.extend(
                    [
                        f"shhs{int(row['visitnumber'])}-{int(row['nsrrid'])}"
                        for _, row in self._tables[table_name].iterrows()
                    ]
                )
        self.rec_with_hrv_detailed_ann = sorted(
            list(set(self.rec_with_hrv_detailed_ann))
        )

        self.logger.info("Finding records with rpeaks annotations....")
        # find available rpeak annotation files
        self.rec_with_rpeaks_ann = sorted(
            [
                f.stem.replace("-rpoint", "")
                for f in self.wave_deli_path.rglob("shhs*-rpoint.csv")
            ]
        )

        self.logger.info("Finding records with event annotations....")
        # find available event annotation files
        self.rec_with_event_ann = sorted(
            [
                f.stem.replace("-nsrr", "")
                for f in self.event_ann_path.rglob("shhs*-nsrr.xml")
            ]
        )
        self.rec_with_event_profusion_ann = sorted(
            [
                f.stem.replace("-profusion", "")
                for f in self.event_profusion_ann_path.rglob("shhs*-profusion.xml")
            ]
        )

        self._df_records["available_signals"] = None
        if not self.lazy:
            self.get_available_signals(None)

        # END OF `_ls_rec`

    def list_table_names(self) -> List[str]:
        """List available table names."""
        return list(self._tables.keys())

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Get table by name.

        Parameters
        ----------
        table_name : str
            Table name.
            For available table names, call method :meth:`list_table_names`.

        Returns
        -------
        table : pandas.DataFrame
            The loaded table.

        """
        return self._tables[table_name]

    def update_sleep_stage_names(self) -> None:
        """Update :attr:`self.sleep_stage_names`
        according to :attr:`self.sleep_stage_protocol`.
        """
        if self.sleep_stage_protocol == "aasm":
            nb_stages = 5
        elif self.sleep_stage_protocol == "simplified":
            nb_stages = 4
        elif self.sleep_stage_protocol == "shhs":
            nb_stages = 6
        else:
            raise ValueError(f"No stage protocol named `{self.sleep_stage_protocol}`")

        self.sleep_stage_names = self.all_sleep_stage_names[:nb_stages]

    def get_subject_id(self, rec: Union[str, int]) -> int:
        """Attach a unique subject ID for the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.

        Returns
        -------
        pid : int
            Subject ID derived from (attached to) the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        head_shhs1, head_shhs2v3, head_shhs2v4 = "30000", "30001", "30002"
        tranche, nsrrid, visitnumber = [
            self.split_rec_name(rec)[k] for k in ["tranche", "nsrrid", "visitnumber"]
        ]
        if visitnumber == "2":
            raise ValueError(
                "SHHS2 has two different sampling frequencies, "
                "currently could not be distinguished using only `rec`"
            )
        pid = int(head_shhs1 + str(visitnumber) + str(nsrrid))
        return pid

    def get_available_signals(
        self, rec: Union[str, int, None]
    ) -> Union[List[str], None]:
        """Get available signals for a record.

        If input `rec` is None,
        this function finds available signals for all records,
        and assign to :attr:`self._df_records['available_signals']`.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.

        Returns
        -------
        available_signals : List[str]
            Names of available signals for `rec`.

        """
        if rec is None:
            # iterrows with tqdm
            for _, row in tqdm(
                self._df_records.iterrows(),
                total=len(self._df_records),
                desc="Finding available signals",
                unit="record",
                dynamic_ncols=True,
                mininterval=1.0,
                disable=(self.verbose < 1),
            ):
                rec = row.name
                if self._df_records.loc[rec, "available_signals"] is not None:
                    continue
                available_signals = self.get_available_signals(rec)
                self._df_records.at[rec, "available_signals"] = available_signals
            return

        if isinstance(rec, int):
            rec = self[rec]

        if rec in self._df_records.index:
            available_signals = self._df_records.loc[rec, "available_signals"]
            if available_signals is not None and len(available_signals) > 0:
                return available_signals

            frp = self.get_absolute_path(rec)
            try:
                # perhaps broken file
                # or the downloading is not finished
                self.safe_edf_file_operation("open", frp)
            except OSError:
                return None
            available_signals = [s.lower() for s in self.file_opened.getSignalLabels()]
            self.safe_edf_file_operation("close")
            self._df_records.at[rec, "available_signals"] = available_signals
            self.all_signals = self.all_signals.union(set(available_signals))
        else:
            available_signals = []
        return available_signals

    def split_rec_name(self, rec: Union[str, int]) -> Dict[str, Union[str, int]]:
        """Split `rec` into `tranche`, `visitnumber`, `nsrrid`

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in attr:`self.all_records`.

        Returns
        -------
        dict
            Keys: "tranche", "visitnumber", "nsrrid".

        """
        if isinstance(rec, int):
            rec = self[rec]
        assert isinstance(rec, str) and re.match(
            self.rec_name_pattern, rec
        ), f"Invalid record name: `{rec}`"
        tranche, nsrrid = rec.split("-")
        visitnumber = tranche[-1]
        return {
            "tranche": tranche,
            "visitnumber": int(visitnumber),
            "nsrrid": int(nsrrid),
        }

    def get_visitnumber(self, rec: Union[str, int]) -> int:
        """Get ``visitnumber`` from `rec`.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.

        Returns
        -------
        int
            Visit number extracted from `rec`.

        """
        return self.split_rec_name(rec)["visitnumber"]

    def get_tranche(self, rec: Union[str, int]) -> str:
        """Get ``tranche`` ("shhs1" or "shhs2") from `rec`.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.

        Returns
        -------
        str
            Tranche extracted from `rec`.

        """
        return self.split_rec_name(rec)["tranche"]

    def get_nsrrid(self, rec: Union[str, int]) -> int:
        """Get ``nsrrid`` from `rec`.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.

        Returns
        -------
        int
            ``nsrrid`` extracted from `rec`.

        """
        return self.split_rec_name(rec)["nsrrid"]

    def get_fs(
        self,
        rec: Union[str, int],
        sig: str = "ECG",
        rec_path: Optional[Union[str, Path]] = None,
    ) -> Real:
        """Get the sampling frequency of a signal of a record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        sig : str, default "ECG"
            Signal name or annotation name (e.g. "rpeak").
            Some annotation files (\\*-rpeak.csv) have
            a sampling frequency column.
        rec_path : str or path.Path, optional
            Path of the file which contains the PSG data.
            If is None, default path will be used.

        Returns
        -------
        fs : numbers.Real
            Sampling frequency of the signal `sig` of the record `rec`.
            If corresponding signal (.edf) file is not available,
            or the signal file does not contain the signal `sig`,
            -1 will be returned.

        """
        if isinstance(rec, int):
            rec = self[rec]
        sig = self.match_channel(sig, raise_error=False)
        assert sig in self.all_signals.union({"rpeak"}), f"Invalid signal name: `{sig}`"
        if sig.lower() == "rpeak":
            df_rpeaks_with_type_info = self.load_wave_delineation_ann(rec)
            if df_rpeaks_with_type_info.empty:
                self.logger.info(
                    f"Rpeak annotation file corresponding to `{rec}` is not available."
                )
                return -1
            return df_rpeaks_with_type_info.iloc[0]["samplingrate"]

        frp = self.get_absolute_path(rec, rec_path)
        if not frp.exists():
            self.logger.info(
                f"Signal (.edf) file corresponding to `{rec}` is not available."
            )
            return -1
        self.safe_edf_file_operation("open", frp)
        sig = self.match_channel(sig)
        available_signals = [s.lower() for s in self.file_opened.getSignalLabels()]
        if sig not in available_signals:
            self.logger.info(
                f"Signal `{sig}` is not available in signal file corresponding to `{rec}`."
            )
            return -1
        chn_num = available_signals.index(sig)
        fs = self.file_opened.getSampleFrequency(chn_num)
        self.safe_edf_file_operation("close")
        return fs

    def get_chn_num(
        self,
        rec: Union[str, int],
        sig: str = "ECG",
        rec_path: Optional[Union[str, Path]] = None,
    ) -> int:
        """Get the index of the channel of the signal in the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        sig : str, default "ECG"
            Signal name.
        rec_path : str or pathlib.Path, optional
            Path of the file which contains the PSG data.
            If is None, default path will be used.

        Returns
        -------
        chn_num : int
            Index of channel of the signal `sig` of the record `rec`.
            Returns -1
            if corresponding signal (.edf) file is not available,
            or the signal file does not contain the signal `sig`.

        """
        sig = self.match_channel(sig)
        available_signals = self.get_available_signals(rec)
        if sig not in available_signals:
            if isinstance(rec, int):
                rec = self[rec]
            self.logger.info(
                f"Signal (.edf) file corresponding to `{rec}` is not available, or"
                f"signal `{sig}` is not available in signal file corresponding to `{rec}`."
            )
            return -1
        chn_num = available_signals.index(self.match_channel(sig))
        return chn_num

    def match_channel(self, channel: str, raise_error: bool = True) -> str:
        """Match the channel name to the standard channel name.

        Parameters
        ----------
        channel : str
            Channel name.
        raise_error : bool, default True
            Whether to raise error if no match is found.
            If False, returns the input `channel` directly.

        Returns
        -------
        sig : str
            Standard channel name in SHHS.
            If no match is found, and `raise_error` is False,
            returns the input `channel` directly.

        """
        if channel.lower() in self.all_signals:
            return channel.lower()
        if raise_error:
            raise ValueError(f"No channel named `{channel}`")
        return channel

    def get_absolute_path(
        self,
        rec: Union[str, int],
        rec_path: Optional[Union[str, Path]] = None,
        rec_type: str = "psg",
    ) -> Path:
        """Get the absolute path of specific type of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rec_path : str or pathlib.Path, optional
            Path of the file which contains the desired data.
            If is None, default path will be used.
        rec_type : str, default "psg"
            Record type, either data (psg, etc.) or annotations.

        Returns
        -------
        rp : pathlib.Path
            Absolute path of the record `rec` with type `rec_type`.

        """
        if rec_path is not None:
            rp = Path(rec_path)
            return rp

        assert rec_type in self.folder_or_file, (
            "`rec_type` should be one of "
            f"`{list(self.folder_or_file.keys())}`, but got `{rec_type}`"
        )

        if isinstance(rec, int):
            rec = self[rec]

        tranche, nsrrid = [self.split_rec_name(rec)[k] for k in ["tranche", "nsrrid"]]
        # rp = self._df_records.loc[rec, rec_type]
        rp = (
            self.folder_or_file[rec_type] / tranche / f"{rec}{self.extension[rec_type]}"
        )
        return rp

    def database_stats(self) -> None:
        raise NotImplementedError

    def show_rec_stats(
        self, rec: Union[str, int], rec_path: Optional[Union[str, Path]] = None
    ) -> None:
        """Print the statistics of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rec_path : str or pathlib.Path, optional
            Path of the file which contains the PSG data.
            If is None, default path will be used.

        """
        frp = self.get_absolute_path(rec, rec_path, rec_type="psg")
        self.safe_edf_file_operation("open", frp)
        for chn, lb in enumerate(self.file_opened.getSignalLabels()):
            print("SignalLabel:", lb)
            print("Prefilter:", self.file_opened.getPrefilter(chn))
            print("Transducer:", self.file_opened.getTransducer(chn))
            print("PhysicalDimension:", self.file_opened.getPhysicalDimension(chn))
            print("SampleFrequency:", self.file_opened.getSampleFrequency(chn))
            print("*" * 40)
        self.safe_edf_file_operation("close")

    def load_psg_data(
        self,
        rec: Union[str, int],
        channel: str = "all",
        rec_path: Optional[Union[str, Path]] = None,
        sampfrom: Optional[Real] = None,
        sampto: Optional[Real] = None,
        fs: Optional[int] = None,
        physical: bool = True,
    ) -> Union[Dict[str, Tuple[np.ndarray, Real]], Tuple[np.ndarray, Real]]:
        """Load PSG data of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        channel : str, default "all"
            Name of the channel of PSG.
            If is "all", then all channels will be returned.
        rec_path : str or pathlib.Path, optional
            Path of the file which contains the PSG data.
            If is None, default path will be used.
        sampfrom : numbers.Real, optional
            Start time (units in seconds) of the data to be loaded,
            valid only when `channel` is some specific channel.
        sampto : numbers.Real, optional
            End time (units in seconds) of the data to be loaded,
            valid only when `channel` is some specific channel
        fs : numbers.Real, optional
            Sampling frequency of the loaded data.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
            Valid only when `channel` is some specific channel.
        physical : bool, default True
            If True, then the data will be converted to physical units,
            otherwise, the data will be in digital units.

        Returns
        -------
        dict or tuple
            If `channel` is "all", then a dictionary will be returned:

                - keys: PSG channel names;
                - values: PSG data and sampling frequency

            Otherwise, a 2-tuple will be returned:
            (:class:`numpy.ndarray`, :class:`numbers.Real`), which is the
            PSG data of the channel `channel` and its sampling frequency.

        """
        chn = self.match_channel(channel) if channel.lower() != "all" else "all"
        frp = self.get_absolute_path(rec, rec_path, rec_type="psg")
        self.safe_edf_file_operation("open", frp)

        if chn == "all":
            ret_data = {
                k: (
                    self.file_opened.readSignal(idx, digital=not physical),
                    self.file_opened.getSampleFrequency(idx),
                )
                for idx, k in enumerate(self.file_opened.getSignalLabels())
            }
        else:
            all_signals = [s.lower() for s in self.file_opened.getSignalLabels()]
            assert (
                chn in all_signals
            ), f"`channel` should be one of `{self.file_opened.getSignalLabels()}`, but got `{chn}`"
            idx = all_signals.index(chn)
            data_fs = self.file_opened.getSampleFrequency(idx)
            data = self.file_opened.readSignal(idx, digital=not physical)
            # the `readSignal` method of `EdfReader` does NOT treat
            # the parameters `start` and `n` correctly
            # so we have to do it manually
            if sampfrom is not None:
                idx_from = int(round(sampfrom * data_fs))
            else:
                idx_from = 0
            if sampto is not None:
                idx_to = int(round(sampto * data_fs))
            else:
                idx_to = len(data)
            data = data[idx_from:idx_to]
            if fs is not None and fs != data_fs:
                data = SS.resample_poly(data, fs, data_fs).astype(data.dtype)
                data_fs = fs
            ret_data = (data, data_fs)

        self.safe_edf_file_operation("close")

        return ret_data

    def load_ecg_data(
        self,
        rec: Union[str, int],
        rec_path: Optional[Union[str, Path]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[int] = None,
        return_fs: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """Load ECG data of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rec_path : str or pathlib.Path, optional
            Path of the file which contains the ECG data.
            If is None, default path will be used.
        sampfrom : int, optional
            Start index of the data to be loaded.
        sampto : int, optional
            End index of the data to be loaded.
        data_format : str, default "channel_first"
            Format of the ECG data,
            "channel_last" (alias "lead_last"), or
            "channel_first" (alias "lead_first"), or
            "flat" (alias "plain") which is valid only when `leads` is a single lead.
        units : str or None, default "mV"
            Units of the output signal, can also be "μV" (aliases "uV", "muV").
            None for digital data, without digital-to-physical conversion.
        fs : numbers.Real, optional
            Sampling frequency of the loaded data.
            If not None, the loaded data will be resampled to this frequency,
            otherwise, the original sampling frequency will be used.
        return_fs : bool, default True
            Whether to return the sampling frequency of the output signal.

        Returns
        -------
        data : numpy.ndarray
            The loaded ECG data.
        data_fs : numbers.Real
            Sampling frequency of the loaded ECG data.
            Returned if `return_fs` is True.

        """
        allowed_data_format = [
            "channel_first",
            "lead_first",
            "channel_last",
            "lead_last",
            "flat",
            "plain",
        ]
        assert (
            data_format.lower() in allowed_data_format
        ), f"`data_format` should be one of `{allowed_data_format}`, but got `{data_format}`"
        allowed_units = ["mv", "uv", "μv", "muv"]
        assert (
            units is None or units.lower() in allowed_units
        ), f"`units` should be one of `{allowed_units}` or None, but got `{units}`"

        data, data_fs = self.load_psg_data(
            rec=rec,
            channel="ecg",
            rec_path=rec_path,
            sampfrom=sampfrom,
            sampto=sampto,
            fs=fs,
            physical=units is not None,
        )
        data = data.astype(DEFAULTS.DTYPE.NP)

        if units is not None and units.lower() in ["μv", "uv", "muv"]:
            data *= 1e3
        if data_format.lower() in ["channel_first", "lead_first"]:
            data = data[np.newaxis, :]
        elif data_format.lower() in ["channel_last", "lead_last"]:
            data = data[:, np.newaxis]

        if return_fs:
            return data, data_fs
        return data

    @add_docstring(
        " " * 8 + "NOTE: one should call `load_psg_data` to load other channels.",
        mode="append",
    )
    @add_docstring(load_ecg_data.__doc__)
    def load_data(
        self,
        rec: Union[str, int],
        rec_path: Optional[Union[str, Path]] = None,
        sampfrom: Optional[int] = None,
        sampto: Optional[int] = None,
        data_format: str = "channel_first",
        units: Union[str, type(None)] = "mV",
        fs: Optional[int] = None,
        return_fs: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Real]]:
        """alias of `load_ecg_data`"""
        return self.load_ecg_data(
            rec=rec,
            rec_path=rec_path,
            sampfrom=sampfrom,
            sampto=sampto,
            data_format=data_format,
            units=units,
            fs=fs,
            return_fs=return_fs,
        )

    def load_ann(
        self,
        rec: Union[str, int],
        ann_type: str,
        ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, pd.DataFrame, dict]:
        """Load annotations of specific type of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        ann_type : str,
            Type of the annotation, can be
            "event", "event_profusion", "hrv_summary", "hrv_detailed",
            "sleep", "sleep_stage", "sleep_event", "apnea" (alias "sleep_apnea"),
            "wave_delineation", "rpeak", "rr", "nn".
        ann_path : str or pathlib.Path, optional
            Path of the file which contains the annotations.
            If is None, default path will be used.
        kwargs : dict, optional
            Other arguments for specific annotation type.

        Returns
        -------
        annotations : numpy.ndarray or pandas.DataFrame or dict
            The loaded annotations.

        """
        if ann_type.lower() == "event":
            return self.load_event_ann(rec=rec, event_ann_path=ann_path, **kwargs)
        elif ann_type.lower() == "event_profusion":
            return self.load_event_profusion_ann(
                rec=rec, event_profusion_ann_path=ann_path, **kwargs
            )
        elif ann_type.lower() == "hrv_summary":
            return self.load_hrv_summary_ann(rec=rec, hrv_ann_path=ann_path, **kwargs)
        elif ann_type.lower() == "hrv_detailed":
            return self.load_hrv_detailed_ann(rec=rec, hrv_ann_path=ann_path, **kwargs)
        elif ann_type.lower() == "sleep":
            return self.load_sleep_ann(rec=rec, sleep_ann_path=ann_path, **kwargs)
        elif ann_type.lower() == "sleep_stage":
            return self.load_sleep_stage_ann(
                rec=rec, sleep_stage_ann_path=ann_path, **kwargs
            )
        elif ann_type.lower() == "sleep_event":
            return self.load_sleep_event_ann(
                rec=rec, sleep_event_ann_path=ann_path, **kwargs
            )
        elif ann_type.lower() in ["sleep_apnea", "apnea"]:
            return self.load_apnea_ann(rec=rec, apnea_ann_path=ann_path, **kwargs)
        elif ann_type.lower() == "wave_delineation":
            return self.load_wave_delineation_ann(
                rec=rec, wave_deli_path=ann_path, **kwargs
            )
        elif ann_type.lower() == "rpeak":
            return self.load_rpeak_ann(rec=rec, rpeak_ann_path=ann_path, **kwargs)
        elif ann_type.lower() in ["rr", "rr_interval"]:
            return self.load_rr_ann(rec=rec, rpeak_ann_path=ann_path, **kwargs)
        elif ann_type.lower() in ["nn", "nn_interval"]:
            return self.load_nn_ann(rec=rec, rpeak_ann_path=ann_path, **kwargs)

    def load_event_ann(
        self,
        rec: Union[str, int],
        event_ann_path: Optional[Union[str, Path]] = None,
        simplify: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load event annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        event_ann_path : str or pathlib.Path, optional
            Path of the file which contains the events-nsrr annotations.
            If is None, default path will be used.

        Returns
        -------
        df_events : pandas.DataFrame
            Event annotations of the record.

        """
        file_path = self.get_absolute_path(rec, event_ann_path, rec_type="event")
        if not file_path.exists():
            # rec not in `self.rec_with_event_ann`
            return pd.DataFrame()
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
        self,
        rec: Union[str, int],
        event_profusion_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> dict:
        """Load events-profusion annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        event_profusion_ann_path : str or pathlib.Path, optional
            Path of the file which contains the events-profusion annotations.
            If is None, default path will be used.

        Returns
        -------
        dict
            Event-profusions annotations of the record,
            with items "sleep_stage_list", "df_events".

        TODO
        ----
        Merge "sleep_stage_list" and "df_events" into one :class:`~pandas.DataFrame`.

        """
        file_path = self.get_absolute_path(
            rec, event_profusion_ann_path, rec_type="event_profusion"
        )
        if not file_path.exists():
            # rec not in `self.rec_with_event_profusion_ann`
            return {"sleep_stage_list": [], "df_events": pd.DataFrame()}
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
        self,
        rec: Optional[Union[str, int]] = None,
        hrv_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load summary HRV annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        hrv_ann_path : str or pathlib.Path, optional
            Path of the summary HRV annotation file.
            If is None, default path will be used.

        Returns
        -------
        df_hrv_ann : pandas.DataFrame
            If `rec` is not None, `df_hrv_ann` is the summary HRV annotations of `rec`;
            if `rec` is None, `df_hrv_ann` is the summary HRV annotations of all records
            that had HRV annotations (about 10% of all the records in SHHS).

        """
        if rec is None:
            df_hrv_ann = pd.concat(
                [
                    self._tables[table_name]
                    for table_name in ["shhs1-hrv-summary", "shhs2-hrv-summary"]
                    if table_name in self._tables
                ],
                ignore_index=True,
            )
            return df_hrv_ann

        if isinstance(rec, int):
            rec = self[rec]

        if rec not in self.rec_with_hrv_summary_ann:
            return pd.DataFrame()

        tranche, nsrrid = [self.split_rec_name(rec)[k] for k in ["tranche", "nsrrid"]]
        table_name = f"{tranche}-hrv-summary"
        df_hrv_ann = self._tables[table_name][
            self._tables[table_name].nsrrid == int(nsrrid)
        ].reset_index(drop=True)
        return df_hrv_ann

    def load_hrv_detailed_ann(
        self,
        rec: Union[str, int],
        hrv_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load detailed HRV annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        hrv_ann_path : str or pathlib.Path, optional
            Path of the detailed HRV annotation file.
            If is None, default path will be used.

        Returns
        -------
        df_hrv_ann : pandas.DataFrame.
            Detailed HRV annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if rec not in self.rec_with_hrv_detailed_ann:
            return pd.DataFrame()

        tranche, nsrrid = [self.split_rec_name(rec)[k] for k in ["tranche", "nsrrid"]]
        table_name = f"{tranche}-hrv-5min"
        df_hrv_ann = self._tables[table_name][
            self._tables[table_name].nsrrid == int(nsrrid)
        ].reset_index(drop=True)

        return df_hrv_ann

    def load_sleep_ann(
        self,
        rec: Union[str, int],
        source: str = "event",
        sleep_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, dict]:
        """Load sleep annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        source : {"hrv", "event", "event_profusion"}, optional
            Source of the annotations, case insensitive,
            by default "event"
        sleep_ann_path : str or pathlib.Path, optional
            Path of the file which contains the sleep annotations.
            If is None, default path will be used.

        Returns
        -------
        df_sleep_ann : pandas.DataFrame or dict
            All sleep annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        if source.lower() == "hrv":
            df_hrv_ann = self.load_hrv_detailed_ann(
                rec=rec, hrv_ann_path=sleep_ann_path
            )
            if not df_hrv_ann.empty:
                df_sleep_ann = df_hrv_ann[self.sleep_ann_keys_from_hrv].reset_index(
                    drop=True
                )
            else:
                df_sleep_ann = pd.DataFrame(columns=self.sleep_ann_keys_from_hrv)
            self.logger.debug(
                f"record `{rec}` has `{len(df_sleep_ann)}` sleep annotations from corresponding "
                f"hrv-5min (detailed) annotation file, with `{len(self.sleep_ann_keys_from_hrv)}` column(s)"
            )
        elif source.lower() == "event":
            df_event_ann = self.load_event_ann(
                rec, event_ann_path=sleep_ann_path, simplify=False
            )
            _cols = ["EventType", "EventConcept", "Start", "Duration", "SignalLocation"]
            if not df_event_ann.empty:
                df_sleep_ann = df_event_ann[_cols]
            else:
                df_sleep_ann = pd.DataFrame(columns=_cols)
            self.logger.debug(
                f"record `{rec}` has `{len(df_sleep_ann)}` sleep annotations from corresponding "
                f"event-nsrr annotation file, with `{len(_cols)}` column(s)"
            )
        elif source.lower() == "event_profusion":
            dict_event_ann = self.load_event_profusion_ann(rec)
            # temporarily finished
            # latter to make imporvements
            df_sleep_ann = dict_event_ann
            self.logger.debug(
                f"record `{rec}` has `{len(df_sleep_ann['df_events'])}` sleep event annotations "
                "from corresponding event-profusion annotation file, "
                f"with `{len(df_sleep_ann['df_events'].columns)}` column(s)"
            )
        else:
            raise ValueError(
                f"Source `{source}` not supported, "
                "only `hrv`, `event`, `event_profusion` are supported"
            )
        return df_sleep_ann

    def load_sleep_stage_ann(
        self,
        rec: Union[str, int],
        source: str = "event",
        sleep_stage_ann_path: Optional[Union[str, Path]] = None,
        sleep_stage_protocol: str = "aasm",
        with_stage_names: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load sleep stage annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        source : {"hrv", "event", "event_profusion"}, optional
            Source of the annotations, case insensitive,
            by default "event".
        sleep_stage_ann_path : str or pathlib.Path, optional
            Path of the file which contains the sleep stage annotations.
            If is None, default path will be used.
        sleep_stage_protocol : str, default "aasm"
            The protocol to classify sleep stages.
            Currently can be "aasm", "simplified", "shhs".
            The only difference lies in the number of different stages of the NREM periods.
        with_stage_names : bool, default True
            If True, an additional column "sleep_stage_name"
            will be added to the returned :class:`~pandas.DataFrame`.

        Returns
        -------
        df_sleep_stage_ann : pandas.DataFrame
            Sleep stage annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
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
        else:
            raise ValueError(
                f"Source `{source}` not supported, "
                "only `hrv`, `event`, `event_profusion` are supported"
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
            self.logger.debug(
                f"record `{rec}` has `{len(df_tmp)}` raw (epoch_len = 5min) sleep stage annotations, "
                f"with `{len(self.sleep_stage_ann_keys_from_hrv)}` column(s)"
            )
            self.logger.debug(
                f"after being transformed (epoch_len = 30sec), record `{rec}` has {len(df_sleep_stage_ann)} "
                f"sleep stage annotations, with `{len(self.sleep_stage_keys)}` column(s)"
            )

        return df_sleep_stage_ann

    def load_sleep_event_ann(
        self,
        rec: Union[str, int],
        source: str = "event",
        event_types: Optional[List[str]] = None,
        sleep_event_ann_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load sleep event annotations of a record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        source : {"hrv", "event", "event_profusion"}, optional
            Source of the annotations, case insensitive,
            by default "event".
        event_types : List[str], optional
            List of event types to be loaded, by default None.
            The event types are:
            "Respiratory" (including "Apnea", "SpO2"), "Arousal",
            "Apnea" (including "CSA", "OSA", "MSA", "Hypopnea"), "SpO2",
            "CSA", "OSA", "MSA", "Hypopnea".
            Used only when `source` is "event" or "event_profusion".
        sleep_event_ann_path : str or pathlib.Path, optional
            Path of the file which contains the sleep event annotations.
            If is None, default path will be used.

        Returns
        -------
        df_sleep_event_ann : pandas.DataFrame
            Sleep event annotations of the record.

        """
        if isinstance(rec, int):
            rec = self[rec]
        df_sleep_ann = self.load_sleep_ann(
            rec=rec, source=source, sleep_ann_path=sleep_event_ann_path
        )
        if isinstance(df_sleep_ann, pd.DataFrame) and df_sleep_ann.empty:
            return df_sleep_ann
        elif isinstance(df_sleep_ann, dict) and df_sleep_ann["df_events"].empty:
            return df_sleep_ann["df_events"]

        df_sleep_event_ann = pd.DataFrame(columns=self.sleep_event_keys)

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

            self.logger.debug(
                f"record `{rec}` has `{len(df_sleep_ann)}` raw (epoch_len = 5min) sleep event "
                f"annotations from hrv, with `{len(self.sleep_event_ann_keys_from_hrv)}` column(s)"
            )
            self.logger.debug(
                f"after being transformed, record `{rec}` has `{len(df_sleep_event_ann)}` sleep event(s)"
            )
        elif source.lower() == "event":
            if event_types is None:
                event_types = ["respiratory", "arousal"]
            else:
                event_types = [e.lower() for e in event_types]
            assert (
                set()
                < set(event_types)
                <= set(
                    [
                        "respiratory",
                        "arousal",
                        "apnea",
                        "spo2",
                        "csa",
                        "osa",
                        "msa",
                        "hypopnea",
                    ]
                )
            ), (
                "`event_types` should be a subset of "
                "'respiratory', 'arousal', 'apnea', 'spo2', 'csa', 'osa', 'msa', 'hypopnea'",
                f"but got `{event_types}`",
            )
            _cols = set()
            if "respiratory" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[:6])
            if "arousal" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[6:11])
            if "apnea" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[:4])
            if "spo2" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[4:6])
            if "csa" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[0:1])
            if "osa" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[1:2])
            if "msa" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[2:3])
            if "hypopnea" in event_types:
                _cols = _cols | set(self.long_event_names_from_event[3:4])
            _cols = list(_cols)

            self.logger.debug(f"for record `{rec}`, _cols = `{_cols}`")

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
            if event_types is None:
                event_types = ["respiratory", "arousal"]
            else:
                event_types = [e.lower() for e in event_types]
            assert (
                set()
                < set(event_types)
                <= set(
                    [
                        "respiratory",
                        "arousal",
                        "apnea",
                        "spo2",
                        "csa",
                        "osa",
                        "msa",
                        "hypopnea",
                    ]
                )
            ), (
                "`event_types` should be a subset of "
                "'respiratory', 'arousal', 'apnea', 'spo2', 'csa', 'osa', 'msa', 'hypopnea', "
                f"but got `{event_types}`"
            )
            if "respiratory" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[:6])
            if "arousal" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[6:8])
            if "apnea" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[:4])
            if "spo2" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[4:6])
            if "csa" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[0:1])
            if "osa" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[1:2])
            if "msa" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[2:3])
            if "hypopnea" in event_types:
                _cols = _cols | set(self.event_names_from_event_profusion[3:4])
            _cols = list(_cols)

            self.logger.debug(f"for record `{rec}`, _cols = `{_cols}`")

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
        else:
            raise ValueError(
                f"Source `{source}` not supported, "
                "only `hrv`, `event`, `event_profusion` are supported"
            )

        return df_sleep_event_ann

    def load_apnea_ann(
        self,
        rec: Union[str, int],
        source: str = "event",
        apnea_types: Optional[List[str]] = None,
        apnea_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load annotations on apnea events of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        source : {"event", "event_profusion"}, optional
            Source of the annotations, case insensitive,
            by default "event".
        apnea_types : List[str], optional
            Types of apnea events to load, should be a subset of
            "CSA", "OSA", "MSA", "Hypopnea".
            If is None, then all types of apnea will be loaded.
        apnea_ann_path : str or pathlib.Path, optional
            Path of the file which contains the apnea event annotations.
            If is None, default path will be used.

        Returns
        -------
        df_apnea_ann : pandas.DataFrame
            Apnea event annotations of the record.

        """
        event_types = ["apnea"] if apnea_types is None else apnea_types
        if source.lower() not in ["event", "event_profusion"]:
            raise ValueError(
                f"Source `{source}` contains no apnea annotations, "
                "should be one of 'event', 'event_profusion'"
            )
        df_apnea_ann = self.load_sleep_event_ann(
            rec=rec,
            source=source,
            event_types=event_types,
            sleep_event_ann_path=apnea_ann_path,
        )
        return df_apnea_ann

    def load_wave_delineation_ann(
        self,
        rec: Union[str, int],
        wave_deli_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load annotations on wave delineations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        wave_deli_path : str or pathlib.Path, optional
            Path of the file which contains wave delineation annotations.
            If is None, default path will be used.

        Returns
        -------
        df_wave_delineation : pandas.DataFrame
            Wave delineation annotations of the record.

        NOTE
        ----
        See the part describing wave delineation annotations of the docstring of the class,
        or call ``self.database_info(detailed=True)``.

        """
        if isinstance(rec, int):
            rec = self[rec]

        file_path = self.get_absolute_path(
            rec, wave_deli_path, rec_type="wave_delineation"
        )

        if not file_path.is_file():
            self.logger.debug(
                f"The annotation file of wave delineation of record `{rec}` has not been downloaded yet. "
                f"Or the path `{str(file_path)}` is not correct. "
                f"Or `{rec}` does not have `rpeak.csv` annotation file. Please check!"
            )
            return pd.DataFrame()

        df_wave_delineation = pd.read_csv(file_path, engine="python")
        df_wave_delineation = df_wave_delineation[self.wave_deli_keys].reset_index(
            drop=True
        )
        return df_wave_delineation

    def load_rpeak_ann(
        self,
        rec: Union[str, int],
        rpeak_ann_path: Optional[Union[str, Path]] = None,
        exclude_artifacts: bool = True,
        exclude_abnormal_beats: bool = True,
        units: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Load annotations on R peaks of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rpeak_ann_path : str or pathlib.Path, optional
            Path of the file which contains R peak annotations.
            If is None, default path will be used.
        exclude_artifacts : bool, default True
            Whether exlcude those beats (R peaks) that are labelled artifact or not.
        exclude_abnormal_beats : bool, default True
            Whether exlcude those beats (R peaks) that are
            labelled abnormal ("VE" and "SVE") or not.
        units : {None, "s", "ms"}, optional
            Units of the returned R peak locations, case insensitive.
            None for no conversion, using indices of samples.

        Returns
        -------
        numpy.ndarray
            Locations of R peaks of the record,
            of shape ``(n_rpeaks, )``.

        """
        info_items = ["Type", "rpointadj", "samplingrate"]
        df_rpeaks_with_type_info = self.load_wave_delineation_ann(rec, rpeak_ann_path)
        if df_rpeaks_with_type_info.empty:
            return np.array([], dtype=int)
        df_rpeaks_with_type_info = df_rpeaks_with_type_info[info_items]
        exclude_beat_types = []
        # 0 = Artifact, 1 = Normal Sinus Beat, 2 = VE, 3 = SVE
        if exclude_artifacts:
            exclude_beat_types.append(0)
        if exclude_abnormal_beats:
            exclude_beat_types += [2, 3]

        rpeaks = df_rpeaks_with_type_info[
            ~df_rpeaks_with_type_info["Type"].isin(exclude_beat_types)
        ]["rpointadj"].values

        if units is None:
            rpeaks = (np.round(rpeaks)).astype(int)
        elif units.lower() == "s":
            fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
            rpeaks = rpeaks / fs
        elif units.lower() == "ms":
            fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
            rpeaks = rpeaks / fs * 1000
            rpeaks = (np.round(rpeaks)).astype(int)
        else:
            raise ValueError(
                "`units` should be one of 's', 'ms', case insensitive, "
                "or None for no conversion, using indices of samples, "
                f"but got `{units}`"
            )

        return rpeaks

    def load_rr_ann(
        self,
        rec: Union[str, int],
        rpeak_ann_path: Optional[Union[str, Path]] = None,
        units: Union[str, None] = "s",
        **kwargs: Any,
    ) -> np.ndarray:
        """Load annotations on RR intervals of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rpeak_ann_path : str or pathlib.Path, optional
            Path of the file which contains R peak annotations.
            If is None, default path will be used.
        units : {None, "s", "ms"}, optional
            units of the returned R peak locations,
            by default "s", case insensitive.
            None for no conversion, using indices of samples.

        Returns
        -------
        rr : numpy.ndarray.
            Array of RR intervals, of shape ``(n_rpeaks - 1, 2)``.
            Each row is a RR interval, and
            the first column is the location of the R peak.

        """
        rpeaks_ts = self.load_rpeak_ann(
            rec=rec,
            rpeak_ann_path=rpeak_ann_path,
            exclude_artifacts=True,
            exclude_abnormal_beats=True,
            units=units,
        )
        rr = np.diff(rpeaks_ts)
        rr = np.column_stack((rpeaks_ts[:-1], rr))
        return rr

    def load_nn_ann(
        self,
        rec: Union[str, int],
        rpeak_ann_path: Optional[Union[str, Path]] = None,
        units: Union[str, None] = "s",
        **kwargs: Any,
    ) -> np.ndarray:
        """Load annotations on NN intervals of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        rpeak_ann_path: str or pathlib.Path, optional
            Path of the file which contains R peak annotations.
            If is None, default path will be used.
        units: {None, "s", "ms"}, optional
            Units of the returned R peak locations,
            by default "s", case insensitive.
            None for no conversion, using indices of samples.

        Returns
        -------
        nn : numpy.ndarray
            Array of nn intervals, of shape (n, 2).
            Each row is a nn interval, and
            the first column is the location of the R peak.

        """
        info_items = ["Type", "rpointadj", "samplingrate"]
        df_rpeaks_with_type_info = self.load_wave_delineation_ann(rec, rpeak_ann_path)
        if df_rpeaks_with_type_info.empty:
            return np.array([]).reshape(0, 2)

        df_rpeaks_with_type_info = df_rpeaks_with_type_info[info_items]
        fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
        rpeaks = df_rpeaks_with_type_info["rpointadj"]

        if units is None:
            rpeaks = (np.round(rpeaks)).astype(int)
        elif units.lower() == "s":
            rpeaks = rpeaks / fs
        elif units.lower() == "ms":
            rpeaks = rpeaks / fs * 1000
            rpeaks = (np.round(rpeaks)).astype(int)
        else:
            raise ValueError(
                "`units` should be one of 's', 'ms', case insensitive, "
                "or None for no conversion, using indices of samples, "
                f"but got `{units}`"
            )

        rr = np.diff(rpeaks)
        rr = np.column_stack((rpeaks[:-1], rr))

        normal_sinus_rpeak_indices = np.where(
            df_rpeaks_with_type_info["Type"].values == 1
        )[
            0
        ]  # 1 = Normal Sinus Beat
        keep_indices = np.where(np.diff(normal_sinus_rpeak_indices) == 1)[0].tolist()
        nn = rr[normal_sinus_rpeak_indices[keep_indices]]
        return nn.reshape(-1, 2)

    def locate_artifacts(
        self,
        rec: Union[str, int],
        wave_deli_path: Optional[Union[str, Path]] = None,
        units: Optional[str] = None,
    ) -> np.ndarray:
        """Locate "artifacts" in the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        wave_deli_path : str or pathlib.Path, optional
            Path of the file which contains wave delineation annotations.
            If is None, default path will be used.
        units : {None, "s", "ms"}, optional
            Units of the returned artifact locations,
            can be one of "s", "ms", case insensitive,
            None for no conversion, using indices of samples.

        Returns
        -------
        artifacts : numpy.ndarray
            Array of indices (or time) of artifacts locations,
            of shape ``(n_artifacts,)``.

        """
        df_rpeaks_with_type_info = self.load_wave_delineation_ann(rec, wave_deli_path)
        if df_rpeaks_with_type_info.empty:
            dtype = int if units is None or units.lower() != "s" else float
            return np.array([], dtype=dtype)
        # df_rpeaks_with_type_info = df_rpeaks_with_type_info[["Type", "rpointadj"]]

        artifacts = (
            np.round(
                df_rpeaks_with_type_info[df_rpeaks_with_type_info["Type"] == 0][
                    "rpointadj"
                ].values
            )
        ).astype(int)

        if units is not None:
            fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
            if units.lower() == "s":
                artifacts = artifacts / fs
            elif units.lower() == "ms":
                artifacts = artifacts / fs * 1000
                artifacts = (np.round(artifacts)).astype(int)
            else:
                raise ValueError(
                    "`units` should be one of 's', 'ms', case insensitive, "
                    "or None for no conversion, using indices of samples, "
                    f"but got `{units}`"
                )

        return artifacts

    def locate_abnormal_beats(
        self,
        rec: Union[str, int],
        wave_deli_path: Optional[Union[str, Path]] = None,
        abnormal_type: Optional[str] = None,
        units: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Locate "abnormal beats" in the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        wave_deli_path : str or pathlib.Path, optional
            Path of the file which contains wave delineation annotations.
            If is None, default path will be used.
        abnormal_type : {"VE", "SVE"}, optional
            Type of abnormal beat type to locate.
            If is None, both "VE" and "SVE" will be located.
        units : {None, "s", "ms"}, optional
            Units of the returned R peak locations,
            by default None, case insensitive.
            None for no conversion, using indices of samples.

        Returns
        -------
        abnormal_rpeaks : dict or numpy.ndarray
            If `abnormal_type` is None,
            return a dictionary of abnormal beat locations,
            which contains two keys "VE" and/or "SVE", and
            values are indices (or time) of abnormal beats,
            of shape ``(n,)``.
            If `abnormal_type` is not None,
            return a :class:`~numpy.ndarray` of abnormal beat locations,
            of shape ``(n,)``.

        """
        if abnormal_type is not None and abnormal_type not in ["VE", "SVE"]:
            raise ValueError(
                f"No abnormal type of `{abnormal_type}` in "
                "wave delineation annotation (*-rpeak.csv) files"
            )

        df_rpeaks_with_type_info = self.load_wave_delineation_ann(rec, wave_deli_path)

        if not df_rpeaks_with_type_info.empty:
            # df_rpeaks_with_type_info = df_rpeaks_with_type_info[["Type", "rpointadj"]]
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
        else:
            dtype = int if units is None or units.lower() != "s" else float
            abnormal_rpeaks = {
                "VE": np.array([], dtype=dtype),
                "SVE": np.array([], dtype=dtype),
            }

        if units is not None and not df_rpeaks_with_type_info.empty:
            fs = df_rpeaks_with_type_info.iloc[0]["samplingrate"]
            if units.lower() == "s":
                abnormal_rpeaks = {
                    abnormal_type: abnormal_rpeaks[abnormal_type] / fs
                    for abnormal_type in abnormal_rpeaks
                }
            elif units.lower() == "ms":
                abnormal_rpeaks = {
                    abnormal_type: abnormal_rpeaks[abnormal_type] / fs * 1000
                    for abnormal_type in abnormal_rpeaks
                }
                abnormal_rpeaks = {
                    abnormal_type: (np.round(abnormal_rpeaks[abnormal_type])).astype(
                        int
                    )
                    for abnormal_type in abnormal_rpeaks
                }
            else:
                raise ValueError(
                    "`units` should be one of 's', 'ms', case insensitive, "
                    "or None for no conversion, using indices of samples, "
                    f"but got `{units}`"
                )

        if abnormal_type is None:
            return abnormal_rpeaks
        elif abnormal_type in ["VE", "SVE"]:
            return abnormal_rpeaks[abnormal_type]

    def load_eeg_band_ann(
        self,
        rec: Union[str, int],
        eeg_band_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load annotations on EEG bands of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        eeg_band_ann_path : str or pathlib.Path, optional
            Path of the file which contains EEG band annotations.
            if is None, default path will be used.

        Returns
        -------
        pandas.DataFrame
            A :class:`~pandas.DataFrame` of EEG band annotations.

        """
        if self.current_version >= "0.15.0":
            self.logger.info(
                f"EEG spectral summary variables are removed in version {self.current_version}"
            )
        else:
            raise NotImplementedError

    def load_eeg_spectral_ann(
        self,
        rec: Union[str, int],
        eeg_spectral_ann_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load annotations on EEG spectral summary of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        eeg_spectral_ann_path : str or pathlib.Path, optional
            Path of the file which contains EEG spectral summary annotations.
            If is None, default path will be used.

        Returns
        -------
        pandas.DataFrame
            A :class:`~pandas.DataFrame` of EEG spectral summary annotations.

        """
        if self.current_version >= "0.15.0":
            self.logger.info(
                f"EEG spectral summary variables are removed in version {self.current_version}"
            )
        else:
            raise NotImplementedError

    # TODO: add more functions for annotation reading
    # TODO: add plotting functions

    def plot_ann(
        self,
        rec: Union[str, int],
        stage_source: Optional[str] = None,
        stage_kw: dict = {},
        event_source: Optional[str] = None,
        event_kw: dict = {},
        plot_format: str = "span",
    ) -> None:
        """Plot annotations of the record.

        Plot the sleep stage annotations
        and sleep event annotations of the record.

        Parameters
        ----------
        rec : str or int
            Record name, typically in the form "shhs1-200001",
            or index of the record in :attr:`all_records`.
        stage_source : {"hrv", "event", "event_profusion"}, optional
            Source of the sleep stage annotations, case in-sensitive.
            If is None, then annotations of sleep stages of `rec` won't be plotted.
        stage_kw : dict, optional
            Key word arguments to the function :meth:`load_sleep_stage_ann`.
        event_source : {"hrv", "event", "event_profusion"}, optional
            Source of the sleep event annotations, case in-sensitive.
            If is None, then annotations of sleep events of `rec` won't be plotted.
        event_kw : dict, optional
            Key word arguments to the function :meth:`load_sleep_event_ann`.
        plot_format : {"span", "hypnogram"}, optional
            Format of the plot, case insensitive, by default "span".

        TODO
        ----
        1. ~~Implement the "hypnogram" format.~~
        2. Implement plotting of sleep events.

        """
        if all([stage_source is None, event_source is None]):
            raise ValueError("`stage_source` and `event_source` cannot be both `None`")

        if stage_source is not None:
            df_sleep_stage = self.load_sleep_stage_ann(
                rec, source=stage_source, **stage_kw
            )
            if df_sleep_stage.empty:
                if isinstance(rec, int):
                    rec = self[rec]
                raise ValueError(
                    f"No sleep stage annotations found for record `{rec}` "
                    f"with source `{stage_source}`"
                )
        else:
            df_sleep_stage = None
        if event_source is not None:
            df_sleep_event = self.load_sleep_event_ann(
                rec, source=event_source, **event_kw
            )
            if df_sleep_event.empty:
                if isinstance(rec, int):
                    rec = self[rec]
                raise ValueError(
                    f"No sleep event annotations found for record `{rec}` "
                    f"with source `{event_source}`"
                )
        else:
            df_sleep_event = None

        self._plot_ann(
            df_sleep_stage=df_sleep_stage,
            df_sleep_event=df_sleep_event,
            plot_format=plot_format,
        )

    def _plot_ann(
        self,
        df_sleep_stage: Optional[pd.DataFrame] = None,
        df_sleep_event: Optional[pd.DataFrame] = None,
        plot_format: str = "span",
    ) -> None:
        """Internal function to plot annotations.

        Parameters
        ----------
        df_sleep_stage : pandas.DataFrame, optional
            Sleep stage annotations.
        df_sleep_event : pandas.DataFrame, optional
            Sleep event annotations.
        plot_format : {"span", "hypnogram"}, optional
            Format of the plot, case insensitive, by default "span".

        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        check = [df_sleep_stage is None, df_sleep_event is None]
        nb_axes = len(check) - np.sum(check)

        if nb_axes == 0:
            raise ValueError("No input data!")

        if plot_format.lower() not in ["span", "hypnogram"]:
            raise ValueError(
                f"Unknown plot format `{plot_format}`! "
                f"`plot_format` can only be one of `span`, `hypnogram`"
            )

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
                raise NotImplementedError(
                    "Plotting of some type of events in `df_sleep_event` has not been implemented yet!"
                )

        if plot_format.lower() == "hypnogram":
            stage_mask = df_sleep_stage["sleep_stage"].values
            stage_mask = len(self.sleep_stage_names) - 1 - stage_mask
            fig, ax = self.plot_hypnogram(stage_mask, granularity=30)
            return

        patches = {k: mpatches.Patch(color=c, label=k) for k, c in self.palette.items()}

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
        """Convert a string to a real number.

        Some columns in the annotations might incorrectly
        been converted from numbers.Real to string, using ``xmltodict``.

        Parameters
        ----------
        s : str or numbers.Real
            The string to be converted.

        Returns
        -------
        numbers.Real
            The converted number.

        """
        if isinstance(s, str):
            if "." in s:
                return float(s)
            else:
                return int(s)
        else:  # NaN case
            return s

    def __create_constants(self, **kwargs) -> None:
        """Create constants for the class."""
        self.lazy = kwargs.get("lazy", False)
        self.extension = {
            "psg": ".edf",
            "wave_delineation": "-rpoint.csv",
            "event": "-nsrr.xml",
            "event_profusion": "-profusion.xml",
        }

        # fmt: off

        self.all_signals = [
            "EEG(sec)", "ECG", "EMG", "EOG(L)", "EOG(R)", "EEG",
            "AIRFLOW", "THOR RES", "ABDO RES", "NEW AIR", "OX stat", "SaO2", "H.R.",
            "POSITION", "SOUND", "LIGHT",
            "AUX", "CPAP", "EPMS", "OX STAT", "PR",
        ]
        self.all_signals = set([s.lower() for s in self.all_signals])

        # annotations regarding sleep analysis
        self.hrv_ann_summary_keys = [
            "nsrrid", "visitnumber", "NN_RR", "AVNN", "IHR",
            "SDNN", "SDANN", "SDNNIDX", "rMSSD",
            "pNN10", "pNN20", "pNN30", "pNN40", "pNN50",
            "tot_pwr", "ULF", "VLF", "LF", "HF", "LF_HF", "LF_n", "HF_n",
        ]
        self.hrv_ann_detailed_keys = [
            "nsrrid", "visitnumber", "Start__sec_", "ihr", "hasrespevent",
            "NN_RR", "AVNN", "SDNN", "rMSSD",
            "PNN10", "PNN20", "PNN30", "PNN40", "PNN50",
            "TOT_PWR", "VLF", "LF", "LF_n", "HF", "HF_n", "LF_HF",
            "sleepstage01", "sleepstage02", "sleepstage03", "sleepstage04", "sleepstage05",
            "sleepstage06", "sleepstage07", "sleepstage08", "sleepstage09", "sleepstage10",
            "event01start", "event01end",
            "event02start", "event02end",
            "event03start", "event03end",
            "event04start", "event04end",
            "event05start", "event05end",
            "event06start", "event06end",
            "event07start", "event07end",
            "event08start", "event08end",
            "event09start", "event09end",
            "event10start", "event10end",
            "event11start", "event11end",
            "event12start", "event12end",
            "event13start", "event13end",
            "event14start", "event14end",
            "event15start", "event15end",
            "event16start", "event16end",
            "event17start", "event17end",
            "event18start", "event18end",
        ]
        self.hrv_ann_epoch_len_sec = 300  # 5min
        self.sleep_ann_keys_from_hrv = [
            "Start__sec_", "hasrespevent",
            "sleepstage01", "sleepstage02", "sleepstage03", "sleepstage04", "sleepstage05",
            "sleepstage06", "sleepstage07", "sleepstage08", "sleepstage09", "sleepstage10",
            "event01start", "event01end",
            "event02start", "event02end",
            "event03start", "event03end",
            "event04start", "event04end",
            "event05start", "event05end",
            "event06start", "event06end",
            "event07start", "event07end",
            "event08start", "event08end",
            "event09start", "event09end",
            "event10start", "event10end",
            "event11start", "event11end",
            "event12start", "event12end",
            "event13start", "event13end",
            "event14start", "event14end",
            "event15start", "event15end",
            "event16start", "event16end",
            "event17start", "event17end",
            "event18start", "event18end",
        ]
        self.sleep_stage_ann_keys_from_hrv = [
            "Start__sec_",
            "sleepstage01", "sleepstage02", "sleepstage03", "sleepstage04", "sleepstage05",
            "sleepstage06", "sleepstage07", "sleepstage08", "sleepstage09", "sleepstage10",
        ]
        self.sleep_event_ann_keys_from_hrv = [
            "Start__sec_", "hasrespevent",
            "event01start", "event01end",
            "event02start", "event02end",
            "event03start", "event03end",
            "event04start", "event04end",
            "event05start", "event05end",
            "event06start", "event06end",
            "event07start", "event07end",
            "event08start", "event08end",
            "event09start", "event09end",
            "event10start", "event10end",
            "event11start", "event11end",
            "event12start", "event12end",
            "event13start", "event13end",
            "event14start", "event14end",
            "event15start", "event15end",
            "event16start", "event16end",
            "event17start", "event17end",
            "event18start", "event18end",
        ]

        # annotations from events-nsrr and events-profusion folders
        self.event_keys = [
            "EventType", "EventConcept", "Start", "Duration",
            "SignalLocation", "SpO2Nadir", "SpO2Baseline",
        ]
        # NOTE: the union of names from shhs1-200001 to shhs1-200399
        # NOT a full search
        self.short_event_types_from_event = [
            "Respiratory", "Stages", "Arousals",
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
            "Name", "Start", "Duration",
            "Input", "LowestSpO2", "Desaturation",
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
            "RPoint", "Start", "End",
            "STLevel1", "STSlope1", "STLevel2", "STSlope2",
            "Manual", "Type", "rpointadj",
            "PPoint", "PStart", "PEnd",
            "TPoint", "TStart", "TEnd",
            "TemplateID", "nsrrid", "samplingrate", "seconds", "epoch",
        ]
        self.wave_deli_samp_num_keys = [
            "RPoint", "Start", "End",
            "PPoint", "PStart", "PEnd",
            "TPoint", "TStart", "TEnd",
        ]

        # TODO: other annotation files: EEG

        # self-defined items
        self.sleep_stage_keys = ["start_sec", "sleep_stage"]
        self.sleep_event_keys = [
            "event_name", "event_start", "event_end", "event_duration",
        ]
        self.sleep_epoch_len_sec = 30
        self.ann_sleep_stages = [0, 1, 2, 3, 4, 5, 9]
        """
        0 --- Wake
        1 --- sleep stage 1
        2 --- sleep stage 2
        3 --- sleep stage 3/4
        4 --- sleep stage 3/4
        5 --- REM stage
        9 --- Movement/Wake or Unscored?
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
        """9 to nan?
        0 --- awake
        1 --- REM
        2 --- N1 (NREM1/2), shallow sleep
        3 --- N2 (NREM3/4), deep sleep
        """
        self._to_aasm_states = {9: 0, 0: 0, 5: 1, 1: 2, 2: 3, 3: 4, 4: 4}
        """9 to nan?
        0 --- awake
        1 --- REM
        2 --- N1 (NREM1)
        3 --- N2 (NREM2)
        4 --- N3 (NREM3/4)
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

        # fmt: on

    @property
    def folder_or_file(self) -> Dict[str, Path]:
        return {
            "psg": self.psg_data_path,
            "wave_delineation": self.wave_deli_path,
            "event": self.event_ann_path,
            "event_profusion": self.event_profusion_ann_path,
        }

    @property
    def url(self) -> str:
        warnings.warn(
            "one has to apply for a token from `sleepdata.org` "
            "and uses `nsrr` to download the data",
            RuntimeWarning,
        )
        return ""

    @property
    def database_info(self) -> DataBaseInfo:
        return _SHHS_INFO
