"""
"""

import re
import os
import warnings
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from abc import abstractmethod
from typing import Union, Optional, Any, List, Dict, Tuple, Sequence, NoReturn

import numpy as np
import pandas as pd
import wfdb

try:
    import librosa
except Exception:
    librosa = None
import torch

try:
    import torchaudio
except Exception:
    torchaudio = None
import scipy.signal as ss  # noqa: F401
import scipy.io as sio

try:
    import scipy.io.wavfile as sio_wav
except Exception:
    sio_wav = None
import IPython

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
from torch_ecg.databases.base import PhysioNetDataBase, DataBaseInfo
from torch_ecg.utils.utils_signal import butter_bandpass_filter
from torch_ecg.utils.misc import (
    get_record_list_recursive,
    get_record_list_recursive3,
    ReprMixin,
    list_sum,
    add_docstring,
)
from torch_ecg.utils.download import http_get

from cfg import BaseCfg
from utils.schmidt_spike_removal import schmidt_spike_removal


__all__ = [
    "PCGDataBase",
    "CINC2022Reader",
    "CINC2016Reader",
    "EPHNOGRAMReader",
    "CompositeReader",
]


_BACKEND_PRIORITY = [
    "torchaudio",
    "librosa",
    "scipy",
    "wfdb",
]


_HeartMurmurInfo = """
    About Heart Murmur
    ------------------
    1. A heart murmur is a blowing, whooshing, or rasping sound heard **during a heartbeat**. The sound is caused by turbulent (rough) blood flow through the heart valves or near the heart. ([source](https://medlineplus.gov/ency/article/003266.htm))

    2. A murmur is a series of vibrations of variable duration, audible with a stethoscope at the chest wall, that emanates from the heart or great vessels. A **systolic murmur** is a murmur that begins during or after the first heart sound and ends before or during the second heart sound. ([source](https://www.ncbi.nlm.nih.gov/books/NBK345/)) A **diastolic murmur** is a sound of some duration occurring during diastole. ([source](https://www.ncbi.nlm.nih.gov/books/NBK346/))

    3. ([Wikipedia](https://en.wikipedia.org/wiki/Heart_murmur)) Heart murmurs may have a distinct pitch, duration and timing. Murmurs have seven main characteristics. These include timing, shape, location, radiation, intensity, pitch and quality
        - Timing refers to whether the murmur is a
            * systolic murmur
            * diastolic murmur. Diastolic murmurs are usually abnormal, and may be early, mid or late diastolic ([source](https://www.utmb.edu/pedi_ed/CoreV2/Cardiology/cardiologyV2/cardiologyV24.html))
            * continuous murmur
        - Shape refers to the intensity over time. Murmurs can be crescendo, decrescendo or crescendo-decrescendo
            * Crescendo murmurs increase in intensity over time
            * Decrescendo murmurs decrease in intensity over time
            * Crescendo-decrescendo murmurs have both shapes over time, resembling a diamond or kite shape
        - Location refers to where the heart murmur is usually heard best. There are **four** places on the anterior chest wall to listen for heart murmurs. Each location roughly corresponds to a specific part of the heart.
            | Region    | Location                                  | Heart Valve Association|
            |-----------|-------------------------------------------|------------------------|
            | Aortic    | 2nd right intercostal space               | Aortic valve           |
            | Pulmonic  | 2nd left intercostal spaces               | Pulmonic valve         |
            | Tricuspid | 4th left intercostal space                | Tricuspid valve        |
            | Mitral    | 5th left mid-clavicular intercostal space | Mitral valve           |
        - Radiation refers to where the sound of the murmur travels.
        - Intensity refers to the loudness of the murmur with grades according to the [Levine scale](https://en.wikipedia.org/wiki/Levine_scale), from 1 to 6
            | Levine scale | Murmur Description                                                                                      |
            |--------------|---------------------------------------------------------------------------------------------------------|
            | 1            | only audible on listening carefully for some time                                                       |
            | 2            | faint but immediately audible on placing the stethoscope on the chest                                   |
            | 3            | loud, readily audible but with no palpable thrill                                                       |
            | 4            | loud with a palpable thrill                                                                             |
            | 5            | loud with a palpable thrill, audible with only the rim of the stethoscope touching the chest            |
            | 6            | loud with a palpable thrill, audible with the stethoscope not touching the chest but lifted just off it |
        - Pitch may be
            * low
            * medium
            * high
        This depends on whether auscultation is best with the bell or diaphragm of a stethoscope.
        - Quality refers to **unusual characteristics** of a murmur. For example
            * blowing
            * harsh
            * rumbling
            * musical

    4. Heart sounds usually has frequency lower than 500 Hz (mostly lower than 300 Hz) (inferred from [source](https://biologicalproceduresonline.biomedcentral.com/articles/10.1186/1480-9222-13-7) Figure 2). frequency of heart sounds is low in range between 20 and 150 Hz.

    5. Instantaneous dominant heart sound frequencies ranged from 130 to 410 Hz (mean ± standard deviation 282 ± 70 Hz). Peak murmur frequencies ranged from 200 to 410 Hz (308 ± 70 Hz) ([source](https://www.ajconline.org/article/0002-9149(89)90491-8/pdf))

    6. innocent murmurs had lower frequencies (below 200 Hz) and a frequency spectrum with a more harmonic structure than pathological cases ([source](https://bmcpediatr.biomedcentral.com/articles/10.1186/1471-2431-7-23)). [Table 4](https://bmcpediatr.biomedcentral.com/articles/10.1186/1471-2431-7-23/tables/4) is very important and copied as follows
        | Group        | Amplitude (%) | Low freq limit (Hz) | High freq limit (Hz) |
        |--------------|---------------|---------------------|----------------------|
        | Vibratory    | 23 ± 9        | 72 ± 15             | 161 ± 22             |
        | Ejection     | 20 ± 9        | 60 ± 9              | 142 ± 51             |
        | Pathological | 30 ± 20       | 52 ± 19             | 299 ± 133            |
        | p-value      | 0.013         | < 0.001             | < 0.001              |

    7. the principal frequencies of heart sounds and murmurs are at the lower end of this range, from 20 to 500 Hz; The murmur containing the highest frequency sound is aortic regurgitation, whose dominant frequencies are approximately 400 Hz. The principal frequencies of other sounds and murmurs are between 100 and 400 Hz ([source1](https://www.sciencedirect.com/science/article/pii/B9780323392761000391), [source2](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/heart-sounds))
"""


@add_docstring(_HeartMurmurInfo)
class PCGDataBase(PhysioNetDataBase):
    """ """

    __name__ = "PCGDataBase"

    def __init__(
        self,
        db_name: str,
        db_dir: str,
        fs: int = 1000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Parameters
        ----------
        db_name: str,
            name of the database
        db_dir: str, optional,
            storage path of the database
        fs: int, default 1000,
            (re-)sampling frequency of the audio
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(db_name, db_dir, working_dir, verbose, **kwargs)
        self.db_dir = (
            Path(self.db_dir).resolve().absolute()
        )  # will be fixed in `torch_ecg`
        self.fs = fs
        self.dtype = kwargs.get("dtype", BaseCfg.np_dtype)
        self.audio_backend = audio_backend.lower()
        if self.audio_backend not in self.available_backends():
            self.audio_backend = self.available_backends()[0]
            warnings.warn(
                f"audio backend {audio_backend.lower()} is not available, "
                f"using {self.audio_backend} instead"
            )
        if self.audio_backend == "torchaudio":

            def torchaudio_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                try:
                    data, new_fs = torchaudio.load(file, normalize=True)
                except Exception:
                    data, new_fs = torchaudio.load(file, normalization=True)
                return data, new_fs

            self._audio_load_func = torchaudio_load
        elif self.audio_backend == "librosa":

            def librosa_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                data, _ = librosa.load(file, sr=fs, mono=False)
                return torch.from_numpy(data.reshape((-1, data.shape[-1]))), fs

            self._audio_load_func = librosa_load
        elif self.audio_backend == "scipy":

            def scipy_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                new_fs, data = sio_wav.read(file)
                data = (data / (2**15)).astype(self.dtype)[np.newaxis, :]
                return torch.from_numpy(data), new_fs

            self._audio_load_func = scipy_load
        elif self.audio_backend == "wfdb":
            warnings.warn(
                "loading result using wfdb is inconsistent with other backends"
            )

            def wfdb_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
                record = wfdb.rdrecord(file, physical=True)  # channel last
                sig = record.p_signal.T.astype(self.dtype)
                return torch.from_numpy(sig), record.fs[0]

            self._audio_load_func = wfdb_load
        self.data_ext = None
        self.ann_ext = None
        self.header_ext = "hea"
        self._all_records = None

    @staticmethod
    def available_backends() -> List[str]:
        """ """
        ab = ["wfdb"]
        if torchaudio is not None:
            ab.append("torchaudio")
        if librosa is not None:
            ab.append("librosa")
        if sio_wav is not None:
            ab.append("scipy")
        ab = sorted(ab, key=lambda x: _BACKEND_PRIORITY.index(x))
        return ab

    def _auto_infer_units(self) -> NoReturn:
        """
        disable this function implemented in the base class
        """
        print("DO NOT USE THIS FUNCTION for a PCG database!")

    @abstractmethod
    def play(self, rec: str, **kwargs) -> IPython.display.Audio:
        """ """
        raise NotImplementedError

    def _reset_fs(self, new_fs: int) -> NoReturn:
        """ """
        self.fs = new_fs


_CINC2022_INFO = DataBaseInfo(
    title="""
    The CirCor DigiScope Phonocardiogram Dataset (main resource for CinC2022)
    """,
    about="""
    1. 5272 heart sound recordings (.wav format, sampling rate 4000 Hz) were collected from the main 4 auscultation locations of 1568 subjects, aged between 0 and 21 years (mean ± STD = 6.1 ± 4.3 years), with a duration between 4.8 to 80.4 seconds (mean ± STD = 22.9 ± 7.4 s)
    2. segmentation annotations (.tsv format) regarding the location of fundamental heart sounds (S1 and S2) in the recordings have been obtained using a semi-supervised scheme. The annotation files are composed of three distinct columns: the first column corresponds to the time instant (in seconds) where the wave was detected for the first time, the second column corresponds to the time instant (in seconds) where the wave was detected for the last time, and the third column corresponds to an identifier that uniquely identifies the detected wave. Here, we use the following convention:
        - The S1 wave is identified by the integer 1.
        - The systolic period is identified by the integer 2.
        - The S2 wave is identified by the integer 3.
        - The diastolic period is identified by the integer 4.
        - The unannotated segments of the signal are identified by the integer 0.
    """,
    usage=[
        "Heart murmur detection",
        "Heart sound segmentation",
    ],
    note="""
    1. the "Murmur" column (records whether heart murmur can be heard or not) and the "Outcome" column (the expert cardiologist's overall diagnosis using **clinical history, physical examination, analog auscultation, echocardiogram, etc.**) are **NOT RELATED**. All of the 6 combinations (["Present", "Absent", "Unknown"] x ["Abnormal", "Normal"]) occur in the dataset.
    2. the segmentation files do NOT in general (totally 132 such files) have the same length (namely the second column of the last row of these .tsv files) as the audio files.
    """,
    issues="""
    1. the segmentation file `50782_MV_1.tsv` (versions 1.0.2, 1.0.3) is broken.
    2. the challenge website states that the `Age` variable takes values in `Neonate`, `Infant`, `Child`, `Adolescent`, and `Young adult`. However, from the statistics csv file (training_data.csv), there's no subject whose `Age` column has value `Young adult`. Instead, there are 74 subject with null `Age` value, which only indicates that their ages were not recorded and may or may not belong to the “Young adult” age group.
    """,
    references=[
        "https://moody-challenge.physionet.org/2022/",
        "https://physionet.org/content/circor-heart-sound/1.0.3/",
    ],
    doi=[
        "10.1109/JBHI.2021.3137048",
        "10.13026/tshs-mw03",
    ],
)


@add_docstring(f"\n{_HeartMurmurInfo}\n", mode="append")
@add_docstring(_CINC2022_INFO.format_database_docstring())
class CINC2022Reader(PCGDataBase):
    """ """

    __name__ = "CINC2022Reader"
    stats_fillna_val = "NA"

    def __init__(
        self,
        db_dir: str,
        fs: int = 4000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Parameters
        ----------
        db_dir: str,
            storage path of the database
        fs: int, default 4000,
            (re-)sampling frequency of the audio
        audio_backend: str, default "torchaudio",
            audio backend to use, can be one of
            "librosa", "torchaudio", "scipy",  "wfdb",
            case insensitive.
            "librosa" or "torchaudio" is recommended.
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="circor-heart-sound",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        if "training_data" in os.listdir(self.db_dir):
            self.data_dir = self.db_dir / "training_data"
        else:
            self.data_dir = self.db_dir
        self.data_ext = "wav"
        self.ann_ext = "hea"
        self.segmentation_ext = "tsv"
        self.segmentation_states = deepcopy(BaseCfg.states)
        self.ignore_unannotated = kwargs.get("ignore_unannotated", True)
        if self.ignore_unannotated:
            self.segmentation_states = [
                s for s in self.segmentation_states if s != "unannotated"
            ]
        self.segmentation_map = {n: s for n, s in enumerate(self.segmentation_states)}
        if self.ignore_unannotated:
            self.segmentation_map[BaseCfg.ignore_index] = "unannotated"
        self.auscultation_locations = {
            "PV",
            "AV",
            "MV",
            "TV",
            "Phc",
        }

        self._rec_pattern = f"(?P<sid>[\\d]+)\\_(?P<loc>{'|'.join(self.auscultation_locations)})((?:\\_)(?P<num>\\d))?"

        self._all_records = None
        self._all_subjects = None
        self._subject_records = None
        self._exceptional_records = ["50782_MV_1"]
        self._ls_rec()

        self._df_stats = None
        self._stats_cols = [
            "Patient ID",
            "Locations",
            "Age",
            "Sex",
            "Height",
            "Weight",
            "Pregnancy status",
            "Outcome",  # added in version 1.0.2 in the official phase
            "Murmur",
            "Murmur locations",
            "Most audible location",
            "Systolic murmur timing",
            "Systolic murmur shape",
            "Systolic murmur grading",
            "Systolic murmur pitch",
            "Systolic murmur quality",
            "Diastolic murmur timing",
            "Diastolic murmur shape",
            "Diastolic murmur grading",
            "Diastolic murmur pitch",
            "Diastolic murmur quality",
            "Campaign",
            "Additional ID",
        ]
        self._df_stats_records = None
        self._stats_records_cols = [
            "Patient ID",
            "Location",
            "rec",
            "siglen",
            "siglen_sec",
            "Murmur",
        ]
        self._load_stats()

        # attributes for plot
        self.palette = {
            "systolic": "#d62728",
            "diastolic": "#2ca02c",
            "S1": "#17becf",
            "S2": "#bcbd22",
            "default": "#7f7f7f",
        }

    def _ls_rec(self) -> NoReturn:
        """
        list all records in the database
        """
        write_file = False
        self._df_records = pd.DataFrame(columns=["record", "path"])
        records_file = self.db_dir / "RECORDS"
        if records_file.exists():
            self._df_records["record"] = records_file.read_text().splitlines()
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
        else:
            write_file = True

        # self._all_records = wfdb.get_record_list(self.db_name)

        if len(self._df_records) == 0:
            write_file = True
            self._df_records["path"] = get_record_list_recursive3(
                self.db_dir, f"{self._rec_pattern}\\.{self.data_ext}", relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))

        data_dir = self._df_records["path"].apply(lambda x: x.parent).unique()
        assert len(data_dir) <= 1, "data_dir should be a single directory"
        if len(data_dir) == 1:  # in case no data found
            self.data_dir = data_dir[0]

        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records = self._df_records[
            ~self._df_records["record"].isin(self._exceptional_records)
        ]
        self._df_records.set_index("record", inplace=True)

        self._all_records = [
            item
            for item in self._df_records.index.tolist()
            if item not in self._exceptional_records
        ]
        self._all_subjects = sorted(
            set([item.split("_")[0] for item in self._all_records]),
            key=lambda x: int(x),
        )
        self._subject_records = defaultdict(list)
        for rec in self._all_records:
            self._subject_records[self.get_subject(rec)].append(rec)
        self._subject_records = dict(self._subject_records)

        if write_file:
            records_file.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

    def _load_stats(self) -> NoReturn:
        """
        collect statistics of the database
        """
        print("Reading the statistics from local file...")
        stats_file = self.db_dir / "training_data.csv"
        if stats_file.exists():
            self._df_stats = pd.read_csv(stats_file)
        elif self._all_records is not None and len(self._all_records) > 0:
            print("No cached statistics found, gathering from scratch...")
            self._df_stats = pd.DataFrame()
            with tqdm(
                self.all_subjects, total=len(self.all_subjects), desc="loading stats"
            ) as pbar:
                for s in pbar:
                    f = self.data_dir / f"{s}.txt"
                    content = f.read_text().splitlines()
                    new_row = {"Patient ID": s}
                    locations = set()
                    for line in content:
                        if not line.startswith("#"):
                            if line.split()[0] in self.auscultation_locations:
                                locations.add(line.split()[0])
                            continue
                        k, v = line.replace("#", "").split(":")
                        k, v = k.strip(), v.strip()
                        if v == "nan":
                            v = self.stats_fillna_val
                        new_row[k] = v
                    new_row["Recording locations:"] = "+".join(locations)
                    self._df_stats = self._df_stats.append(
                        new_row,
                        ignore_index=True,
                    )
            self._df_stats.to_csv(stats_file, index=False)
        else:
            print("No data found locally!")
            return
        self._df_stats = self._df_stats.fillna(self.stats_fillna_val)
        try:
            # the column "Locations" is changed to "Recording locations:" in version 1.0.2
            self._df_stats.Locations = self._df_stats.Locations.apply(
                lambda s: s.split("+")
            )
        except AttributeError:
            self._df_stats["Locations"] = self._df_stats["Recording locations:"].apply(
                lambda s: s.split("+")
            )
        self._df_stats["Murmur locations"] = self._df_stats["Murmur locations"].apply(
            lambda s: s.split("+")
        )
        self._df_stats["Patient ID"] = self._df_stats["Patient ID"].astype(str)
        self._df_stats = self._df_stats[self._stats_cols]
        for idx, row in self._df_stats.iterrows():
            for c in ["Height", "Weight"]:
                if row[c] == self.stats_fillna_val:
                    self._df_stats.at[idx, c] = np.nan

        # load stats of the records
        print("Reading the statistics of the records from local file...")
        stats_file = self.db_dir / "stats_records.csv"
        if stats_file.exists():
            self._df_stats_records = pd.read_csv(stats_file)
        else:
            self._df_stats_records = pd.DataFrame(columns=self._stats_records_cols)
            with tqdm(
                self._df_stats.iterrows(),
                total=len(self._df_stats),
                desc="loading record stats",
            ) as pbar:
                for _, row in pbar:
                    sid = row["Patient ID"]
                    for loc in row["Locations"]:
                        rec = f"{sid}_{loc}"
                        if rec not in self._all_records:
                            continue
                        header = wfdb.rdheader(str(self.data_dir / f"{rec}"))
                        if row["Murmur"] == "Unknown":
                            murmur = "Unknown"
                        if loc in row["Murmur locations"]:
                            murmur = "Present"
                        else:
                            murmur = "Absent"
                        new_row = {
                            "Patient ID": sid,
                            "Location": loc,
                            "rec": rec,
                            "siglen": header.sig_len,
                            "siglen_sec": header.sig_len / header.fs,
                            "Murmur": murmur,
                        }
                        self._df_stats_records = self._df_stats_records.append(
                            new_row,
                            ignore_index=True,
                        )
            self._df_stats_records.to_csv(stats_file, index=False)
        self._df_stats_records = self._df_stats_records.fillna(self.stats_fillna_val)

    def _decompose_rec(self, rec: Union[str, int]) -> Dict[str, str]:
        """
        decompose a record name into its components (subject, location, and number)

        Parameters
        ----------
        rec: str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        dict,
            the components (subject, location, and number) of the record,
            with keys "sid", "loc", and "num" respectively

        """
        if isinstance(rec, int):
            rec = self[rec]
        return list(re.finditer(self._rec_pattern, rec))[0].groupdict()

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """
        get the absolute path of the record `rec`

        Parameters
        ----------
        rec: str or int,
            record name or index of the record in `self.all_records`
        extension: str, optional,
            extension of the file

        Returns
        -------
        Path,
            absolute path of the file

        """
        if isinstance(rec, int):
            rec = self[rec]
        path = self._df_records.loc[rec, "path"]
        if extension is not None and not extension.startswith("."):
            extension = f".{extension}"
        return path.with_suffix(extension or "").resolve()

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """
        load data from the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the record, defaults to `self.fs`,
            -1 for the sampling frequency from the audio file
        data_format : str, optional,
            the format of the returned data, defaults to `channel_first`
            can be `channel_last`, `channel_first`, `flat`,
            case insensitive
        data_type : str, default "np",
            the type of the returned data, can be one of "pt", "np",
            case insensitive

        Returns
        -------
        data : np.ndarray,
            the data of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.get_absolute_path(rec, self.data_ext)
        data, data_fs = self._audio_load_func(data_file, fs)
        # data of shape (n_channels, n_samples), of type torch.Tensor
        if fs is not None and data_fs != fs:
            data = torchaudio.transforms.Resample(data_fs, fs)(data)
        if data_format.lower() == "channel_last":
            data = data.T
        elif data_format.lower() == "flat":
            data = data.reshape(-1)
        if data_type.lower() == "np":
            data = data.numpy()
        elif data_type.lower() != "pt":
            raise ValueError(f"Unsupported data type: {data_type}")
        return data

    @add_docstring(load_data.__doc__)
    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """alias of `load_data`"""
        return self.load_data(rec, fs, data_format, data_type)

    def load_ann(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """
        load classification annotation of the record `rec` or the subject `sid`

        Parameters
        ----------
        rec_or_sid : str or int,
            the record name or the index of the record in `self.all_records`
            or the subject id
        class_map : dict, optional,
            the mapping of the annotation classes

        Returns
        -------
        ann : str or int,
            the class of the record,
            or the number of the class if `class_map` is provided

        """
        if isinstance(rec_or_sid, int):
            rec_or_sid = self[rec_or_sid]
        _class_map = class_map or {}
        if rec_or_sid in self.all_subjects:
            ann = self.df_stats[self.df_stats["Patient ID"] == rec_or_sid].iloc[0][
                "Murmur"
            ]
        elif rec_or_sid in self.all_records:
            decom = self._decompose_rec(rec_or_sid)
            sid, loc = decom["sid"], decom["loc"]
            row = self.df_stats[self.df_stats["Patient ID"] == sid].iloc[0]
            if row["Murmur"] == "Unknown":
                ann = "Unknown"
            if loc in row["Murmur locations"]:
                ann = "Present"
            else:
                ann = "Absent"
        else:
            raise ValueError(f"{rec_or_sid} is not a valid record or patient ID")
        ann = _class_map.get(ann, ann)
        return ann

    @add_docstring(load_ann.__doc__)
    def load_murmur(
        self, rec_or_sid: Union[str, int], class_map: Optional[Dict[str, int]] = None
    ) -> Union[str, int]:
        """alias of `load_ann`"""
        return self.load_ann(rec_or_sid, class_map)

    def load_segmentation(
        self,
        rec: Union[str, int],
        seg_format: str = "df",
        ensure_same_len: bool = True,
        fs: Optional[int] = None,
    ) -> Union[pd.DataFrame, np.ndarray, dict]:
        """
        load the segmentation of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        seg_format : str, default `df`,
            the format of the returned segmentation,
            can be `df`, `dict`, `mask`, `binary`,
            case insensitive
        ensure_same_len : bool, default True,
            if True, the length of the segmentation will be
            the same as the length of the audio data
        fs : int, optional,
            the sampling frequency, defaults to `self.fs`,
            -1 for the sampling frequency from the audio file

        Returns
        -------
        pd.DataFrame or np.ndarray or dict,
            the segmentation of the record

        NOTE
        ----
        segmentation files do NOT have the same length (namely the second column of the last row of these .tsv files) as the audio files.

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = self.get_fs(rec)
        segmentation_file = self.get_absolute_path(rec, self.segmentation_ext)
        df_seg = pd.read_csv(segmentation_file, sep="\t", header=None)
        df_seg.columns = ["start_t", "end_t", "label"]
        if self.ignore_unannotated:
            df_seg["label"] = df_seg["label"].apply(
                lambda x: x - 1 if x > 0 else BaseCfg.ignore_index
            )
        df_seg["wave"] = df_seg["label"].apply(lambda s: self.segmentation_map[s])
        df_seg["start"] = (fs * df_seg["start_t"]).apply(round)
        df_seg["end"] = (fs * df_seg["end_t"]).apply(round)
        if ensure_same_len:
            sig_len = wfdb.rdheader(str(self.get_absolute_path(rec))).sig_len
            if sig_len != df_seg["end"].max():
                df_seg = df_seg.append(
                    dict(
                        start_t=df_seg["end"].max() / fs,
                        end_t=sig_len / fs,
                        start=df_seg["end"].max(),
                        end=sig_len,
                        wave="unannotated",
                        label=BaseCfg.ignore_index,
                    ),
                    ignore_index=True,
                )
        if seg_format.lower() in [
            "dataframe",
            "df",
        ]:
            return df_seg
        elif seg_format.lower() in [
            "dict",
            "dicts",
        ]:
            # dict of intervals
            return {
                k: [
                    [row["start"], row["end"]]
                    for _, row in df_seg[df_seg["wave"] == k].iterrows()
                ]
                for _, k in self.segmentation_map.items()
            }
        elif seg_format.lower() in [
            "mask",
        ]:
            # mask = np.zeros(df_seg.end.values[-1], dtype=int)
            mask = np.full(df_seg.end.values[-1], BaseCfg.ignore_index, dtype=int)
            for _, row in df_seg.iterrows():
                mask[row["start"] : row["end"]] = int(row["label"])
            return mask
        elif seg_format.lower() in [
            "binary",
        ]:
            bin_mask = np.zeros(
                (df_seg.end.values[-1], len(self.segmentation_states)), dtype=self.dtype
            )
            for _, row in df_seg.iterrows():
                if row["wave"] in self.segmentation_states:
                    bin_mask[
                        row["start"] : row["end"],
                        self.segmentation_states.index(row["wave"]),
                    ] = 1
            return bin_mask
        else:
            raise ValueError(f"{seg_format} is not a valid format")

    def load_meta_data(
        self,
        subject: str,
        keys: Optional[Union[Sequence[str], str]] = None,
    ) -> Union[dict, str, float, int]:
        """
        load meta data of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id
        keys : str or sequence of str, optional,
            the keys of the meta data to be returned,
            if None, return all meta data

        Returns
        -------
        meta_data : dict or str or float or int,
            the meta data of the subject

        """
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        meta_data = row.to_dict()
        if keys:
            if isinstance(keys, str):
                for k, v in meta_data.items():
                    if k.lower() == keys.lower():
                        return v
            else:
                _keys = [k.lower() for k in keys]
                return {k: v for k, v in meta_data.items() if k.lower() in _keys}
        return meta_data

    def load_outcome(self, rec_or_subject: Union[str, int]) -> str:
        """
        load the outcome of the subject or the subject related to the record

        Parameters
        ----------
        rec_or_subject : str or int,
            the record name or the index of the record in `self.all_records`,
            or the subject id (Patient ID)

        Returns
        -------
        outcome : str,
            the outcome of the record

        """
        if isinstance(rec_or_subject, int):
            rec_or_subject = self[rec_or_subject]
        if rec_or_subject in self.all_subjects:
            pass
        elif rec_or_subject in self.all_records:
            decom = self._decompose_rec(rec_or_subject)
            rec_or_subject = decom["sid"]
        else:
            raise ValueError(f"{rec_or_subject} is not a valid record or patient ID")
        outcome = self.load_outcome_(rec_or_subject)
        return outcome

    def load_outcome_(self, subject: str) -> str:
        """
        load the expert cardiologist's overall diagnosis of  of the subject `subject`

        Parameters
        ----------
        subject : str,
            the subject id

        Returns
        -------
        str,
            the expert cardiologist's overall diagnosis,
            one of `Normal`, `Abnormal`

        """
        if isinstance(subject, int) or subject in self.all_records:
            raise ValueError("subject should be chosen from `self.all_subjects`")
        row = self._df_stats[self._df_stats["Patient ID"] == subject].iloc[0]
        return row.Outcome

    def _load_preprocessed_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        passband: Sequence[int] = BaseCfg.passband,
        order: int = BaseCfg.filter_order,
        spike_removal: bool = True,
    ) -> np.ndarray:
        """
        load preprocessed data of the record `rec`,
        with preprocessing procedure:
            - resample to `fs` (if `fs` is not None)
            - bandpass filter
            - spike removal

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the returned data
        data_format : str, default `channel_first`,
            the format of the returned data,
            can be `channel_first`, `channel_last` or `flat`,
            case insensitive
        passband : sequence of int, default `BaseCfg.passband`,
            the passband of the bandpass filter
        order : int, default `BaseCfg.filter_order`,
            the order of the bandpass filter
        spike_removal : bool, default True,
            whether to remove spikes using the Schmmidt algorithm

        Returns
        -------
        data : np.ndarray,
            the preprocessed data of the record

        """
        fs = fs or self.fs
        data = butter_bandpass_filter(
            self.load_data(rec, fs=fs, data_format="flat"),
            lowcut=passband[0],
            highcut=passband[1],
            fs=fs,
            order=order,
        ).astype(self.dtype)
        if spike_removal:
            data = schmidt_spike_removal(data, fs=fs)
        if data_format.lower() == "flat":
            return data
        data = np.atleast_2d(data)
        if data_format.lower() == "channel_last":
            data = data.T
        return data

    def get_fs(self, rec: Union[str, int]) -> int:
        """
        get the original sampling frequency of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        int,
            the original sampling frequency of the record

        """
        return wfdb.rdheader(str(self.get_absolute_path(rec))).fs

    def get_subject(self, rec: Union[str, int]) -> str:
        """
        get the subject id (Patient ID) of the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`

        Returns
        -------
        str,
            the subject id (Patient ID) of the record

        """
        return self._decompose_rec(rec)["sid"]

    @property
    def all_subjects(self) -> List[str]:
        return self._all_subjects

    @property
    def subject_records(self) -> Dict[str, List[str]]:
        return self._subject_records

    @property
    def df_stats(self) -> pd.DataFrame:
        if self._df_stats is None or self._df_stats.empty:
            self._load_stats()
        return self._df_stats

    @property
    def df_stats_records(self) -> pd.DataFrame:
        if self._df_stats_records is None or self._df_stats_records.empty:
            self._load_stats()
        return self._df_stats_records

    @property
    def murmur_feature_cols(self) -> List[str]:
        return [
            "Systolic murmur timing",
            "Systolic murmur shape",
            "Systolic murmur grading",
            "Systolic murmur pitch",
            "Systolic murmur quality",
            "Diastolic murmur timing",
            "Diastolic murmur shape",
            "Diastolic murmur grading",
            "Diastolic murmur pitch",
            "Diastolic murmur quality",
        ]

    def play(self, rec: Union[str, int], **kwargs) -> IPython.display.Audio:
        """
        play the record `rec` in a Juptyer Notebook

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        kwargs : dict,
            optional keyword arguments including `data`, `fs`,
            if specified, the data will be played instead of the record

        Returns
        -------
        IPython.display.Audio,
            the audio object of the record

        """
        if "data" in kwargs:
            return IPython.display.Audio(
                kwargs["data"], rate=kwargs.get("fs", self.get_fs(rec))
            )
        audio_file = self.get_absolute_path(rec)
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: Union[str, int], **kwargs) -> NoReturn:
        """
        plot the record `rec`, with metadata and segmentation

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        kwargs : dict,
            not used currently

        Returns
        -------
        fig: matplotlib.figure.Figure,
            the figure of the record
        ax: matplotlib.axes.Axes,
            the axes of the figure

        """
        import matplotlib.pyplot as plt

        waveforms = self.load_pcg(rec, data_format="flat")
        df_segmentation = self.load_segmentation(rec)
        meta_data = self.load_meta_data(self.get_subject(rec))
        labels = {
            "Outcome": meta_data["Outcome"],
            "Murmur": meta_data["Murmur"],
        }
        meta_data = {
            k: "NA" if meta_data[k] == self.stats_fillna_val else meta_data[k]
            for k in ["Age", "Sex", "Height", "Weight", "Pregnancy status"]
        }
        rec_dec = self._decompose_rec(rec)
        rec_dec = {
            "SubjectID": rec_dec["sid"],
            "Location": rec_dec["loc"],
            "Number": rec_dec["num"],
        }
        rec_dec = {k: v for k, v in rec_dec.items() if v is not None}
        figsize = (5 * len(waveforms) / self.fs, 5)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(
            np.arange(len(waveforms)) / self.fs,
            waveforms,
            color=self.palette["default"],
        )
        counter = {
            "systolic": 0,
            "diastolic": 0,
            "S1": 0,
            "S2": 0,
        }
        for _, row in df_segmentation.iterrows():
            if row.wave != "unannotated":
                # labels starting with "_" are ignored
                # ref. https://stackoverflow.com/questions/44632903/setting-multiple-axvspan-labels-as-one-element-in-legend
                ax.axvspan(
                    row.start_t,
                    row.end_t,
                    color=self.palette[row.wave],
                    alpha=0.3,
                    label="_" * counter[row.wave] + row.wave,
                )
                counter[row.wave] += 1
        ax.legend(loc="upper right")
        bbox_prop = {
            "boxstyle": "round",
            "facecolor": "#EAEAF2",
            "edgecolor": "black",
        }
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in rec_dec.items()]),
            (0.01, 0.95),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in meta_data.items()]),
            (0.01, 0.80),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )
        ax.annotate(
            "\n".join(["{}: {}".format(k, v) for k, v in labels.items()]),
            (0.01, 0.15),
            xycoords="axes fraction",
            va="top",
            bbox=bbox_prop,
        )

        return fig, ax

    def plot_outcome_correlation(self, col: str = "Murmur", **kwargs: Any) -> object:
        """
        plot the correlation between the outcome and the feature `col`

        Parameters
        ----------
        col: str, default "Murmur",
            the feature to be used for the correlation, can be one of
            "Murmur", "Age", "Sex", "Pregnancy status"
        kwargs: dict,
            key word arguments,
            passed to the function `pd.DataFrame.plot`

        Returns
        -------
        ax: mpl.axes.Axes

        """
        # import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns

        # sns.set()
        sns.set_theme(style="white")  # darkgrid, whitegrid, dark, white, ticks
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["axes.labelsize"] = 24
        plt.rcParams["legend.fontsize"] = 18
        plt.rcParams["hatch.linewidth"] = 2.5

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        hatches = ["/", "\\", "|", ".", "x"]

        assert col in ["Murmur", "Age", "Sex", "Pregnancy status"]
        prefix_sep = " - "
        df_dummies = pd.get_dummies(
            self.df_stats[col], prefix=col, prefix_sep=prefix_sep
        )
        columns = df_dummies.columns.tolist()
        if f"{col}{prefix_sep}{self.stats_fillna_val}" in columns:
            idx = columns.index(f"{col}{prefix_sep}{self.stats_fillna_val}")
            columns[idx] = f"{col}{prefix_sep}{'NA'}"
            df_dummies.columns = columns
        df_stats = pd.concat((self.df_stats, df_dummies), axis=1)
        plot_kw = dict(
            kind="bar",
            figsize=(8, 8),
            ylabel="Number of Subjects (n.u.)",
            stacked=True,
            rot=0,
            ylim=(0, 620),
            yticks=np.arange(0, 700, 100),
            width=0.3,
            fill=True,
            # hatch=hatches[: len(columns)],
        )
        plot_kw.update(kwargs)
        ax = (
            df_stats.groupby("Outcome")
            .agg("sum")[df_dummies.columns.tolist()]
            .plot(**plot_kw)
        )
        for idx in range(len(columns)):
            ax.patches[2 * idx].set_hatch(hatches[idx])
            ax.patches[2 * idx + 1].set_hatch(hatches[idx])
        ax.legend(loc="upper left", ncol=int(np.ceil(len(columns) / 3)))
        plt.tight_layout()

        # mpl.rc_file_defaults()

        return ax


_CINC2016_INFO = DataBaseInfo(  # NOT finished yet
    title="""
    Classification of Heart Sound Recordings:
    The PhysioNet/Computing in Cardiology Challenge 2016
    """,
    about="""
    1. The Challenge training set consists of five databases (A through E) containing a total of 3,126 heart sound recordings collected from different locations on the body (4 typical locations are aortic area, pulmonic area, tricuspid area and mitral area, but could be one of 9 different locations).
    2. The recordings last from several (5) seconds to up to more than one hundred seconds (just over 120 seconds). All recordings have been resampled to 2,000 Hz and have been provided as .wav format. Each recording contains only one PCG lead.
    3. The training and test (unavailable to the public) sets have each been divided so that they are two sets of mutually exclusive populations.
    4. Heart sound recordings (.wav files) were divided into two types: normal and abnormal heart sound recordings. The normal recordings were from healthy subjects and the abnormal ones were from patients with a confirmed cardiac diagnosis.
    5. Due to the uncontrolled environment of the recordings, many recordings are corrupted by various noise sources, such as talking, stethoscope motion, breathing and intestinal sounds. Some recordings were difficult or even impossible to classify as normal or abnormal. Hence a third class "Uncertain" is added for scoring.
    6. Extra ECG recordings (.dat files) were provided along with the heart sound recordings.
    """,
    usage=[
        "Heart murmur detection",
        "Heart abnormality detection from ECG",
    ],
    references=[
        "https://physionet.org/content/challenge-2016/1.0.0/",
    ],
    doi=[
        "10.1088/0967-3334/37/12/2181",
    ],
)


@add_docstring(f"\n{_HeartMurmurInfo}\n", mode="append")
@add_docstring(_CINC2016_INFO.format_database_docstring())
class CINC2016Reader(PCGDataBase):
    """ """

    __name__ = "CINC2016Reader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 2000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="challenge-2016",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "wav"
        self.ecg_ext = "dat"
        self.ann_ext = "hea"

        self._subsets = [f"training-{s}" for s in "abcde"]

        self._all_records = None
        self._ls_rec()

    def _ls_rec(self) -> NoReturn:
        """ """
        records_file = self.db_dir / "RECORDS"
        write_file = False
        self._df_records = pd.DataFrame()
        if records_file.exists():
            self._df_records["record"] = records_file.read_text().splitlines()
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
        else:
            write_file = True
        if len(self._df_records) == 0:
            write_file = True
            self._df_records["path"] = get_record_list_recursive(
                self.db_dir, self.header_ext, relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
        self._df_records["subset"] = self._df_records["path"].apply(
            lambda x: x.parent.name
        )
        self._df_records = self._df_records[
            self._df_records["subset"].isin(self._subsets)
        ]
        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()
        if write_file:
            records_file.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        path = self._df_records.loc[rec, "path"]
        if extension is not None:
            path = path.with_suffix(
                extension if extension.startswith(".") else f".{extension}"
            )
        return path

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        if isinstance(rec, int):
            rec = self[rec]
        data = {
            "PCG": self.load_pcg(rec, fs, data_format, data_type),
            "ECG": self.load_ecg(rec, fs, data_format, data_type),
        }
        return data

    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data_file = self.get_absolute_path(rec, self.data_ext)
        pcg, pcg_fs = self._audio_load_func(data_file, fs)
        # data of shape (n_channels, n_samples), of type torch.Tensor
        if fs is not None and pcg_fs != fs:
            pcg = torchaudio.transforms.Resample(pcg_fs, fs)(pcg)
        if data_format.lower() == "channel_last":
            pcg = pcg.T
        elif data_format.lower() == "flat":
            pcg = pcg.reshape(-1)
        if data_type.lower() == "np":
            pcg = pcg.numpy()
        elif data_type.lower() != "pt":
            raise ValueError(f"Unsupported data type: {data_type}")
        return pcg

    def load_ecg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        wfdb_rec = wfdb.rdrecord(
            str(self.get_absolute_path(rec)), channel_names=["ECG"], physical=True
        )
        ecg = wfdb_rec.p_signal.T
        if fs is not None and fs != wfdb_rec.fs:
            # ecg = ss.resample_poly(ecg, fs, wfdb_rec.fs, axis=-1)
            ecg = librosa.resample(ecg, wfdb_rec.fs, fs, res_type="kaiser_best")
        if data_format.lower() == "channel_last":
            ecg = ecg.T
        if data_type.lower() == "pt":
            ecg = torch.from_numpy(ecg).float()
        return ecg

    def load_ann(self, rec: Union[str, int]) -> str:
        """
        load annotations of the record `rec`
        """
        return wfdb.rdheader(self.get_absolute_path(rec)).comments[0]

    def play(self, rec: Union[str, int], **kwargs) -> IPython.display.Audio:
        """ """
        audio_file = self.get_absolute_path(rec, self.data_ext)
        return IPython.display.Audio(filename=str(audio_file))

    def plot(self, rec: Union[str, int], **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError

    @property
    def url(self) -> List[str]:
        return [
            f"https://physionet.org/files/challenge-2016/1.0.0/{subset}.zip?download"
            for subset in ["training", "validation"]
        ]

    def download(self) -> NoReturn:
        """ """
        for url in self.url:
            http_get(url, self.db_dir / Path(url.split("?")[0]).stem, extract=True)

    validation_set = (
        "a0001\na0002\na0003\na0004\na0005\na0006\na0007\na0008\na0009\na0010\na0011\na0012\na0013\n"
        "a0014\na0015\na0016\na0017\na0018\na0019\na0020\na0021\na0022\na0023\na0024\na0025\na0026\n"
        "a0027\na0028\na0029\na0030\na0031\na0032\na0033\na0034\na0035\na0036\na0037\na0038\na0039\n"
        "a0040\na0041\na0042\na0043\na0044\na0045\na0046\na0047\na0048\na0049\na0050\na0051\na0052\n"
        "a0053\na0054\na0055\na0056\na0057\na0068\na0069\na0070\na0071\na0080\na0081\na0085\na0086\n"
        "a0088\na0091\na0093\na0094\na0102\na0105\na0106\na0108\na0109\na0118\na0125\na0127\na0129\n"
        "a0136\na0139\nb0001\nb0002\nb0003\nb0004\nb0005\nb0006\nb0007\nb0008\nb0009\nb0010\nb0011\n"
        "b0012\nb0013\nb0014\nb0015\nb0016\nb0017\nb0018\nb0019\nb0020\nb0021\nb0022\nb0023\nb0024\n"
        "b0025\nb0026\nb0027\nb0028\nb0029\nb0030\nb0031\nb0032\nb0033\nb0034\nb0035\nb0036\nb0037\n"
        "b0038\nb0039\nb0040\nb0041\nb0042\nb0043\nb0044\nb0045\nb0046\nb0047\nb0048\nb0049\nb0050\n"
        "b0051\nb0052\nb0053\nb0054\nb0055\nb0056\nb0057\nb0058\nb0059\nb0060\nb0061\nb0062\nb0063\n"
        "b0064\nb0065\nb0066\nb0067\nb0068\nb0077\nb0081\nb0086\nb0096\nb0106\nb0117\nb0120\nb0130\n"
        "b0136\nb0137\nb0140\nb0148\nb0155\nb0159\nb0164\nb0171\nb0176\nb0190\nb0197\nb0208\nb0221\n"
        "b0224\nb0232\nb0233\nb0235\nb0238\nb0239\nb0242\nb0243\nb0248\nc0001\nc0002\nc0003\nc0004\n"
        "c0006\nc0007\nc0031\nd0001\nd0002\nd0003\nd0004\nd0005\nd0006\nd0007\nd0008\nd0009\nd0013\n"
        "e00001\ne00002\ne00003\ne00004\ne00005\ne00006\ne00007\ne00008\ne00009\ne00010\ne00011\n"
        "e00012\ne00013\ne00014\ne00015\ne00016\ne00017\ne00018\ne00019\ne00020\ne00021\ne00022\n"
        "e00023\ne00024\ne00025\ne00026\ne00027\ne00028\ne00029\ne00030\ne00031\ne00032\ne00033\n"
        "e00034\ne00035\ne00036\ne00037\ne00038\ne00039\ne00040\ne00041\ne00042\ne00043\ne00044\n"
        "e00045\ne00046\ne00047\ne00048\ne00049\ne00050\ne00051\ne00052\ne00053\ne00054\ne00055\n"
        "e00059\ne00071\ne00087\ne00097\ne00114\ne00120\ne00135\ne00140\ne00142\ne00152\ne00176\n"
        "e00191\ne00195\ne00216\ne00228\ne00266\ne00275\ne00295\ne00304\ne00305\ne00321\ne00328\n"
        "e00330\ne00336\ne00359\ne00373\ne00388\ne00435\ne00456\ne00457\ne00461\ne00475\ne00477\n"
        "e00523\ne00526\ne00528\ne00536\ne00537\ne00539\ne00551\ne00562\ne00591\ne00601\ne00603\n"
        "e00605\ne00619\ne00622\ne00627\ne00648\ne00657\ne00670\n"
    ).splitlines()  # read from validation/RECORDS


_EPHNOGRAM_INFO = DataBaseInfo(  # NOT finished yet
    title="""
    EPHNOGRAM:
    A Simultaneous Electrocardiogram and Phonocardiogram Database
    """,
    about="""
    1. 61 recordings were acquired from 24 healthy male adults aged between 23 and 29 (average: 25.4 ± 1.9 years) in 30min stress-test sessions during resting, walking, running and biking conditions, using indoor fitness center equipment. The database also contains several (8) 30s sample records acquired during rest conditions.
    2. The device for collecting the database includes three-lead ECG, two digital stethoscope channels for PCG acquisition and two auxiliary channels to capture the ambient audio noise.
    3. The three channels PCG2, AUX1, and AUX2 (which are available for some of the records), are mostly very weak in amplitude (at quantization noise level). However, through visual inspection and by listening to these audio channels, it is noticed that they have captured some of the electronic device noises and the weak background sounds in the environment
    4. The analog signals are filtered by an anti-aliasing analog filter and sampled at 8kHz with a resolution of 12-bits (with 10.5 effective number of bits).
    5. The front-end anti-aliasing and baseline wander rejection filter consists of a first-order passive high-pass filter with a -3dB cutoff frequency of 0.1Hz, followed by an active 5th order low-pass Butterworth filter, which form bandpass filters that cover the major ECG (upper -3dB cutoff frequency was set to 150Hz, with 30dB of attenuation at 1kHz and a 30dB gain in the passband) and PCG (upper cutoff frequency of 1kHz, 30dB of attenuation at 5kHz, and a passband gain of 5dB) bandwidths.
    6. Additional filtering, including power-line cancellation (50Hz) is performed in the digital domain.
    """,
    usage=[
        "Simultaneous multi-modal analysis of ECG and PCG",
        "Mathematical PCG models for generating synthetic signals"
        "PCG quality enhancement",
        "PCG model pretraining",
        "ECG model pretraining",
    ],
    references=[
        "https://physionet.org/content/ephnogram/1.0.0/",
        "A. Kazemnejad, P. Gordany, and R. Sameni. An open-access simultaneous electrocardiogram and phonocardiogram database. bioRxiv, 2021. DOI: https://doi.org/10.1101/2021.05.17.444563",
        "R. Sameni, The Open-Source Electrophysiological Toolbox (OSET), v 3.14, URL: https://github.com/alphanumericslab/OSET",
    ],
    doi=[
        "10.13026/tjtq-5911",
        "10.1101/2021.05.17.444563",
    ],
)


@add_docstring(f"\n{_HeartMurmurInfo}\n", mode="append")
@add_docstring(_EPHNOGRAM_INFO.format_database_docstring())
class EPHNOGRAMReader(PCGDataBase):
    """ """

    __name__ = "EPHNOGRAMReader"

    def __init__(
        self,
        db_dir: str,
        fs: int = 8000,
        audio_backend: str = "torchaudio",
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            db_name="ephnogram",
            db_dir=db_dir,
            fs=fs,
            audio_backend=audio_backend,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "mat"
        self.aux_ext = "dat"

        self._ls_rec()
        self.data_dir = self.db_dir / "MAT"
        self.aux_dir = self.db_dir / "WFDB"
        self._channels = [
            "ECG",
            "PCG",
            "PCG2",
            "AUX1",
            "AUX2",
        ]

        self._df_stats = pd.DataFrame()
        self._aggregate_stats()

    def _ls_rec(self) -> NoReturn:
        """ """
        self._df_records = pd.DataFrame()
        try:
            self._df_records["record"] = wfdb.get_record_list(self.db_name)
            self._df_records["record"] = self._df_records["record"].apply(
                lambda x: x.replace("WFDB", "MAT")
            )
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
            self._df_records["record"] = self._df_records["path"].apply(
                lambda x: x.stem
            )
            self._df_records["aux_path"] = self._df_records["path"].apply(
                lambda x: x.parents[1] / "WFDB" / x.stem
            )
            self._df_records.set_index("record", inplace=True)
            self._df_records = self._df_records[
                self._df_records["path"].apply(lambda x: x.exists())
            ]
            self._all_records = self._df_records.index.values.tolist()
        except Exception:
            self._ls_rec_local()
        if len(self._df_records) == 0:
            self._ls_rec_local()

    def _ls_rec_local(self) -> NoReturn:
        """ """
        self._df_records = pd.DataFrame(columns=["record", "path", "aux_path"])
        records_file = self.db_dir / "RECORDS"
        write_file = False
        if records_file.exists():
            self._df_records["record"] = records_file.read_text().splitlines()
            self._df_records["record"] = self._df_records["record"].apply(
                lambda x: x.replace("WFDB", "MAT")
            )
            self._df_records["path"] = self._df_records["record"].apply(
                lambda x: self.db_dir / x
            )
            self._df_records = self._df_records[
                self._df_records["path"].apply(lambda x: x.is_file())
            ]
        else:
            write_file = True
        if len(self._df_records) == 0:
            write_file = True
            self._df_records["path"] = get_record_list_recursive(
                self.db_dir, self.data_ext, relative=False
            )
            self._df_records["path"] = self._df_records["path"].apply(lambda x: Path(x))
        self._df_records["record"] = self._df_records["path"].apply(lambda x: x.stem)
        self._df_records["aux_path"] = self._df_records["path"].apply(
            lambda x: x.parents[1] / "WFDB" / x.stem
        )
        self._df_records.set_index("record", inplace=True)
        self._all_records = self._df_records.index.values.tolist()
        if write_file:
            records_file.write_text(
                "\n".join(
                    self._df_records["path"]
                    .apply(lambda x: x.relative_to(self.db_dir).as_posix())
                    .tolist()
                )
            )

        records_file.write_text("\n".join(self._all_records))

    def _aggregate_stats(self) -> NoReturn:
        """ """
        if len(self) == 0:
            return
        self._df_stats = pd.read_csv(self.db_dir / "ECGPCGSpreadsheet.csv")
        self._df_stats = self._df_stats[
            [
                "Record Name",
                "Subject ID",
                "Record Duration (min)",
                "Age (years)",
                "Gender",
                "Recording Scenario",
                "Num Channels",
                "ECG Notes",
                "PCG Notes",
                "PCG2 Notes",
                "AUX1 Notes",
                "AUX2 Notes",
                "Database Housekeeping",
            ]
        ]
        self._df_stats = self._df_stats[
            ~self._df_stats["Record Name"].isna()
        ].reset_index(drop=True)

    def get_absolute_path(
        self, rec: Union[str, int], extension: Optional[str] = None
    ) -> Path:
        """ """
        if isinstance(rec, int):
            rec = self[rec]
        key = "path"
        if extension is not None and extension.strip(".") == self.aux_ext:
            key = "aux_path"
        path = self._df_records.loc[rec, key]
        if extension is not None:
            path = path.with_suffix(
                extension if extension.startswith(".") else f".{extension}"
            )
        return path

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
        channels: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        load data from the record `rec`
        """
        assert data_format in ["channel_first", "channel_last"]
        assert data_type in ["np", "pt"]
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        data = sio.loadmat(self.get_absolute_path(rec, self.data_ext))
        data_fs = data["fs"][0][0]
        channels = channels or self._channels
        if isinstance(channels, str):
            channels = [channels]
        assert set(channels).issubset(self._channels), "invalid channels"
        data = {
            k: torch.from_numpy(data[k].astype(np.float32))
            if data_format.lower() == "channel_first"
            else torch.from_numpy(data[k].astype(np.float32).T)
            for k in channels
            if k in data
        }
        if fs is not None and fs != data_fs:
            resampler = torchaudio.transforms.Resample(data_fs, fs)
            for k in data:
                data[k] = resampler(data[k])
        if data_type.lower() == "np":
            data = {k: v.numpy() for k, v in data.items()}
        elif data_type.lower() != "pt":
            raise ValueError(f"Unsupported data type: {data_type}")
        return data

    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """ """
        concat_dim = -1 if data_format.lower() == "channel_last" else 0
        pcg = self.load_data(rec, fs, data_format, data_type, ["PCG", "PCG2"])
        if data_type.lower() == "pt":
            pcg = torch.cat(list(pcg.values()), dim=concat_dim)
        else:
            pcg = np.concatenate(list(pcg.values()), axis=concat_dim)
        return pcg

    def load_ann(self, rec: str) -> str:
        """
        load annotations of the record `rec`
        """
        raise NotImplementedError("No annotation for this database")

    def play(self, rec: Union[str, int], channel: str = "PCG") -> IPython.display.Audio:
        """ """
        data = self.load_data(rec, channels=channel)[0]
        return IPython.display.Audio(data=data, rate=8000)

    def plot(self, rec: Union[str, int], **kwargs) -> NoReturn:
        """ """
        raise NotImplementedError

    @property
    def df_stats(self) -> pd.DataFrame:
        if self._df_stats.empty:
            self._aggregate_stats()
        return self._df_stats


@add_docstring(_HeartMurmurInfo, mode="append")
class CompositeReader(ReprMixin):
    """
    Database reader that combines multiple readers,
    for the purpose of pretraining.

    """

    __name__ = "CompositeReader"

    def __init__(
        self, databases: Sequence[PCGDataBase], fs: Optional[int] = None
    ) -> NoReturn:
        """ """
        self.databases = databases
        self.fs = fs

        self._sep = "^" * (
            max(
                [
                    len(item)
                    for item in list_sum(
                        [
                            re.findall("[\\^]+", rec)
                            for dr in databases
                            for rec in dr.all_records
                        ]
                    )
                    + ["^"]
                ]
            )
            + 1
        )
        self._all_records = [
            self.get_composite_record_name(dr, rec)
            for dr in databases
            for rec in dr.all_records
        ]
        self._db_name_map = {dr.db_name: dr for dr in self.databases}

    def get_composite_record_name(self, database: PCGDataBase, rec: str) -> str:
        """
        get the composite record name of the record `rec` in the database `database`

        Parameters
        ----------
        database: PCGDataBase,
            database reader
        rec: str,
            record name

        Returns
        -------
        str,
            composite record name

        """
        assert rec in database.all_records
        return f"{database.db_name}{self._sep}{rec}"

    @property
    def all_records(self) -> List[str]:
        """ """
        return self._all_records

    def __len__(self) -> int:
        """
        number of records in the database
        """
        return len(self.all_records)

    def __getitem__(self, index: int) -> str:
        """
        get the record name by index
        """
        return self.all_records[index]

    def load_data(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """
        load data from the record `rec`

        Parameters
        ----------
        rec : str or int,
            the record name or the index of the record in `self.all_records`
        fs : int, optional,
            the sampling frequency of the record, defaults to `self.fs`
        data_format : str, optional,
            the format of the returned data, defaults to `channel_first`
            can be `channel_last`, `channel_first`, `flat`,
            case insensitive
        data_type : str, default "np",
            the type of the returned data, can be one of "pt", "np",
            case insensitive

        Returns
        -------
        data : np.ndarray,
            the data of the record

        """
        if isinstance(rec, int):
            rec = self[rec]
        fs = fs or self.fs
        if fs == -1:
            fs = None
        db_name, rec = rec.split(self._sep)
        dr = self._db_name_map[db_name]
        data = dr.load_pcg(rec, fs=fs, data_format=data_format, data_type=data_type)
        return data

    @add_docstring(load_data.__doc__)
    def load_pcg(
        self,
        rec: Union[str, int],
        fs: Optional[int] = None,
        data_format: str = "channel_first",
        data_type: str = "np",
    ) -> np.ndarray:
        """alias of `load_data`"""
        return self.load_data(rec, fs, data_format, data_type)

    def extra_repr_keys(self) -> List[str]:
        return ["databases", "fs"]
