import os
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

from ...cfg import DEFAULTS
from ...utils.misc import add_docstring, get_record_list_recursive3
from ..base import DataBaseInfo, PhysioNetDataBase

__all__ = [
    "PTBXL",
]


_PTBXL_INFO = DataBaseInfo(
    title="""
    PTB-XL, a large publicly available electrocardiography dataset
    """,
    about="""
    1. The PTB-XL database [1]_ is a large database of 21799 clinical 12-lead ECGs from 18869 patients of 10 second length collected with devices from Schiller AG over the course of nearly seven years between October 1989 and June 1996.
    2. The raw waveform data of the PTB-XL database was annotated by up to two cardiologists, who assigned potentially multiple ECG statements to each recording which were converted into a standardized set of SCP-ECG statements (scp_codes).
    3. The PTB-XL database contains 71 different ECG statements conforming to the SCP-ECG standard, including diagnostic, form, and rhythm statements.
    4. The waveform files of the PTB-XL database are stored in WaveForm DataBase (WFDB) format with 16 bit precision at a resolution of 1μV/LSB and a sampling frequency of 500Hz. A downsampled versions of the waveform data at a sampling frequency of 100Hz is also provided.
    5. In the metadata file (ptbxl_database.csv), each record of the PTB-XL database is identified by a unique ecg_id. The corresponding patient is encoded via patient_id. The paths to the original record (500 Hz) and a downsampled version of the record (100 Hz) are stored in `filename_hr` and `filename_lr`. The `report` field contains the diagnostic statements assigned to the record by the cardiologists. The `scp_codes` field contains the SCP-ECG statements assigned to the record which are formed as a dictionary with entries of the form `statement: likelihood`, where likelihood is set to 0 if unknown).
    6. The PTB-XL database underwent a 10-fold train-test splits (stored in the `strat_fold` field of the metadata file) obtained via stratified sampling while respecting patient assignments, i.e. all records of a particular patient were assigned to the same fold. Records in fold 9 and 10 underwent at least one human evaluation and are therefore of a particularly high label quality. It is proposed to use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
    """,
    usage=[
        "Classification of ECG images",
    ],
    note="""
    1. A new comprehensive feature database named PTB-XL+ [2]_ was created to supplement the PTB-XL database.
    """,
    issues="""
    """,
    references=[
        "https://physionet.org/content/ptb-xl/",
        "https://physionet.org/content/ptb-xl-plus/",
    ],
    doi=[
        "https://doi.org/10.1038/s41597-023-02153-8",  # PTB-XL+ paper
        "https://doi.org/10.1038/s41597-020-0495-6",  # PTB-XL paper
        "https://doi.org/10.13026/nqsf-pc74",  # PTB-XL+ physionet
        "https://doi.org/10.13026/6sec-a640",  # PTB-XL physionet
    ],
)


@add_docstring(_PTBXL_INFO.format_database_docstring(), mode="prepend")
class PTBXL(PhysioNetDataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    feature_db_dir : `path-like`, optional
        Whether to include the feature database (the `PTB-XL+` database).
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "PTBXL"
    __metadata_file__ = "ptbxl_database.csv"
    __scp_statements_file__ = "scp_statements.csv"
    __100Hz_dir__ = "records100"
    __500Hz_dir__ = "records500"

    def __init__(
        self,
        db_dir: Union[str, bytes, os.PathLike],
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        feature_db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(db_name="ptb-xl", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
        self.feature_db_dir = Path(feature_db_dir).resolve() if feature_db_dir is not None else None
        if self.feature_db_dir is not None:
            self._feature_reader = PTBXLPlus(db_dir=self.feature_db_dir, verbose=verbose)
        else:
            self._feature_reader = None
        if self.fs is None:
            self.fs = 500
        self.data_ext = "dat"
        self.header_ext = "hea"
        self.record_pattern = "[\\d]{5}_[lh]r"

        self._df_records = None
        self._df_metadata = None
        self._df_scp_statements = None
        self._all_records = None
        self._all_subjects = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        # locate the true database directory using the metadata file
        try:
            metadata_file = list(self.db_dir.rglob(self.__metadata_file__))[0]
        except IndexError:
            # raise FileNotFoundError(f"metadata file {self.__metadata_file__} not found in {self.db_dir}")
            self.logger.info(
                f"metadata file {self.__metadata_file__} not found in {self.db_dir}. "
                "Download the database first using the `download` method."
            )
            self._df_records = pd.DataFrame()
            self._df_metadata = pd.DataFrame()
            self._df_scp_statements = pd.DataFrame()
            self._all_records = []
            self._all_subjects = []
            return
        self.db_dir = metadata_file.parent.resolve()
        assert (self.db_dir / self.__scp_statements_file__).exists(), f"scp_statements file not found in {self.db_dir}"

        # read metadata file and scp_statements file
        self._df_metadata = pd.read_csv(self.db_dir / self.__metadata_file__)
        self._df_metadata["ecg_id"] = self._df_metadata["ecg_id"].apply(lambda x: f"{x:05d}")
        self._df_metadata.set_index("ecg_id", inplace=True)
        self._df_metadata["patient_id"] = self._df_metadata["patient_id"].astype(int)
        self._df_scp_statements = pd.read_csv(self.db_dir / self.__scp_statements_file__, index_col=0)

        self._df_records = self._df_metadata.copy()
        if self.fs == 100:
            self._df_records["path"] = self._df_records["filename_lr"].apply(lambda x: self.db_dir / x)
        else:
            self._df_records["path"] = self._df_records["filename_hr"].apply(lambda x: self.db_dir / x)
        # keep only records that exist
        self._df_records = self._df_records[
            self._df_records["path"].apply(lambda x: x.with_suffix(f".{self.data_ext}").exists())
        ]
        self._df_metadata = self._df_metadata.loc[self._df_records.index]
        if self._subsample is not None:
            size = min(
                len(self._df_records),
                max(1, int(round(self._subsample * len(self._df_records)))),
            )
            self.logger.debug(f"subsample `{size}` records from `{len(self._df_records)}`")
            self._df_records = self._df_records.sample(n=size, random_state=DEFAULTS.SEED, replace=False)
            self._df_metadata = self._df_metadata.loc[self._df_records.index]

        self._all_records = self._df_records.index.tolist()
        self._all_subjects = self._df_records["patient_id"].unique().tolist()

    def reset_fs(self, fs: int) -> None:
        """Reset the default sampling frequency.

        Parameters
        ----------
        fs : int
            The new sampling frequency.

        """
        self.fs = fs
        self._ls_rec()

    def load_metadata(
        self, rec: Union[str, int], items: Optional[Union[str, List[str]]] = None
    ) -> Union[Dict[str, Union[str, int, float]], str, int, float]:
        """Load the metadata of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        items : str or list of str, optional
            The items to load.

        Returns
        -------
        metadata : dict
            The metadata of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        if items is None:
            return self._df_metadata.loc[rec].to_dict()
        if isinstance(items, str):
            metadata = self._df_metadata.loc[rec, items]
            if isinstance(metadata, np.generic):
                return metadata.item()
        return self._df_metadata.loc[rec, items].to_dict()

    def load_ann(self, rec: Union[str, int], with_interpretation: bool = False) -> Dict[str, Union[float, Dict[str, Any]]]:
        """Load the annotation (the "scp_codes" field) of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        with_interpretation : bool, default False
            Whether to include the interpretation of the statement.

        Returns
        -------
        ann : dict
            The annotation of the record, of the form ``{statement: likelihood}``.
            If ``with_interpretation`` is ``True``, the form is
            ``{statement: {"likelihood": likelihood, ...}}``,
            where ``...`` are other information of the statement.

        """
        ann = literal_eval(self.load_metadata(rec)["scp_codes"])
        if with_interpretation:
            for statement, likelihood in ann.items():
                ann[statement] = {"likelihood": likelihood}
                ann[statement].update(self._df_scp_statements.loc[statement].to_dict())
        return ann

    @property
    def database_info(self) -> DataBaseInfo:
        return _PTBXL_INFO

    @property
    def default_train_val_test_split(self) -> Dict[str, List[str]]:
        return {
            "train": self._df_records[self._df_records["strat_fold"] < 9].index.tolist(),
            "val": self._df_records[self._df_records["strat_fold"] == 9].index.tolist(),
            "test": self._df_records[self._df_records["strat_fold"] == 10].index.tolist(),
        }

    @property
    def default_train_val_split(self) -> Dict[str, List[str]]:
        return {
            "train": self._df_records[self._df_records["strat_fold"] < 10].index.tolist(),
            "val": self._df_records[self._df_records["strat_fold"] == 10].index.tolist(),
        }


_PTBXL_PLUS_INFO = DataBaseInfo(
    title="""
    PTB-XL+, a comprehensive electrocardiographic feature dataset
    """,
    about="""
    1. This database [1]_ is a comprehensive feature dataset that supplements the PTB-XL database [2]_.
    2. The features were extracted via two commercial (the University of Glasgow ECG Analysis Program (Uni-G) [3]_, GE Healthcare's Marquette™ 12SL™ (12SL) [4]_) and one open-source algorithm (ECGDeli [5]_).
    3. There are also automatic diagnosis statements from one commercial ECG analysis algorithm.
    4. Features are given a tabular format in the csv files features/12sl_features.csv, features/unig_features.csv and features/ecgdeli_features.csv, with mappings to standardized LOINC IDs given in features/feature_description.csv.
    5. Median beats are provided in the median_beats/12sl and median_beats/unig directories.
    6. Fiducial points are provided in the fiducial_points/ecgdeli directory.
    7. Diagnostic statements are provided in the labels/12sl_statements.csv, labels/unig_statements.csv and labels/ecgdeli_statements.csv files. The subdirectory labels/mapping contains mappings from the 12SL and PTB-XL diagnostic statements to SNOMED codes.
    """,
    usage=[
        "Evaluation of ECG (classification) models",
    ],
    note="""
    """,
    issues="""
    1. The file labels/mapping/ptbxlToSNOMED.csv contains bad lines with inconsistent number of columns.
    """,
    references=[
        "https://physionet.org/content/ptb-xl-plus/",
        "https://physionet.org/content/ptb-xl/",
        "Macfarlane, P., Devine, B. & Clark, E. (2005). The university of glasgow (uni-g) ecg analysis program. In Computers in Cardiology, 451–454.",
        "GE Healthcare (2019). Marquette 12SL ECG Analysis Program: Physician’s Guide. General Electric Company. 2056246-002C.",
        "Pilia, N., Nagel, C., Lenis, G., Becker, S., Dössel, O., Loewe, A. (2021). ECGdeli - an open source ECG delineation toolbox for MATLAB. SoftwareX 13, 100639.",
    ],
    doi=[
        "https://doi.org/10.1038/s41597-023-02153-8",  # PTB-XL+ paper
        "https://doi.org/10.13026/nqsf-pc74",  # PTB-XL+ physionet
    ],
)


@add_docstring(_PTBXL_PLUS_INFO.format_database_docstring(), mode="prepend")
class PTBXLPlus(PhysioNetDataBase):
    """
    Parameters
    ----------
    db_dir : `path-like`
        Storage path of the database.
        If not specified, data will be fetched from Physionet.
    working_dir : `path-like`, optional
        Working directory, to store intermediate files and log files.
    verbose : int, default 1
        Level of logging verbosity.
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "PTBXLPlus"

    def __init__(
        self,
        db_dir: Union[str, bytes, os.PathLike],
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_dir=db_dir,
            db_name="ptb-xl-plus",
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.fs = None
        self.data_ext = "dat"  # median beats
        self.ann_ext = "atr"  # fiducial points
        self._feature_tables = None
        self._label_tables = None
        self._label_mappings = None
        self._ls_rec()

    def _ls_rec(self) -> None:
        """Find all records in the database directory
        and store them (path, metadata, etc.) in a dataframe.
        """
        records_file = list(self.db_dir.rglob("RECORDS"))
        if len(records_file) == 0:
            self._df_records = pd.DataFrame()
            self._all_records = []

            return
        self.db_dir = records_file[0].parent.resolve()

        # features: csv files, exclude the files in the "old" subdirectory
        feature_files = list((self.db_dir / "features").glob("*.csv"))
        self._feature_tables = {f.stem: pd.read_csv(f).dropna(how="all", axis=0) for f in feature_files}
        # turn the "ecg_id" column into string of 5 digits
        for table_name, table in self._feature_tables.items():
            if "ecg_id" in table.columns:
                table["ecg_id"] = table["ecg_id"].apply(lambda x: f"{x:05d}")

        # labels: csv files: mainly label descriptions and mappings (from 12sl/ptbxl to SNOMED)
        label_files = list((self.db_dir / "labels").glob("*.csv"))
        self._label_tables = {f.stem: pd.read_csv(f).dropna(how="all", axis=0) for f in label_files}
        mapping_files = list((self.db_dir / "labels" / "mapping").glob("*.csv"))
        # self._label_mappings = {f.stem: pd.read_csv(f) for f in mapping_files}
        for f in mapping_files:
            try:
                self._label_tables[f.stem] = pd.read_csv(f)
            except pd.errors.ParserError:
                file_content = [line.split(",") for line in f.read_text().splitlines()]
                common_len = max([max([idx for idx, item in enumerate(line) if item.strip()] + [0]) for line in file_content])
                file_content = [line[: common_len + 1] for line in file_content]
                self._label_tables[f.stem] = pd.DataFrame(file_content[1:], columns=file_content[0])
            self._label_tables[f.stem] = self._label_tables[f.stem].dropna(how="all", axis=0)
        # turn the "ecg_id" column into string of 5 digits
        for table_name, table in self._label_tables.items():
            if "ecg_id" in table.columns:
                table["ecg_id"] = table["ecg_id"].apply(lambda x: f"{x:05d}")

        # median beats: dat files, including "12sl" and "unig"
        self._df_records["12sl_path"] = get_record_list_recursive3(
            db_dir=self.db_dir / "median_beats" / "12sl",
            rec_patterns=f"[\\d]+_medians\\.{self.data_ext}",
            relative=False,
        )
        self._df_records["ecg_id"] = self._df_records["12sl_path"].apply(lambda x: int(Path(x).stem.split("_")[0]))
        df_unig = pd.DataFrame()
        df_unig["unig_path"] = get_record_list_recursive3(
            db_dir=self.db_dir / "median_beats" / "unig",
            rec_patterns=f"[\\d]+_medians\\.{self.data_ext}",
            relative=False,
        )
        df_unig["ecg_id"] = df_unig["unig_path"].apply(lambda x: int(Path(x).stem.split("_")[0]))
        # merge the two dataframes on "ecg_id" and keep empty cells
        self._df_records = self._df_records.merge(df_unig, on="ecg_id", how="outer")
        del df_unig

        # fiducial points: atr files
        df_fiducial = pd.DataFrame()
        df_fiducial["fiducial_path"] = get_record_list_recursive3(
            db_dir=self.db_dir / "fiducial_points" / "ecgdeli",
            rec_patterns=f"[\\d]+_points_[\\w_]+\\.{self.ann_ext}",
            relative=False,
        )
        df_fiducial["ecg_id"] = df_fiducial["fiducial_path"].apply(lambda x: int(Path(x).stem.split("_")[0]))
        fiducial_point_files = df_fiducial.groupby("ecg_id")["fiducial_path"].apply(list).to_dict()
        df_fiducial["fiducial_point_files"] = df_fiducial["ecg_id"].apply(lambda x: fiducial_point_files.get(x, []))
        # drop duplicates on "ecg_id"
        df_fiducial.drop_duplicates(subset="ecg_id", keep="first", inplace=True)
        # merge the two dataframes on "ecg_id" and keep empty cells
        self._df_records = self._df_records.merge(df_fiducial, on="ecg_id", how="outer")
        del df_fiducial

        # turn the "ecg_id" column into string of 5 digits
        self._df_records["ecg_id"] = self._df_records["ecg_id"].apply(lambda x: f"{x:05d}")
        self._df_records.set_index("ecg_id", inplace=True)
        self._all_records = self._df_records.index.tolist()

        # Fix potential bugs in the database
        self._fix_bugs()

    def load_data(self, rec: Union[str, int], source: Literal["12sl", "unig"] = "12sl") -> np.ndarray:
        """Load the data of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        source : {"12sl", "unig"}, default "12sl"
            The data source to load.

        Returns
        -------
        data : ndarray
            The data of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        if source == "12sl":
            path = self._df_records.loc[rec, "12sl_path"]
        elif source == "unig":
            path = self._df_records.loc[rec, "unig_path"]
        else:
            raise ValueError(f"unknown data source: {source}")
        if path is None:
            print(f"record {rec} not found in {source}")
            return None
        return wfdb.rdrecord(path).p_signal

    @add_docstring(load_data.__doc__)
    def load_median_beats(self, rec: Union[str, int], source: str = "12sl") -> np.ndarray:
        """alias of `load_data`."""
        return self.load_data(rec, source)

    def load_ann(self, rec: Union[str, int], source: Literal["12sl", "ptbxl"] = "12sl") -> Dict[str, Any]:
        """Load the annotation (diagnostic statements) of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        source : {"12sl", "ptbxl"}, default "12sl"
            The annotation source to load.

        Returns
        -------
        ann : dict
            The annotation of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        assert f"{source}_statements" in self._label_tables, f"source {source} not found in label tables"
        df = self._label_tables[f"{source}_statements"]
        if rec not in df["ecg_id"].values:
            return {}
        ann = df[df["ecg_id"] == rec].iloc[0].to_dict()
        ann.pop("ecg_id")
        ann = {key: literal_eval(val) for key, val in ann.items()}
        return ann

    def load_features(self, rec: Union[str, int], source: Literal["12sl", "unig", "ecgdeli"] = "12sl") -> Dict[str, float]:
        """Load the features of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        source : {"12sl", "unig", "ecgdeli"}, default "12sl"
            The feature source to load.

        Returns
        -------
        features : dict
            The features of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        assert f"{source}_features" in self._feature_tables, f"source {source} not found in feature tables"
        df = self._feature_tables[f"{source}_features"]
        if rec not in df["ecg_id"].values:
            return {}
        features = df[df["ecg_id"] == rec].iloc[0].to_dict()
        features.pop("ecg_id")
        return features

    def load_fiducial_points(
        self, rec: Union[str, int], leads: Optional[Union[str, List[str]]] = None
    ) -> Union[Dict[str, list], Dict[str, Dict[str, list]]]:
        """Load the fiducial points of a record.

        Parameters
        ----------
        rec : str or int
            The record name (ecg_id) or the index of the record.
        leads : str or list of str, optional
            The leads to load.
            If is None, load all leads.

        Returns
        -------
        fiducial_points : dict
            The fiducial points of the record.

        """
        if isinstance(rec, int):
            rec = self._all_records[rec]
        fiducial_point_files = self._df_records.loc[rec, "fiducial_point_files"]
        if not fiducial_point_files:
            return {}
        if leads is not None and isinstance(leads, str):
            leads = [leads]
        fiducial_points = {}
        for file in fiducial_point_files:
            lead = Path(file).stem.split("_")[-1]
            if leads is not None and lead not in leads:
                continue
            ann = wfdb.rdann(file, extension=self.ann_ext)
            fiducial_points[lead] = {
                "indices": ann.sample.tolist(),
                "labels": ann.aux_note,
            }
        return fiducial_points

    @property
    def database_info(self) -> DataBaseInfo:
        return _PTBXL_PLUS_INFO

    def _fix_bugs(self):
        """Fix bugs in the database.

        See https://github.com/MIT-LCP/wfdb-python/issues/528
        """
        flag_file = self.db_dir / "median_beats" / "12sl" / ".fixed"
        if flag_file.exists() or self.version > "1.0.1":
            return

        # fix the bug in the 12sl median beats
        with tqdm(
            self._df_records["12sl_path"].dropna(),
            total=len(self._df_records["12sl_path"].dropna()),
            desc="Fixing bugs in 12sl median beats header files",
            unit="record",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=(self.verbose < 1),
        ) as pbar:
            for path in pbar:
                header_file = Path(path).with_suffix(".hea")
                header_content = header_file.read_text()
                header_content = header_content.replace("ge_median_beats_wfdb/", "")
                header_file.write_text(header_content)

        flag_file.touch()
