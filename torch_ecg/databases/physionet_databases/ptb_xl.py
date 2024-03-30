import os
from ast import literal_eval
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...cfg import DEFAULTS
from ...utils.misc import add_docstring
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
        "Re-digitization of ECG images",
        "Classification of ECG images",
    ],
    note="""
    """,
    issues="""
    """,
    references=[
        "https://physionet.org/content/ptb-xl/",
    ],
    doi=[
        "https://doi.org/10.13026/kfzx-aw45",
        "https://doi.org/10.1038/s41597-020-0495-6",
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
    kwargs : dict, optional
        Auxilliary key word arguments.

    """

    __name__ = "PTBXLReader"
    __metadata_file__ = "ptbxl_database.csv"
    __scp_statements_file__ = "scp_statements.csv"
    __100Hz_dir__ = "records100"
    __500Hz_dir__ = "records500"

    def __init__(
        self,
        db_dir: Union[str, bytes, os.PathLike],
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(db_name="ptb-xl", db_dir=db_dir, working_dir=working_dir, verbose=verbose, **kwargs)
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
            self._df_images = pd.DataFrame()
            self._all_records = []
            self._all_subjects = []
            self._all_images = []
            return
        self.db_dir = metadata_file.parent.resolve()
        assert (self.db_dir / self.__scp_statements_file__).exists(), f"scp_statements file not found in {self.db_dir}"

        # read metadata file and scp_statements file
        self._df_metadata = pd.read_csv(self.db_dir / self.__metadata_file__)
        self._df_metadata["ecg_id"] = self._df_metadata["ecg_id"].apply(lambda x: f"{x:05d}")
        self._df_metadata.set_index("ecg_id", inplace=True)
        self._df_metadata["patient_id"] = self._df_metadata["patient_id"].astype(int)
        self._df_scp_statements = pd.read_csv(self.db_dir / self.__scp_statements_file__, index_col=0)

        if self._subsample is not None:
            size = min(
                len(self._df_metadata),
                max(1, int(round(self._subsample * len(self._df_metadata)))),
            )
            self.logger.debug(f"subsample `{size}` records from `{len(self._df_metadata)}`")
            self._df_records = self._df_metadata.sample(n=size, random_state=DEFAULTS.SEED, replace=False)
        else:
            self._df_records = self._df_metadata.copy()

        if self.fs == 100:
            self._df_records["path"] = self._df_records["filename_lr"].apply(lambda x: self.db_dir / x)
        else:
            self._df_records["path"] = self._df_records["filename_hr"].apply(lambda x: self.db_dir / x)

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
