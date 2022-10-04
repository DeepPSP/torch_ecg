# -*- coding: utf-8 -*-
"""
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ...cfg import DEFAULTS  # noqa: F401
from ...utils import add_docstring
from ..base import PhysioNetDataBase, DataBaseInfo


__all__ = [
    "CINC2018",
]


_CINC2018_INFO = DataBaseInfo(
    title="""
    You Snooze You Win - The PhysioNet Computing in Cardiology Challenge 2018
    """,
    about="""
    1. includes 1,985 subjects, partitioned into balanced training (n = 994), and test sets (n = 989)
    2. signals include
        electrocardiogram (ECG),
        electroencephalography (EEG),
        electrooculography (EOG),
        electromyography (EMG),
        electrocardiology (EKG),
        oxygen saturation (SaO2)
    3. frequency of all signal channels is 200 Hz
    4. units of signals:
        mV for ECG, EEG, EOG, EMG, EKG
        percentage for SaO2
    5. six sleep stages were annotated in 30 second contiguous intervals:
        wakefulness,
        stage 1,
        stage 2,
        stage 3,
        rapid eye movement (REM),
        undefined
    6. annotated arousals were classified as either of the following:
        spontaneous arousals,
        respiratory effort related arousals (RERA),
        bruxisms,
        hypoventilations,
        hypopneas,
        apneas (central, obstructive and mixed),
        vocalizations,
        snores,
        periodic leg movements,
        Cheyne-Stokes breathing,
        partial airway obstructions
    """,
    usage=[
        "sleep stage",
        "sleep apnea",
    ],
    references=[
        "https://physionet.org/content/challenge-2018/1.0.0/",
    ],
    status="NOT Finished",
    doi=[
        "10.22489/CinC.2018.049",
        "10.13026/6phb-r450",
    ],
)


@add_docstring(_CINC2018_INFO.format_database_docstring())
class CINC2018(PhysioNetDataBase):
    """ """

    __name__ = "CINC2018"

    def __init__(
        self,
        db_dir: Optional[Union[str, Path]] = None,
        working_dir: Optional[str] = None,
        verbose: int = 2,
        **kwargs: Any
    ) -> None:
        """NOT finished, NOT checked,

        Parameters
        ----------
        db_dir: str or Path, optional,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
            log verbosity
        kwargs: auxilliary key word arguments

        """
        super().__init__(
            db_name="challenge-2018",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs
        )
        self.fs = None
        self.training_dir = self.db_dir / "training"
        self.test_dir = self.db_dir / "test"
        self.training_records = []
        self.test_records = []
        self._all_records = []

    def get_subject_id(self, rec: str) -> int:
        """
        Parameters
        ----------
        rec: str,
            name of the record

        Returns
        -------
        pid: int,
            the `subject_id` corr. to `rec`

        """
        head = "2018"
        mid = rec[2:4]
        tail = rec[-4:]
        pid = int(head + mid + tail)
        return pid

    def load_data(self) -> np.ndarray:
        """ """
        raise NotImplementedError

    def load_ann(self):
        """ """
        raise NotImplementedError

    def plot(self) -> None:
        """ """
        raise NotImplementedError

    @property
    def database_info(self) -> DataBaseInfo:
        return _CINC2018_INFO
