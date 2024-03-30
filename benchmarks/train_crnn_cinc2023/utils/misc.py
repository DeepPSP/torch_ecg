"""
Miscellaneous functions.
"""

import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import scipy as sp
import wfdb
import yaml
from cfg import BaseCfg
from helper_code import get_end_time, load_text_file
from tqdm.auto import tqdm

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils.misc import get_record_list_recursive3

__all__ = [
    "load_challenge_eeg_data",
    "find_eeg_recording_files",
    "load_challenge_metadata",
    "load_unofficial_phase_metadata",
    "func_indicator",
    "get_outcome_from_cpc",
    "predict_proba_ordered",
    "url_is_reachable",
    "get_leaderboard",
    "load_submission_log",
]


def load_challenge_eeg_data(
    data_folder: str,
    patient_id: str,
    hour_limit: Optional[int] = None,
    backend: str = "scipy",
) -> List[Tuple[np.ndarray, int, List[str]]]:
    """Load challenge EEG data given the data folder and patient ID.

    Adapted from the ``load_challenge_data`` function of the unofficial phase.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.
    hour_limit : int, optional
        The hour limit of the recordings to load.
        None for no limit, by default None.
    backend : {"scipy", "wfdb"}
        The backend to use, by default "scipy".

    Returns
    -------
    List[Tuple[numpy.ndarray, int, List[str]]]
        A list of tuples, each tuple contains
        the recording data, sampling frequency, and channel names.

    """
    patient_folder = Path(data_folder) / patient_id
    # Load recordings.
    recording_files = find_eeg_recording_files(data_folder, patient_id, hour_limit)
    recordings = list()
    with tqdm(recording_files, desc=f"Loading {patient_id} recordings", mininterval=1) as pbar:
        for recording_location in pbar:
            if os.path.exists(recording_location + ".hea"):
                recording_data, channels, sampling_frequency = load_recording_data(recording_location, backend=backend)
                # utility_frequency = get_utility_frequency(recording_location + ".hea")
                recordings.append((recording_data, int(sampling_frequency), channels))
    return recordings


def find_eeg_recording_files(data_folder: str, patient_id: str, hour_limit: Optional[int] = None) -> List[str]:
    """Find the EEG recording files.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.
    hour_limit : int, optional
        The hour limit of the recordings to load.
        None for no limit, by default None

    Returns
    -------
    List[str]
        Absolute paths of the EEG recording files, without file extension.

    """
    patient_folder = Path(data_folder) / patient_id
    # resolve to avoid comparison fails if `data_folder` is a symlink
    patient_folder = patient_folder.expanduser().resolve()
    recording_files = get_record_list_recursive3(patient_folder, f"{BaseCfg.recording_pattern}\\.hea", relative=False)
    recording_files = [fp for fp in recording_files if fp.endswith("EEG") and Path(fp).parent == patient_folder]
    if hour_limit is not None:
        end_hours = [
            # get_end_time returns a tuple of (hour, minute, second)
            get_end_time(Path(fp + ".hea").read_text())[0]
            for fp in recording_files
        ]
        recording_files = [fp for fp, end_hour in zip(recording_files, end_hours) if end_hour < hour_limit]
    return recording_files


def load_challenge_metadata(data_folder: str, patient_id: str) -> str:
    """Load the patient metadata.

    Adapted from the ``load_challenge_data`` function of the unofficial phase.
    Now deprecated by the ``load_challenge_data`` function of the official phase.

    Parameters
    ----------
    data_folder : str
        The data folder.
    patient_id : str
        The patient ID.

    Returns
    -------
    str
        The patient metadata.

    """
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + ".txt")

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)

    return patient_metadata


def load_unofficial_phase_metadata() -> pd.DataFrame:
    """Load the unofficial phase metadata."""
    path = Path(__file__).parent / "unofficial_phase_metadata.csv.gz"
    df = pd.read_csv(path, index_col=0)
    df["subject"] = df["subject"].apply(lambda x: f"{x:04d}")
    df["start_sec"] = df["time"].apply(lambda x: 60 * int(x.split(":")[1]))
    df["end_sec"] = df["start_sec"] + 60 * 5
    return df


def func_indicator(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("\n" + "-" * 100)
            print(f"  Start {name}  ".center(100, "-"))
            print("-" * 100 + "\n")
            func(*args, **kwargs)
            print("\n" + "-" * 100)
            print(f"  End {name}  ".center(100, "-"))
            print("-" * 100 + "\n")

        return wrapper

    return decorator


def load_recording_data(
    record_name: str, check_values: bool = False, backend: str = "scipy"
) -> Tuple[np.ndarray, List[str], float]:
    """Load a recording, including the data, channel names, and sampling frequency.

    Modified from `helper_code.load_recording_data`, ref:
    https://github.com/physionetchallenges/python-example-2023/issues/8

    Parameters
    ----------
    record_name : str
        The record name.
    check_values : bool, optional
        Whether to check the values, by default False.
    backend : {"scipy", "wfdb"}
        The backend to use, by default "scipy".

    Returns
    -------
    rescaled_data : numpy.ndarray
        The loaded signal data.
    channels : List[str]
        The channel names.
    sampling_frequency : float
        The sampling frequency.

    .. note::

        "ValueError: Mat 4 mopt wrong format, byteswapping problem?" encountered
        when using `scipy.io.loadmat` to load the record "0430_086_098_EEG".

    """
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext == "":
        header_file = record_name + ".hea"
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError("{} recording not found.".format(record_name))

    with open(header_file, "r") as f:
        header = [line_.strip() for line_ in f.readlines() if line_.strip()]

    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    offsets = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, line_ in enumerate(header):
        arrs = [arr.strip() for arr in line_.split(" ")]
        # Parse the record line.
        if i == 0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        elif not line_.startswith("#") or len(line_.strip()) == 0:
            signal_file = arrs[0]
            gain = float(arrs[2].split("/")[0])
            offset = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            offsets.append(offset)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format allows for multiple signal files, but, for
    # simplicity, we have not done that here.
    num_signal_files = len(set(signal_files))
    if num_signal_files != 1:
        raise NotImplementedError(
            "The header file {}".format(header_file)
            + " references {} signal files; one signal file expected.".format(num_signal_files)
        )

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    if backend == "wfdb":
        signal_file_ = str(Path(signal_file).with_suffix(""))
        data = wfdb.rdrecord(signal_file_, physical=False, return_res=16).d_signal.T
    elif backend == "scipy":
        data = np.asarray(sp.io.loadmat(signal_file)["val"])
    else:
        raise ValueError(f"backend must be one of ['wfdb', 'scipy'], got {backend}.")

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data) != (num_channels, num_samples):
        raise ValueError("The header file {}".format(header_file) + " is inconsistent with the dimensions of the signal file.")

    # Check that the initial value and checksums in the signal file are consistent with the initial value and checksums in the
    # header file.
    if check_values:
        for i in range(num_channels):
            if data[i, 0] != initial_values[i]:
                raise ValueError(
                    "The initial value in header file {}".format(header_file)
                    + " is inconsistent with the initial value for channel {} in the signal data".format(channels[i])
                )
            if np.sum(np.asarray(data[i, :], dtype=np.int16)) != checksums[i]:
                raise ValueError(
                    "The checksum in header file {}".format(header_file)
                    + " is inconsistent with the checksum value for channel {} in the signal data".format(channels[i])
                )

    # Convert the signal data to float32 (original data is int16)
    # to avoid integer overflow when subtracting the offset.
    data = data.astype(dtype=DEFAULTS.DTYPE.NP)
    # Rescale the signal data using the gains and offsets.
    rescaled_data = np.zeros(np.shape(data), dtype=np.float32)
    for i in range(num_channels):
        rescaled_data[i, :] = (data[i, :] - offsets[i]) / gains[i]

    return rescaled_data, channels, sampling_frequency


def get_outcome_from_cpc(cpc_value: Union[int, str, Iterable[Union[int, str]]]) -> Union[str, List[str]]:
    """Get the outcome from the CPC value.

    Parameters
    ----------
    cpc_value : int or str or Iterable[int] or Iterable[str]
        The CPC value.

    Returns
    -------
    outcome : str or List[str]
        The outcome.

    """
    if isinstance(cpc_value, (list, tuple, np.ndarray)):
        return [get_outcome_from_cpc(v) for v in cpc_value]
    # convert numpy type to python type
    if isinstance(cpc_value, np.generic):
        cpc_value = cpc_value.item()
    # convert dtype of cpc_value to str
    if not isinstance(cpc_value, str):
        assert isinstance(cpc_value, (int, float))
        cpc_value = str(int(cpc_value))
    outcome = BaseCfg.cpc2outcome_map[cpc_value]
    return outcome


def predict_proba_ordered(probs: np.ndarray, classes_: np.ndarray, all_classes: np.ndarray) -> np.ndarray:
    """Workaround for the problem that one can not set explicitly
    the list of classes for models in sklearn.

    Modified from https://stackoverflow.com/a/32191708

    Parameters
    ----------
    probs : numpy.ndarray
        Array of probabilities, output of `predict_proba` method of sklearn models.
    classes_ : numpy.ndarray
        Array of classes, output of `classes_` attribute of sklearn models.
    all_classes : numpy.ndarray
        All possible classes (superset of `classes_`).

    Returns
    -------
    numpy.ndarray
        Array of probabilities, ordered according to all_classes.

    """
    proba_ordered = np.zeros((probs.shape[0], all_classes.size), dtype=float)
    sorter = np.argsort(all_classes)  # http://stackoverflow.com/a/32191125/395857
    idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
    proba_ordered[:, idx] = probs
    return proba_ordered


def url_is_reachable(url: str) -> bool:
    """Check if a URL is reachable.

    Parameters
    ----------
    url : str
        The URL.

    Returns
    -------
    bool
        Whether the URL is reachable.

    """
    try:
        r = requests.head(url, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_leaderboard(by_team: Union[bool, str] = False, sort_by: int = 72) -> pd.DataFrame:
    """Get the leaderboard of the official phase.

    Parameters
    ----------
    by_team : bool or str, default False
        If True, sort by team and keep only the best score of each team;
        if False, return the original leaderboard;
        if str and is a team name, return the scores of the team.
    sort_by : {12, 24, 48, 72}, default 72
        Sort by the score evaluated on the specified hour limit.

    Returns
    -------
    df_leaderboard : pandas.DataFrame
        The leaderboard.

    """
    url = (
        "https://docs.google.com/spreadsheets/u/0/d/e/"
        "2PACX-1vTa94VmPIbywGJEBYjNkzJiGZuPLaajzPIZpoxsi12_"
        "X5DF66ccUFB6Qi3U41UEpVu2q1rzTF7nlSpY/pubhtml"
    )
    if not url_is_reachable(url):
        raise ConnectionError("The leaderboard is unreachable.")
    df_leaderboard = pd.read_html(url, index_col=0)[0].dropna()
    df_leaderboard = pd.DataFrame(
        df_leaderboard.values[1:, 1:],
        columns=df_leaderboard.values[0, 1:],
        index=df_leaderboard.values[1:, 0],
    )
    if sort_by not in [12, 24, 48, 72]:
        raise ValueError(f"sort_by must be one of [12, 24, 48, 72], got {sort_by}.")
    col = f"Challenge score for {int(sort_by)} hours"
    if sort_by != 72:
        df_leaderboard = df_leaderboard.sort_values(by=col, ascending=False)
        # reset index (Rank)
        df_leaderboard.index = np.arange(1, df_leaderboard.shape[0] + 1)
        # set the rows with the same score to the same rank
        df_leaderboard.index = df_leaderboard.apply(
            lambda row: df_leaderboard.index[df_leaderboard[col] == row[col]][0],
            axis=1,
        )
    if isinstance(by_team, bool):
        if by_team:
            df_leaderboard = df_leaderboard.drop_duplicates(subset="Team", keep="first")
            df_leaderboard.index = np.arange(1, df_leaderboard.shape[0] + 1)
            # set the rows with the same score to the same rank
            df_leaderboard.index = df_leaderboard.apply(
                lambda row: df_leaderboard.index[df_leaderboard[col] == row[col]][0],
                axis=1,
            )
    elif isinstance(by_team, str):
        if by_team not in df_leaderboard["Team"].values:
            raise ValueError(f"Team {by_team} not found.")
        df_leaderboard = df_leaderboard[df_leaderboard["Team"] == by_team]
    else:
        raise TypeError(f"by_team must be bool or str, got {type(by_team)}.")
    df_leaderboard.index.name = "Rank"
    return df_leaderboard


def load_submission_log() -> pd.DataFrame:
    """Load the submission log.

    Returns
    -------
    df_sub_log : pandas.DataFrame
        The submission log,
        sorted by challenge score in descending order.

    """
    path = Path(__file__).parents[1] / "submissions"
    df_sub_log = pd.DataFrame.from_dict(yaml.safe_load(path.read_text())["Official Phase"], orient="index").sort_values(
        "score", ascending=False
    )
    return df_sub_log
