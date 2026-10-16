""" """

from copy import deepcopy
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import scipy.io.wavfile as sio_wav
import torch

# local-env compatibility shim: torch_audiomentations 0.11.0 predates
# torchaudio 2.x (set_audio_backend removed); CI installs the current
# release where this is not needed
import torchaudio
from torch.utils.data import DataLoader

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **kw: None

from benchmarks.train_mtl_cinc2022.cfg import BaseCfg, ModelCfg, TrainCfg
from benchmarks.train_mtl_cinc2022.data_reader import CINC2022Reader
from benchmarks.train_mtl_cinc2022.dataset import CinC2022Dataset
from benchmarks.train_mtl_cinc2022.models import CRNN_CINC2022
from benchmarks.train_mtl_cinc2022.outputs import CINC2022Outputs
from benchmarks.train_mtl_cinc2022.train import _MODEL_MAP, _set_task
from benchmarks.train_mtl_cinc2022.trainer import CINC2022Trainer
from benchmarks.train_mtl_cinc2022.utils.scoring_metrics import compute_challenge_metrics
from torch_ecg.components.outputs import ClassificationOutput
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

try:
    import librosa
except ImportError:
    librosa = None

_CWD = Path(__file__).absolute().parents[1] / "tmp" / "test_mtl_cinc2022_pipeline"
_CWD.mkdir(parents=True, exist_ok=True)
_DB_DIR = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2022"
_DB_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK = "classification"  # "multi_task"


def torchaudio_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
    try:
        data, new_fs = torchaudio.load(file, normalize=True)
    except Exception:
        data, new_fs = torchaudio.load(file, normalization=True)
    return data, new_fs


def librosa_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
    data, _ = librosa.load(file, sr=fs, mono=False)
    return torch.from_numpy(data.reshape((-1, data.shape[-1]))), fs


def scipy_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
    new_fs, data = sio_wav.read(file)
    data = (data / (2**15)).astype(BaseCfg.np_dtype)[np.newaxis, :]
    return torch.from_numpy(data), new_fs


def wfdb_load(file: str, fs: int) -> Tuple[torch.Tensor, int]:
    import wfdb

    data = wfdb.rdrecord(file.replace(".wav", "")).p_signal.T
    return torch.from_numpy(data.astype(BaseCfg.np_dtype)), fs


@pytest.fixture(scope="session")
def db_dir():
    """ """
    try:
        dr = CINC2022Reader(db_dir)
        dr.download(compressed=True)
    except Exception as err:
        pytest.skip(f"failed to download the Circulatory Sounds database: {err}")
    return _DB_DIR


@pytest.mark.db
def test_data_reader(db_dir):
    with pytest.warns(RuntimeWarning, match="audio backend xxx is not available"):
        dr = CINC2022Reader(db_dir, audio_backend="xxx")
    with pytest.warns(
        RuntimeWarning,
        match="loading result using wfdb is inconsistent with other backends",
    ):
        dr = CINC2022Reader(db_dir, audio_backend="wfdb")

    dr = CINC2022Reader(db_dir, audio_backend="librosa")

    stats_file = dr.db_dir / "training_data.csv"
    if stats_file.is_file():
        stats_file.unlink()
    dr._df_stats = pd.DataFrame()
    dr._df_stats_records = pd.DataFrame()
    assert not dr.df_stats.empty
    assert not dr.df_stats_records.empty

    assert dr.get_fs(0) == dr.fs

    # TODO: test
    # `reader.load_segmentation`, `reader.load_meta_data`, `reader._load_preprocessed_data`


@pytest.mark.db
def test_dataset(db_dir):
    """ """
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = db_dir

    ds_train = CinC2022Dataset(ds_config, TASK, training=True, lazy=True)
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)

    ds_train._load_all_data()
    ds_val._load_all_data()

    # TODO: test more tasks

    print("dataset test passed")


@pytest.mark.db
def test_models(db_dir):
    """ """
    model = CRNN_CINC2022(ModelCfg[TASK])
    model.to(DEVICE)
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = db_dir
    ds_val = CinC2022Dataset(ds_config, TASK, training=False, lazy=True)
    ds_val._load_all_data()
    dl = DataLoader(
        dataset=ds_val,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    for idx, input_tensors in enumerate(dl):
        waveforms = input_tensors.pop("waveforms").to(DEVICE)
        print(model.inference(waveforms))
        if idx >= 2:
            break


def test_challenge_metrics():
    """ """
    outputs = [
        CINC2022Outputs(
            murmur_output=ClassificationOutput(
                classes=["Present", "Unknown", "Absent"],
                prob=np.array([[0.75, 0.15, 0.1]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0, 0]]),
            ),
            outcome_output=ClassificationOutput(
                classes=["Abnormal", "Normal"],
                prob=np.array([[0.6, 0.4]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0]]),
            ),
            segmentation_output=None,
        ),
        CINC2022Outputs(
            murmur_output=ClassificationOutput(
                classes=["Present", "Unknown", "Absent"],
                prob=np.array([[0.3443752, 0.32366553, 0.33195925]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0, 0]]),
            ),
            outcome_output=ClassificationOutput(
                classes=["Abnormal", "Normal"],
                prob=np.array([[0.5230, 0.0202]]),
                pred=np.array([0]),
                bin_pred=np.array([[1, 0]]),
            ),
            segmentation_output=None,
        ),
    ]
    labels = [
        {
            "murmur": np.array([[0.0, 0.0, 1.0]]),
            "outcome": np.array([0]),
        },
        {
            "murmur": np.array([[0.0, 1.0, 0.0]]),
            "outcome": np.array([1]),
        },
    ]

    compute_challenge_metrics(labels, outputs)

    print("challenge metrics test passed")


@pytest.mark.db
def test_trainer(db_dir):
    """ """
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = db_dir
    train_config.debug = True

    train_config.n_epochs = 1
    train_config.batch_size = 2

    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    assert train_config[TASK].model_name == "crnn"
    model_name = model_config.model_name = train_config[TASK].model_name
    model_config[model_name].cnn_name = train_config[TASK].cnn_name
    model_config[model_name].rnn_name = train_config[TASK].rnn_name
    model_config[model_name].attn_name = train_config[TASK].attn_name

    model_cls = _MODEL_MAP[model_config.model_name]
    model_cls.__DEBUG__ = False

    model = model_cls(config=model_config)
    model.to(device=DEVICE)

    trainer = CINC2022Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )

    best_state_dict = trainer.train()
