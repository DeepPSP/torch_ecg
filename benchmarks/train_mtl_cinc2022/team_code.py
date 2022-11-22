#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################

import time
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401
from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import PreprocManager

from cfg import TrainCfg, ModelCfg, OutcomeCfg, remove_extra_heads
from dataset import CinC2022Dataset
from inputs import (  # noqa: F401
    InputConfig,
    BaseInput,
    WaveformInput,
    SpectrogramInput,
    MelSpectrogramInput,
    MFCCInput,
    SpectralInput,
)  # noqa: F401
from models import (  # noqa: F401
    CRNN_CINC2022,
    SEQ_LAB_NET_CINC2022,
    UNET_CINC2022,
    Wav2Vec2_CINC2022,
    HFWav2Vec2_CINC2022,
    OutComeClassifier_CINC2022,
)  # noqa: F401
from trainer import (  # noqa: F401
    CINC2022Trainer,
    _MODEL_MAP,
    _set_task,
)  # noqa: F401

from helper_code import find_patient_files, get_locations


################################################################################
# NOTE: configurable options

USE_AUX_OUTCOME_MODEL = True  # True, False
MURMUR_UNKNOWN_AS_POSITIVE = True  # for OutcomeClassifier_CINC2022

TASK = "multi_task"  # "classification", "multi_task"

# choices of the models
TrainCfg[TASK].model_name = "crnn"  # "wav2vec", "crnn", "wav2vec2_hf"

# "tresnetS"  # "resnet_nature_comm", "tresnetF", etc.
TrainCfg[TASK].cnn_name = "resnet_nature_comm_bottle_neck_se"

# TrainCfg[TASK].rnn_name = "none"  # "none", "lstm"
# TrainCfg[TASK].attn_name = "se"  # "none", "se", "gc", "nl"
# TrainCfg[TASK].encoder_name = "wav2vec2_nano"
################################################################################


################################################################################
# NOTE: constants

FS = 4000
MURMUR_POSITIVE_CLASS = "Present"
MURMUR_UNKNOWN_CLASS = "Unknown"
OUTCOME_POSITIVE_CLASS = "Abnormal"

_ModelFilename = "final_model_main.pth.tar"
_ModelFilename_outcome = "final_model_outcome.pkl"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


CinC2022Dataset.__DEBUG__ = False

CRNN_CINC2022.__DEBUG__ = False
SEQ_LAB_NET_CINC2022.__DEBUG__ = False
UNET_CINC2022.__DEBUG__ = False
Wav2Vec2_CINC2022.__DEBUG__ = False
HFWav2Vec2_CINC2022.__DEBUG__ = False

CINC2022Trainer.__DEBUG__ = False
################################################################################


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder: str, model_folder: str, verbose: int) -> None:
    """

    Parameters
    ----------
    data_folder: str,
        path to the folder containing the training data
    model_folder: str,
        path to the folder to save the trained model
    verbose: int,
        verbosity level

    """

    print("\n" + "*" * 100)
    msg = "   CinC2022 challenge training entry starts   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")

    # Find data files.
    if verbose >= 1:
        print("Finding data files...")

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception("No data was provided.")

    # Create a folder for the model if it does not already exist.
    # os.makedirs(model_folder, exist_ok=True)
    Path(model_folder).mkdir(parents=True, exist_ok=True)

    classes = ["Present", "Unknown", "Absent"]
    num_classes = len(classes)

    ###############################################################################
    # Train the model.
    ###############################################################################
    # general configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = Path(data_folder).resolve().absolute()
    train_config.model_dir = Path(model_folder).resolve().absolute()
    train_config.debug = False

    if train_config.get("entry_test_flag", False):
        # to test in the file test_docker.py or in test_local.py
        train_config.n_epochs = 1
        train_config.batch_size = 4
        train_config.log_step = 4
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = 20
    else:
        train_config.n_epochs = 60
        train_config.freeze_backbone_at = 40
        train_config.batch_size = 32  # 16G (Tesla T4)
        train_config.log_step = 50
        # train_config.max_lr = 1.5e-3
        train_config.early_stopping.patience = int(train_config.n_epochs * 0.5)

    train_config.final_model_name = _ModelFilename
    train_config[TASK].final_model_name = _ModelFilename
    _set_task(TASK, train_config)

    model_config = deepcopy(ModelCfg[TASK])

    # adjust model choices if needed
    model_name = model_config.model_name = train_config[TASK].model_name
    if "cnn" in model_config[model_name]:
        model_config[model_name].cnn.name = train_config[TASK].cnn_name
    if "rnn" in model_config[model_name]:
        model_config[model_name].rnn.name = train_config[TASK].rnn_name
    if "attn" in model_config[model_name]:
        model_config[model_name].attn.name = train_config[TASK].attn_name
    # if "encoder" in model_config[model_name]:
    #     model_config[model_name].encoder.name = train_config[TASK].encoder_name

    # NOTE: choose whether to remove some heads
    if USE_AUX_OUTCOME_MODEL:
        # from torch_ecg.utils.misc import dict_to_str
        # print(dict_to_str(train_config))
        remove_extra_heads(
            train_config=train_config[TASK],
            model_config=model_config,
            heads=["outcome"],  # "outcome", "segmentation", None
        )

    start_time = time.time()

    # ds_train = CinC2022Dataset(train_config, TASK, training=True, lazy=False)
    # ds_val = CinC2022Dataset(train_config, TASK, training=False, lazy=False)

    model_cls = _MODEL_MAP[model_config.model_name]
    model_cls.__DEBUG__ = False

    model = model_cls(config=model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=DEVICE)

    if isinstance(model, DP):
        print(model.module.config)
    else:
        print(model.config)

    trainer = CINC2022Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )

    best_state_dict = trainer.train()  # including saving model

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()

    if USE_AUX_OUTCOME_MODEL:
        # NOTE: train an auxilliary outcome model
        OutcomeCfg.db_dir = Path(data_folder).resolve().absolute()
        OutcomeCfg.model_dir = Path(model_folder).resolve().absolute()
        outcome_clf = OutComeClassifier_CINC2022(OutcomeCfg)
        outcome_clf.search(model_name="rf", cv=None)
        outcome_clf.save_best_model(model_name=_ModelFilename_outcome)

    if verbose >= 1:
        print("Done.")

    print("\n" + "*" * 100)
    msg = "   CinC2022 challenge training entry ends   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n\n")


# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def load_challenge_model(
    model_folder: str, verbose: int
) -> Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]]:
    """

    Parameters
    ----------
    model_folder: str,
        path to the folder containing the trained model
    verbose: int,
        verbosity level

    Returns
    -------
    dict, with items:
        - main_model: torch.nn.Module,
            the loaded model, for murmur predictions,
            or for both murmur and outcome predictions
        - train_cfg: CFG,
            the training configuration,
            including the list of classes (the ordering is important),
            and the preprocessing configurations
        - outcome_model: BaseEstimator, optional,
            the loaded model, for outcome predictions

    """
    print("\n" + "*" * 100)
    msg = "   loading CinC2022 challenge model   ".center(100, "#")
    print(msg)
    # murmur model
    model_cls = _MODEL_MAP[TrainCfg[TASK].model_name]
    main_model, train_cfg = model_cls.from_checkpoint(
        path=Path(model_folder) / _ModelFilename,
        device=DEVICE,
    )
    main_model.eval()
    if USE_AUX_OUTCOME_MODEL:
        # outcome model
        outcome_model = OutComeClassifier_CINC2022.from_file(
            Path(model_folder) / _ModelFilename_outcome
        )
    else:
        outcome_model = None
    msg = "   CinC2022 challenge model loaded   ".center(100, "#")
    print(msg)
    print("*" * 100 + "\n")
    model = dict(main_model=main_model, train_cfg=train_cfg)
    if USE_AUX_OUTCOME_MODEL:
        model["outcome_model"] = outcome_model
    return model


# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments and outputs of this function.
def run_challenge_model(
    model: Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]],
    data: str,
    recordings: Union[List[np.ndarray], Tuple[List[np.ndarray], List[int]]],
    verbose: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    model: Dict[str, Union[CFG, torch.nn.Module, BaseEstimator]],
        the trained main model (key "main_model"),
        along with the training configuration (key "train_cfg");
        and (optionally) the trained outcome model (key "outcome_model")
    data: str,
        patient metadata file data, read from a text file
    recordings: List[np.ndarray],
        list of recordings, each recording is a 1D numpy array
    verbose: int,
        verbosity level

    Returns
    -------
    classes: list of str,
        list of class names
    labels: np.ndarray,
        binary prediction
    probabilities: np.ndarray,
        probability prediction

    NOTE
    ----
    the `recordings` are read by `scipy.io.wavfile.read`, with the following possible data types:
    =====================  ===========  ===========  =============
        WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============
    Note that 8-bit PCM is unsigned.

    """

    locations = get_locations(data)

    if USE_AUX_OUTCOME_MODEL:
        outcome_model = model["outcome_model"]
    else:
        outcome_model = None
    main_model = model["main_model"]
    main_model.to(device=DEVICE)
    train_cfg = model["train_cfg"]
    ppm_config = CFG(random=False)
    ppm_config.update(deepcopy(train_cfg[TASK]))
    ppm = PreprocManager.from_config(ppm_config)

    if not isinstance(recordings[0], np.ndarray):
        recordings, frequencies = recordings
        num_recordings = len(recordings)
    else:
        num_recordings = len(recordings)
        frequencies = list(repeat(FS, num_recordings))

    murmur_classes = train_cfg[TASK].classes
    murmur_positive_class_id = murmur_classes.index(MURMUR_POSITIVE_CLASS)
    murmur_unknown_class_id = murmur_classes.index(MURMUR_UNKNOWN_CLASS)

    if USE_AUX_OUTCOME_MODEL:
        outcome_classes = outcome_model.config.classes
    else:
        outcome_classes = train_cfg[TASK].outcomes

    outcome_positive_class_id = outcome_classes.index(OUTCOME_POSITIVE_CLASS)

    murmur_probabilities, murmur_labels, murmur_cls_labels, murmur_forward_outputs = (
        [],
        [],
        [],
        [],
    )
    (
        outcome_probabilities,
        outcome_labels,
        outcome_cls_labels,
        outcome_forward_outputs,
    ) = ([], [], [], [])
    # features = []
    # if BaseCfg.merge_rule.lower() == "avg":
    #     pooler = torch.nn.AdaptiveAvgPool1d((1,))
    # elif BaseCfg.merge_rule.lower() == "max":
    #     pooler = torch.nn.AdaptiveMaxPool1d((1,))

    murmur_pred_dict = dict()  # potentialy used for OutComeClassifier_CINC2022

    for loc, rec, fs in zip(locations, recordings, frequencies):
        rec = _to_dtype(rec, DTYPE)
        rec, _ = ppm(rec, fs)
        for _ in range(3 - rec.ndim):
            rec = rec[np.newaxis, :]
        model_output = main_model.inference(rec.copy().astype(DTYPE))
        murmur_probabilities.append(model_output.murmur_output.prob)
        murmur_labels.append(model_output.murmur_output.bin_pred)
        murmur_cls_labels.append(model_output.murmur_output.pred)
        murmur_forward_outputs.append(model_output.murmur_output.forward_output)
        # rec = torch.from_numpy(rec.copy().astype(DTYPE)).to(device=DEVICE)
        # # rec of shape (1, 1, n_samples)
        # features.append(main_model.extract_features(rec))  # shape (1, n_features, n_samples)
        if MURMUR_UNKNOWN_AS_POSITIVE:
            murmur_pred_dict[loc] = int(
                model_output.murmur_output.pred.item()
                in [murmur_positive_class_id, murmur_unknown_class_id]
            )
        else:
            murmur_pred_dict[loc] = int(
                model_output.murmur_output.pred.item() == murmur_positive_class_id
            )

        if not USE_AUX_OUTCOME_MODEL:
            outcome_probabilities.append(model_output.outcome_output.prob)
            outcome_labels.append(model_output.outcome_output.bin_pred)
            outcome_cls_labels.append(model_output.outcome_output.pred)
            outcome_forward_outputs.append(model_output.outcome_output.forward_output)

    # features = torch.cat(features, dim=-1)  # shape (1, n_features, n_samples)
    # features = pooler(features).squeeze(dim=-1)  # shape (1, n_features)
    # forward_output = main_model.clf(features)  # shape (1, n_classes)
    # probabilities = main_model.softmax(forward_output)
    # labels = (probabilities == probabilities.max(dim=-1, keepdim=True).values).to(int)
    # probabilities = probabilities.squeeze(dim=0).cpu().detach().numpy()
    # labels = labels.squeeze(dim=0).cpu().detach().numpy()

    # get final prediction for murmurs:
    # strategy:
    # 1. (at least) one positive -> positive
    # 2. no positive, (at least) one unknown -> unknown
    # 3. all negative -> negative
    murmur_probabilities = np.concatenate(murmur_probabilities, axis=0)
    murmur_labels = np.concatenate(murmur_labels, axis=0)
    murmur_cls_labels = np.concatenate(murmur_cls_labels, axis=0)
    murmur_forward_outputs = np.concatenate(murmur_forward_outputs, axis=0)

    murmur_positive_indices = np.where(murmur_cls_labels == murmur_positive_class_id)[0]
    murmur_unknown_indices = np.where(murmur_cls_labels == murmur_unknown_class_id)[0]

    if len(murmur_positive_indices) > 0:
        # if exists at least one positive recording,
        # then the subject is diagnosed with the positive class
        murmur_probabilities = murmur_probabilities[murmur_positive_indices, ...].mean(
            axis=0
        )
        murmur_labels = murmur_labels[murmur_positive_indices[0]]
    elif len(murmur_unknown_indices) > 0:
        # no positive recording, but at least one unknown recording
        murmur_probabilities = murmur_probabilities[murmur_unknown_indices, ...].mean(
            axis=0
        )
        murmur_labels = murmur_labels[murmur_unknown_indices[0]]
    else:
        # no positive or unknown recording,
        # only negative class recordings
        murmur_probabilities = murmur_probabilities.mean(axis=0)
        murmur_labels = murmur_labels[0]

    if not USE_AUX_OUTCOME_MODEL:
        # get final prediction for outcomes
        # strategy:
        # 1. (at least) one positive -> positive
        # 2. all negative -> negative
        # TODO:
        # 1. consider using patient's metadata (`data`) to determine the outcome class,
        #    since at least `Preganancy status` has high correlation with outcome
        outcome_probabilities = np.concatenate(outcome_probabilities, axis=0)
        outcome_labels = np.concatenate(outcome_labels, axis=0)
        outcome_cls_labels = np.concatenate(outcome_cls_labels, axis=0)
        outcome_forward_outputs = np.concatenate(outcome_forward_outputs, axis=0)
        outcome_positive_indices = np.where(
            outcome_cls_labels == outcome_positive_class_id
        )[0]

        if len(outcome_positive_indices) > 0:
            # if exists at least one positive recording,
            # then the subject is diagnosed with the positive class
            outcome_probabilities = outcome_probabilities[
                outcome_positive_indices, ...
            ].mean(axis=0)
            outcome_labels = outcome_labels[outcome_positive_indices[0]]
        else:
            # no positive recording, only negative class recordings
            outcome_probabilities = outcome_probabilities.mean(axis=0)
            outcome_labels = outcome_labels[0]
    else:
        outcome_output = outcome_model.inference(
            data,
            murmur_pred_dict,
        )
        outcome_probabilities = outcome_output.prob[0]
        outcome_labels = outcome_output.bin_pred[0]
        outcome_cls_labels = outcome_output.pred
        outcome_forward_outputs = None

    # Concatenate classes, labels, and probabilities.
    classes = murmur_classes + outcome_classes
    labels = np.concatenate((murmur_labels, outcome_labels))
    probabilities = np.concatenate((murmur_probabilities, outcome_probabilities))

    return classes, labels, probabilities


def _to_dtype(data: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """ """
    if data.dtype == dtype:
        return data
    if data.dtype in (np.int8, np.uint8, np.int16, np.int32, np.int64):
        data = data.astype(dtype) / (np.iinfo(data.dtype).max + 1)
    return data
