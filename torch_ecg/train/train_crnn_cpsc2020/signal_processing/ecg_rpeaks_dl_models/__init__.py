"""
NOTE:
-----
.h5 files are gitignored, corresponding files can be downloaded at
https://opensz.oss-cn-beijing.aliyuncs.com/ICBEB2020/file/CPSC2019-opensource.zip

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] to add more
"""
import os
from typing import Union, Optional, Tuple

try:
    from keras.models import model_from_json, Model
except:
    from tensorflow.keras.models import model_from_json, Model
import torch
from torch import nn


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "load_model",
]


def load_model(name:str, **kwargs) -> Union[Model, Tuple[Model,...], nn.Module, Tuple[nn.Module,...]]:
    """ finished, checked,

    Parameters:
    -----------
    name: str,
        name of the model

    Returns:
    --------
    model, or sequence of models, either keras or pytorch
    """
    if name.lower() == "keras_ecg_seq_lab_net":
        models = _load_keras_ecg_seq_lab_net(**kwargs)
        return models
    elif name.lower() == "pytorch_ecg_seq_lab_net":
        raise NotImplementedError
    else:
        raise NotImplementedError


def _load_keras_ecg_seq_lab_net(which:str="both", **kwargs) -> Union[Tuple[Model,Model],Model]:
    """ finished, checked,

    load the CNN model and CRNN model from the entry 0416 of CPSC2019

    Parameters:
    -----------
    which: str, default "both",
        choice of model(s) to load,
        can be one of "both", "cnn", "crnn", case insensitive

    Returns:
    --------
    cnn_model, crnn_model (both or one): Model
    """
    _which = which.lower()
    if _which in ["both", "cnn"]:
        cnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.json")
        cnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.h5")
        cnn_model = model_from_json(open(cnn_config_path).read())
        cnn_model.load_weights(cnn_h5_path)
        cnn_model.trainable = False
        if _which == "cnn":
            return cnn_model
    if _which in ["both", "cnn"]:
        crnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.json")
        crnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.h5")
        crnn_model = model_from_json(open(crnn_config_path).read())
        crnn_model.load_weights(crnn_h5_path)
        crnn_model.trainable = False
        if _which == "crnn":
            return crnn_model
    return cnn_model, crnn_model


def _load_pytorch_ecg_seq_lab_net():
    """
    """
    raise NotImplementedError
