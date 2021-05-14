"""
NOTE:
-----
corresponding model (weight) files can be downloaded at
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
    if _which in ["both", "crnn"]:
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



"""
Observations:
-------------
Weights of the CNN part of both the keras CNN model and the keras CRNN model
are very similar (or close).
These weights can serve as "pretrained backbone" as in the field of vision.
The weights can be extracted as follows
(to use these weights in pytorch models, one has to do some permutations):

d_key_map = {
    "conv1d_1": "branch_0.block_0.ca_0.conv1d",
    "conv1d_2": "branch_0.block_1.ca_0.conv1d",
    "conv1d_3": "branch_0.block_1.ca_1.conv1d",
    "conv1d_4": "branch_0.block_2.ca_0.conv1d",
    "conv1d_5": "branch_0.block_2.ca_1.conv1d",
    "conv1d_6": "branch_0.block_2.ca_2.conv1d",
    "conv1d_7": "branch_1.block_0.ca_0.conv1d",
    "conv1d_8": "branch_1.block_1.ca_0.conv1d",
    "conv1d_9": "branch_1.block_1.ca_1.conv1d",
    "conv1d_10": "branch_1.block_2.ca_0.conv1d",
    "conv1d_11": "branch_1.block_2.ca_1.conv1d",
    "conv1d_12": "branch_1.block_2.ca_2.conv1d",
    "conv1d_13": "branch_2.block_0.ca_0.conv1d",
    "conv1d_14": "branch_2.block_1.ca_0.conv1d",
    "conv1d_15": "branch_2.block_1.ca_1.conv1d",
    "conv1d_16": "branch_2.block_2.ca_0.conv1d",
    "conv1d_17": "branch_2.block_2.ca_1.conv1d",
    "conv1d_18": "branch_2.block_2.ca_2.conv1d",
    "batch_normalization_1": "branch_0.block_0.bn",
    "batch_normalization_2": "branch_0.block_1.bn",
    "batch_normalization_3": "branch_0.block_2.bn",
    "batch_normalization_4": "branch_0.block_0.bn",
    "batch_normalization_5": "branch_0.block_1.bn",
    "batch_normalization_6": "branch_0.block_2.bn",
    "batch_normalization_7": "branch_0.block_0.bn",
    "batch_normalization_8": "branch_0.block_1.bn",
    "batch_normalization_9": "branch_0.block_2.bn",
    "kernel": "weight",
    "bias": "bias",
    "gamma": "weight",
    "beta": "bias",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
}

d_weights_cnn = {}
prefix = "cnn.branches"
for l in cnn_model.layers[1:41]:
    if len(l.weights) > 0:
        key = d_key_map[l.name]
        for v in l.variables:
            minor_key = d_key_map[v.name.split("/")[1].split(":")[0]]
            d_weights_cnn[f"{prefix}.{key}.{minor_key}"] = v.value().numpy()

torch_model = ECG_SEQ_LAB_NET_CPSC2020(classes=["S", "V"], n_leads=1, input_len=4000)
with torch.no_grad():
    for k, v in d_weights_cnn.items():
        if v.ndim == 3:
            exec(f"torch_model.{k} = torch.nn.Parameter(torch.from_numpy(v).permute((2,1,0)))")
        else:
            exec(f"torch_model.{k} = torch.nn.Parameter(torch.from_numpy(v))")

However, a better "pretrained backbone" I think, should be one trained for the task of
wave delineation (detection of P,T waves and QRS complexes, rather than just R peaks).
"""