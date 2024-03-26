"""
NOTE
----
corresponding model (weight) files can be downloaded at
https://opensz.oss-cn-beijing.aliyuncs.com/ICBEB2020/file/CPSC2019-opensource.zip

References
----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] to add more
"""

import os
from typing import Tuple, Union

try:
    from keras.models import Model, model_from_json
except ImportError:
    from tensorflow.keras.models import Model, model_from_json

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from torch import nn

from torch_ecg.utils.download import http_get, url_is_reachable

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "load_model",
]


_MODEL_URLS = {
    "keras": {
        "cnn": "https://www.dropbox.com/scl/fi/6n82j69ut9j8sn79es5kk/CNN.h5?rlkey=o6m0etkm1lj7sg0mjsqsima49&dl=1",
        "crnn": "https://www.dropbox.com/scl/fi/rkqlcmt6uikowfoaav177/CRNN.h5?rlkey=nqw347n7ek2ueldwglqkeat6j&dl=1",
    },
}

_MODEL_ALT_URLS = {
    "keras": {
        "cnn": "https://deep-psp.tech/Models/CPSC2019-0416/CNN.h5",
        "crnn": "https://deep-psp.tech/Models/CPSC2019-0416/CRNN.h5",
    },
}


def load_model(name: str, **kwargs) -> Union[Model, Tuple[Model, ...], nn.Module, Tuple[nn.Module, ...]]:
    """Load model(s) by name.

    Parameters
    ----------
    name : str
        Name of the model, case insensitive, can be one of the followings:
        - "keras_ecg_seq_lab_net"
        - "pytorch_ecg_seq_lab_net" (not implemented yet)
    **kwargs : dict
        Other keyword arguments to be passed to the model loading function.

    Returns
    -------
    model : Model, nn.Module, Tuple[Model, ...], Tuple[nn.Module, ...]
        Model, or sequence of models, either keras or pytorch.

    """
    if name.lower() == "keras_ecg_seq_lab_net":
        models = _load_keras_ecg_seq_lab_net(**kwargs)
        return models
    elif name.lower() == "pytorch_ecg_seq_lab_net":
        raise NotImplementedError("pytorch model is not implemented yet")
    else:
        raise NotImplementedError("Unknown model name")


def _load_keras_ecg_seq_lab_net(which: str = "both", **kwargs) -> Union[Tuple[Model, Model], Model]:
    """Load the CNN model and CRNN model from the entry 0416 of CPSC2019.

    Parameters
    ----------
    which : str, default "both"
        Choice of model(s) to load,
        can be one of "both", "cnn", "crnn", case insensitive.

    Returns
    -------
    cnn_model, crnn_model (both or one) : Model

    """
    _which = which.lower()
    if _which in ["both", "cnn"]:
        download_model_if_not_exist("keras-cnn")
        cnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.json")
        cnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.h5")
        cnn_model = model_from_json(open(cnn_config_path).read())
        cnn_model.load_weights(cnn_h5_path)
        cnn_model.trainable = False
        if _which == "cnn":
            return cnn_model
    if _which in ["both", "crnn"]:
        download_model_if_not_exist("keras-crnn")
        crnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.json")
        crnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.h5")
        crnn_model = model_from_json(open(crnn_config_path).read())
        crnn_model.load_weights(crnn_h5_path)
        crnn_model.trainable = False
        if _which == "crnn":
            return crnn_model
    return cnn_model, crnn_model


def _load_pytorch_ecg_seq_lab_net():
    """ """
    raise NotImplementedError


def download_model_if_not_exist(name: str) -> None:
    """Download model(s) by name if not exist locally.

    Parameters
    ----------
    name : str
        Name of the model, case insensitive, can be one of the followings:
        - "keras-cnn"
        - "keras-crnn"

    Returns
    -------
    None

    """
    if name.lower() == "keras-cnn" and not os.path.exists(os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.h5")):
        http_get(
            url=(
                _MODEL_URLS["keras"]["cnn"]
                if url_is_reachable(_MODEL_URLS["keras"]["cnn"])
                else _MODEL_ALT_URLS["keras"]["cnn"]
            ),
            dst_dir=os.path.join(_BASE_DIR, "CPSC2019_0416"),
            filename="CNN.h5",
            extract=False,
        )
    elif name.lower() == "keras-crnn" and not os.path.exists(os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.h5")):
        http_get(
            url=(
                _MODEL_URLS["keras"]["crnn"]
                if url_is_reachable(_MODEL_URLS["keras"]["crnn"])
                else _MODEL_ALT_URLS["keras"]["crnn"]
            ),
            dst_dir=os.path.join(_BASE_DIR, "CPSC2019_0416"),
            filename="CRNN.h5",
            extract=False,
        )
    else:
        raise NotImplementedError(f"model {name} is not implemented yet")


"""
Observations
------------
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
