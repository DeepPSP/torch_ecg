"""
"""
import os
from typing import Union, Optional, Tuple

import torch
from torch import nn

from train.train_crnn_cpsc2020.model import (
    ECG_CRNN_CPSC2020,
    ECG_SEQ_LAB_NET_CPSC2020,
)
from train.train_crnn_cpsc2020.cfg import ModelCfg


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "load_model",
]


def load_model(which: str = "both") -> Union[nn.Module, Tuple[nn.Module, ...]]:
    """finished, checked,

    Parameters
    ----------
    which: str,
        choice of the models

    Returns
    -------
    nn.Module, or sequence of nn.Module
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    _which = which.lower()
    if _which in ["both", "crnn"]:
        ckpt = torch.load(
            os.path.join(_BASE_DIR, "crnn_10s.pth.tar"), map_location=device
        )
        # crnn_cfg = ModelCfg.crnn
        crnn_cfg = ckpt["model_config"]
        crnn_model = ECG_CRNN_CPSC2020(
            classes=crnn_cfg.classes,
            n_leads=crnn_cfg.n_leads,
            input_len=4000,
            config=crnn_cfg,
        )
        crnn_state_dict = ckpt["model_state_dict"]
        crnn_model.load_state_dict(crnn_state_dict)
        crnn_model.eval()
        if _which == "crnn":
            return crnn_model
    if _which in ["both", "seq_lab"]:
        ckpt = torch.load(
            os.path.join(_BASE_DIR, "seq_lab_10s.pth.tar"), map_location=device
        )
        # seq_lab_cfg = ModelCfg.seq_lab
        seq_lab_cfg = ckpt["model_config"]
        seq_lab_model = ECG_SEQ_LAB_NET_CPSC2020(
            classes=seq_lab_cfg.classes,
            n_leads=seq_lab_cfg.n_leads,
            input_len=4000,
            config=seq_lab_cfg,
        )
        seq_lab_state_dict = ckpt["model_config"]
        seq_lab_model.load_state_dict(seq_lab_state_dict)
        seq_lab_model.eval()
        if _which == "seq_lab":
            return seq_lab_model
    return crnn_model, seq_lab_model
