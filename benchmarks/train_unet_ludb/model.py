"""
"""
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, Sequence, Tuple, List, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from easydict import EasyDict as ED
import biosppy.signals.ecg as BSE

# from torch_ecg.models.ecg_subtract_unet import ECG_SUBTRACT_UNET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from train.train_unet_ludb.cfg import ModelCfg
from train.train_unet_ludb.utils import mask_to_intervals, _remove_spikes_naive


__all__ = [
    "ECG_UNET_LUDB",
]


class ECG_UNET_LUDB(ECG_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_UNET_LUDB"
    
    def __init__(self, n_leads:int, config:Optional[ED]=None, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        model_config = deepcopy(ModelCfg.unet)
        model_config.update(deepcopy(config) or {})
        super().__init__(model_config.classes, n_leads, model_config, **kwargs)

    @torch.no_grad()
    def inference(self, input:Union[Sequence[float],np.ndarray,Tensor]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ NOT finished, NOT checked,
        """
        self.eval()
        _device = next(self.parameters()).device
        _dtype = next(self.parameters()).dtype
        _input = torch.as_tensor(input, dtype=_dtype, device=_device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        pred = self.forward(_input)
        pred = self.softmax(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)

        raise NotImplementedError

    def inference_LUDB(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200, correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)
