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

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.models.unets.ecg_subtract_unet import ECG_SUBTRACT_UNET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils.misc import mask_to_intervals
from torch_ecg.utils.utils_signal import remove_spikes_naive

from cfg import ModelCfg


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
        if config:
            model_config.update(deepcopy(config[config.model_name]))
            ModelCfg.update(deepcopy(config))
        _inv_class_map = {v: k for k, v in ModelCfg.class_map.items()}
        self._mask_map = ED({
            k: _inv_class_map[v] \
                for k, v in ModelCfg.mask_class_map.items()
        })
        super().__init__(ModelCfg.mask_classes, n_leads, model_config)

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,) -> Tuple[np.ndarray, np.ndarray]:
        """ finished, NOT checked,

        Parameters
        ----------
        input: array-like,
            input ECG signal
        bin_pred_thr: float, default 0.5,
            threshold for binary prediction,
            used only when the `background` class "i" is not included in `mask_classes`

        Returns
        -------
        pred: np.ndarray,
            predicted probability map, of shape (n_samples, seq_len, n_classes)
        mask: np.ndarray,
            predicted mask, of shape (n_samples, seq_len)
        """
        self.eval()
        _device = next(self.parameters()).device
        _dtype = next(self.parameters()).dtype
        _input = torch.as_tensor(input, dtype=_dtype, device=_device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        pred = self.forward(_input)
        if "i" in self.classes:
            pred = self.softmax(pred)
        else:
            pred = torch.sigmoid(pred)
        pred = pred.cpu().detach().numpy()

        if "i" in self.classes:
            mask = np.argmax(pred, axis=-1)
        else:
            mask = np.vectorize(lambda n: self._mask_map[n])(np.argmax(pred, axis=-1))
            mask *= (pred > bin_pred_thr).any(axis=-1)  # class "i" mapped to 0

        # TODO: shoule one add more post-processing to filter out false positives of the waveforms?

        return pred, mask

    def inference_LUDB(self,
                       input:Union[np.ndarray,Tensor],
                       bin_pred_thr:float=0.5,) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)
