"""
"""
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, Sequence, Tuple, List, NoReturn

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from easydict import EasyDict as ED
import biosppy.signals.ecg as BSE

from torch_ecg.models.ecg_subtract_unet import ECG_SUBTRACT_UNET
from torch_ecg.models.ecg_unet import ECG_UNET
from .cfg import ModelCfg
from .utils import mask_to_intervals, _remove_spikes_naive


class ECG_SUBTRACT_UNET_CPSC2019(ECG_SUBTRACT_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SUBTRACT_UNET_CPSC2019"
    
    def __init__(self, n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        n_leads: int,
            number of leads (number of input channels)
        input_len: int, optional,
            sequence length (last dim.) of the input,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200, correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ finished, NOT checked,

        auxiliary function to `forward`, for CPSC2019,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters:
        -----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).
        correction: bool, default False,
            if True, correct rpeaks to local maximum in a small nbh
            of rpeaks detected by DL model using `BSE.correct_rpeaks`

        Returns:
        --------
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        raise NotImplementedError

    def _inference_post_process(self, pred:np.ndarray, bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200) -> List[np.ndarray]:
        """ finished, NOT checked,

        prob --> qrs mask --> qrs intervals --> rpeaks

        Parameters: ref. `self.inference`
        """
        raise NotImplementedError

    def inference_CPSC2019(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200, correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)


class ECG_UNET_CPSC2019(ECG_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_UNET_CPSC2019"
    
    def __init__(self, n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        n_leads: int,
            number of leads (number of input channels)
        input_len: int, optional,
            sequence length (last dim.) of the input,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200, correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ finished, NOT checked,

        auxiliary function to `forward`, for CPSC2019,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters:
        -----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).
        correction: bool, default False,
            if True, correct rpeaks to local maximum in a small nbh
            of rpeaks detected by DL model using `BSE.correct_rpeaks`

        Returns:
        --------
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        raise NotImplementedError

    def _inference_post_process(self, pred:np.ndarray, bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200) -> List[np.ndarray]:
        """ finished, NOT checked,

        prob --> qrs mask --> qrs intervals --> rpeaks

        Parameters: ref. `self.inference`
        """
        raise NotImplementedError

    def inference_CPSC2019(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=200, correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)
