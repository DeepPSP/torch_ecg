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

from ...models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from .cfg import ModelCfg
from .utils import mask_to_intervals


class ECG_SEQ_LAB_NET_CPSC2019(ECG_SEQ_LAB_NET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2019"
    
    def __init__(self, classes:Sequence[str], n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for sequence labeling
        n_leads: int,
            number of leads (number of input channels)
        input_len: int, optional,
            sequence length (last dim.) of the input,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        model_config = deepcopy(ModelCfg.seq_lab_crnn)
        model_config.update(config or {})
        super().__init__(classes, n_leads, input_len, model_config)

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        """
        raise NotImplementedError

    def inference_CPSC2019(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        """
        raise NotImplementedError
