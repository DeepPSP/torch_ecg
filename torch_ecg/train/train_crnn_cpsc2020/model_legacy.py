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

from torch_ecg.models._legacy.legacy_ecg_crnn_v03 import ECG_CRNN
from torch_ecg.models import ECG_SEQ_LAB_NET
from .cfg import ModelCfg
from .utils import mask_to_intervals


__all__ = [
    "ECG_CRNN_CPSC2020",
    "ECG_SEQ_LAB_NET_CPSC2020",
]


class ECG_CRNN_CPSC2020(ECG_CRNN):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_CRNN_CPSC2020"

    def __init__(self, classes:Sequence[str], n_leads:int, input_len:Optional[int]=None, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        input_len: int, optional,
            sequence length (last dim.) of the input,
            will not be used in the inference mode
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        model_config = deepcopy(ModelCfg.crnn)
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, input_len, model_config)

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], class_names:bool=False, bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """ finished, checked,

        auxiliary function to `forward`, for CPSC2020,

        Parameters:
        -----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions

        Returns:
        --------
        pred: ndarray or DataFrame,
            scalar predictions, (and binary predictions if `class_names` is True)
        bin_pred: ndarray,
            the array (with values 0, 1 for each class) of binary prediction
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)
        if isinstance(input, np.ndarray):
            _input = torch.from_numpy(input).to(device)
        else:
            _input = input.to(device)
        pred = self.forward(_input)
        pred = self.sigmoid(pred)
        bin_pred = (pred>=bin_pred_thr).int()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()
        for row_idx, row in enumerate(bin_pred):
            row_max_prob = pred[row_idx,...].max()
            if row.sum() == 0:
                bin_pred[row_idx,...] = (pred[row_idx,...]==np.max(pred[row_idx,...])).astype(int)
        if class_names:
            pred = pd.DataFrame(pred)
            pred.columns = self.classes
            pred['bin_pred'] = ''
            for row_idx in range(len(pred)):
                pred.at[row_idx, 'bin_pred'] = \
                    np.array(self.classes)[np.where(bin_pred==1)[0]].tolist()
        return pred, bin_pred

    def inference_CPSC2020(self, input:Union[np.ndarray,Tensor], class_names:bool=False, bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)



class ECG_SEQ_LAB_NET_CPSC2020(ECG_SEQ_LAB_NET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2020"

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
        model_config = deepcopy(ModelCfg.seq_lab)
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, input_len, model_config)
        self.reduction = reduce(
            lambda a,b: a*b,
            self.config.cnn.multi_scopic.subsample_lengths,
            1,
        )

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5, rpeak_inds:Optional[List[np.ndarray]]=None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """ finished, checked,

        auxiliary function to `forward`, for CPSC2020,

        Parameters:
        -----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeak_inds: list of ndarray, optional,
            indices of rpeaks for each batch data

        Returns:
        --------
        pred: ndarray or DataFrame,
            scalar predictions, (and binary predictions if `class_names` is True)
        SPB_indices: list,
            list of predicted indices of SPB
        PVC_indices: list,
            list of predicted indices of PVC
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.to(device)
        batch_size, channels, seq_len = input.shape
        if isinstance(input, np.ndarray):
            _input = torch.from_numpy(input).to(device)
        else:
            _input = input.to(device)
        pred = self.forward(_input)
        if self.n_classes == 2:
            pred = self.sigmoid(pred)  # (batch_size, seq_len, 2)
            bin_pred = (pred>=bin_pred_thr).int()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            # aux used to filter out potential simultaneous predictions of SPB and PVC
            aux = (pred == np.max(pred, axis=2, keepdims=True)).astype(int)
            bin_pred = aux * bin_pred
        elif self.n_classes == 3:
            pred = self.softmax(pred)  # (batch_size, seq_len, 3)
            pred = pred.cpu().detach().numpy()
            bin_pred = np.argmax(pred, axis=2)
        
        if rpeak_inds is not None:
            assert len(rpeak_inds) == batch_size
            rpeak_mask = np.zeros((batch_size, seq_len//self.reduction), dtype=int)
            for i in range(batch_size):
                batch_rpeak_inds = (rpeak_inds[i]/self.reduction).astype(int)
                rpeak_mask[i, batch_rpeak_inds] = 1
        else:
            rpeak_mask = np.ones((batch_size, seq_len//self.reduction), dtype=int)

        SPB_intervals = [
            mask_to_intervals(seq*rpeak_mask[idx], 1) \
                for idx, seq in enumerate(bin_pred[..., self.classes.index("S")])
        ]
        SPB_indices = [
            [self.reduction * (itv[0]+itv[1])//2 for itv in l_itv] if len(l_itv) > 0 else [] \
                for l_itv in SPB_intervals
        ]
        PVC_intervals = [
            mask_to_intervals(seq*rpeak_mask[idx], 1) \
                for idx, seq in enumerate(bin_pred[..., self.classes.index("V")])
        ]
        PVC_indices = [
            [self.reduction * (itv[0]+itv[1])//2 for itv in l_itv]  if len(l_itv) > 0 else [] \
                for l_itv in PVC_intervals
        ]
        return pred, SPB_indices, PVC_indices

    def inference_CPSC2020(self, input:Union[np.ndarray,Tensor], bin_pred_thr:float=0.5) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr)
