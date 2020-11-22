"""
"""
from copy import deepcopy
from typing import Union, Optional, Sequence, Tuple, NoReturn

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from easydict import EasyDict as ED

from torch_ecg.models._legacy.legacy_ecg_crnn_v03 import ECG_CRNN
from .cfg import ModelCfg


__all__ = [
    "ECG_CRNN_CINC2020",
]


class ECG_CRNN_CINC2020(ECG_CRNN):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_CRNN_CINC2020"

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
        model_config = deepcopy(ModelCfg)
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, input_len, model_config)


    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], class_names:bool=False, bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """ finished, checked,

        auxiliary function to `forward`, for CINC2020,

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
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
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
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                bin_pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                bin_pred[row_idx,...] = \
                    (((pred[row_idx,...]+ModelCfg.bin_pred_look_again_tol) >= row_max_prob) & (pred[row_idx,...] >= ModelCfg.bin_pred_nsr_thr)).astype(int)
        if class_names:
            pred = pd.DataFrame(pred)
            pred.columns = self.classes
            # pred['bin_pred'] = pred.apply(
            #     lambda row: np.array(self.classes)[np.where(row.values>=bin_pred_thr)[0]],
            #     axis=1
            # )
            pred['bin_pred'] = ''
            for row_idx in range(len(pred)):
                pred.at[row_idx, 'bin_pred'] = \
                    np.array(self.classes)[np.where(bin_pred==1)[0]].tolist()
        return pred, bin_pred


    def inference_CINC2020(self, input:Union[np.ndarray,Tensor], class_names:bool=False, bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)
