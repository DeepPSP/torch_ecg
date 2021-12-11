"""
"""
from copy import deepcopy
from typing import Union, Optional, Sequence, Tuple, NoReturn

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from easydict import EasyDict as ED

from torch_ecg.models.ecg_crnn import ECG_CRNN

from cfg import ModelCfg


__all__ = [
    "ECG_CRNN_CINC2021",
]


class ECG_CRNN_CINC2021(ECG_CRNN):
    """
    """
    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2021"

    def __init__(self, classes:Sequence[str], n_leads:int, config:Optional[ED]=None) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        model_config = ED(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, model_config)


    @torch.no_grad()
    def inference(self,
                  input:Union[np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray,pd.DataFrame],np.ndarray]:
        """ finished, checked,

        auxiliary function to `forward`, for CINC2021,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions

        Returns
        -------
        pred: ndarray or DataFrame,
            scalar predictions, (and binary predictions if `class_names` is True)
        bin_pred: ndarray,
            the array (with values 0, 1 for each class) of binary prediction

        NOTE that when `input` is ndarray, one should make sure that it is transformed,
        e.g. bandpass filtered, normalized, etc.
        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        _device = next(self.parameters()).device
        _dtype = next(self.parameters()).dtype
        _input = torch.as_tensor(input, dtype=_dtype, device=_device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
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
            # pred["bin_pred"] = pred.apply(
            #     lambda row: np.array(self.classes)[np.where(row.values>=bin_pred_thr)[0]],
            #     axis=1
            # )
            pred["bin_pred"] = ""
            for row_idx in range(len(pred)):
                pred.at[row_idx, "bin_pred"] = \
                    np.array(self.classes)[np.where(bin_pred[row_idx]==1)[0]].tolist()
        return pred, bin_pred


    @torch.no_grad()
    def inference_CINC2021(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray,pd.DataFrame],np.ndarray]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


    @staticmethod
    def from_checkpoint(path:str, device:Optional[torch.device]=None) -> Tuple[nn.Module, dict]:
        """

        Parameters
        ----------
        path: str,
            path of the checkpoint
        device: torch.device, optional,
            map location of the model parameters,
            defaults "cuda" if available, otherwise "cpu"

        Returns
        -------
        model: Module,
            the model loaded from a checkpoint
        aux_config: dict,
            auxiliary configs that are needed for data preprocessing, etc.
        """
        _device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        model = ECG_CRNN_CINC2021(
            classes=aux_config["classes"],
            n_leads=aux_config["n_leads"],
            config=ckpt["model_config"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model, aux_config
