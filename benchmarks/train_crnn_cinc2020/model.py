"""
"""

from copy import deepcopy
from typing import Union, Optional, Sequence, Tuple, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import Tensor

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.outputs import MultiLabelClassificationOutput

from cfg import ModelCfg


__all__ = [
    "ECG_CRNN_CINC2020",
]


class ECG_CRNN_CINC2020(ECG_CRNN):
    """
    """
    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2020"

    def __init__(self, classes:Sequence[str], n_leads:int, config:Optional[CFG]=None, **kwargs:Any) -> NoReturn:
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
        model_config = deepcopy(ModelCfg)
        model_config.update(deepcopy(config) or {})
        assert n_leads == 12, "CinC2020 only supports 12-lead models"
        super().__init__(classes, n_leads, model_config, **kwargs)

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  class_names:bool=False,
                  bin_pred_thr:float=0.5) -> MultiLabelClassificationOutput:
        """ finished, checked,

        auxiliary function to `forward`, for CINC2020,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions

        Returns
        -------
        MultiLabelClassificationOutput, with the following items:
            classes: list,
                list of the classes for classification
            thr: float,
                threshold for making binary predictions from scalar predictions
            prob: ndarray or DataFrame,
                scalar predictions, (and binary predictions if `class_names` is True)
            prob: ndarray,
                the array (with values 0, 1 for each class) of binary prediction
        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        prob = self.sigmoid(self.forward(_input))
        pred = (prob>=bin_pred_thr).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        for row_idx, row in enumerate(pred):
            row_max_prob = prob[row_idx,...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                pred[row_idx,...] = \
                    (
                        ((prob[row_idx,...]+ModelCfg.bin_pred_look_again_tol) >= row_max_prob) \
                            & (prob[row_idx,...] >= ModelCfg.bin_pred_nsr_thr )
                    ).astype(int)
        if class_names:
            prob = pd.DataFrame(prob)
            prob.columns = self.classes
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = \
                    np.array(self.classes)[np.where(pred==1)[0]].tolist()
        return MultiLabelClassificationOutput(
            classes=self.classes, thr=bin_pred_thr, prob=prob, pred=pred,
        )

    @torch.no_grad()
    def inference_CINC2020(self,
                           input:Union[np.ndarray,Tensor],
                           class_names:bool=False,
                           bin_pred_thr:float=0.5) -> MultiLabelClassificationOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)
