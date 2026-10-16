"""The CINC2021 model: ECG_CRNN with challenge-specific inference."""

from copy import deepcopy
from typing import Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import MultiLabelClassificationOutput
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.misc import add_docstring

from .cfg import ModelCfg

__all__ = [
    "ECG_CRNN_CINC2021",
]


class ECG_CRNN_CINC2021(ECG_CRNN):
    """ECG_CRNN with CINC2021-specific inference post-processing.

    The challenge metric allows empty predictions to fall back to NSR;
    `inference` implements that rule on top of the plain sigmoid outputs.
    """

    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2021"

    def __init__(self, classes: Sequence[str], n_leads: int, config: CFG | None = None) -> None:
        model_config = CFG(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, model_config)

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> MultiLabelClassificationOutput:
        """Inference with the challenge-specific NSR fallback rule.

        Parameters
        ----------
        input : ndarray or Tensor
            input tensor, of shape (batch_size, channels, seq_len)
        class_names : bool, default False
            if True, the returned scalar predictions are a `DataFrame`
            with the predicted class names appended
        bin_pred_thr : float, default 0.5
            threshold for making binary predictions

        Returns
        -------
        MultiLabelClassificationOutput
            with items `classes`, `thr`, `prob`, `pred`.

        NOTE when `input` is an ndarray, it must already be preprocessed
        (bandpass filtered, normalized, etc.).

        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        prob = self.sigmoid(self.forward(_input))
        pred = (prob >= bin_pred_thr).int()
        prob = prob.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        for row_idx, row in enumerate(pred):
            row_max_prob = prob[row_idx, ...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                pred[row_idx, ...] = (
                    ((prob[row_idx, ...] + ModelCfg.bin_pred_look_again_tol) >= row_max_prob)
                    & (prob[row_idx, ...] >= ModelCfg.bin_pred_nsr_thr)
                ).astype(int)
        if class_names:
            prob = pd.DataFrame(prob)
            prob.columns = self.classes
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = np.array(self.classes)[np.where(pred[row_idx] == 1)[0]].tolist()
        return MultiLabelClassificationOutput(
            classes=self.classes,
            thr=bin_pred_thr,
            prob=prob,
            pred=pred,
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2021(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> MultiLabelClassificationOutput:
        """alias of `inference`"""
        return self.inference(input, class_names, bin_pred_thr)
