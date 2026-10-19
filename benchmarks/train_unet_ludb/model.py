"""The LUDB model: ECG_UNET with delineation-specific inference."""

from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import WaveDelineationOutput
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils.misc import add_docstring

from .cfg import ModelCfg

__all__ = [
    "ECG_UNET_LUDB",
]


class ECG_UNET_LUDB(ECG_UNET):
    """ECG_UNET over the LUDB mask classes, with mask-decoding inference.

    The forward pass predicts per-sample probability maps over the mask
    classes; `inference` turns them into integer masks (argmax over the
    classes, with the softmax/sigmoid choice depending on whether the
    background class "i" is part of `mask_classes`).
    """

    __DEBUG__ = True
    __name__ = "ECG_UNET_LUDB"

    def __init__(self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any) -> None:
        model_config = deepcopy(ModelCfg.unet)
        if config:
            model_config.update(deepcopy(config[config.model_name]))
            ModelCfg.update(deepcopy(config))
        _inv_class_map = {v: k for k, v in ModelCfg.class_map.items()}
        self._mask_map = CFG({k: _inv_class_map[v] for k, v in ModelCfg.mask_class_map.items()})
        super().__init__(ModelCfg.mask_classes, n_leads, model_config)

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
    ) -> WaveDelineationOutput:
        """Delineation inference.

        Parameters
        ----------
        input : array-like
            input ECG signal
        bin_pred_thr : float, default 0.5
            threshold for binary prediction, used only when the
            background class "i" is not included in `mask_classes`

        Returns
        -------
        WaveDelineationOutput
            with items `classes`, `prob` (n_samples, seq_len, n_classes)
            and `mask` (n_samples, seq_len).

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        prob = self.forward(_input)
        if "i" in self.classes:
            prob = self.softmax(prob)
        else:
            prob = torch.sigmoid(prob)
        prob = prob.detach().cpu().numpy()

        if "i" in self.classes:
            mask = np.argmax(prob, axis=-1)
        else:
            mask = np.vectorize(lambda n: self._mask_map[n])(np.argmax(prob, axis=-1))
            mask *= (prob > bin_pred_thr).any(axis=-1)  # class "i" mapped to 0

        return WaveDelineationOutput(
            classes=self.classes,
            prob=prob,
            mask=mask,
        )

    @add_docstring(inference.__doc__)
    def inference_LUDB(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
    ) -> WaveDelineationOutput:
        """alias of `inference`"""
        return self.inference(input, bin_pred_thr)
