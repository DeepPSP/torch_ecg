"""Model of the CinC2023 benchmark.

`CRNN_CINC2023` predicts the neurological outcome (CPC level, or the
binary Good/Poor outcome) of comatose post-cardiac-arrest patients from
bipolar EEG segments, with the waveform CRNN from the library.
"""

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import ClassificationOutput
from torch_ecg.models.ecg_crnn import ECG_CRNN

from .outputs import CINC2023Outputs

__all__ = [
    "CRNN_CINC2023",
]


class CRNN_CINC2023(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2023"

    def __init__(self, config: CFG, **kwargs) -> None:
        """
        Parameters
        ----------
        config : dict
            Configurations of the model, ref. `cfg.ModelCfg`,
            with the following key fields:
                - model_name: str, name of the model architecture,
                - classes: list of str, the classes of the output target,
                - output_target: str, "cpc" or "outcome",
                - num_channels: int, number of input channels,
                - fs: int, sampling frequency of the input.

        """
        model_config = deepcopy(config[config.model_name])
        self.output_target = config.output_target
        super().__init__(
            list(config.classes),
            config.num_channels,
            model_config,
            **kwargs,
        )

    def forward(self, waveforms: Tensor, input_tensors: Optional[dict] = None) -> dict:
        """

        Parameters
        ----------
        waveforms : Tensor,
            Input tensor of shape ``(batch_size, n_channels, siglen)``.
        input_tensors : dict, optional,
            The full dict of input tensors (labels, hospitals, etc.),
            unused by the CRNN which consumes waveforms only.

        Returns
        -------
        out_tensors : dict,
            With the item `self.output_target`: logits of shape
            ``(batch_size, n_classes)``.

        """
        return {self.output_target: super().forward(waveforms)}

    @torch.no_grad()
    def inference(self, waveforms: Tensor) -> CINC2023Outputs:
        """

        Parameters
        ----------
        waveforms : Tensor,
            Input tensor of shape ``(batch_size, n_channels, siglen)``.

        Returns
        -------
        CINC2023Outputs,
            With the `cpc_output` (or `outcome_output`) item filled;
            the other target is inferred via the cpc-outcome mapping.

        """
        self.eval()
        _input = torch.as_tensor(waveforms, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        logits = super().forward(_input)
        prob = self.sigmoid(logits).detach().cpu().numpy()
        # single-label targets (cpc level / outcome), hence argmax
        pred = np.argmax(prob, axis=1)
        output = ClassificationOutput(
            classes=list(self.classes),
            prob=prob,
            pred=pred,
        )
        if self.output_target == "cpc":
            return CINC2023Outputs(cpc_output=output)
        return CINC2023Outputs(outcome_output=output)

    @torch.no_grad()
    def inference_CinC2023(self, waveforms: Tensor) -> CINC2023Outputs:
        """alias for `self.inference`"""
        return self.inference(waveforms)
