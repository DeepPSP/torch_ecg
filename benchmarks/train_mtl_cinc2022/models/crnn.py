"""
"""

from copy import deepcopy
from typing import Union, Optional, Any, Dict

import numpy as np
import torch
from torch import Tensor
from einops import rearrange
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)
from torch_ecg.utils import add_docstring

from cfg import ModelCfg
from outputs import CINC2022Outputs
from .heads import MultiTaskHead


__all__ = ["CRNN_CINC2022"]


class CRNN_CINC2022(ECG_CRNN):
    """ """

    __DEBUG__ = True
    __name__ = "CRNN_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "classification"
        model_cfg = deepcopy(ModelCfg[task])
        model = ECG_CRNN_CINC2022(model_cfg)
        ````

        """
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        super().__init__(
            _config.classes,
            _config.num_channels,
            _config[_config.model_name],
        )
        self.outcomes = _config.outcomes
        self.states = _config.states
        self.extra_heads = MultiTaskHead(
            in_channels=self.clf.in_channels,
            config=_config,
        )
        if self.extra_heads.empty:
            self.extra_heads = None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        freeze the backbone (CRNN part, excluding the heads) of the model

        Parameters
        ----------
        freeze: bool, default True,
            whether to freeze the backbone

        """
        for params in self.cnn.parameters():
            params.requires_grad = not freeze
        if getattr(self, "rnn") is not None:
            for params in self.rnn.parameters():
                params.requires_grad = not freeze
        if getattr(self, "attn") is not None:
            for params in self.attn.parameters():
                params.requires_grad = not freeze

    def forward(
        self,
        waveforms: Tensor,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """

        Parameters
        ----------
        waveforms: Tensor,
            of shape (batch_size, channels, seq_len)
        labels: dict of Tensor, optional,
            the labels of the waveforms data, including:
            - "murmur": the murmur labels, of shape (batch_size, n_classes) or (batch_size,)
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes) or (batch_size,)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of Tensor, with items (some are optional):
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        batch_size, channels, seq_len = waveforms.shape

        features = self.extract_features(waveforms)

        if self.pool:
            pooled_features = self.pool(features)  # (batch_size, channels, pool_size)
            # features = features.squeeze(dim=-1)
            pooled_features = rearrange(
                pooled_features,
                "batch_size channels pool_size -> batch_size (channels pool_size)",
            )
        else:
            # pooled_features of shape (batch_size, channels) or (batch_size, seq_len, channels)
            pooled_features = features

        # print(f"clf in shape = {x.shape}")
        pred = self.clf(pooled_features)  # batch_size, n_classes

        if self.extra_heads is not None:
            out = self.extra_heads(features, pooled_features, seq_len, labels)
            out["murmur"] = pred
        else:
            out = {"murmur": pred}

        return out

    @torch.no_grad()
    def inference(
        self,
        waveforms: Union[np.ndarray, Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        auxiliary function to `forward`, for CINC2022,

        Parameters
        ----------
        waveforms: ndarray or Tensor,
            waveforms tensor, of shape (batch_size, channels, seq_len)
        seg_thr: float, default 0.5,
            threshold for making binary predictions for
            the optional segmentaion head

        Returns
        -------
        CINC2022Outputs, with attributes:
            - murmur_output: ClassificationOutput, with items:
                - classes: list of str,
                    list of the class names
                - prob: ndarray or DataFrame,
                    scalar (probability) predictions,
                    (and binary predictions if `class_names` is True)
                - pred: ndarray,
                    the array of class number predictions
                - bin_pred: ndarray,
                    the array of binary predictions
                - forward_output: ndarray,
                    the array of output of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
            outcome_output: ClassificationOutput, optional, with items:
                - classes: list of str,
                    list of the outcome class names
                - prob: ndarray,
                    scalar (probability) predictions,
                - pred: ndarray,
                    the array of outcome class number predictions
                - forward_output: ndarray,
                    the array of output of the outcome head of the model's forward function,
                    useful for producing challenge result using
                    multiple recordings
            segmentation_output: SequenceLabellingOutput, optional, with items:
                - classes: list of str,
                    list of the state class names
                - prob: ndarray,
                    scalar (probability) predictions,
                - pred: ndarray,
                    the array of binarized prediction

        """
        self.eval()
        _input = torch.as_tensor(waveforms, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        forward_output = self.forward(_input)

        prob = self.softmax(forward_output["murmur"])
        pred = torch.argmax(prob, dim=-1)
        bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        bin_pred = bin_pred.cpu().detach().numpy()

        murmur_output = ClassificationOutput(
            classes=self.classes,
            prob=prob,
            pred=pred,
            bin_pred=bin_pred,
            forward_output=forward_output["murmur"].cpu().detach().numpy(),
        )

        if forward_output.get("outcome", None) is not None:
            prob = self.softmax(forward_output["outcome"])
            pred = torch.argmax(prob, dim=-1)
            bin_pred = (prob == prob.max(dim=-1, keepdim=True).values).to(int)
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            bin_pred = bin_pred.cpu().detach().numpy()
            outcome_output = ClassificationOutput(
                classes=self.outcomes,
                prob=prob,
                pred=pred,
                bin_pred=bin_pred,
                forward_output=forward_output["outcome"].cpu().detach().numpy(),
            )
        else:
            outcome_output = None

        if forward_output.get("segmentation", None) is not None:
            # if "unannotated" in self.states, use softmax
            # else use sigmoid
            if "unannotated" in self.states:
                prob = self.softmax(forward_output["segmentation"])
                pred = torch.argmax(prob, dim=-1)
            else:
                prob = self.sigmoid(forward_output["segmentation"])
                pred = (prob > seg_thr).int() * (
                    prob == prob.max(dim=-1, keepdim=True).values
                ).int()
            prob = prob.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            segmentation_output = SequenceLabellingOutput(
                classes=self.states,
                prob=prob,
                pred=pred,
                forward_output=forward_output["segmentation"].cpu().detach().numpy(),
            )
        else:
            segmentation_output = None

        return CINC2022Outputs(murmur_output, outcome_output, segmentation_output)

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        waveforms: Union[np.ndarray, Tensor],
        seg_thr: float = 0.5,
    ) -> CINC2022Outputs:
        """
        alias for `self.inference`
        """
        return self.inference(waveforms, seg_thr)
