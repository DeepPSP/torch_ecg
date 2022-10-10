"""
"""

import warnings
from typing import NoReturn, Optional, Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import scalar_tensor
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import MLP
from torch_ecg.utils.utils_nn import SizeMixin
from torch_ecg.models.loss import (
    AsymmetricLoss,
    BCEWithLogitsWithClassWeightLoss,
    FocalLoss,
    MaskedBCEWithLogitsLoss,
)


class MultiTaskHead(nn.Module, SizeMixin):
    """ """

    __name__ = "MultiTaskHead"

    def __init__(self, in_channels: int, config: CFG) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            the number of input channels
        config: dict,
            configurations, ref. `cfg.ModelCfg`

        """
        super().__init__()
        self.in_channels = in_channels
        self.config = config

        self.heads = nn.ModuleDict()
        self.criteria = nn.ModuleDict()

        if self.config.get("outcome_head", None) is not None:
            self.outcomes = self.config.get("outcomes")
            self.config.outcome_head.out_channels.append(len(self.outcomes))
            self.heads["outcome"] = MLP(
                in_channels=self.in_channels,
                skip_last_activation=True,
                **self.config.outcome_head,
            )
            self.criteria["outcome"] = self._setup_criterion(
                loss=self.config.outcome_head.loss,
                loss_kw=self.config.outcome_head.loss_kw,
            )
        else:
            self.outcomes = None
        if self.config.get("segmentation_head", None) is not None:
            self.states = self.config.get("states")
            self.config.segmentation_head.out_channels.append(len(self.states))
            self.heads["segmentation"] = MLP(
                in_channels=self.in_channels,
                skip_last_activation=True,
                **self.config.segmentation_head,
            )
            self.criteria["segmentation"] = self._setup_criterion(
                loss=self.config.segmentation_head.loss,
                loss_kw=self.config.segmentation_head.loss_kw,
            )
        else:
            self.states = None

    def forward(
        self,
        features: Tensor,
        pooled_features: Tensor,
        original_len: int,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """

        Parameters
        ----------
        features: Tensor,
            the feature tensor,
            of shape (batch_size, n_channels, seq_len)
        pooled_features: Tensor,
            the pooled features of the input data,
            of shape (batch_size, n_channels)
        original_len: int,
            the original length of the input data,
            used when segmentation head's `recover_length` config is set `True`
        labels: dict of Tensor, optional,
            the labels of the input data, including:
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes) or (batch_size,)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of Tensor,
            the output of the model, including (some are optional):
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        if self.empty:
            warnings.warn("Empty model, DO NOT call forward function!")
        out = dict(total_extra_loss=scalar_tensor(0.0))
        if "segmentation" in self.heads:
            out["segmentation"] = self.heads["segmentation"](features.permute(0, 2, 1))
            if self.config.segmentation_head.get("recover_length", True):
                out["segmentation"] = F.interpolate(
                    out["segmentation"].permute(0, 2, 1),
                    size=original_len,
                    mode="linear",
                    align_corners=True,
                ).permute(0, 2, 1)
            if labels is not None and labels.get("segmentation", None) is not None:
                out["segmentation_loss"] = self.criteria["segmentation"](
                    out["segmentation"].reshape(-1, out["segmentation"].shape[0]),
                    labels["segmentation"].reshape(-1, labels["segmentation"].shape[0]),
                )
                out["total_extra_loss"] = (
                    out["total_extra_loss"].to(dtype=out["segmentation_loss"].dtype)
                    + out["segmentation_loss"]
                )
        if "outcome" in self.heads:
            out["outcome"] = self.heads["outcome"](pooled_features)
            if labels is not None and labels.get("outcome", None) is not None:
                out["outcome_loss"] = self.criteria["outcome"](
                    out["outcome"], labels["outcome"]
                )
                out["total_extra_loss"] = (
                    out["total_extra_loss"].to(dtype=out["outcome_loss"].dtype)
                    + out["outcome_loss"]
                )
        return out

    def _setup_criterion(self, loss: str, loss_kw: Optional[dict] = None) -> NoReturn:
        """ """
        if loss_kw is None:
            loss_kw = {}
        if loss == "BCEWithLogitsLoss":
            criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif loss == "BCEWithLogitsWithClassWeightLoss":
            criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif loss == "BCELoss":
            criterion = nn.BCELoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "MaskedBCEWithLogitsLoss":
            criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif loss == "FocalLoss":
            criterion = FocalLoss(**loss_kw)
        elif loss == "AsymmetricLoss":
            criterion = AsymmetricLoss(**loss_kw)
        elif loss == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(
                f"loss `{loss}` not implemented! "
                "Please use one of the following: `BCEWithLogitsLoss`, `BCEWithLogitsWithClassWeightLoss`, "
                "`BCELoss`, `MaskedBCEWithLogitsLoss`, `MaskedBCEWithLogitsLoss`, `FocalLoss`, "
                "`AsymmetricLoss`, `CrossEntropyLoss`, or override this method to setup your own criterion."
            )
        return criterion

    @property
    def empty(self) -> bool:
        return len(self.heads) == 0
