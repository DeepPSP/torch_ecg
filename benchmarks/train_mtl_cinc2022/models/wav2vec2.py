"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Any, Tuple, Dict

import numpy as np
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models.cnn.vgg import VGG16
from torch_ecg.models.cnn.resnet import ResNet
from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.models.cnn.densenet import DenseNet
from torch_ecg.models._nets import MLP
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)
from torch_ecg.utils import add_docstring, SizeMixin, CkptMixin
from transformers import (
    Wav2Vec2Model as HFWav2Vec2Model,
    Wav2Vec2PreTrainedModel as HFWav2Vec2PreTrainedModel,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _HIDDEN_STATES_START_POSITION as _HF_HIDDEN_STATES_START_POSITION,
)

from cfg import ModelCfg
from outputs import CINC2022Outputs
from wav2vec2_ta import Wav2Vec2Model, components as w2v2_components
from wav2vec2_hf import PreTrainModelCfg as HFPreTrainModelCfg
from .heads import MultiTaskHead


__all__ = [
    "Wav2Vec2_CINC2022",
    "HFWav2Vec2_CINC2022",
]


class Wav2Vec2_CINC2022(Wav2Vec2Model):
    """ """

    __DEBUG__ = True
    __name__ = "Wav2Vec2_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        self.config = _config[_config.model_name]
        assert "encoder" in self.config, "`encoder` is a required key in config"
        if "rnn" in self.config:
            self.config.pop("rnn", None)
        if "attn" in self.config:
            self.config.pop("attn", None)
        self.classes = deepcopy(_config.classes)
        self.n_classes = len(_config.classes)

        self.outcomes = _config.outcomes
        self.states = _config.states

        self.squeeze = None

        cnn_choice = self.config.cnn.name.lower()
        cnn_config = self.config.cnn[self.config.cnn.name]
        encoder_in_features = None
        if "wav2vec2" in cnn_choice:
            cnn = w2v2_components._get_feature_extractor(
                norm_mode=cnn_config.norm_mode,
                shapes=cnn_config.ch_ks_st,
                bias=cnn_config.bias,
            )
            encoder_in_features = cnn_config.ch_ks_st[-1][0]
        elif "vgg16" in cnn_choice:
            cnn = VGG16(_config.num_channels, **cnn_config)
        elif "resnet" in cnn_choice:
            cnn = ResNet(_config.num_channels, **cnn_config)
        elif "multi_scopic" in cnn_choice:
            cnn = MultiScopicCNN(_config.num_channels, **cnn_config)
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            cnn = DenseNet(_config.num_channels, **cnn_config)
        else:
            raise NotImplementedError(
                f"the CNN \042{cnn_choice}\042 not implemented yet"
            )
        if encoder_in_features is None:
            _, encoder_in_features, _ = cnn.compute_output_shape()
            cnn = nn.Sequential(
                cnn, Rearrange("batch chan seqlen -> batch seqlen chan")
            )

        encoder_config = self.config.encoder[self.config.encoder.name]
        encoder = w2v2_components._get_encoder(
            in_features=encoder_in_features,
            embed_dim=encoder_config.embed_dim,
            dropout_input=encoder_config.projection_dropout,
            pos_conv_kernel=encoder_config.pos_conv_kernel,
            pos_conv_groups=encoder_config.pos_conv_groups,
            num_layers=encoder_config.num_layers,
            num_heads=encoder_config.num_heads,
            attention_dropout=encoder_config.attention_dropout,
            ff_interm_features=encoder_config.ff_interm_features,
            ff_interm_dropout=encoder_config.ff_interm_dropout,
            dropout=encoder_config.dropout,
            layer_norm_first=encoder_config.layer_norm_first,
            layer_drop=encoder_config.layer_drop,
        )
        # encoder output shape: (batch, seq_len, embed_dim)

        super().__init__(cnn, encoder)

        if self.config.global_pool.lower() == "max":
            pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.global_pool.lower() == "avg":
            pool = nn.AdaptiveAvgPool1d((1,))
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")

        self.pool = nn.Sequential(
            Rearrange("batch seqlen chan -> batch chan seqlen"),
            pool,
            Rearrange("batch chan seqlen -> batch (chan seqlen)"),
        )

        self.clf = MLP(
            in_channels=encoder_config.embed_dim,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        self.extra_heads = MultiTaskHead(
            in_channels=self.clf.in_channels,
            config=_config,
        )
        if self.extra_heads.empty:
            self.extra_heads = None

        if "wav2vec2" in cnn_choice:
            self.squeeze = Rearrange("batch 1 seqlen -> batch seqlen")

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if "unannotated" counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        freeze the backbone (feature_extractor and encoder) of the model

        Parameters
        ----------
        freeze: bool, default True,
            whether to freeze the backbone

        """
        for params in self.feature_extractor.parameters():
            params.requires_grad = not freeze
        for params in self.encoder.parameters():
            params.requires_grad = not freeze

    def forward(
        self,
        waveforms: Tensor,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """

        Parameters
        ----------
        waveforms: Tensor,
            shape: (batch, num_channels, seq_len)
        labels: dict of Tensor, optional,
            the labels of the input data, including:
            - "murmur": the murmur labels, shape: (batch_size, n_classes) or (batch_size,)
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

        if self.squeeze is not None:
            waveforms = self.squeeze(waveforms)
        if "wav2vec2" in self.config.cnn.name.lower():
            features, _ = self.feature_extractor(waveforms, seq_len)
        else:
            features = self.feature_extractor(waveforms)
        features = self.encoder(features)
        pooled_features = self.pool(features)

        pred = self.clf(pooled_features)

        if self.extra_heads is not None:
            out = self.extra_heads(
                features.permute(0, 2, 1), pooled_features, seq_len, labels
            )
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


class HFWav2Vec2_CINC2022(nn.Module, CkptMixin, SizeMixin):
    """ """

    __name__ = "HFWav2Vec2_CINC2022"

    def __init__(self, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        super().__init__()
        if config is None:
            _config = deepcopy(ModelCfg.classification)
        else:
            _config = deepcopy(config)
        self.config = _config[_config.model_name]
        assert "backbone" in self.config, "`backbone` is a required key in config"
        if self.config["backbone"] not in HFPreTrainModelCfg.list_models():
            assert self.config["backbone"] in self.config["backbone_cfg"], (
                f"backbone {self.config['backbone']} is not in the list of built-in models, "
                f"one has to provide a config in `config.backbone_cfg['backbone']` "
                "for it to register a new backbone"
            )
            HFPreTrainModelCfg.register_model(
                self.config["backbone"],
                self.config["backbone_cfg"][self.config["backbone"]],
            )
        HFPreTrainModelCfg.change_model(self.config["backbone"])
        self.backbone_config = HFPreTrainModelCfg.get_Wav2Vec2Config()
        self.backbone = HFWav2Vec2Model(self.backbone_config)

        if "cnn" in self.config:
            self.config.pop("cnn", None)
        if "rnn" in self.config:
            self.config.pop("rnn", None)
        if "attn" in self.config:
            self.config.pop("attn", None)
        self.classes = deepcopy(_config.classes)
        self.n_classes = len(_config.classes)

        self.outcomes = _config.outcomes
        self.states = _config.states

        self.squeeze = Rearrange("batch 1 seqlen -> batch seqlen")

        self.dropout = nn.Dropout(self.backbone_config.final_dropout)

        num_layers = (
            self.backbone_config.num_hidden_layers + 1
        )  # transformer layers + input embeddings
        if self.backbone_config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        if self.config.global_pool.lower() == "max":
            pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.global_pool.lower() == "avg":
            pool = nn.AdaptiveAvgPool1d((1,))
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")

        self.pool = nn.Sequential(
            Rearrange("batch seqlen chan -> batch chan seqlen"),
            pool,
            Rearrange("batch chan seqlen -> batch (chan seqlen)"),
        )

        self.clf = MLP(
            in_channels=self.backbone_config.hidden_size,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        self.extra_heads = MultiTaskHead(
            in_channels=self.clf.in_channels,
            config=_config,
        )
        if self.extra_heads.empty:
            self.extra_heads = None

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if "unannotated" counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.

        Parameters
        ----------
        freeze: bool, default True,
            whether to freeze the feature extractor

        """
        for param in self.backbone.feature_extractor.parameters():
            param.requires_grad = not freeze
        self.backbone.feature_extractor._requires_grad = not freeze

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        freeze the backbone (feature_extractor and encoder) of the model

        Parameters
        ----------
        freeze: bool, default True,
            whether to freeze the backbone

        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def load_pretrained_backbone(
        self, path_or_model: Union[str, Path, HFWav2Vec2PreTrainedModel]
    ) -> None:
        """
        load the pretrained backbone from a given path

        Parameters
        ----------
        path_or_model: str or Path or transformers.Wav2Vec2PreTrainedModel,
            path to the pretrained backbone model, or the backbone model itself,
            which is a `transformers` `Wav2Vec2Model` (or related classes) checkpoint

        """
        if isinstance(path_or_model, (str, Path)):
            state_dict = self.backbone.__class__.from_pretrained(
                path_or_model
            ).state_dict()
        elif isinstance(path_or_model, HFWav2Vec2PreTrainedModel):
            state_dict = path_or_model.state_dict()
        else:
            raise ValueError(f"{path_or_model} is not a valid path or model")
        self.backbone.load_state_dict(state_dict)
        del state_dict

    def forward(
        self,
        waveforms: Tensor,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """

        Parameters
        ----------
        waveforms: Tensor,
            shape: (batch, num_channels, seq_len)
        labels: dict of Tensor, optional,
            the labels of the input data, including:
            - "murmur": the murmur labels, shape: (batch_size, n_classes) or (batch_size,)
            - "outcome": the outcome labels, of shape (batch_size, n_outcomes) or (batch_size,)
            - "segmentation": the segmentation labels, of shape (batch_size, seq_len, n_states)

        Returns
        -------
        dict of Tensor, with items:
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        batch_size, channels, seq_len = waveforms.shape

        waveforms = self.squeeze(waveforms)
        backbone_outputs = self.backbone(waveforms)

        if self.backbone_config.use_weighted_layer_sum:
            features = backbone_outputs[_HF_HIDDEN_STATES_START_POSITION]
            features = torch.stack(features, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            features = (features * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            features = backbone_outputs[0]
        # features of shape (batch_size, seq_len, n_channels)

        pooled_features = self.pool(features)

        pred = self.clf(pooled_features)

        if self.extra_heads is not None:
            out = self.extra_heads(
                features.permute(0, 2, 1), pooled_features, seq_len, labels
            )
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
