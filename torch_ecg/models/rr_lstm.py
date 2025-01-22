"""
AF (and perhaps other arrhythmias like preamature beats) detection
using rr time series as input and using lstm as model.

References
----------
1. https://github.com/al3xsh/rnn-based-af-detection

"""

import warnings
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..cfg import CFG
from ..components.outputs import BaseOutput
from ..model_configs.rr_lstm import RR_LSTM_CONFIG
from ..models._nets import MLP, ExtendedCRF, GlobalContextBlock, NonLocalBlock, SEBlock, SelfAttention, StackedLSTM
from ..utils.misc import CitationMixin
from ..utils.utils_nn import CkptMixin, SizeMixin

__all__ = [
    "RR_LSTM",
]


class RR_LSTM(nn.Module, CkptMixin, SizeMixin, CitationMixin):
    """LSTM model for RR time series classification or sequence labeling.

    LSTM model using RR time series as input is studied in
    [:footcite:ct:`faust2018automated`] for atrial fibrillation detection.
    It is further improved in [:footcite:ct:`af_detection`] via
    incorporating attention mechanism and conditional random fields.

    Parameters
    ----------
    classes : List[str]
        List of the names of the classes.
    config : dict
        Other hyper-parameters, including kernel sizes, etc.
        Refer to corresponding config file for details.


    .. footbibliography::

    """

    __name__ = "RR_LSTM"

    def __init__(self, classes: Sequence[str], config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.config = deepcopy(RR_LSTM_CONFIG)
        if not config:
            warnings.warn("No config is provided, using default config.", RuntimeWarning)
        self.config.update(deepcopy(config) or {})
        self.config.batch_first = self.config.get("batch_first", False)

        if self.config.batch_first:
            self.in_rearrange = Rearrange("batch_size n_channels seq_len -> seq_len batch_size n_channels")
        else:
            self.in_rearrange = nn.Identity()

        self.lstm = StackedLSTM(
            input_size=1,
            hidden_sizes=self.config.lstm.hidden_sizes,
            bias=self.config.lstm.bias,
            dropouts=self.config.lstm.dropouts,
            bidirectional=self.config.lstm.bidirectional,
            return_sequences=self.config.lstm.retseq,
        )

        attn_input_size = self.lstm.compute_output_shape(None, None)[-1]

        if not self.config.lstm.retseq:
            if self.config.attn.name.lower() != "none":
                warnings.warn(
                    "Attention is not supported when lstm is not returning sequences",
                    RuntimeWarning,
                )
                self.config.attn.name = "none"
            self.attn_in_rearrange = nn.Identity()
            self.attn = nn.Identity()
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "none":
            self.attn_in_rearrange = Rearrange("seq_len batch_size n_channels -> batch_size n_channels seq_len")
            self.attn = nn.Identity()
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # non_local
            self.attn_in_rearrange = Rearrange("seq_len batch_size n_channels -> batch_size n_channels seq_len")
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,
                filter_lengths=self.config.attn.nl.filter_lengths,
                subsample_length=self.config.attn.nl.subsample_length,
                batch_norm=self.config.attn.nl.batch_norm,
            )
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # squeeze_exitation
            self.attn_in_rearrange = Rearrange("seq_len batch_size n_channels -> batch_size n_channels seq_len")
            self.attn = SEBlock(
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # global_context
            self.attn_in_rearrange = Rearrange("seq_len batch_size n_channels -> batch_size n_channels seq_len")
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,
                ratio=self.config.attn.gc.ratio,
                reduction=self.config.attn.gc.reduction,
                pooling_type=self.config.attn.gc.pooling_type,
                fusion_types=self.config.attn.gc.fusion_types,
            )
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # self_attention
            # NOTE: this branch NOT tested
            self.attn_in_rearrange = nn.Identity()
            self.attn = SelfAttention(
                embed_dim=attn_input_size,
                num_heads=self.config.attn.sa.get("num_heads", self.config.attn.sa.get("head_num")),
                dropout=self.config.attn.sa.dropout,
                bias=self.config.attn.sa.bias,
            )
            self.attn_out_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"Attn module \042{self.config.attn.name}\042 not implemented yet.")

        if not self.config.lstm.retseq:
            if self.config.global_pool.lower() != "none":
                warnings.warn(
                    f"Global pooling \042{self.config.global_pool}\042 is ignored in non-sequence mode",
                    RuntimeWarning,
                )
                self.config.global_pool = "none"
            self.pool = nn.Identity()
            self.pool_rearrange = nn.Identity()
            if self.config.clf.name.lower() == "crf":
                warnings.warn(
                    "CRF layer is not supported in non-sequence mode, using linear instead.",
                    RuntimeWarning,
                )
                self.config.clf.name = "linear"
                assert "linear" in self.config.clf, "Linear layer not defined in `config`."
            self.clf = MLP(
                in_channels=clf_input_size,
                out_channels=self.config.clf.linear.out_channels + [self.n_classes],
                activation=self.config.clf.linear.activation,
                bias=self.config.clf.linear.bias,
                dropouts=self.config.clf.linear.dropouts,
                skip_last_activation=True,
            )
        elif self.config.clf.name.lower() == "linear":
            if self.config.global_pool.lower() == "max":
                self.pool = nn.AdaptiveMaxPool1d((self.config.global_pool_size,), return_indices=False)
                self.pool_rearrange = Rearrange("batch_size channels pool_size -> batch_size (channels pool_size)")
                clf_input_size *= self.config.global_pool_size
            elif self.config.global_pool.lower() == "avg":
                self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))
                self.pool_rearrange = Rearrange("batch_size channels pool_size -> batch_size (channels pool_size)")
                clf_input_size *= self.config.global_pool_size
            elif self.config.global_pool.lower() == "none":
                self.pool = nn.Identity()
                self.pool_rearrange = Rearrange("batch_size n_channels seq_len -> batch_size seq_len n_channels")
            else:
                raise NotImplementedError(f"Pooling type \042{self.config.global_pool}\042 not supported")
            self.clf = MLP(
                in_channels=clf_input_size,
                out_channels=self.config.clf.linear.out_channels + [self.n_classes],
                activation=self.config.clf.linear.activation,
                bias=self.config.clf.linear.bias,
                dropouts=self.config.clf.linear.dropouts,
                skip_last_activation=True,
            )
        elif self.config.clf.name.lower() == "crf":
            if self.config.global_pool.lower() != "none":
                warnings.warn(
                    f"Global pooling \042{self.config.global_pool}\042 is ignored for CRF prediction head.",
                    RuntimeWarning,
                )
                self.config.global_pool = "none"
            self.pool = nn.Identity()
            self.pool_rearrange = Rearrange("batch_size n_channels seq_len -> batch_size seq_len n_channels")
            self.clf = ExtendedCRF(
                in_channels=clf_input_size,
                num_tags=self.n_classes,
                bias=self.config.clf.crf.proj_bias,
            )

        # for inference, except for crf
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor
            Input RR series tensor
            of shape ``(seq_len, batch_size, n_channels)``,
            or ``(batch_size, n_channels, seq_len)``
            if `config.batch_first` is True.

        Returns
        -------
        torch.Tensor
            Output tensor,
            of shape ``(batch_size, seq_len, n_classes)``
            or ``(batch_size, n_classes)``.

        """
        x = self.in_rearrange(input)
        x = self.lstm(x)

        x = self.attn_in_rearrange(x)
        x = self.attn(x)
        x = self.attn_out_rearrange(x)

        x = self.pool(x)
        x = self.pool_rearrange(x)

        output = self.clf(x)

        return output

    @torch.no_grad()
    def inference(self, input: Tensor, bin_pred_thr: float = 0.5) -> BaseOutput:
        """Inference method for the model."""
        raise NotImplementedError("Implement a task-specific inference method.")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input series tensor.
        batch_size : int, optional
            Batch size of the input series tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the model.

        """
        if (not self.config.lstm.retseq) or (self.config.global_pool.lower() != "none"):
            output_shape = (batch_size, self.n_classes)
        else:
            output_shape = (batch_size, seq_len, self.n_classes)
        return output_shape

    @property
    def doi(self) -> List[str]:
        return list(
            set(
                self.config.get("doi", [])
                + [
                    "10.1162/neco.1997.9.8.1735",
                    "10.1016/j.compbiomed.2018.07.001",
                    "10.1142/s0219519421400212",
                ]
            )
        )

    @classmethod
    def from_v1(
        cls, v1_ckpt: str, device: Optional[torch.device] = None, return_config: bool = False
    ) -> Union["RR_LSTM", Tuple["RR_LSTM", dict]]:
        """Restore an instance of the model from a v1 checkpoint.

        Parameters
        ----------
        v1_ckpt : str
            Path to the v1 checkpoint file.
        device : torch.device, optional
            The device to load the model to.
            Defaults to "cuda" if available, otherwise "cpu".
        return_config : bool, default False
            Whether to return the config dict.

        Returns
        -------
        model : RR_LSTM
            The model instance restored from the v1 checkpoint.

        """
        v1_model, train_config = RR_LSTM_v1.from_checkpoint(v1_ckpt, device=device)
        model = cls(classes=v1_model.classes, config=v1_model.config)
        model = model.to(v1_model.device)
        model.lstm.load_state_dict(v1_model.lstm.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(v1_model.attn.state_dict())
        model.clf.load_state_dict(v1_model.clf.state_dict())
        del v1_model
        if return_config:
            return model, train_config
        return model


class RR_LSTM_v1(nn.Module, CkptMixin, SizeMixin, CitationMixin):
    """LSTM model for RR time series classification or sequence labeling.

    LSTM model using RR time series as input is studied in [1]_ for
    atrial fibrillation detection. It is further improved in [2]_ via
    incorporating attention mechanism and conditional random fields.

    Parameters
    ----------
    classes : List[str]
        List of the names of the classes.
    config : dict
        Other hyper-parameters, including kernel sizes, etc.
        Refer to corresponding config file for details.

    References
    ----------
    .. [1] Faust, Oliver, et al. "Automated detection of atrial fibrillation
           using long short-term memory network with RR interval signals."
           Computers in biology and medicine 102 (2018): 327-335.
    .. [2] Wen, Hao, et al. "A Scalable Hybrid Model for Atrial Fibrillation Detection."
           Journal of Mechanics in Medicine and Biology 21.05 (2021): 2140021.

    """

    __name__ = "RR_LSTM_v1"

    def __init__(self, classes: Sequence[str], config: Optional[CFG] = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        classes: list,
            list of the classes for classification
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.config = deepcopy(RR_LSTM_CONFIG)
        if not config:
            warnings.warn("No config is provided, using default config.", RuntimeWarning)
        self.config.update(deepcopy(config) or {})

        self.lstm = StackedLSTM(
            input_size=1,
            hidden_sizes=self.config.lstm.hidden_sizes,
            bias=self.config.lstm.bias,
            dropouts=self.config.lstm.dropouts,
            bidirectional=self.config.lstm.bidirectional,
            return_sequences=self.config.lstm.retseq,
        )

        attn_input_size = self.lstm.compute_output_shape(None, None)[-1]

        if not self.config.lstm.retseq:
            if self.config.attn.name.lower() != "none":
                warnings.warn(
                    "Attention is not supported when lstm is not returning sequences",
                    RuntimeWarning,
                )
                self.config.attn.name = "none"
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "none":
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # non_local
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,
                filter_lengths=self.config.attn.nl.filter_lengths,
                subsample_length=self.config.attn.nl.subsample_length,
                batch_norm=self.config.attn.nl.batch_norm,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # squeeze_exitation
            self.attn = SEBlock(
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # global_context
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,
                ratio=self.config.attn.gc.ratio,
                reduction=self.config.attn.gc.reduction,
                pooling_type=self.config.attn.gc.pooling_type,
                fusion_types=self.config.attn.gc.fusion_types,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # self_attention
            # NOTE: this branch NOT tested
            self.attn = SelfAttention(
                embed_dim=attn_input_size,
                num_heads=self.config.attn.sa.get("num_heads", self.config.attn.sa.get("head_num")),
                dropout=self.config.attn.sa.dropout,
                bias=self.config.attn.sa.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"Attn module \042{self.config.attn.name}\042 not implemented yet.")

        if not self.config.lstm.retseq:
            if self.config.global_pool.lower() != "none":
                warnings.warn(
                    f"Global pooling \042{self.config.global_pool}\042 is ignored in non-sequence mode",
                    RuntimeWarning,
                )
                self.config.global_pool = "none"
            self.pool = None
            if self.config.clf.name.lower() == "crf":
                warnings.warn(
                    "CRF layer is not supported in non-sequence mode, using linear instead.",
                    RuntimeWarning,
                )
                self.config.clf.name = "linear"
                assert "linear" in self.config.clf, "Linear layer not defined in `config`."
            self.clf = MLP(
                in_channels=clf_input_size,
                out_channels=self.config.clf.linear.out_channels + [self.n_classes],
                activation=self.config.clf.linear.activation,
                bias=self.config.clf.linear.bias,
                dropouts=self.config.clf.linear.dropouts,
                skip_last_activation=True,
            )
        elif self.config.clf.name.lower() == "linear":
            if self.config.global_pool.lower() == "max":
                self.pool = nn.AdaptiveMaxPool1d((self.config.global_pool_size,), return_indices=False)
                clf_input_size *= self.config.global_pool_size
            elif self.config.global_pool.lower() == "avg":
                self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))
                clf_input_size *= self.config.global_pool_size
            elif self.config.global_pool.lower() == "none":
                self.pool = None
            else:
                raise NotImplementedError(f"Pooling type \042{self.config.global_pool}\042 not supported")
            self.clf = MLP(
                in_channels=clf_input_size,
                out_channels=self.config.clf.linear.out_channels + [self.n_classes],
                activation=self.config.clf.linear.activation,
                bias=self.config.clf.linear.bias,
                dropouts=self.config.clf.linear.dropouts,
                skip_last_activation=True,
            )
        elif self.config.clf.name.lower() == "crf":
            if self.config.global_pool.lower() != "none":
                warnings.warn(
                    f"Global pooling \042{self.config.global_pool}\042 is ignored for CRF prediction head.",
                    RuntimeWarning,
                )
                self.config.global_pool = "none"
            self.pool = None
            self.clf = ExtendedCRF(
                in_channels=clf_input_size,
                num_tags=self.n_classes,
                bias=self.config.clf.crf.proj_bias,
            )

        # for inference, except for crf
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor
            Input RR series tensor
            of shape ``(seq_len, batch_size, n_channels)``,
            or ``(batch_size, n_channels, seq_len)``
            if `config.batch_first` is True.

        Returns
        -------
        torch.Tensor
            Output tensor,
            of shape ``(batch_size, seq_len, n_classes)``
            or ``(batch_size, n_classes)``.

        """
        # (batch_size, n_channels, seq_len) --> (seq_len, batch_size, n_channels)
        # x = input.permute(1,2,0)
        x = self.lstm(input)  # (seq_len, batch_size, n_channels) or (batch_size, n_channels)
        if self.attn:
            # (seq_len, batch_size, n_channels) --> (batch_size, n_channels, seq_len)
            x = x.permute(1, 2, 0)
            x = self.attn(x)  # (batch_size, n_channels, seq_len)
        elif x.ndim == 3:
            # (seq_len, batch_size, n_channels) --> (batch_size, n_channels, seq_len)
            x = x.permute(1, 2, 0)
        if self.pool:
            x = self.pool(x)  # (batch_size, n_channels, 1)
            # x = x.squeeze(dim=-1)  # (batch_size, n_channels)
            x = rearrange(
                x,
                "batch_size n_channels pool_size -> batch_size (n_channels pool_size)",
            )
        elif x.ndim == 3:
            x = x.permute(0, 2, 1)  # (batch_size, n_channels, seq_len) --> (batch_size, seq_len, n_channels)
        else:
            # x of shape (batch_size, n_channels),
            # in the case where config.lstm.retseq = False
            pass
        if self.config.clf.name.lower() == "linear":
            x = self.clf(x)  # (batch_size, seq_len, n_classes) or (batch_size, n_classes)
        elif self.config.clf.name.lower() == "crf":
            x = self.clf(x)  # (batch_size, seq_len, n_classes)
        output = x

        return output

    @torch.no_grad()
    def inference(self, input: Tensor, bin_pred_thr: float = 0.5) -> BaseOutput:
        """Inference method for the model."""
        raise NotImplementedError("Implement a task-specific inference method.")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input series tensor.
        batch_size : int, optional
            Batch size of the input series tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the model.

        """
        if (not self.config.lstm.retseq) or (self.config.global_pool.lower() != "none"):
            output_shape = (batch_size, self.n_classes)
        else:
            output_shape = (batch_size, seq_len, self.n_classes)
        return output_shape

    @property
    def doi(self) -> List[str]:
        return list(
            set(
                self.config.get("doi", [])
                + [
                    "10.1162/neco.1997.9.8.1735",
                    "10.1016/j.compbiomed.2018.07.001",
                    "10.1142/s0219519421400212",
                ]
            )
        )
