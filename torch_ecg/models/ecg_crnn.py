"""
C(R)NN structure models, for classifying ECG arrhythmias, and other tasks.
"""

import os
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from ..cfg import CFG
from ..components.outputs import BaseOutput
from ..model_configs.ecg_crnn import ECG_CRNN_CONFIG
from ..utils.misc import CitationMixin
from ..utils.utils_nn import CkptMixin, SizeMixin
from ._nets import MLP, GlobalContextBlock, NonLocalBlock, SEBlock, SelfAttention, StackedLSTM
from .cnn.densenet import DenseNet
from .cnn.mobilenet import MobileNetV1, MobileNetV2, MobileNetV3
from .cnn.multi_scopic import MultiScopicCNN
from .cnn.regnet import RegNet
from .cnn.resnet import ResNet
from .cnn.vgg import VGG16
from .cnn.xception import Xception
from .transformers import Transformer

__all__ = [
    "ECG_CRNN",
]


class ECG_CRNN(nn.Module, CkptMixin, SizeMixin, CitationMixin):
    """Convolutional (Recurrent) Neural Network for ECG tasks.

    This C(R)NN architecture is adapted from [:footcite:ct:`yao2018ti_cnn,yao2020ati_cnn`]
    in the first place,and then modified to be more general, and more flexible.
    The most famous model is perhaps [:footcite:ct:`awni2019stanford_ecg`],
    which is a modified 1D-ResNet34 model. The website of this model is
    `<https://stanfordmlgroup.github.io/projects/ecg2/>`_, and the code is hosted on
    `<https://github.com/awni/ecg>`_.

    The C(R)NN models have long been competitive in various ECG tasks,
    e.g. CPSC2018 entry 0236, CPSC2019 entry 0416.
    The models are also used in the PhysioNet/CinC Challenges.

    Parameters
    ----------
    classes : List[str]
        List of the names of the classes.
    n_leads : int
        Number of leads (number of input channels).
    config : dict
        Other hyper-parameters, including kernel sizes, etc.
        Refer to corresponding config files.


    .. footbibliography::

    """

    __name__ = "ECG_CRNN"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = n_leads
        self.config = deepcopy(ECG_CRNN_CONFIG)
        if not config:
            warnings.warn("No config is provided, using default config.", RuntimeWarning)
        self.config.update(deepcopy(config) or {})

        cnn_choice = self.config.cnn.name.lower()  # type: ignore
        cnn_config = self.config.cnn[self.config.cnn.name]  # type: ignore
        if "resnet" in cnn_choice or "resnext" in cnn_choice:
            self.cnn = ResNet(self.n_leads, **cnn_config)
        elif "regnet" in cnn_choice:
            self.cnn = RegNet(self.n_leads, **cnn_config)
        elif "multi_scopic" in cnn_choice:
            self.cnn = MultiScopicCNN(self.n_leads, **cnn_config)
        elif "mobile_net" in cnn_choice or "mobilenet" in cnn_choice:
            if "v1" in cnn_choice:
                self.cnn = MobileNetV1(self.n_leads, **cnn_config)
            elif "v2" in cnn_choice:
                self.cnn = MobileNetV2(self.n_leads, **cnn_config)
            elif "v3" in cnn_choice:
                self.cnn = MobileNetV3(self.n_leads, **cnn_config)
            else:
                raise ValueError(f"CNN \042{cnn_choice}\042 is not supported for {self.__name__}")
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            self.cnn = DenseNet(self.n_leads, **cnn_config)
        elif "vgg16" in cnn_choice:
            self.cnn = VGG16(self.n_leads, **cnn_config)
        elif "xception" in cnn_choice:
            self.cnn = Xception(self.n_leads, **cnn_config)
        else:
            raise NotImplementedError(f"CNN \042{cnn_choice}\042 not implemented yet")
        rnn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.config.rnn.name.lower() == "none":  # type: ignore
            self.rnn_in_rearrange = Rearrange("batch_size channels seq_len -> seq_len batch_size channels")
            self.rnn = nn.Identity()
            self.__rnn_seqlen_dim = 0
            self.rnn_out_rearrange = nn.Identity()
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":  # type: ignore
            self.rnn_in_rearrange = Rearrange("batch_size channels seq_len -> seq_len batch_size channels")
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,  # type: ignore
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,  # type: ignore
                bias=self.config.rnn.lstm.bias,  # type: ignore
                dropouts=self.config.rnn.lstm.dropouts,  # type: ignore
                bidirectional=self.config.rnn.lstm.bidirectional,  # type: ignore
                return_sequences=self.config.rnn.lstm.retseq,  # type: ignore
            )
            self.__rnn_seqlen_dim = 0
            self.rnn_out_rearrange = nn.Identity()
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        elif self.config.rnn.name.lower() == "linear":  # type: ignore
            # abuse of notation, to put before the global attention module
            self.rnn_in_rearrange = Rearrange("batch_size channels seq_len -> batch_size seq_len channels")
            self.rnn = MLP(
                in_channels=rnn_input_size,  # type: ignore
                out_channels=self.config.rnn.linear.out_channels,  # type: ignore
                activation=self.config.rnn.linear.activation,  # type: ignore
                bias=self.config.rnn.linear.bias,  # type: ignore
                dropouts=self.config.rnn.linear.dropouts,  # type: ignore
            )
            self.__rnn_seqlen_dim = 1
            self.rnn_out_rearrange = Rearrange("batch_size seq_len channels -> seq_len batch_size channels")
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"RNN \042{self.config.rnn.name}\042 not implemented yet")  # type: ignore

        # attention
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:  # type: ignore
            self.attn_in_rearrange = nn.Identity()
            self.attn = nn.Identity()
            self.__attn_seqlen_dim = 0
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = attn_input_size
            if self.config.attn.name.lower() != "none":  # type: ignore
                warnings.warn(
                    f"since `retseq` of rnn is False, hence attention `{self.config.attn.name}` is ignored",  # type: ignore
                    RuntimeWarning,
                )
        elif self.config.attn.name.lower() == "none":  # type: ignore
            self.attn_in_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            self.attn = nn.Identity()
            self.__attn_seqlen_dim = -1
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # type: ignore
            # non_local
            self.attn_in_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,  # type: ignore
                filter_lengths=self.config.attn.nl.filter_lengths,  # type: ignore
                subsample_length=self.config.attn.nl.subsample_length,  # type: ignore
                batch_norm=self.config.attn.nl.batch_norm,  # type: ignore
            )
            self.__attn_seqlen_dim = -1
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # type: ignore
            # squeeze_exitation
            self.attn_in_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            self.attn = SEBlock(
                in_channels=attn_input_size,  # type: ignore
                reduction=self.config.attn.se.reduction,  # type: ignore
                activation=self.config.attn.se.activation,  # type: ignore
                kw_activation=self.config.attn.se.kw_activation,  # type: ignore
                bias=self.config.attn.se.bias,  # type: ignore
            )
            self.__attn_seqlen_dim = -1
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # type: ignore
            # global_context
            self.attn_in_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,  # type: ignore
                ratio=self.config.attn.gc.ratio,  # type: ignore
                reduction=self.config.attn.gc.reduction,  # type: ignore
                pooling_type=self.config.attn.gc.pooling_type,  # type: ignore
                fusion_types=self.config.attn.gc.fusion_types,  # type: ignore
            )
            self.__attn_seqlen_dim = -1
            self.attn_out_rearrange = nn.Identity()
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # type: ignore
            # self_attention
            # NOTE: this branch NOT tested
            self.attn_in_rearrange = nn.Identity()
            self.attn = SelfAttention(  # type: ignore
                embed_dim=attn_input_size,
                num_heads=self.config.attn.sa.get("num_heads", self.config.attn.sa.get("head_num")),  # type: ignore
                dropout=self.config.attn.sa.dropout,  # type: ignore
                bias=self.config.attn.sa.bias,  # type: ignore
            )
            self.__attn_seqlen_dim = 0
            self.attn_out_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        elif self.config.attn.name.lower() == "transformer":  # type: ignore
            self.attn = Transformer(
                input_size=attn_input_size,  # type: ignore
                hidden_size=self.config.attn.transformer.hidden_size,  # type: ignore
                num_layers=self.config.attn.transformer.num_layers,  # type: ignore
                num_heads=self.config.attn.transformer.num_heads,  # type: ignore
                dropout=self.config.attn.transformer.dropout,  # type: ignore
                activation=self.config.attn.transformer.activation,  # type: ignore
            )
            if self.attn.batch_first:
                self.attn_in_rearrange = Rearrange("seq_len batch_size channels -> batch_size seq_len channels")
                self.attn_out_rearrange = Rearrange("batch_size seq_len channels -> batch_size channels seq_len")
                self.__attn_seqlen_dim = 1
            else:
                self.attn_in_rearrange = nn.Identity()
                self.attn_out_rearrange = Rearrange("seq_len batch_size channels -> batch_size channels seq_len")
                self.__attn_seqlen_dim = 0
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"Attention \042{self.config.attn.name}\042 not implemented yet")  # type: ignore

        # global pooling
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:  # type: ignore
            self.pool = nn.Identity()
            if self.config.global_pool.lower() != "none":  # type: ignore
                warnings.warn(
                    f"since `retseq` of rnn is False, hence global pooling `{self.config.global_pool}` is ignored",  # type: ignore
                    RuntimeWarning,
                )
            self.pool_rearrange = nn.Identity()
            self.__clf_input_seq = False
        elif self.config.global_pool.lower() == "max":  # type: ignore
            self.pool = nn.AdaptiveMaxPool1d((self.config.global_pool_size,), return_indices=False)  # type: ignore
            clf_input_size *= self.config.global_pool_size  # type: ignore
            self.pool_rearrange = Rearrange("batch_size channels pool_size -> batch_size (channels pool_size)")
            self.__clf_input_seq = False
        elif self.config.global_pool.lower() == "avg":  # type: ignore
            self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))  # type: ignore
            clf_input_size *= self.config.global_pool_size  # type: ignore
            self.pool_rearrange = Rearrange("batch_size channels pool_size -> batch_size (channels pool_size)")
            self.__clf_input_seq = False
        elif self.config.global_pool.lower() == "attn":  # type: ignore
            raise NotImplementedError("Attentive pooling not implemented yet!")
        elif self.config.global_pool.lower() == "none":  # type: ignore
            self.pool = nn.Identity()
            self.pool_rearrange = Rearrange("batch_size channels seq_len -> batch_size seq_len channels")
            self.__clf_input_seq = True
        else:
            raise NotImplementedError(f"Global Pooling \042{self.config.global_pool}\042 not implemented yet!")  # type: ignore

        # input of `self.clf` has shape: batch_size, channels
        self.clf = MLP(
            in_channels=clf_input_size,  # type: ignore
            out_channels=self.config.clf.out_channels + [self.n_classes],  # type: ignore
            activation=self.config.clf.activation,  # type: ignore
            bias=self.config.clf.bias,  # type: ignore
            dropouts=self.config.clf.dropouts,  # type: ignore
            skip_last_activation=True,
        )

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if background counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def extract_features(self, input: Tensor) -> Tensor:
        """Extract feature map before the
        dense (linear) classifying layer(s).

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        features : torch.Tensor
            Feature map tensor,
            of shape ``(batch_size, channels, seq_len)``
            or ``(batch_size, channels)``.

        """
        # CNN
        features = self.cnn(input)  # batch_size, channels, seq_len

        # RNN (optional)
        features = self.rnn_in_rearrange(features)
        features = self.rnn(features)
        features = self.rnn_out_rearrange(features)

        # Attention (optional)
        features = self.attn_in_rearrange(features)
        features = self.attn(features)
        features = self.attn_out_rearrange(features)

        return features

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        pred : torch.Tensor
            Predictions tensor,
            of shape ``(batch_size, seq_len, channels)``
            or ``(batch_size, channels)``.

        """
        features = self.extract_features(input)

        # global pooling (optional)
        features = self.pool(features)
        features = self.pool_rearrange(features)

        pred = self.clf(features)

        return pred

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> BaseOutput:
        """Inference method for the model.

        Parameters
        ----------
        input : numpy.ndarray or torch.Tensor
            Input tensor, of shape ``(batch_size, channels, seq_len)``.
        class_names : bool, default False
            If True, the returned scalar predictions will be
            a :class:`~pandas.DataFrame`,
            with class names for each scalar prediction.
        bin_pred_thr : float, default 0.5
            Threshold for making binary predictions from scalar predictions.

        Returns
        -------
        output : BaseOutput
            The output of the inference method, including the following items:

                - prob: numpy.ndarray or torch.Tensor,
                  scalar predictions, (and binary predictions if `class_names` is True).
                - pred: numpy.ndarray or torch.Tensor,
                  the array (with values 0, 1 for each class) of binary prediction.

        """
        raise NotImplementedError("Implement a task-specific inference method.")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input signal tensor.
        batch_size : int, optional
            Batch size of the input signal tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the model.

        """
        output_shape = self.cnn.compute_output_shape(seq_len, batch_size)
        _, _, _seq_len = output_shape
        if self.rnn.__class__.__name__ != "Identity":
            output_shape = self.rnn.compute_output_shape(_seq_len, batch_size)
            _seq_len = output_shape[self.__rnn_seqlen_dim]
        if self.attn.__class__.__name__ != "Identity":
            output_shape = self.attn.compute_output_shape(_seq_len, batch_size)
            _seq_len = output_shape[self.__attn_seqlen_dim]
        if self.clf.__class__.__name__ != "Identity":
            output_shape = self.clf.compute_output_shape(_seq_len, batch_size, input_seq=self.__clf_input_seq)
        return output_shape

    @property
    def doi(self) -> List[str]:
        doi = []
        candidates = [self.config]
        while len(candidates) > 0:
            new_candidates = []
            for candidate in candidates:
                if hasattr(candidate, "doi"):
                    if isinstance(candidate.doi, str):  # type: ignore
                        doi.append(candidate.doi)  # type: ignore
                    else:
                        doi.extend(list(candidate.doi))  # type: ignore
                for k, v in candidate.items():
                    if isinstance(v, CFG):
                        new_candidates.append(v)
            candidates = new_candidates
        doi = list(set(doi + ["10.1016/j.inffus.2019.06.024", "10.1088/1361-6579/ac6aa3"]))
        return doi

    @classmethod
    def from_v1(
        cls, v1_ckpt: Union[str, bytes, os.PathLike], device: Optional[torch.device] = None, return_config: bool = False
    ) -> Union["ECG_CRNN", Tuple["ECG_CRNN", dict]]:
        """Restore an instance of the model from a v1 checkpoint.

        Parameters
        ----------
        v1_ckpt : path_like
            Path to the v1 checkpoint file.
        device : torch.device, optional
            The device to load the model to.
            Defaults to "cuda" if available, otherwise "cpu".
        return_config : bool, default False
            Whether to return the config dict.

        Returns
        -------
        model : ECG_CRNN
            The model instance restored from the v1 checkpoint.
        config : dict
            The config dict. (if `return_config` is `True`)

        """
        v1_model, train_config = ECG_CRNN_v1.from_checkpoint(v1_ckpt, device=device)
        model = cls(classes=v1_model.classes, n_leads=v1_model.n_leads, config=v1_model.config)
        model = model.to(v1_model.device)
        model.cnn.load_state_dict(v1_model.cnn.state_dict())
        if model.rnn.__class__.__name__ != "Identity":
            model.rnn.load_state_dict(v1_model.rnn.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(v1_model.attn.state_dict())
        model.clf.load_state_dict(v1_model.clf.state_dict())
        del v1_model
        if return_config:
            return model, train_config
        return model


class ECG_CRNN_v1(nn.Module, CkptMixin, SizeMixin, CitationMixin):
    """Convolutional (Recurrent) Neural Network for ECG tasks.

    This C(R)NN architecture is adapted from [1]_, [2]_ in the first place,
    and then modified to be more general, and more flexible.
    The most famous model is perhaps [3]_, which is a modified 1D-ResNet34 model.
    The website of this model is [4]_, and the code is hosted on [5]_.
    The C(R)NN models have long been competitive in various ECG tasks,
    e.g. [6]_, [7]_. The models are also used in the PhysioNet/CinC Challenges.

    Parameters
    ----------
    classes : List[str]
        List of the names of the classes.
    n_leads : int
        Number of leads (number of input channels).
    config : dict
        Other hyper-parameters, including kernel sizes, etc.
        Refer to corresponding config files.

    References
    ----------
    .. [1] Yao, Qihang, et al.
           "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram."
           2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing,
           16th Intl Conf on Pervasive Intelligence and Computing,
           4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress
           (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    .. [2] Yao, Qihang, et al.
           "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based
           Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    .. [3] Hannun, Awni Y., et al.
           "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms
           using a deep neural network." Nature medicine 25.1 (2019): 65.
    .. [4] https://stanfordmlgroup.github.io/projects/ecg2/
    .. [5] https://github.com/awni/ecg
    .. [6] CPSC2018 entry 0236
    .. [7] CPSC2019 entry 0416

    """

    __name__ = "ECG_CRNN_v1"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = n_leads
        self.config = deepcopy(ECG_CRNN_CONFIG)
        if not config:
            warnings.warn("No config is provided, using default config.", RuntimeWarning)
        self.config.update(deepcopy(config) or {})

        cnn_choice = self.config.cnn.name.lower()  # type: ignore
        cnn_config = self.config.cnn[self.config.cnn.name]  # type: ignore
        if "resnet" in cnn_choice or "resnext" in cnn_choice:
            self.cnn = ResNet(self.n_leads, **cnn_config)
        elif "regnet" in cnn_choice:
            self.cnn = RegNet(self.n_leads, **cnn_config)
        elif "multi_scopic" in cnn_choice:
            self.cnn = MultiScopicCNN(self.n_leads, **cnn_config)
        elif "mobile_net" in cnn_choice or "mobilenet" in cnn_choice:
            if "v1" in cnn_choice:
                self.cnn = MobileNetV1(self.n_leads, **cnn_config)
            elif "v2" in cnn_choice:
                self.cnn = MobileNetV2(self.n_leads, **cnn_config)
            elif "v3" in cnn_choice:
                self.cnn = MobileNetV3(self.n_leads, **cnn_config)
            else:
                raise ValueError(f"CNN \042{cnn_choice}\042 is not supported for {self.__name__}")
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            self.cnn = DenseNet(self.n_leads, **cnn_config)
        elif "vgg16" in cnn_choice:
            self.cnn = VGG16(self.n_leads, **cnn_config)
        elif "xception" in cnn_choice:
            self.cnn = Xception(self.n_leads, **cnn_config)
        else:
            raise NotImplementedError(f"CNN \042{cnn_choice}\042 not implemented yet")
        rnn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.config.rnn.name.lower() == "none":  # type: ignore
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":  # type: ignore
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,  # type: ignore
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,  # type: ignore
                bias=self.config.rnn.lstm.bias,  # type: ignore
                dropouts=self.config.rnn.lstm.dropouts,  # type: ignore
                bidirectional=self.config.rnn.lstm.bidirectional,  # type: ignore
                return_sequences=self.config.rnn.lstm.retseq,  # type: ignore
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        elif self.config.rnn.name.lower() == "linear":  # type: ignore
            # abuse of notation, to put before the global attention module
            self.rnn = MLP(
                in_channels=rnn_input_size,  # type: ignore
                out_channels=self.config.rnn.linear.out_channels,  # type: ignore
                activation=self.config.rnn.linear.activation,  # type: ignore
                bias=self.config.rnn.linear.bias,  # type: ignore
                dropouts=self.config.rnn.linear.dropouts,  # type: ignore
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"RNN \042{self.config.rnn.name}\042 not implemented yet")  # type: ignore

        # attention
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:  # type: ignore
            self.attn = None
            clf_input_size = attn_input_size
            if self.config.attn.name.lower() != "none":  # type: ignore
                warnings.warn(
                    f"since `retseq` of rnn is False, hence attention `{self.config.attn.name}` is ignored",  # type: ignore
                    RuntimeWarning,
                )
        elif self.config.attn.name.lower() == "none":  # type: ignore
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # type: ignore
            # non_local
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,  # type: ignore
                filter_lengths=self.config.attn.nl.filter_lengths,  # type: ignore
                subsample_length=self.config.attn.nl.subsample_length,  # type: ignore
                batch_norm=self.config.attn.nl.batch_norm,  # type: ignore
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # type: ignore
            # squeeze_exitation
            self.attn = SEBlock(
                in_channels=attn_input_size,  # type: ignore
                reduction=self.config.attn.se.reduction,  # type: ignore
                activation=self.config.attn.se.activation,  # type: ignore
                kw_activation=self.config.attn.se.kw_activation,  # type: ignore
                bias=self.config.attn.se.bias,  # type: ignore
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # type: ignore
            # global_context
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,  # type: ignore
                ratio=self.config.attn.gc.ratio,  # type: ignore
                reduction=self.config.attn.gc.reduction,  # type: ignore
                pooling_type=self.config.attn.gc.pooling_type,  # type: ignore
                fusion_types=self.config.attn.gc.fusion_types,  # type: ignore
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # type: ignore
            # self_attention
            # NOTE: this branch NOT tested
            self.attn = SelfAttention(  # type: ignore
                embed_dim=attn_input_size,
                num_heads=self.config.attn.sa.get("num_heads", self.config.attn.sa.get("head_num")),  # type: ignore
                dropout=self.config.attn.sa.dropout,  # type: ignore
                bias=self.config.attn.sa.bias,  # type: ignore
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        elif self.config.attn.name.lower() == "transformer":  # type: ignore
            self.attn = Transformer(
                input_size=attn_input_size,  # type: ignore
                hidden_size=self.config.attn.transformer.hidden_size,  # type: ignore
                num_layers=self.config.attn.transformer.num_layers,  # type: ignore
                num_heads=self.config.attn.transformer.num_heads,  # type: ignore
                dropout=self.config.attn.transformer.dropout,  # type: ignore
                activation=self.config.attn.transformer.activation,  # type: ignore
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError(f"Attention \042{self.config.attn.name}\042 not implemented yet")  # type: ignore

        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:  # type: ignore
            self.pool = None
            if self.config.global_pool.lower() != "none":  # type: ignore
                warnings.warn(
                    f"since `retseq` of rnn is False, hence global pooling `{self.config.global_pool}` is ignored",  # type: ignore
                    RuntimeWarning,
                )
        elif self.config.global_pool.lower() == "max":  # type: ignore
            self.pool = nn.AdaptiveMaxPool1d((self.config.global_pool_size,), return_indices=False)  # type: ignore
            clf_input_size *= self.config.global_pool_size  # type: ignore
        elif self.config.global_pool.lower() == "avg":  # type: ignore
            self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))  # type: ignore
            clf_input_size *= self.config.global_pool_size  # type: ignore
        elif self.config.global_pool.lower() == "attn":  # type: ignore
            raise NotImplementedError("Attentive pooling not implemented yet!")
        elif self.config.global_pool.lower() == "none":  # type: ignore
            self.pool = None
        else:
            raise NotImplementedError(f"Global Pooling \042{self.config.global_pool}\042 not implemented yet!")  # type: ignore

        # input of `self.clf` has shape: batch_size, channels
        self.clf = MLP(
            in_channels=clf_input_size,  # type: ignore
            out_channels=self.config.clf.out_channels + [self.n_classes],  # type: ignore
            activation=self.config.clf.activation,  # type: ignore
            bias=self.config.clf.bias,  # type: ignore
            dropouts=self.config.clf.dropouts,  # type: ignore
            skip_last_activation=True,
        )

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if background counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def extract_features(self, input: Tensor) -> Tensor:
        """Extract feature map before the
        dense (linear) classifying layer(s).

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        features : torch.Tensor
            Feature map tensor,
            of shape ``(batch_size, channels, seq_len)``
            or ``(batch_size, channels)``.

        """
        # CNN
        features = self.cnn(input)  # batch_size, channels, seq_len

        # RNN (optional)
        if self.config.rnn.name.lower() in ["lstm"]:  # type: ignore
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2, 0, 1)
            features = self.rnn(features)  # type: ignore
            # (seq_len, batch_size, channels) or (batch_size, channels)
        elif self.config.rnn.name.lower() in ["linear"]:  # type: ignore
            # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
            features = features.permute(0, 2, 1)
            # (batch_size, seq_len, channels)
            features = self.rnn(features)  # type: ignore
            # (batch_size, seq_len, channels) --> (seq_len, batch_size, channels)
            features = features.permute(1, 0, 2)
        else:
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2, 0, 1)

        # Attention (optional)
        if self.attn is None and features.ndim == 3:
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        elif self.config.attn.name.lower() in ["nl", "se", "gc"]:  # type: ignore
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
            # (batch_size, channels, seq_len)
            features = self.attn(features)  # type: ignore
        elif self.config.attn.name.lower() in ["sa"]:  # type: ignore
            # (seq_len, batch_size, channels)
            features = self.attn(features)  # type: ignore
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        elif self.config.attn.name.lower() in ["transformer"]:  # type: ignore
            features = self.attn(features)  # type: ignore
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        return features

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        input : torch.Tensor
            Input signal tensor,
            of shape ``(batch_size, channels, seq_len)``.

        Returns
        -------
        pred : torch.Tensor
            Predictions tensor,
            of shape ``(batch_size, seq_len, channels)``
            or ``(batch_size, channels)``.

        """
        features = self.extract_features(input)

        if self.pool:
            features = self.pool(features)  # (batch_size, channels, pool_size)
            # features = features.squeeze(dim=-1)
            features = rearrange(
                features,
                "batch_size channels pool_size -> batch_size (channels pool_size)",
            )
        elif features.ndim == 3:
            # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
            features = features.permute(0, 2, 1)

        # print(f"clf in shape = {features.shape}")
        pred = self.clf(features)  # batch_size, n_classes

        return pred

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> BaseOutput:
        """Inference method for the model.

        Parameters
        ----------
        input : numpy.ndarray or torch.Tensor
            Input tensor, of shape ``(batch_size, channels, seq_len)``.
        class_names : bool, default False
            If True, the returned scalar predictions will be
            a :class:`~pandas.DataFrame`,
            with class names for each scalar prediction.
        bin_pred_thr : float, default 0.5
            Threshold for making binary predictions from scalar predictions.

        Returns
        -------
        output : BaseOutput
            The output of the inference method, including the following items:

                - prob: numpy.ndarray or torch.Tensor,
                  scalar predictions, (and binary predictions if `class_names` is True).
                - pred: numpy.ndarray or torch.Tensor,
                  the array (with values 0, 1 for each class) of binary prediction.

        """
        raise NotImplementedError("Implement a task-specific inference method.")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input signal tensor.
        batch_size : int, optional
            Batch size of the input signal tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the model.

        """
        if self.pool:
            return (batch_size, len(self.classes))
        else:
            _seq_len = seq_len
            output_shape = self.cnn.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
            if self.rnn:
                output_shape = self.rnn.compute_output_shape(_seq_len, batch_size)
                _seq_len = output_shape[0]
            if self.attn:
                output_shape = self.attn.compute_output_shape(_seq_len, batch_size)
                _seq_len = output_shape[-1]
            output_shape = self.clf.compute_output_shape(_seq_len, batch_size)
            return output_shape

    @property
    def doi(self) -> List[str]:
        doi = []
        candidates = [self.config]
        while len(candidates) > 0:
            new_candidates = []
            for candidate in candidates:
                if hasattr(candidate, "doi"):
                    if isinstance(candidate.doi, str):  # type: ignore
                        doi.append(candidate.doi)  # type: ignore
                    else:
                        doi.extend(list(candidate.doi))  # type: ignore
                for k, v in candidate.items():
                    if isinstance(v, CFG):
                        new_candidates.append(v)
            candidates = new_candidates
        doi = list(set(doi + ["10.1016/j.inffus.2019.06.024", "10.1088/1361-6579/ac6aa3"]))
        return doi
