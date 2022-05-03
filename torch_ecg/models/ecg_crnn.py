"""
validated C(R)NN structure models, for classifying ECG arrhythmias

"""

from copy import deepcopy
from typing import Any, NoReturn, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from einops import rearrange

from ..cfg import CFG, DEFAULTS
from ..components.outputs import BaseOutput
from ..model_configs.ecg_crnn import ECG_CRNN_CONFIG
from ..utils.misc import dict_to_str
from ..utils.utils_nn import CkptMixin, SizeMixin
from ._nets import (  # noqa: F401
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
    SelfAttention,
    MLP,
    StackedLSTM,
)
from .cnn.densenet import DenseNet
from .cnn.multi_scopic import MultiScopicCNN
from .cnn.resnet import ResNet
from .cnn.mobilenet import MobileNetV1, MobileNetV2, MobileNetV3
from .cnn.vgg import VGG16
from .cnn.xception import Xception  # noqa: F401
from .transformers import Transformer  # noqa: F401

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_CRNN",
]


class ECG_CRNN(nn.Module, CkptMixin, SizeMixin):
    """finished, continuously improving,

    C(R)NN models modified from the following refs.

    References
    ----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    [3] Hannun, Awni Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65.
    [4] https://stanfordmlgroup.github.io/projects/ecg2/
    [5] https://github.com/awni/ecg
    [6] CPSC2018 entry 0236
    [7] CPSC2019 entry 0416

    """

    __DEBUG__ = False
    __name__ = "ECG_CRNN"

    def __init__(
        self,
        classes: Sequence[str],
        n_leads: int,
        config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """

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
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = n_leads
        self.config = deepcopy(ECG_CRNN_CONFIG)
        self.config.update(deepcopy(config) or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )
            debug_input_len = 4000

        cnn_choice = self.config.cnn.name.lower()
        cnn_config = self.config.cnn[self.config.cnn.name]
        if "resnet" in cnn_choice:
            self.cnn = ResNet(self.n_leads, **cnn_config)
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
                raise ValueError(f"{cnn_choice} is not supported for {self.__name__}")
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            self.cnn = DenseNet(self.n_leads, **cnn_config)
        elif "vgg16" in cnn_choice:
            self.cnn = VGG16(self.n_leads, **cnn_config)
        else:
            raise NotImplementedError(
                f"the CNN \042{cnn_choice}\042 not implemented yet"
            )
        rnn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(debug_input_len, None)
            print(
                f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, "
                f"given input_len = {debug_input_len}"
            )

        if self.config.rnn.name.lower() == "none":
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropouts=self.config.rnn.lstm.dropouts,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=self.config.rnn.lstm.retseq,
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        elif self.config.rnn.name.lower() == "linear":
            # abuse of notation, to put before the global attention module
            self.rnn = MLP(
                in_channels=rnn_input_size,
                out_channels=self.config.rnn.linear.out_channels,
                activation=self.config.rnn.linear.activation,
                bias=self.config.rnn.linear.bias,
                dropouts=self.config.rnn.linear.dropouts,
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        # attention
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.attn = None
            clf_input_size = attn_input_size
            if self.config.attn.name.lower() != "none":
                print(
                    f"since `retseq` of rnn is False, hence attention `{self.config.attn.name}` is ignored"
                )
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
                in_features=attn_input_size,
                head_num=self.config.attn.sa.head_num,
                dropout=self.config.attn.sa.dropout,
                bias=self.config.attn.sa.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        elif self.config.attn.name.lower() == "transformer":
            self.attn = Transformer(
                input_size=attn_input_size,
                hidden_size=self.config.attn.transformer.hidden_size,
                num_layers=self.config.attn.transformer.num_layers,
                num_heads=self.config.attn.transformer.num_heads,
                dropout=self.config.attn.transformer.dropout,
                activation=self.config.attn.transformer.activation,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        if self.__DEBUG__:
            print(f"clf_input_size = {clf_input_size}")

        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.pool = None
            if self.config.global_pool.lower() != "none":
                print(
                    f"since `retseq` of rnn is False, hence global pooling `{self.config.global_pool}` is ignored"
                )
        elif self.config.global_pool.lower() == "max":
            self.pool = nn.AdaptiveMaxPool1d(
                (self.config.global_pool_size,), return_indices=False
            )
            clf_input_size *= self.config.global_pool_size
        elif self.config.global_pool.lower() == "avg":
            self.pool = nn.AdaptiveAvgPool1d((self.config.global_pool_size,))
            clf_input_size *= self.config.global_pool_size
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")
        elif self.config.global_pool.lower() == "none":
            self.pool = None
        else:
            raise NotImplementedError(
                f"pooling method {self.config.global_pool} not implemented yet!"
            )

        # input of `self.clf` has shape: batch_size, channels
        self.clf = MLP(
            in_channels=clf_input_size,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        # for inference
        # classification: if single-label, use softmax; otherwise (multi-label) use sigmoid
        # sequence tagging: if background counted in `classes`, use softmax; otherwise use sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

    def extract_features(self, input: Tensor) -> Tensor:
        """

        extract feature map before the dense (linear) classifying layer(s)

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        features: Tensor,
            of shape (batch_size, channels, seq_len) or (batch_size, channels)

        """
        # CNN
        features = self.cnn(input)  # batch_size, channels, seq_len

        # RNN (optional)
        if self.config.rnn.name.lower() in ["lstm"]:
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2, 0, 1)
            features = self.rnn(
                features
            )  # (seq_len, batch_size, channels) or (batch_size, channels)
        elif self.config.rnn.name.lower() in ["linear"]:
            # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
            features = features.permute(0, 2, 1)
            features = self.rnn(features)  # (batch_size, seq_len, channels)
            # (batch_size, seq_len, channels) --> (seq_len, batch_size, channels)
            features = features.permute(1, 0, 2)
        else:
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2, 0, 1)

        # Attention (optional)
        if self.attn is None and features.ndim == 3:
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        elif self.config.attn.name.lower() in ["nl", "se", "gc"]:
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
            features = self.attn(features)  # (batch_size, channels, seq_len)
        elif self.config.attn.name.lower() in ["sa"]:
            features = self.attn(features)  # (seq_len, batch_size, channels)
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        elif self.config.attn.name.lower() in ["transformer"]:
            features = self.attn(features)
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1, 2, 0)
        return features

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        pred: Tensor,
            of shape (batch_size, n_classes)

        """
        features = self.extract_features(input)

        if self.pool:
            features = self.pool(features)  # (batch_size, channels, pool_size)
            # features = features.squeeze(dim=-1)
            features = rearrange(
                features,
                "batch_size channels pool_size -> batch_size (channels pool_size)",
            )
        else:
            # features of shape (batch_size, channels) or (batch_size, seq_len, channels)
            pass

        # print(f"clf in shape = {x.shape}")
        pred = self.clf(features)  # batch_size, n_classes

        return pred

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> BaseOutput:
        """

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
        output: BaseOutput, including the following items:
            prob: ndarray or DataFrame,
                scalar predictions, (and binary predictions if `class_names` is True)
            pred: ndarray,
                the array (with values 0, 1 for each class) of binary prediction

        """
        raise NotImplementedError("implement a task specific inference method")

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this model, given `seq_len` and `batch_size`

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
