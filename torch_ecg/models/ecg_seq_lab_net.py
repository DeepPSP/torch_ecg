"""
Sequence labeling nets, for wave delineation,

the labeling granularity is the frequency of the input signal,
divided by the length (counted by the number of basic blocks) of each branch

pipeline:
multi-scopic cnn --> (bidi-lstm -->) "attention" (se block) --> seq linear

References
----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).

"""

import warnings
from copy import deepcopy
from typing import Optional, Sequence, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .ecg_crnn import ECG_CRNN, ECG_CRNN_v1
from ..model_configs.ecg_seq_lab_net import ECG_SEQ_LAB_NET_CONFIG
from ..cfg import CFG


__all__ = [
    "ECG_SEQ_LAB_NET",
]


class ECG_SEQ_LAB_NET(ECG_CRNN):
    """
    SOTA model from CPSC2019 challenge (entry 0416)

    pipeline
    --------
    (multi-scopic, etc.) cnn --> head ((bidi-lstm -->) "attention" --> seq linear) -> output

    References
    ----------
    [1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).

    """

    __name__ = "ECG_SEQ_LAB_NET"
    __DEFAULT_CONFIG__ = {"recover_length": False}
    __DEFAULT_CONFIG__.update(deepcopy(ECG_SEQ_LAB_NET_CONFIG))

    def __init__(
        self, classes: Sequence[str], n_leads: int, config: Optional[CFG] = None
    ) -> None:
        """
        Parameters
        ----------
        classes: list,
            list of the classes for sequence labeling
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        _config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        if not config:
            warnings.warn(
                "No config is provided, using default config.", RuntimeWarning
            )
        _config.update(deepcopy(config) or {})
        _config.global_pool = "none"
        super().__init__(classes, n_leads, _config)

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)

        Returns
        -------
        pred: Tensor,
            of shape (batch_size, seq_len, n_classes)

        """
        batch_size, channels, seq_len = input.shape

        pred = super().forward(input)  # (batch_size, len, n_classes)

        if self.config.recover_length:
            pred = F.interpolate(
                pred.permute(0, 2, 1),
                size=seq_len,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)

        return pred

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
        output_shape = super().compute_output_shape(seq_len, batch_size)
        if self.config.recover_length:
            output_shape = (batch_size, seq_len, output_shape[-1])
        return output_shape

    @classmethod
    def from_v1(
        cls, v1_ckpt: str, device: Optional[torch.device] = None
    ) -> "ECG_SEQ_LAB_NET":
        """
        Parameters
        ----------
        v1_ckpt: str,
            the path to the v1 checkpoint file

        Returns
        -------
        model: ECG_SEQ_LAB_NET,
            the converted model

        """
        v1_model, _ = ECG_SEQ_LAB_NET_v1.from_checkpoint(v1_ckpt, device=device)
        model = cls(
            classes=v1_model.classes, n_leads=v1_model.n_leads, config=v1_model.config
        )
        model = model.to(v1_model.device)
        model.cnn.load_state_dict(v1_model.cnn.state_dict())
        if model.rnn.__class__.__name__ != "Identity":
            model.rnn.load_state_dict(v1_model.rnn.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(v1_model.attn.state_dict())
        model.clf.load_state_dict(v1_model.clf.state_dict())
        del v1_model
        return model

    @property
    def doi(self) -> List[str]:
        return list(set(super().doi + ["10.1109/access.2020.2997473"]))


class ECG_SEQ_LAB_NET_v1(ECG_CRNN_v1):
    """
    SOTA model from CPSC2019 challenge (entry 0416)

    pipeline
    --------
    (multi-scopic, etc.) cnn --> head ((bidi-lstm -->) "attention" --> seq linear) -> output

    References
    ----------
    [1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).

    """

    __name__ = "ECG_SEQ_LAB_NET_v1"
    __DEFAULT_CONFIG__ = {"recover_length": False}
    __DEFAULT_CONFIG__.update(deepcopy(ECG_SEQ_LAB_NET_CONFIG))

    def __init__(
        self, classes: Sequence[str], n_leads: int, config: Optional[CFG] = None
    ) -> None:
        """
        Parameters
        ----------
        classes: list,
            list of the classes for sequence labeling
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        _config = CFG(deepcopy(self.__DEFAULT_CONFIG__))
        if not config:
            warnings.warn(
                "No config is provided, using default config.", RuntimeWarning
            )
        _config.update(deepcopy(config) or {})
        _config.global_pool = "none"
        super().__init__(classes, n_leads, _config)

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
            of shape (batch_size, seq_len, channels)

        """
        # cnn
        cnn_output = self.cnn(input)  # (batch_size, channels, seq_len)

        # rnn or none
        if self.rnn:
            rnn_output = cnn_output.permute(2, 0, 1)  # (seq_len, batch_size, channels)
            rnn_output = self.rnn(rnn_output)  # (seq_len, batch_size, channels)
            rnn_output = rnn_output.permute(1, 2, 0)  # (batch_size, channels, seq_len)
        else:
            rnn_output = cnn_output

        # attention
        if self.attn:
            features = self.attn(rnn_output)  # (batch_size, channels, seq_len)
        else:
            features = rnn_output
        # features = features.permute(0, 2, 1)  # (batch_size, seq_len, channels)
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
            of shape (batch_size, seq_len, n_classes)

        """
        batch_size, channels, seq_len = input.shape

        pred = super().forward(input)  # (batch_size, len, n_classes)

        if self.config.recover_length:
            pred = F.interpolate(
                pred.permute(0, 2, 1),
                size=seq_len,
                mode="linear",
                align_corners=True,
            ).permute(0, 2, 1)

        return pred

    @property
    def doi(self) -> List[str]:
        return list(set(super().doi + ["10.1109/access.2020.2997473"]))
