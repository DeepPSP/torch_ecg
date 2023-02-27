"""
"""

import warnings
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ...utils.utils_nn import SizeMixin
from ...utils.misc import get_kwargs


__all__ = [
    "Transformer",
]


class Transformer(nn.Module, SizeMixin):
    """Transformer feature extractor.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    hidden_size : int
        Number of hidden units in the encoding layer.
    num_heads : int, default 8
        Number of attention heads.
    num_layers : int, default 1
        Number of encoding layers.
    dropout : float, default 0.1
        Dropout probability
    batch_first : bool, default True
        If True, the input is of shape ``(batch_size, seq_len, input_size)``,
        otherwise, the input is of shape ``(seq_len, batch_size, input_size)``.
    kwargs : dict
        Other hyper-parameters of the Module, including
        activation, kernel_initializer, etc.

    """

    __name__ = "Transformer"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        self.__num_heads = num_heads
        self.__batch_first = batch_first

        if self.__input_size % self.__num_heads != 0:
            input_size = self.__input_size
            self.__input_size = self.__input_size // self.__num_heads * self.__num_heads
            warnings.warn(
                f"`input_size` {input_size} is not divisible by `num_heads` {self.__num_heads}, "
                f"adjusted to {self.__input_size}",
                RuntimeWarning,
            )
            if self.__batch_first:
                self.project = nn.Linear(input_size, self.__input_size)
            else:
                self.project = nn.Sequential(
                    Rearrange(
                        "seq_len batch_size input_size -> batch_size seq_len input_size"
                    ),
                    nn.Linear(input_size, self.__input_size),
                    Rearrange(
                        "batch_size seq_len input_size -> seq_len batch_size input_size"
                    ),
                )
        else:
            self.project = nn.Identity()

        if "batch_first" in get_kwargs(nn.TransformerEncoderLayer):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.__input_size,
                nhead=self.__num_heads,
                dim_feedforward=self.__hidden_size,
                dropout=dropout,
                batch_first=self.__batch_first,
                activation=kwargs.get("activation", "relu"),
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.__input_size,
                nhead=self.__num_heads,
                dim_feedforward=self.__hidden_size,
                dropout=dropout,
                activation=kwargs.get("activation", "relu"),
            )
            if self.__batch_first:
                warnings.warn(
                    "`batch_first` only supports torch >= 1.9, defaults to False",
                    RuntimeWarning,
                )
            self.__batch_first = False
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.__num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor,
            of shape ``(seq_len, batch_size, input_size)`` if `batch_first` is False,
            else ``(batch_size, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Output tensor,
            of shape ``(seq_len, batch_size, output_size)`` if `batch_first` is False,
            else ``(batch_size, seq_len, output_size)``.

        """
        x = self.project(x)
        return self.encoder(x)

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the transformer feature extractor.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            Batch size of the input tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the transformer feature extractor.

        """
        if self.__batch_first:
            return (batch_size, seq_len, self.__input_size)
        else:
            return (seq_len, batch_size, self.__input_size)

    @property
    def batch_first(self) -> bool:
        return self.__batch_first
