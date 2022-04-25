"""
"""

from typing import Any, NoReturn, Optional, Sequence, Union

import torch
import torch.nn as nn

from ...cfg import DEFAULTS
from ...utils.utils_nn import SizeMixin

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "Transformer",
]


class Transformer(nn.Module, SizeMixin):
    """ """

    __DEBUG__ = True
    __name__ = "Transformer"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        input_size: int,
            number of input channels
        hidden_size: int,
            number of hidden units in the encoding layer
        num_heads: int, default 8,
            number of attention heads
        num_layers: int, default 1,
            number of encoding layers
        dropout: float, default 0.1,
            dropout probability
        kwargs: keyword arguments,

        """
        super().__init__()
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        self.__num_heads = num_heads
        self.__batch_first = kwargs.get("batch_first", False)
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.__input_size,
                nhead=self.__num_heads,
                dim_feedforward=self.__hidden_size,
                dropout=dropout,
                batch_first=self.__batch_first,
                activation=kwargs.get("activation", "relu"),
            )
        except Exception:
            self.__batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.__input_size,
                nhead=self.__num_heads,
                dim_feedforward=self.__hidden_size,
                dropout=dropout,
                activation=kwargs.get("activation", "relu"),
            )
            print("batch_first only supports torch >= 1.9, defaults to False")
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.__num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor,
            the input tensor,
            of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)

        Returns
        -------
        torch.Tensor:
            of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)

        """
        return self.encoder(x)

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
            the output shape of this layer, given `seq_len` and `batch_size`

        """
        if self.__batch_first:
            return (batch_size, seq_len, self.__input_size)
        else:
            return (seq_len, batch_size, self.__input_size)
