"""
Higher Order ResNet

References
----------
[1] Luo, Z., Sun, Z., Zhou, W., & Kamata, S. I. (2021). Rethinking ResNets: Improved Stacking Strategies With High Order Schemes. arXiv preprint arXiv:2103.15244.

"""

from typing import Optional, Sequence, Union

from torch import Tensor, nn

from ...cfg import CFG  # noqa: F401
from ...models._nets import (  # noqa: F401
    Activations,
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
    ZeroPadding,
)
from ...utils import CitationMixin, SizeMixin

__all__ = [
    "MidPointResNet",
    "RK4ResNet",
    "RK8ResNet",
]


class MidPointResNet(nn.Module, SizeMixin, CitationMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model."""
        raise NotImplementedError

    def forward_features(self, input: Tensor) -> Tensor:
        """Forward pass of the model to extract features."""
        raise NotImplementedError

    def compute_features_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the features."""
        raise NotImplementedError


class RK4ResNet(nn.Module, SizeMixin, CitationMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model."""
        raise NotImplementedError

    def forward_features(self, input: Tensor) -> Tensor:
        """Forward pass of the model to extract features."""
        raise NotImplementedError

    def compute_features_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the features."""
        raise NotImplementedError


class RK8ResNet(nn.Module, SizeMixin, CitationMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the model."""
        raise NotImplementedError

    def forward_features(self, input: Tensor) -> Tensor:
        """Forward pass of the model to extract features."""
        raise NotImplementedError

    def compute_features_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the features."""
        raise NotImplementedError
