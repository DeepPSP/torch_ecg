"""
Higher Order ResNet

References
----------
[1] Luo, Z., Sun, Z., Zhou, W., & Kamata, S. I. (2021). Rethinking ResNets: Improved Stacking Strategies With High Order Schemes. arXiv preprint arXiv:2103.15244.

"""

from torch import nn

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
        raise NotImplementedError


class RK4ResNet(nn.Module, SizeMixin, CitationMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        raise NotImplementedError


class RK8ResNet(nn.Module, SizeMixin, CitationMixin):
    """ """

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        raise NotImplementedError
