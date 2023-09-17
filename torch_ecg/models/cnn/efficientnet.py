"""
EfficientNet.

References
----------
1. Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
2. Tan, M., & Le, Q. V. (2021). Efficientnetv2: Smaller models and faster training. arXiv preprint arXiv:2104.00298.
3. https://github.com/google/automl

"""

from typing import List

from torch import nn

from ...models._nets import Conv_Bn_Activation, DownSample, GlobalContextBlock, NonLocalBlock, SEBlock  # noqa: F401
from ...utils import CitationMixin, SizeMixin

__all__ = [
    "EfficientNet",
]


class EfficientNet(nn.Module, SizeMixin, CitationMixin):
    """
    Reference
    ---------
    1. Tan, M., & Le, Q. V. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946.
    2. https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    3. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    3. https://github.com/google/automl

    """

    __name__ = "EfficientNet"

    def __init__(self, in_channels: int, **config) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def compute_output_shape(self):
        raise NotImplementedError

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.48550/ARXIV.1905.11946"]))


class EfficientNetV2(nn.Module, SizeMixin):
    """
    Reference
    ---------
    1. Tan, M., & Le, Q. V. (2021). Efficientnetv2: Smaller models and faster training. arXiv preprint arXiv:2104.00298.
    2. https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    3. https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
    4. https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py
    5. https://github.com/google/automl

    """

    __name__ = "EfficientNetV2"

    def __init__(self, in_channels: int, **config) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def compute_output_shape(self):
        raise NotImplementedError
