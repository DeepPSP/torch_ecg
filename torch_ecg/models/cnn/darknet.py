"""
darknet, backbone for the famous image object detector,

References
----------
[1] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
[2] Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).
[3] Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
[4] Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.
[5] Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2020). Scaled-YOLOv4: Scaling Cross Stage Partial Network. arXiv preprint arXiv:2011.08036.
"""

from typing import NoReturn

import torch
from torch import nn

from ...cfg import DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
    GlobalContextBlock,
    NonLocalBlock,
    SEBlock,
)
from ...utils.utils_nn import SizeMixin

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "DarkNet",
]


class DarkNet(nn.Sequential, SizeMixin):
    """ """

    __DEBUG__ = True
    __name__ = "DarkNet"

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """ """
        super().__init__()
        raise NotImplementedError
