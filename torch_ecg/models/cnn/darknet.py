"""
Darknet, backbone for the famous image object detector.

References
----------
1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
2. Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).
3. Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
4. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.
5. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2020). Scaled-YOLOv4: Scaling Cross Stage Partial Network. arXiv preprint arXiv:2011.08036.
"""

from typing import List

from torch import nn

from ...models._nets import Conv_Bn_Activation, DownSample, GlobalContextBlock, NonLocalBlock, SEBlock  # noqa: F401
from ...utils import CitationMixin, SizeMixin

__all__ = [
    "DarkNet",
]


class DarkNet(nn.Sequential, SizeMixin, CitationMixin):
    """ """

    __name__ = "DarkNet"

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
        raise NotImplementedError

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.1109/CVPR.2016.91"]))
