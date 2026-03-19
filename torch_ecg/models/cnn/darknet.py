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

from typing import List, Optional, Sequence, Union

from torch import Tensor, nn

from ...models._nets import Conv_Bn_Activation, DownSample, GlobalContextBlock, NonLocalBlock, SEBlock  # noqa: F401
from ...utils import CitationMixin, SizeMixin


class DarkNet(SizeMixin, nn.Sequential, CitationMixin):
    """ """

    __name__ = "DarkNet"

    def __init__(self, in_channels: int, **config) -> None:
        """ """
        super().__init__()
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

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.1109/CVPR.2016.91"]))
