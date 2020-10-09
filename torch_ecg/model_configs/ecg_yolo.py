"""
configs of the yolo model for qrs complex (or more?) detection
"""
from itertools import repeat
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "ECG_YOLO_CONFIG",
]


ECG_YOLO_CONFIG = ED()


ECG_YOLO_CONFIG.cnn = ED()
ECG_CRNN_CONFIG.cnn.name = "resnet_gc"


ECG_CRNN_CONFIG.cnn.resnet_gc = ED()

ECG_CRNN_CONFIG.cnn.resnet_gcincrease_channels_method = "conv"
ECG_CRNN_CONFIG.cnn.resnet_gcsubsample_mode = "conv"

ECG_CRNN_CONFIG.cnn.resnet_gc.gc = ED()
ECG_CRNN_CONFIG.cnn.resnet_gc.gc.ratio = 16
ECG_CRNN_CONFIG.cnn.resnet_gc.gc.reduction = True
ECG_CRNN_CONFIG.cnn.resnet_gc.gc.pooling_type = "attn"
ECG_CRNN_CONFIG.cnn.resnet_gc.gc.fusion_types = ["mul",]
