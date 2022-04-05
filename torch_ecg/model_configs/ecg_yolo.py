"""
configs of the yolo model for qrs complex (or more?) detection
"""

from ..cfg import CFG

__all__ = [
    "ECG_YOLO_CONFIG",
]


ECG_YOLO_CONFIG = CFG()


ECG_YOLO_CONFIG.cnn = CFG()
ECG_YOLO_CONFIG.cnn.name = "resnet_gc"


ECG_YOLO_CONFIG.cnn.resnet_gc = CFG()

ECG_YOLO_CONFIG.cnn.resnet_gc.increase_channels_method = "conv"
ECG_YOLO_CONFIG.cnn.resnet_gc.subsample_mode = "conv"

ECG_YOLO_CONFIG.cnn.resnet_gc.gc = CFG()
ECG_YOLO_CONFIG.cnn.resnet_gc.gc.ratio = 16
ECG_YOLO_CONFIG.cnn.resnet_gc.gc.reduction = True
ECG_YOLO_CONFIG.cnn.resnet_gc.gc.pooling_type = "attn"
ECG_YOLO_CONFIG.cnn.resnet_gc.gc.fusion_types = [
    "mul",
]


ECG_YOLO_CONFIG.stage = CFG()
ECG_YOLO_CONFIG.stage.resnet_gc = CFG()

ECG_YOLO_CONFIG.stage.resnet_gc.increase_channels_method = "conv"
ECG_YOLO_CONFIG.stage.resnet_gc.subsample_mode = "conv"

ECG_YOLO_CONFIG.stage.resnet_gc.gc = CFG()
ECG_YOLO_CONFIG.stage.resnet_gc.gc.ratio = 16
ECG_YOLO_CONFIG.stage.resnet_gc.gc.reduction = True
ECG_YOLO_CONFIG.stage.resnet_gc.gc.pooling_type = "attn"
ECG_YOLO_CONFIG.stage.resnet_gc.gc.fusion_types = [
    "mul",
]
