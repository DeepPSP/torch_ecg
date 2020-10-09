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
