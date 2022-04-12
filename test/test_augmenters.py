"""
"""

# import pytest

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

# from torch_ecg.augmenters.augmenter_manager import AugmenterManager
# from torch_ecg.augmenters.baseline_wander import BaselineWanderAugmenter
# from torch_ecg.augmenters.label_smooth import LabelSmooth
# from torch_ecg.augmenters.mixup import Mixup
# from torch_ecg.augmenters.random_flip import RandomFlip
# from torch_ecg.augmenters.random_masking import RandomMasking
# from torch_ecg.augmenters.stretch_compress import StretchCompress
