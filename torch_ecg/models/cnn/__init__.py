"""
named CNNs, which are frequently used by more complicated models, including
1. vgg
2. resnet
3. variants of resnet (with se, gc, etc.)
4. multi_scopic
5. densenet
6. to add more
"""

from .densenet import (
    DenseBasicBlock, DenseBottleNeck,
    DenseMacroBlock, DenseTransition,
    DenseNet,
)
from .multi_scopic import MultiScopicBasicBlock, MultiScopicBranch, MultiScopicCNN
from .resnet import ResNetBasicBlock, ResNetBottleNeck, ResNet
from .vgg import VGGBlock, VGG16
# from .xception import


__all__ = [
    # VGG
    "VGGBlock", "VGG16",

    # ResNet
    "ResNetBasicBlock", "ResNetBottleNeck", "ResNet",

    # MultiScopic
    "MultiScopicBasicBlock", "MultiScopicBranch", "MultiScopicCNN",

    # DenseNet
    "DenseBasicBlock", "DenseBottleNeck", "DenseMacroBlock", "DenseTransition", "DenseNet",
]
