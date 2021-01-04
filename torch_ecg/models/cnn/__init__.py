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
    DenseNet,
    DenseBasicBlock, DenseBottleNeck,
    DenseMacroBlock, DenseTransition,
)
from .multi_scopic import MultiScopicCNN, MultiScopicBasicBlock, MultiScopicBranch
from .resnet import ResNet, ResNetBasicBlock, ResNetBottleNeck
from .vgg import VGG16, VGGBlock
from .xception import (
    Xception,
    XceptionEntryFlow, XceptionMiddleFlow, XceptionExitFlow,
    XceptionMultiConv,
)


__all__ = [
    # VGG
    "VGG16",
    "VGGBlock",

    # ResNet
    "ResNet",
    "ResNetBasicBlock", "ResNetBottleNeck",

    # MultiScopic
    "MultiScopicCNN",
    "MultiScopicBasicBlock", "MultiScopicBranch",

    # DenseNet
    "DenseNet",
    "DenseBasicBlock", "DenseBottleNeck", "DenseMacroBlock", "DenseTransition",

    # Xception
    "Xception",
    "XceptionEntryFlow", "XceptionMiddleFlow", "XceptionExitFlow",
    "XceptionMultiConv",
]
