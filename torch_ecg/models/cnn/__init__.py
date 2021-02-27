"""
named CNNs, which are frequently used by more complicated models, including

## Implemented
1. VGG
2. ResNet
3. MultiScopicNet
4. DenseNet
5. Xception
  
## Ongoing
1. MobileNet
2. DarkNet
3. EfficientNet

## TODO
1. MobileNeXt
2. GhostNet
3. etc.
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
