# named Convolution Neural Networks

purely used as feature extractors

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

## Issues
1. Ordering of (batch) normalization and activation after convolution, should it be
   - cba (conv -> bn -> act), or
   - cab (conv -> act -> bn)
