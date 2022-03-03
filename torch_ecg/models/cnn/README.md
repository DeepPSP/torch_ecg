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
3. ReXNet
4. CSPNet
5. DLA
6. HarDNet
7. HO-ResNet
8. ResNet-RS
9. etc.

## Issues
1. Ordering of (batch) normalization and activation after convolution, should it be
   - cba (conv -> bn -> act), or
   - cab (conv -> act -> bn)

## References:
1. VGG
   1. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
   2. https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
2. ResNet
   1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
   2. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
   3. https://github.com/awni/ecg
   4. https://github.com/antonior92/automatic-ecg-diagnosis
3. MultiScopicNet
   1. Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
4. DenseNet
   1. G. Huang, Z. Liu, L. Van Der Maaten and K. Q. Weinberger, "Densely Connected Convolutional Networks," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017, pp. 2261-2269, doi: 10.1109/CVPR.2017.243.
   2. G. Huang, Z. Liu, G. Pleiss, L. Van Der Maaten and K. Weinberger, "Convolutional Networks with Dense Connectivity," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2019.2918284.
   3. https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
   4. https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
   5. https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
   6. https://github.com/liuzhuang13/DenseNet/tree/master/models
5. Xception
   1. Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
   2. https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py
   3. https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py
6. MobileNet
   1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
   2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
   3. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for mobilenetv3. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1314-1324).
   4. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
   5. https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py
7. DarkNet
   1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
   2. Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).
   3. Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
   4. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934.
   5. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2020). Scaled-YOLOv4: Scaling Cross Stage Partial Network. arXiv preprint arXiv:2011.08036.
8. EfficientNet
   1. Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning (pp. 6105-6114). PMLR.
   2. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. arXiv preprint arXiv:2104.00298.
   3. https://github.com/lukemelas/EfficientNet-PyTorch
   4. https://github.com/rwightman/gen-efficientnet-pytorch
   5. https://github.com/google/automl
9. MobileNeXt
   1. Zhou, D., Hou, Q., Chen, Y., Feng, J., & Yan, S. (2020). Rethinking bottleneck structure for efficient mobile network design. ECCV, August, 2.
   2. https://github.com/yitu-opensource/MobileNeXt
10. GhostNet
    1. Han, K., Wang, Y., Tian, Q., Guo, J., Xu, C., & Xu, C. (2020). Ghostnet: More features from cheap operations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1580-1589).
    2. https://github.com/huawei-noah/CV-Backbones
    3. https://github.com/iamhankai/ghostnet.pytorch
11. ReXNet
    1. to add
    2. https://github.com/clovaai/rexnet
12. CSPNet
    1. Wang, C. Y., Liao, H. Y. M., Wu, Y. H., Chen, P. Y., Hsieh, J. W., & Yeh, I. H. (2020). CSPNet: A new backbone that can enhance learning capability of CNN. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops (pp. 390-391).
    2. https://github.com/WongKinYiu/CrossStagePartialNetworks
13. HO-ResNet
    1. Luo, Z., Sun, Z., Zhou, W., & Kamata, S. I. (2021). Rethinking ResNets: Improved Stacking Strategies With High Order Schemes. arXiv preprint arXiv:2103.15244.
    2. to add
14. ResNet-RS
    1. Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T. Y., ... & Zoph, B. (2021). Revisiting resnets: Improved training and scaling strategies. arXiv preprint arXiv:2103.07579.
    2. to add

Many more from the CinC proceedings are not listed here.
