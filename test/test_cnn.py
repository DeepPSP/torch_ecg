"""
"""

import time
from typing import NoReturn

import pytest
import torch
from easydict import EasyDict as ED

try:
    import torch_ecg
except:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    import torch_ecg

from torch_ecg.model_configs import (
    # building blocks
    # CNN
    # vgg
    vgg16,
    vgg16_leadwise,
    # vanilla resnet
    resnet_vanilla_18,
    resnet_vanilla_34,
    resnet_vanilla_50,
    resnet_vanilla_101,
    resnet_vanilla_152,
    resnext_vanilla_50_32x4d,
    resnext_vanilla_101_32x8d,
    resnet_vanilla_wide_50_2,
    resnet_vanilla_wide_101_2,
    # custom resnet
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    tresnetN,
    tresnetP,
    tresnetF,
    tresnetS,
    tresnetM,
    tresnetL,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_bottle_neck_nl,
    resnet_nature_comm_bottle_neck_gc,
    resnet_cpsc2018,
    resnet_cpsc2018_leadwise,
    # stanford resnet
    resnet_block_stanford,
    resnet_stanford,
    # cpsc2018 SOTA, legacy
    cpsc_2018,
    cpsc_2018_leadwise,
    # multi_scopic
    multi_scopic,
    multi_scopic_leadwise,
    # vanilla dense_net
    densenet_vanilla,
    # custom dense_net
    densenet_leadwise,
    # vanilla xception
    xception_vanilla,
    # custom xception
    xception_leadwise,
    # vanilla mobilenets
    mobilenet_v1_vanilla,
)

from torch_ecg.models.cnn.darknet import DarkNet
from torch_ecg.models.cnn.densenet import DenseNet
from torch_ecg.models.cnn.efficientnet import EfficientNet, EfficientNetV2
from torch_ecg.models.cnn.ho_resnet import (
    MidPointResNet,
    RK4ResNet,
    RK8ResNet,
)
from torch_ecg.models.cnn.mobilenet import (
    MobileNetV1,
    MobileNetV2,
    MobileNetV3,
)
from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.models.cnn.resnet import ResNet
from torch_ecg.models.cnn.vgg import VGG16
from torch_ecg.models.cnn.xception import Xception


_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_IN_CHANNELS = 12
_BATCH_SIZE = 2
_SIG_LEN = 4000
_TEST_EXAMPLE = torch.rand((_BATCH_SIZE, _IN_CHANNELS, _SIG_LEN)).to(_DEVICE)
_TEST_CLF_CLASSES = [
    "nsr",
    "af",
    "pvc",
]


@torch.no_grad()
def test_cnn() -> NoReturn:
    """ """
    start = time.time()
    print("\n" + " Test CNN configs ".center(80, "#") + "\n")
    # test vgg configs
    print("\n" + " Test VGG configs ".center(50, "-") + "\n")
    VGG16.__DEBUG__ = False
    for cfg in [
        "vgg16",
        "vgg16_leadwise",
    ]:
        _test_cnn("VGG16", cfg)

    # test resnet configs
    print("\n" + " Test ResNet configs ".center(50, "-") + "\n")
    ResNet.__DEBUG__ = False
    for cfg in [
        "resnet_vanilla_18",
        "resnet_vanilla_34",
        "resnet_vanilla_50",
        "resnet_vanilla_101",
        "resnet_vanilla_152",
        "resnet_vanilla_wide_50_2",
        "resnet_vanilla_wide_101_2",
        # custom resnet
        "resnetN",
        "resnetNB",
        "resnetNBS",
        "resnetNS",
        "tresnetN",
        "tresnetP",
        "tresnetF",
        "tresnetS",
        "tresnetM",
        "tresnetL",
        "resnet_nature_comm",
        "resnet_nature_comm_bottle_neck",
        "resnet_nature_comm_bottle_neck_se",
        "resnet_nature_comm_bottle_neck_nl",
        "resnet_nature_comm_bottle_neck_gc",
        # "resnet",
        # "resnet_leadwise",
        # TODO: fix bugs in the following
        # "resnext_vanilla_50_32x4d", "resnext_vanilla_101_32x8d",
        # stanford resnet
        # "resnet_stanford",
    ]:
        _test_cnn("ResNet", cfg)

    # test multi_scopic
    print("\n" + " Test MultiScopic configs ".center(50, "-") + "\n")
    MultiScopicCNN.__DEBUG__ = False
    for cfg in [
        "multi_scopic",
        "multi_scopic_leadwise",
    ]:
        _test_cnn("MultiScopicCNN", cfg)

    # test densenet
    print("\n" + " Test DenseNet configs ".center(50, "-") + "\n")
    DenseNet.__DEBUG__ = False
    for cfg in [
        "densenet_vanilla",
        "densenet_leadwise",
    ]:
        _test_cnn("DenseNet", cfg)

    # test xception
    print("\n" + " Test Xception configs ".center(50, "-") + "\n")
    Xception.__DEBUG__ = False
    for cfg in [
        "xception_vanilla",
        "xception_leadwise",
    ]:
        _test_cnn("Xception", cfg)

    # test mobilenet
    print("\n" + " Test MobileNet configs ".center(50, "-") + "\n")
    MobileNetV1.__DEBUG__ = False
    MobileNetV2.__DEBUG__ = False
    MobileNetV3.__DEBUG__ = False
    for cfg in [
        "mobilenet_v1_vanilla",
    ]:
        _test_cnn("MobileNetV1", cfg)

    print(f"total time cost: {time.time()-start:.2f} seconds")
    print("\n" + " Finish testing CNN configs ".center(80, "#") + "\n")


@torch.no_grad()
def _test_cnn(model_name: str, cfg: ED) -> NoReturn:
    """ """
    try:
        test_model = eval(
            f"{model_name}(in_channels=_IN_CHANNELS, **{cfg}).to(_DEVICE)"
        )
        test_model.eval()
        test_output = test_model(_TEST_EXAMPLE)
        print(f"{cfg} output shape = {test_output.shape}")
        # assert list(test_output) == list(test_model.compute_output_shape(seq_len=_SIG_LEN, batch_size=_BATCH_SIZE))
        del test_model
        del test_output
    except Exception as e:
        print(f"{cfg} raises errors\n")
        raise e


if __name__ == "__main__":
    test_cnn()
