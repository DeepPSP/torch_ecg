"""
"""

from copy import deepcopy

import torch
from tqdm.auto import tqdm

from torch_ecg.models.cnn.resnet import ResNet
from torch_ecg.model_configs.cnn.resnet import (
    # smaller resnets
    resnetN,
    resnetNB,
    resnetNS,
    resnetNBS,
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
    # cpsc2018 resnet
    resnet_cpsc2018,
    resnet_cpsc2018_leadwise,
    # stanford resnet
    resnet_stanford,
    # ResNet Nature Communications
    resnet_nature_comm,
    resnet_nature_comm_se,
    resnet_nature_comm_nl,
    resnet_nature_comm_gc,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_bottle_neck_gc,
    resnet_nature_comm_bottle_neck_nl,
    # TresNet
    tresnetF,
    tresnetP,
    tresnetN,
    tresnetS,
    tresnetM,
    tresnetL,
    tresnetXL,
    tresnetM_V2,
)


IN_CHANNELS = 12


@torch.no_grad()
def test_resnet():
    inp = torch.randn(2, IN_CHANNELS, 5000)

    for item in tqdm(
        [
            # smaller resnets
            resnetN,
            resnetNB,
            resnetNS,
            resnetNBS,
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
            # cpsc2018 resnet
            resnet_cpsc2018,
            resnet_cpsc2018_leadwise,
            # stanford resnet
            resnet_stanford,
            # ResNet Nature Communications
            resnet_nature_comm,
            resnet_nature_comm_se,
            resnet_nature_comm_nl,
            resnet_nature_comm_gc,
            resnet_nature_comm_bottle_neck,
            resnet_nature_comm_bottle_neck_se,
            resnet_nature_comm_bottle_neck_gc,
            resnet_nature_comm_bottle_neck_nl,
            # TresNet
            tresnetF,
            tresnetP,
            tresnetN,
            tresnetS,
            tresnetM,
            tresnetL,
            tresnetXL,
            tresnetM_V2,
        ],
        mininterval=1,
        desc="Testing ResNet",
    ):
        config = deepcopy(item)
        model = ResNet(in_channels=IN_CHANNELS, **config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
