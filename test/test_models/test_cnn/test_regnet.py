"""
"""

from copy import deepcopy

import torch
from tqdm.auto import tqdm

from torch_ecg.models.cnn.regnet import RegNet
from torch_ecg.model_configs.cnn.regnet import (
    regnet_16_8,
    regnet_27_24,
    regnet_23_168,
)


IN_CHANNELS = 12


@torch.no_grad()
def test_regnet():
    inp = torch.randn(2, IN_CHANNELS, 2000)

    for item in tqdm(
        [regnet_16_8, regnet_27_24, regnet_23_168],
        mininterval=1,
        desc="Testing RegNet",
    ):
        config = deepcopy(item)
        model = RegNet(IN_CHANNELS, **config)
        out = model(inp)
        assert out.shape == model.compute_output_shape(2000, 2)
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
