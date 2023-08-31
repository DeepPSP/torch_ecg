"""
"""

from copy import deepcopy

import torch

from torch_ecg.models.cnn.mobilenet import (
    MobileNetV1,
    MobileNetV2,
    MobileNetV3,
)
from torch_ecg.model_configs.cnn.mobilenet import (
    mobilenet_v1_vanilla,
    mobilenet_v2_vanilla,
    mobilenet_v3_small,
)


IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_mobilenet():
    inp = torch.randn(2, IN_CHANNELS, 2000).to(DEVICE)

    config = deepcopy(mobilenet_v1_vanilla)
    model = MobileNetV1(in_channels=IN_CHANNELS, **config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(
        seq_len=inp.shape[-1], batch_size=inp.shape[0]
    )
    assert model.in_channels == IN_CHANNELS
    assert isinstance(model.doi, list)

    config = deepcopy(mobilenet_v2_vanilla)
    model = MobileNetV2(in_channels=IN_CHANNELS, **config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(
        seq_len=inp.shape[-1], batch_size=inp.shape[0]
    )
    assert model.in_channels == IN_CHANNELS
    assert isinstance(model.doi, list)

    config = deepcopy(mobilenet_v3_small)
    model = MobileNetV3(in_channels=IN_CHANNELS, **config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(
        seq_len=inp.shape[-1], batch_size=inp.shape[0]
    )
    assert model.in_channels == IN_CHANNELS
    assert isinstance(model.doi, list)
