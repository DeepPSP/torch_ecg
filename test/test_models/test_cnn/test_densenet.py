"""
"""

from copy import deepcopy

import torch

from torch_ecg.models.cnn.densenet import DenseNet
from torch_ecg.model_configs.cnn.densenet import (
    densenet_vanilla,
    densenet_leadwise,
)


IN_CHANNELS = 12


@torch.no_grad()
def test_densenet():
    inp = torch.randn(2, IN_CHANNELS, 5000)

    for item in [densenet_vanilla, densenet_leadwise]:
        config = deepcopy(item)
        model = DenseNet(in_channels=IN_CHANNELS, **config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
