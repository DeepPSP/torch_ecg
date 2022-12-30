"""
"""

from copy import deepcopy

import torch

from torch_ecg.models.cnn.xception import Xception
from torch_ecg.model_configs.cnn.xception import xception_vanilla, xception_leadwise


IN_CHANNELS = 12


@torch.no_grad()
def test_xception():
    inp = torch.randn(2, IN_CHANNELS, 2000)

    for item in [xception_vanilla, xception_leadwise]:
        config = deepcopy(item)
        model = Xception(in_channels=IN_CHANNELS, **config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
