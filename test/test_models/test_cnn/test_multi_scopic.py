"""
"""

from copy import deepcopy

import torch

from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.model_configs.cnn.multi_scopic import multi_scopic, multi_scopic_leadwise


IN_CHANNELS = 12


@torch.no_grad()
def test_multi_scopic():
    inp = torch.randn(2, IN_CHANNELS, 5000)

    for item in [multi_scopic, multi_scopic_leadwise]:
        config = deepcopy(item)
        model = MultiScopicCNN(in_channels=IN_CHANNELS, **config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
