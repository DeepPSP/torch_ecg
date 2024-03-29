"""
"""

from copy import deepcopy

import torch

from torch_ecg.model_configs.cnn.vgg import vgg16, vgg16_leadwise
from torch_ecg.models.cnn.vgg import VGG16

IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_vgg():
    inp = torch.randn(2, IN_CHANNELS, 2000).to(DEVICE)

    for item in [vgg16, vgg16_leadwise]:
        config = deepcopy(item)
        model = VGG16(in_channels=IN_CHANNELS, **config).to(DEVICE)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(seq_len=inp.shape[-1], batch_size=inp.shape[0])
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
