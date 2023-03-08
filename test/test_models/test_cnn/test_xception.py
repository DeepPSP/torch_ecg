"""
"""

from copy import deepcopy

import torch
from tqdm.auto import tqdm

from torch_ecg.models.cnn.xception import Xception
from torch_ecg.model_configs.cnn.xception import xception_vanilla, xception_leadwise


IN_CHANNELS = 12


@torch.no_grad()
def test_xception():
    inp = torch.randn(2, IN_CHANNELS, 2000)

    dropouts = [{"p": 0.1, "type": "1d"}, {"p": 0.2, "type": None}, 0.1]
    model_configs = [xception_vanilla, xception_leadwise]
    for flow in ["entry_flow", "middle_flow", "exit_flow"]:
        for d in dropouts:
            config = deepcopy(xception_vanilla)
            config[flow]["dropouts"] = d
            model_configs.append(config)
    for item in tqdm(
        model_configs,
        mininterval=1,
        desc="Testing Xception",
    ):
        model = Xception(in_channels=IN_CHANNELS, **config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)
