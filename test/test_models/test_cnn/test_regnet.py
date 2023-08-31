"""
"""

from copy import deepcopy

import pytest
import torch
from tqdm.auto import tqdm

from torch_ecg.models.cnn.regnet import RegNet
from torch_ecg.model_configs.cnn.regnet import (
    regnet_16_8,
    regnet_27_24,
    regnet_23_168,
    regnet_S,
)


IN_CHANNELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_regnet():
    inp = torch.randn(2, IN_CHANNELS, 2000).to(DEVICE)

    for item in tqdm(
        [regnet_16_8, regnet_27_24, regnet_23_168, regnet_S],
        mininterval=1,
        desc="Testing RegNet",
    ):
        config = deepcopy(item)
        model = RegNet(IN_CHANNELS, **config).to(DEVICE)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(2000, 2)
        assert model.in_channels == IN_CHANNELS
        assert isinstance(model.doi, list)

    regnet_config = deepcopy(regnet_16_8)
    regnet_config.num_filters = 13
    with pytest.warns(
        RuntimeWarning,
        match=(
            "`num_filters` are computed from `config.w_a`, `config.w_0`, `config.w_m`, "
            "if `config.num_blocks` is not provided"
        ),
    ):
        RegNet(IN_CHANNELS, **regnet_config)

    regnet_config = deepcopy(regnet_16_8)
    regnet_config.pop("tot_blocks")
    with pytest.raises(
        AssertionError,
        match=(
            "If `config.num_blocks` is not provided, then `config.w_a`, `config.w_0`, `config.w_m`, "
            "and `config.tot_blocks` must be provided"
        ),
    ):
        RegNet(IN_CHANNELS, **regnet_config)

    regnet_config = deepcopy(regnet_16_8)
    regnet_config.w_a = -1
    with pytest.raises(ValueError, match="Invalid RegNet settings"):
        RegNet(IN_CHANNELS, **regnet_config)
    regnet_config = deepcopy(regnet_16_8)
    regnet_config.w_0 = 17
    with pytest.raises(ValueError, match="Invalid RegNet settings"):
        RegNet(IN_CHANNELS, **regnet_config)
    regnet_config = deepcopy(regnet_16_8)
    regnet_config.w_m = 1.0
    with pytest.raises(ValueError, match="Invalid RegNet settings"):
        RegNet(IN_CHANNELS, **regnet_config)

    regnet_config = deepcopy(regnet_S)
    regnet_config.stem.subsample_mode = "s2d"
    regnet_config.filter_lengths = 11
    regnet_config.subsample_lengths = 2
    regnet_config.group_widths = 16
    regnet_config.dropouts = [0.1, 0.2, 0.2, 0.1]
    model = RegNet(IN_CHANNELS, **regnet_config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(2000, 2)
