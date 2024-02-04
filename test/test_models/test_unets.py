"""
"""

import pytest
import torch

from torch_ecg.model_configs import ECG_SUBTRACT_UNET_CONFIG, ECG_UNET_VANILLA_CONFIG
from torch_ecg.models.unets.ecg_subtract_unet import ECG_SUBTRACT_UNET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def test_ecg_unet():
    inp = torch.randn(2, 12, 5000).to(DEVICE)
    fs = 400
    classes = ["p", "N", "t", "i"]

    config = adjust_cnn_filter_lengths(ECG_UNET_VANILLA_CONFIG, fs)

    model = ECG_UNET(classes=classes, n_leads=12, config=config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(seq_len=inp.shape[-1], batch_size=inp.shape[0])

    with pytest.warns(RuntimeWarning, match="No config is provided, using default config"):
        model = ECG_UNET(classes=classes, n_leads=12).to(DEVICE)
        model = model.eval()

    doi = model.doi
    assert isinstance(doi, list)
    assert all([isinstance(d, str) for d in doi]), doi

    with pytest.raises(NotImplementedError, match="implement a task specific inference method"):
        model.inference(inp)


@torch.no_grad()
def test_ecg_subtract_unet():
    inp = torch.randn(2, 12, 5000).to(DEVICE)
    fs = 400
    classes = ["p", "N", "t", "i"]

    config = adjust_cnn_filter_lengths(ECG_SUBTRACT_UNET_CONFIG, fs)

    model = ECG_SUBTRACT_UNET(classes=classes, n_leads=12, config=config).to(DEVICE)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(seq_len=inp.shape[-1], batch_size=inp.shape[0])

    with pytest.warns(RuntimeWarning, match="No config is provided, using default config"):
        model = ECG_SUBTRACT_UNET(classes=classes, n_leads=12).to(DEVICE)
        model = model.eval()

    doi = model.doi
    assert isinstance(doi, list)
    assert all([isinstance(d, str) for d in doi]), doi

    with pytest.raises(NotImplementedError, match="implement a task specific inference method"):
        model.inference(inp)
