"""
"""

import itertools
from copy import deepcopy

import pytest
import torch
from tqdm.auto import tqdm

from torch_ecg.models import ECG_CRNN
from torch_ecg.model_configs import ECG_CRNN_CONFIG


def test_ecg_crnn():
    n_leads = 12
    classes = ["NSR", "AF", "PVC", "LBBB", "RBBB", "PAB", "VFL"]
    inp = torch.randn(2, n_leads, 5000)

    with pytest.warns(
        RuntimeWarning, match="No config is provided, using default config"
    ):
        model = ECG_CRNN(classes=classes, n_leads=n_leads)

    grid = itertools.product(
        [cnn_name for cnn_name in ECG_CRNN_CONFIG.cnn.keys() if cnn_name != "name"],
        [rnn_name for rnn_name in ECG_CRNN_CONFIG.rnn.keys() if rnn_name != "name"]
        + ["none"],
        [attn_name for attn_name in ECG_CRNN_CONFIG.attn.keys() if attn_name != "name"]
        + ["none"],
    )
    total = (
        (len(ECG_CRNN_CONFIG.cnn.keys()) - 1)
        * len(ECG_CRNN_CONFIG.rnn.keys())
        * len(ECG_CRNN_CONFIG.attn.keys())
    )

    for cnn_name, rnn_name, attn_name in tqdm(grid, total=total):
        config = deepcopy(ECG_CRNN_CONFIG)
        config.cnn.name = cnn_name
        config.rnn.name = rnn_name
        config.attn.name = attn_name
        model = ECG_CRNN(classes=classes, n_leads=n_leads, config=config)
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[-1], batch_size=inp.shape[0]
        )

    doi = model.doi
    assert isinstance(doi, list)
    assert all([isinstance(d, str) for d in doi])
