"""
"""

from copy import deepcopy
from pathlib import Path

import torch
import pytest

from torch_ecg.models.rr_lstm import RR_LSTM, RR_LSTM_v1
from torch_ecg.model_configs.rr_lstm import (
    RR_AF_VANILLA_CONFIG,
    RR_AF_CRF_CONFIG,
    RR_LSTM_CONFIG,
)


_TMP_DIR = Path(__file__).parents[1] / "tmp"
_TMP_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def test_rr_lstm():
    in_channels = 1
    classes = ["NSR", "AF", "PVC", "LBBB", "RBBB", "PAB", "VFL"]
    inp = torch.randn(100, 2, in_channels)
    inp_bf = torch.randn(2, in_channels, 100)

    config = deepcopy(RR_LSTM_CONFIG)
    config.clf.name = "crf"
    for attn_name in ["none"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[0], batch_size=inp.shape[1]
        )
        model_v1 = RR_LSTM_v1(classes=classes, config=config)
        model_v1 = model_v1.eval()
        out_v1 = model_v1(inp)
        model_v1.compute_output_shape(seq_len=inp.shape[0], batch_size=inp.shape[1])
        model.lstm.load_state_dict(model_v1.lstm.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(model_v1.attn.state_dict())
        model.clf.load_state_dict(model_v1.clf.state_dict())
    config = deepcopy(RR_LSTM_CONFIG)
    config.clf.name = "crf"
    config.batch_first = True
    for attn_name in ["none"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp_bf)
        assert out.shape == model.compute_output_shape(
            seq_len=inp_bf.shape[-1], batch_size=inp_bf.shape[0]
        )

    config = deepcopy(RR_LSTM_CONFIG)
    config.clf.name = "linear"
    for attn_name in ["none", "gc", "nl", "se"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[0], batch_size=inp.shape[1]
        )
        model_v1 = RR_LSTM_v1(classes=classes, config=config)
        model_v1 = model_v1.eval()
        out_v1 = model_v1(inp)
        model_v1.compute_output_shape(seq_len=inp.shape[0], batch_size=inp.shape[1])
        model.lstm.load_state_dict(model_v1.lstm.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(model_v1.attn.state_dict())
        model.clf.load_state_dict(model_v1.clf.state_dict())
    config = deepcopy(RR_LSTM_CONFIG)
    config.clf.name = "linear"
    config.batch_first = True
    for attn_name in ["none", "gc", "nl", "se"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp_bf)
        assert out.shape == model.compute_output_shape(
            seq_len=inp_bf.shape[-1], batch_size=inp_bf.shape[0]
        )

    config = deepcopy(RR_AF_VANILLA_CONFIG)
    for attn_name in ["none", "gc", "nl", "se"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[0], batch_size=inp.shape[1]
        )
        model_v1 = RR_LSTM_v1(classes=classes, config=config)
        model_v1 = model_v1.eval()
        out_v1 = model_v1(inp)
        model_v1.compute_output_shape(seq_len=inp.shape[0], batch_size=inp.shape[1])
        model.lstm.load_state_dict(model_v1.lstm.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(model_v1.attn.state_dict())
        model.clf.load_state_dict(model_v1.clf.state_dict())
    config = deepcopy(RR_AF_VANILLA_CONFIG)
    config.batch_first = True
    for attn_name in ["none", "gc", "nl", "se"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp_bf)
        assert out.shape == model.compute_output_shape(
            seq_len=inp_bf.shape[-1], batch_size=inp_bf.shape[0]
        )

    config = deepcopy(RR_AF_CRF_CONFIG)
    for attn_name in ["none"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp)
        assert out.shape == model.compute_output_shape(
            seq_len=inp.shape[0], batch_size=inp.shape[1]
        )
        model_v1 = RR_LSTM_v1(classes=classes, config=config)
        model_v1 = model_v1.eval()
        out_v1 = model_v1(inp)
        model_v1.compute_output_shape(seq_len=inp.shape[0], batch_size=inp.shape[1])
        model.lstm.load_state_dict(model_v1.lstm.state_dict())
        if model.attn.__class__.__name__ != "Identity":
            model.attn.load_state_dict(model_v1.attn.state_dict())
        model.clf.load_state_dict(model_v1.clf.state_dict())
    config = deepcopy(RR_AF_CRF_CONFIG)
    config.batch_first = True
    for attn_name in ["none"]:
        config.attn.name = attn_name
        model = RR_LSTM(classes=classes, config=config)
        model = model.eval()
        out = model(inp_bf)
        assert out.shape == model.compute_output_shape(
            seq_len=inp_bf.shape[-1], batch_size=inp_bf.shape[0]
        )

    config = deepcopy(RR_LSTM_CONFIG)
    config.lstm.retseq = False
    config.clf.name = "linear"
    config.attn.name = "none"
    model = RR_LSTM(classes=classes, config=config)
    model = model.eval()
    out = model(inp)
    assert out.shape == model.compute_output_shape(
        seq_len=inp.shape[0], batch_size=inp.shape[1]
    )
    model_v1 = RR_LSTM_v1(classes=classes, config=config)
    model_v1 = model_v1.eval()
    out_v1 = model_v1(inp)
    model_v1.compute_output_shape(seq_len=inp.shape[0], batch_size=inp.shape[1])
    model.lstm.load_state_dict(model_v1.lstm.state_dict())
    if model.attn.__class__.__name__ != "Identity":
        model.attn.load_state_dict(model_v1.attn.state_dict())
    model.clf.load_state_dict(model_v1.clf.state_dict())
    config = deepcopy(RR_LSTM_CONFIG)
    config.lstm.retseq = False
    config.clf.name = "linear"
    config.batch_first = True
    config.attn.name = "none"
    model = RR_LSTM(classes=classes, config=config)
    model = model.eval()
    out = model(inp_bf)
    assert out.shape == model.compute_output_shape(
        seq_len=inp_bf.shape[-1], batch_size=inp_bf.shape[0]
    )

    doi = model.doi
    assert isinstance(doi, list)
    assert all([isinstance(d, str) for d in doi])

    doi = model_v1.doi
    assert isinstance(doi, list)
    assert all([isinstance(d, str) for d in doi])


def test_warns_errors():
    in_channels = 1
    classes = ["NSR", "AF", "PVC", "LBBB", "RBBB", "PAB", "VFL"]
    inp = torch.randn(100, 2, in_channels)

    with pytest.warns(
        RuntimeWarning, match="No config is provided, using default config"
    ):
        model = RR_LSTM(classes=classes)
    with pytest.warns(
        RuntimeWarning, match="No config is provided, using default config"
    ):
        model_v1 = RR_LSTM_v1(classes=classes)

    config = deepcopy(RR_LSTM_CONFIG)
    config.lstm.retseq = False
    config.attn.name = "gc"
    config.clf.name = "linear"
    with pytest.warns(
        RuntimeWarning,
        match="Attention is not supported when lstm is not returning sequences",
    ):
        model = RR_LSTM(classes=classes, config=config)
    with pytest.warns(
        RuntimeWarning,
        match="Attention is not supported when lstm is not returning sequences",
    ):
        model_v1 = RR_LSTM_v1(classes=classes, config=config)

    config = deepcopy(RR_LSTM_CONFIG)
    config.lstm.retseq = False
    config.attn.name = "none"
    config.clf.name = "crf"
    with pytest.warns(
        RuntimeWarning,
        match="CRF layer is not supported in non-sequence mode, using linear instead",
    ):
        model = RR_LSTM(classes=classes, config=config)
    with pytest.warns(
        RuntimeWarning,
        match="CRF layer is not supported in non-sequence mode, using linear instead",
    ):
        model_v1 = RR_LSTM_v1(classes=classes, config=config)

    config = deepcopy(RR_AF_CRF_CONFIG)
    config.global_pool = "max"
    with pytest.warns(
        RuntimeWarning,
        match="Global pooling \042.+\042 is ignored for CRF prediction head",
    ):
        model = RR_LSTM(classes=classes, config=config)
    with pytest.warns(
        RuntimeWarning,
        match="Global pooling \042.+\042 is ignored for CRF prediction head",
    ):
        model_v1 = RR_LSTM_v1(classes=classes, config=config)

    with pytest.raises(
        NotImplementedError, match="implement a task specific inference method"
    ):
        model.inference(inp)
    with pytest.raises(
        NotImplementedError, match="implement a task specific inference method"
    ):
        model_v1.inference(inp)

    config = deepcopy(RR_LSTM_CONFIG)
    config.attn.name = "not_implemented"
    config.attn.not_implemented = {}
    with pytest.raises(
        NotImplementedError, match="Attn module \042.+\042 not implemented yet"
    ):
        model = RR_LSTM(classes=classes, config=config)
    with pytest.raises(
        NotImplementedError, match="Attn module \042.+\042 not implemented yet"
    ):
        model_v1 = RR_LSTM_v1(classes=classes, config=config)

    config = deepcopy(RR_LSTM_CONFIG)
    config.clf.name = "linear"
    config.global_pool = "not_supported"
    with pytest.raises(
        NotImplementedError, match="Pooling type \042.+\042 not supported"
    ):
        model = RR_LSTM(classes=classes, config=config)
    with pytest.raises(
        NotImplementedError, match="Pooling type \042.+\042 not supported"
    ):
        model_v1 = RR_LSTM_v1(classes=classes, config=config)


def test_from_v1():
    config = deepcopy(RR_LSTM_CONFIG)
    classes = ["NSR", "AF", "PVC", "LBBB", "RBBB", "PAB", "VFL"]
    model_v1 = RR_LSTM_v1(classes=classes, config=config)
    model_v1.save(_TMP_DIR / "rr_lstm_v1.pth", {"classes": classes})
    model = RR_LSTM.from_v1(_TMP_DIR / "rr_lstm_v1.pth")
    (_TMP_DIR / "rr_lstm_v1.pth").unlink()
    del model_v1, model
