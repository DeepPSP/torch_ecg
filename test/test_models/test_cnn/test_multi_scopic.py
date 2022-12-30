"""
"""

from copy import deepcopy

import numpy as np
import torch

from torch_ecg.models import ECG_CRNN
from torch_ecg.model_configs import ECG_CRNN_CONFIG
from torch_ecg.models.cnn.multi_scopic import MultiScopicCNN
from torch_ecg.model_configs.cnn.multi_scopic import multi_scopic, multi_scopic_leadwise
from torch_ecg.utils.misc import list_sum


IN_CHANNELS = 12


@torch.no_grad()
def test_multi_scopic():
    inp = torch.randn(2, IN_CHANNELS, 2000)

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


@torch.no_grad()
def test_assign_weights_lead_wise():
    # we create models using 12-lead ECGs and using reduced 6-lead ECGs
    indices = [0, 1, 2, 3, 4, 10]  # chosen randomly, no special meaning
    # chose the lead-wise models
    lead_12_config = deepcopy(ECG_CRNN_CONFIG)
    lead_12_config.cnn.name = "multi_scopic_leadwise"
    lead_6_config = deepcopy(ECG_CRNN_CONFIG)
    lead_6_config.cnn.name = "multi_scopic_leadwise"
    # adjust groups and numbers of filters
    lead_6_config.cnn.multi_scopic_leadwise.groups = 6
    # numbers of filters should be proportional to numbers of groups
    lead_6_config.cnn.multi_scopic_leadwise.num_filters = (
        (np.array([[192, 384, 768], [192, 384, 768], [192, 384, 768]]) / 2)
        .astype(int)
        .tolist()
    )
    # we assume that model12 is a trained model on 12-lead ECGs
    model12 = ECG_CRNN(["AF", "PVC", "NSR"], 12, lead_12_config)
    model6 = ECG_CRNN(["AF", "PVC", "NSR"], 6, lead_6_config)
    model12.eval()
    model6.eval()
    # we create tensor12, tensor6 to check the correctness of the assignment of the weights
    tensor12 = torch.zeros(1, 12, 200)  # batch, leads, seq_len
    tensor6 = torch.randn(1, 6, 200)
    # we make tensor12 has identical values as tensor6 at the given leads, and let the other leads have zero values
    tensor12[:, indices, :] = tensor6
    b = "branch_0"  # similarly for other branches
    _, output_shape_12, _ = model12.cnn.branches[b].compute_output_shape()
    _, output_shape_6, _ = model6.cnn.branches[b].compute_output_shape()
    units = output_shape_12 // 12
    out_indices = list_sum([[i * units + j for j in range(units)] for i in indices])

    # different feature maps
    assert (
        model6.cnn.branches[b](tensor6)
        == model12.cnn.branches[b](tensor12)[:, out_indices, :]
    ).all().item() is False

    # here, we assign weights from model12 that correspond to the given leads to model6
    model12.cnn.assign_weights_lead_wise(model6.cnn, indices)
    # identical feature maps
    assert (
        model6.cnn.branches[b](tensor6)
        == model12.cnn.branches[b](tensor12)[:, out_indices, :]
    ).all().item() is True
