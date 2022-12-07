"""
"""

import pytest
import torch

from torch_ecg.models.loss import (
    AsymmetricLoss,
    BCEWithLogitsWithClassWeightLoss,
    FocalLoss,
    MaskedBCEWithLogitsLoss,
)


inp = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]])
targ_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
targ_0 = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
targ_mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
targ_1_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
targ_0_soft = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
targ_mixed_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])

class_weight = torch.tensor([1.0, 2.0])

criterion_bce_cw = BCEWithLogitsWithClassWeightLoss(class_weight=class_weight)
criterion_focal = FocalLoss(class_weight=class_weight, multi_label=True)
criterion_asl = AsymmetricLoss()

criterion_mbce = MaskedBCEWithLogitsLoss()


def test_bce_cw():
    """ """
    for targ in [targ_1, targ_0, targ_1_soft, targ_0_soft]:
        loss_1 = criterion_bce_cw(inp, targ)
        loss_2 = -class_weight * (
            targ * torch.log(torch.sigmoid(inp))
            + (1 - targ) * torch.log(1 - torch.sigmoid(inp))
        )
        loss_2 = loss_2.mean()
        assert torch.allclose(loss_1, loss_2, atol=1e-3)


def test_focal():
    """ """
    assert criterion_focal(inp, targ_1).item() == pytest.approx(0.0, abs=1e-6)
    assert criterion_focal(inp, targ_0).item() > 1.0
    assert criterion_focal(inp, targ_mixed).item() > 1.0 / 3
    assert (
        criterion_focal(inp, targ_1_soft).item() > criterion_focal(inp, targ_1).item()
    )
    assert (
        criterion_focal(inp, targ_0_soft).item() < criterion_focal(inp, targ_0).item()
    )
    assert (
        criterion_focal(inp, targ_1).item()
        < criterion_focal(inp, targ_mixed_soft).item()
        < criterion_focal(inp, targ_0).item()
    )
    assert (
        criterion_focal(inp, targ_1_soft).item()
        < criterion_focal(inp, targ_mixed_soft).item()
        < criterion_focal(inp, targ_0_soft).item()
    )


def test_asl():
    """ """
    assert criterion_asl(inp, targ_1).item() == pytest.approx(0.0, abs=1e-6)
    assert criterion_asl(inp, targ_0).item() > 1.0
    assert criterion_asl(inp, targ_mixed).item() > 1.0 / 3
    assert criterion_asl(inp, targ_1_soft).item() > criterion_asl(inp, targ_1).item()
    assert criterion_asl(inp, targ_0_soft).item() < criterion_asl(inp, targ_0).item()
    assert (
        criterion_asl(inp, targ_1).item()
        < criterion_asl(inp, targ_mixed_soft).item()
        < criterion_asl(inp, targ_0).item()
    )
    assert (
        criterion_asl(inp, targ_1_soft).item()
        < criterion_asl(inp, targ_mixed_soft).item()
        < criterion_asl(inp, targ_0_soft).item()
    )


def test_mbce():
    weight_mask = torch.ones_like(inp)
    weight_mask[:, 0] = 10.0
    assert criterion_mbce(inp, targ_1, weight_mask).item() == pytest.approx(
        0.0, abs=1e-3
    )
    assert criterion_mbce(inp, targ_0, weight_mask).item() > 10.0 / 3
    assert criterion_mbce(inp, targ_mixed, weight_mask).item() > 10.0 / 3
    assert (
        criterion_mbce(inp, targ_1_soft, weight_mask).item()
        > criterion_mbce(inp, targ_1, weight_mask).item()
    )
    assert (
        criterion_mbce(inp, targ_0_soft, weight_mask).item()
        < criterion_mbce(inp, targ_0, weight_mask).item()
    )
    assert (
        criterion_mbce(inp, targ_1, weight_mask).item()
        < criterion_mbce(inp, targ_mixed_soft, weight_mask).item()
        < criterion_mbce(inp, targ_0, weight_mask).item()
    )
