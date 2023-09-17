"""
"""

import pytest
import torch

from torch_ecg.models.loss import (
    AsymmetricLoss,
    BCEWithLogitsWithClassWeightLoss,
    FocalLoss,
    MaskedBCEWithLogitsLoss,
    WeightedBCELoss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inp = torch.tensor([[10.0, -10.0], [-10.0, 10.0], [-10.0, 10.0]]).to(DEVICE)
targ_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]).to(DEVICE)
targ_0 = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]).to(DEVICE)
targ_mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]).to(DEVICE)
targ_1_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]).to(DEVICE)
targ_0_soft = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.9, 0.1]]).to(DEVICE)
targ_mixed_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]).to(DEVICE)

class_weight = torch.tensor([1.0, 2.0]).to(DEVICE)


def test_wbce():
    """ """
    criterion_wbce = WeightedBCELoss(torch.ones((1, 2)), PosWeightIsDynamic=True).to(DEVICE)
    assert criterion_wbce(torch.sigmoid(inp), targ_1).item() == pytest.approx(0.0, abs=1e-4)
    assert criterion_wbce(torch.sigmoid(inp), targ_0).item() > 1.0
    assert criterion_wbce(torch.sigmoid(inp), targ_mixed).item() > 1.0 / 3
    assert criterion_wbce(torch.sigmoid(inp), targ_1_soft).item() > criterion_wbce(torch.sigmoid(inp), targ_1).item()
    assert criterion_wbce(torch.sigmoid(inp), targ_0_soft).item() < criterion_wbce(torch.sigmoid(inp), targ_0).item()
    assert (
        criterion_wbce(torch.sigmoid(inp), targ_1).item()
        < criterion_wbce(torch.sigmoid(inp), targ_mixed_soft).item()
        < criterion_wbce(torch.sigmoid(inp), targ_0).item()
    )
    assert (
        criterion_wbce(torch.sigmoid(inp), targ_1_soft).item()
        < criterion_wbce(torch.sigmoid(inp), targ_mixed_soft).item()
        < criterion_wbce(torch.sigmoid(inp), targ_0_soft).item()
    )

    criterion_wbce = WeightedBCELoss(torch.ones((1, 2)), reduce=False).to(DEVICE)
    criterion_wbce(torch.sigmoid(inp), targ_1)
    criterion_wbce = WeightedBCELoss(torch.ones((1, 2)), size_average=False).to(DEVICE)
    criterion_wbce(torch.sigmoid(inp), targ_1)

    with pytest.raises(ValueError, match="Target size \\(.+\\) must be the same as input size \\(.+\\)"):
        criterion_wbce = WeightedBCELoss(torch.ones((1, 2))).to(DEVICE)
        criterion_wbce(torch.sigmoid(inp), targ_1[:, 0:1])


def test_bce_cw():
    """ """
    criterion_bce_cw = BCEWithLogitsWithClassWeightLoss(class_weight=class_weight).to(DEVICE)
    for targ in [targ_1, targ_0, targ_1_soft, targ_0_soft]:
        loss_1 = criterion_bce_cw(inp, targ)
        loss_2 = -class_weight * (targ * torch.log(torch.sigmoid(inp)) + (1 - targ) * torch.log(1 - torch.sigmoid(inp)))
        loss_2 = loss_2.mean()
        assert torch.allclose(loss_1, loss_2, atol=1e-3)


def test_focal():
    """ """
    criterion_focal = FocalLoss(class_weight=class_weight, multi_label=True).to(DEVICE)
    assert criterion_focal(inp, targ_1).item() == pytest.approx(0.0, abs=1e-6)
    assert criterion_focal(inp, targ_0).item() > 1.0
    assert criterion_focal(inp, targ_mixed).item() > 1.0 / 3
    assert criterion_focal(inp, targ_1_soft).item() > criterion_focal(inp, targ_1).item()
    assert criterion_focal(inp, targ_0_soft).item() < criterion_focal(inp, targ_0).item()
    assert (
        criterion_focal(inp, targ_1).item() < criterion_focal(inp, targ_mixed_soft).item() < criterion_focal(inp, targ_0).item()
    )
    assert (
        criterion_focal(inp, targ_1_soft).item()
        < criterion_focal(inp, targ_mixed_soft).item()
        < criterion_focal(inp, targ_0_soft).item()
    )
    assert torch.allclose(criterion_focal.alpha, class_weight, atol=1e-3)

    criterion_focal = FocalLoss(class_weight=class_weight.unsqueeze(0), multi_label=False, reduction="sum").to(DEVICE)
    criterion_focal(inp, targ_1)


def test_asl():
    """ """
    criterion_asl = AsymmetricLoss().to(DEVICE)
    assert criterion_asl(inp, targ_1).item() == pytest.approx(0.0, abs=1e-6)
    assert criterion_asl(inp, targ_0).item() > 1.0
    assert criterion_asl(inp, targ_mixed).item() > 1.0 / 3
    assert criterion_asl(inp, targ_1_soft).item() > criterion_asl(inp, targ_1).item()
    assert criterion_asl(inp, targ_0_soft).item() < criterion_asl(inp, targ_0).item()
    assert criterion_asl(inp, targ_1).item() < criterion_asl(inp, targ_mixed_soft).item() < criterion_asl(inp, targ_0).item()
    assert (
        criterion_asl(inp, targ_1_soft).item()
        < criterion_asl(inp, targ_mixed_soft).item()
        < criterion_asl(inp, targ_0_soft).item()
    )

    criterion_asl = AsymmetricLoss(implementation="deep-psp").to(DEVICE)
    assert criterion_asl(inp, targ_1).item() == pytest.approx(0.0, abs=1e-6)
    assert criterion_asl(inp, targ_0).item() > 1.0
    assert criterion_asl(inp, targ_mixed).item() > 1.0 / 3
    assert criterion_asl(inp, targ_1_soft).item() > criterion_asl(inp, targ_1).item()
    assert criterion_asl(inp, targ_0_soft).item() < criterion_asl(inp, targ_0).item()
    assert criterion_asl(inp, targ_1).item() < criterion_asl(inp, targ_mixed_soft).item() < criterion_asl(inp, targ_0).item()
    assert (
        criterion_asl(inp, targ_1_soft).item()
        < criterion_asl(inp, targ_mixed_soft).item()
        < criterion_asl(inp, targ_0_soft).item()
    )

    criterion_asl = AsymmetricLoss(disable_torch_grad_focal_loss=True, reduction="sum").to(DEVICE)
    criterion_asl(inp, targ_1)

    criterion_asl = AsymmetricLoss(disable_torch_grad_focal_loss=True, reduction="none").to(DEVICE)
    criterion_asl(inp, targ_1)

    with pytest.raises(ValueError, match="`prob_margin` must be non-negative"):
        AsymmetricLoss(prob_margin=-0.1)


def test_mbce():
    criterion_mbce = MaskedBCEWithLogitsLoss().to(DEVICE)
    weight_mask = torch.ones_like(inp).to(DEVICE)
    weight_mask[:, 0] = 10.0
    assert criterion_mbce(inp, targ_1, weight_mask).item() == pytest.approx(0.0, abs=1e-3)
    assert criterion_mbce(inp, targ_0, weight_mask).item() > 10.0 / 3
    assert criterion_mbce(inp, targ_mixed, weight_mask).item() > 10.0 / 3
    assert criterion_mbce(inp, targ_1_soft, weight_mask).item() > criterion_mbce(inp, targ_1, weight_mask).item()
    assert criterion_mbce(inp, targ_0_soft, weight_mask).item() < criterion_mbce(inp, targ_0, weight_mask).item()
    assert (
        criterion_mbce(inp, targ_1, weight_mask).item()
        < criterion_mbce(inp, targ_mixed_soft, weight_mask).item()
        < criterion_mbce(inp, targ_0, weight_mask).item()
    )
