"""
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from torch_ecg.models.loss import (
    BCEWithLogitsWithClassWeightLoss,
    MaskedBCEWithLogitsLoss,
    FocalLoss,
    AsymmetricLoss,
)


inp = torch.tensor([[10.0,-10.0],[-10.0,10.0]])
targ_1 = torch.tensor([[1.0,0.0],[0.0,1.0]])
targ_0 = torch.tensor([[0.0,1.0],[1.0,0.0]])
targ_1_soft = torch.tensor([[0.9,0.1],[0.1,0.9]])
targ_0_soft = torch.tensor([[0.1,0.9],[0.9,0.1]])

criterion_bce_cw = BCEWithLogitsWithClassWeightLoss(class_weight=torch.tensor([[1.0,2.0]]))
criterion_focal = FocalLoss(class_weight=torch.tensor([[1.0,2.0]]), multi_label=True)
criterion_asl = AsymmetricLoss()

criterion_mbce = MaskedBCEWithLogitsLoss()
