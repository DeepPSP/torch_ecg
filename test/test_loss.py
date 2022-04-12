"""
"""

import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.models.loss import (
    AsymmetricLoss,
    BCEWithLogitsWithClassWeightLoss,
    FocalLoss,
    MaskedBCEWithLogitsLoss,
)

inp = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
targ_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
targ_0 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
targ_1_soft = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
targ_0_soft = torch.tensor([[0.1, 0.9], [0.9, 0.1]])

criterion_bce_cw = BCEWithLogitsWithClassWeightLoss(
    class_weight=torch.tensor([[1.0, 2.0]])
)
criterion_focal = FocalLoss(class_weight=torch.tensor([[1.0, 2.0]]), multi_label=True)
criterion_asl = AsymmetricLoss()

criterion_mbce = MaskedBCEWithLogitsLoss()
