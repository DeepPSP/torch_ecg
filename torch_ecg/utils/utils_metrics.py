"""
utilities for computing metrics.

NOTE that only the widely used metrics are implemented here,
challenge (e.g. CinC, CPSC series) specific metrics are not included.

"""

from typing import Union, Optional, Dict

import numpy as np
import einops
import torch
from torch import Tensor
from torch import nn


__all__ = [
    "top_n_accuracy",
]


def top_n_accuracy(preds: Tensor, labels: Tensor, n: int = 1) -> float:
    """

    Parameters
    ----------
    preds: Tensor,
        of shape (batch_size, num_classes) or (batch_size, num_classes, d_1, ..., d_m)
    labels: Tensor,
        of shape (batch_size,) or (batch_size, d_1, ..., d_m)
    n: int,
        top n to be considered

    Returns
    -------
    acc: float,
        top n accuracy

    """
    assert preds.shape[0] == labels.shape[0]
    batch_size, n_classes, *extra_dims = preds.shape
    _, indices = torch.topk(
        preds, n, dim=1
    )  # of shape (batch_size, n) or (batch_size, n, d_1, ..., d_n)
    pattern = " ".join([f"d_{i+1}" for i in range(len(extra_dims))])
    pattern = f"batch_size {pattern} -> batch_size n {pattern}"
    correct = torch.sum(indices == einops.repeat(labels, pattern, n=n))
    acc = correct.item() / preds.shape[0]
    for d in extra_dims:
        acc = acc / d
    return acc
