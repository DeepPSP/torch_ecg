"""
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from cfg import BaseCfg

from torch_ecg.components.outputs import ClassificationOutput

__all__ = [
    "CINC2023Outputs",
]


cpc_map_inv = {v: k for k, v in BaseCfg.cpc_map.items()}
outcome_map_inv = {v: k for k, v in BaseCfg.outcome_map.items()}
cpc2outcome_map = {
    # map of mapped cpc (int) to mapped outcome (int)
    k: BaseCfg.outcome_map[BaseCfg.cpc2outcome_map[v]]
    for k, v in cpc_map_inv.items()
}  # {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
cpc2outcome_map_inv = {
    v: [k for k, v_ in cpc2outcome_map.items() if v == v_] for v in set(cpc2outcome_map.values())
}  # {0: [0, 1], 1: [2, 3, 4]}


@dataclass
class CINC2023Outputs:
    """Output class for CinC2023.

    Attributes
    ----------
    cpc_output : ClassificationOutput
        Output container for CPC, containing the predicted classes, probabilities, etc.
    cpc_loss : Sequence[float]
        Loss for CPC.
    cpc_value : Sequence[float]
        CPC value (float), i.e., the predicted value of the CPC.
    outcome_output : ClassificationOutput
        Output container for outcome, containing the predicted classes, probabilities, etc.
    outcome_loss : Sequence[float]
        Loss for outcome.
    outcome : Sequence[str]
        Outcome, i.e., the predicted class names (str) of the outcome.

    .. note::

        - If `cpc_output` is provided, then `outcome_output` will be inferred from `cpc_output`.
        - `outcome` will be inferred from `outcome_output` if `outcome` is not provided.
          Otherwise, consistency check will be performed between `outcome` and `outcome_output`.
        - `cpc_value` will be inferred from `cpc_output` if `cpc_value` is not provided.
          Otherwise, consistency check will be performed between `cpc_value` and `cpc_output`.

    """

    cpc_output: Optional[ClassificationOutput] = None
    cpc_loss: Optional[Sequence[float]] = None
    cpc_value: Optional[Sequence[float]] = None
    outcome_output: Optional[ClassificationOutput] = None
    outcome_loss: Optional[Sequence[float]] = None
    outcome: Optional[Sequence[str]] = None

    def __post_init__(self):
        assert any(
            [
                self.cpc_output is not None,
                self.outcome_output is not None,
            ]
        ), "At least one output should be provided"

        if self.outcome_output is None:
            prob = np.zeros((len(self.cpc_output.pred), len(BaseCfg.outcome)))
            # merge the probablities of the same outcome via max
            for k, v in cpc2outcome_map_inv.items():
                prob[:, k] = self.cpc_output.prob[:, v].max(axis=1)
            # apply the softmax
            prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True)
            self.outcome_output = ClassificationOutput(
                classes=BaseCfg.outcome,
                pred=np.array([cpc2outcome_map[p] for p in self.cpc_output.pred]),
                prob=prob,
            )

        if self.outcome_output is not None:
            outcome = [outcome_map_inv[p] for p in self.outcome_output.pred]
            if self.outcome is None:
                self.outcome = outcome
            else:
                assert len(self.outcome) == len(outcome) and all(
                    [o1 == o2 for o1, o2 in zip(self.outcome, outcome)]
                ), "the provided outcome is not consistent with the outcome_output"

        if self.cpc_output is not None:
            cpc_value = [float(cpc_map_inv[p]) for p in self.cpc_output.pred]
            if self.cpc_value is None:
                self.cpc_value = cpc_value
            else:
                assert len(self.cpc_value) == len(cpc_value) and all(
                    [v1 == v2 for v1, v2 in zip(self.cpc_value, cpc_value)]
                ), "the provided cpc_value is not consistent with the cpc_output"
