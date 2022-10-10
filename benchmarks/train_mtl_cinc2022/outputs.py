"""
"""

from typing import Optional
from dataclasses import dataclass

from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)


__all__ = ["CINC2022Outputs"]


@dataclass
class CINC2022Outputs:
    """ """

    murmur_output: ClassificationOutput
    outcome_output: ClassificationOutput
    segmentation_output: SequenceLabellingOutput
    murmur_loss: Optional[float] = None
    outcome_loss: Optional[float] = None
    segmentation_loss: Optional[float] = None
