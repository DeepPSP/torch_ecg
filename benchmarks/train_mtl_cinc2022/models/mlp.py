"""
Currently NOT used, NOT tested.
"""

from copy import deepcopy
from typing import Union, Optional, Any

import numpy as np
import torch
from torch import Tensor
from torch_ecg.cfg import CFG
from torch_ecg.models._nets import MLP
from torch_ecg.components.outputs import (
    ClassificationOutput,
    SequenceLabellingOutput,
)
from torch_ecg.utils import add_docstring, CkptMixin

from cfg import ModelCfg


__all__ = ["OutComeMLP"]


class OutComeMLP(MLP, CkptMixin):
    """ """

    __name__ = "OutComeMLP"

    def __init__(
        self, in_channels: int, config: Optional[CFG] = None, **kwargs: Any
    ) -> None:
        """ """
        _config = CFG(deepcopy(ModelCfg.outcome))
        _config.update(deepcopy(config) or {})
        self.config = _config[_config.mlp]
        super().__init__(
            in_channels,
            out_channels=self.config.out_channels + [len(self.config.classes)],
            activation=self.config.activation,
            bias=self.config.bias,
            dropouts=self.config.dropouts,
            skip_last_activation=True,
        )

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> ClassificationOutput:
        """ """
        self.eval()
        raise NotImplementedError

    @add_docstring(inference.__doc__)
    def inference_CINC2022(
        self,
        input: Union[np.ndarray, Tensor],
    ) -> SequenceLabellingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input)
