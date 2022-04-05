"""
1d analog grad cam,
in order to inspect attention area of the ECG deep models

References
----------
https://github.com/jacobgil/pytorch-grad-cam
"""

from typing import List, NoReturn, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..cfg import DEFAULTS

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "GradCam",
]


class FeatureExtractor(object):
    """
    Class for extracting activations and
    registering gradients from targetted intermediate layers
    """

    def __init__(self, model: nn.Module, target_layers: Sequence[str]) -> NoReturn:
        """

        Parameters
        ----------
        model: Module,
        target_layers: sequence of str,
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad: Tensor) -> NoReturn:
        """ """
        self.gradients.append(grad)

    def __call__(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """ """
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs.append(x)
        last_out = x
        return outputs, last_out


class ModelOutputs(object):
    """
    Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers.
    """

    def __init__(
        self, model: nn.Module, feature_module: nn.Module, target_layers: Sequence[str]
    ) -> NoReturn:
        """ """
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self) -> List[Tensor]:
        """ """
        return self.feature_extractor.gradients

    def __call__(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """ """
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


class GradCam(object):
    """NOT finished,"""

    __DEBUG__ = True
    __name__ = "GradCam"

    def __init__(
        self,
        model: nn.Module,
        feature_module: nn.Module,
        target_layer_names: Sequence[str],
        target_channel_last: bool = False,
        device: str = "cpu",
    ) -> NoReturn:
        """

        Parameters
        ----------
        to write
        """
        self.model = model
        self.feature_module = feature_module
        self.target_layer_names = target_layer_names
        self.target_channel_last = target_channel_last
        self.device = torch.device(device)

        self.model.eval()
        self.model.to(self.device)
        self.extractor = ModelOutputs(
            self.model, self.feature_module, self.target_layer_names
        )

    def forward(self, input: Tensor) -> Tensor:
        """ """
        return self.model(input)

    def __call__(self, input: Tensor, index: Optional[int] = None):
        """NOT finished,

        Parameters
        ----------
        input: Tensor,
            input tensor of shape (batch_size (=1), channels, seq_len)
        index: int, optional,
            the index of the output of the final classifying layer of `self.model`
        """
        # output of shape (batch_size (=1), n_classes)
        features, output = self.extractor(input.to(self.device))
        n_classes = output.shape[-1]

        if index is None:
            index = np.argmax(output.cpu().detach().numpy()[0])

        one_hot = np.zeros((1, n_classes), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device)
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().detach().numpy()

        # of shape (batch_size (=1), channels, seq_len) or (batch_size (=1), seq_len, channels)
        target = features[-1]
        # of shape (channels, seq_len) or (seq_len, channels)
        target = target.cpu().detach().numpy()[0, :]

        if self.target_channel_last:
            weights = np.mean(grads_val, axis=-2)[0, :]
        else:
            weights = np.mean(grads_val, axis=-1)[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # cam = np.maximum(cam, 0)
        # cam = cv2.resize(cam, input.shape[2:])
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        return cam
