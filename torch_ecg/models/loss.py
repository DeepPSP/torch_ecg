"""custom loss functions"""

from numbers import Real
from typing import Union, Sequence, Tuple, List, Optional, NoReturn, Any

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


__all__ = [
    "WeightedBCELoss", "BCEWithLogitsWithClassWeightLoss",
    "MaskedBCEWithLogitsLoss",
    "FocalLoss",
    "AsymmetricLoss",
]


def weighted_binary_cross_entropy(sigmoid_x:Tensor,
                                  targets:Tensor,
                                  pos_weight:Tensor,
                                  weight:Optional[Tensor]=None,
                                  size_average:bool=True,
                                  reduce:bool=True) -> Tensor:
    """ finished, checked,

    Parameters
    ----------
    sigmoid_x: Tensor,
        predicted probability of size [N,C], N sample and C Class.
        Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
    targets: Tensor,
        true value, one-hot-like vector of size [N,C]
    pos_weight: Tensor,
        Weight for postive sample
    weight: Tensor, optional,
    size_average: bool, default True,
    reduce: bool, default True,

    Reference (original source):
    ----------------------------
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight * targets * sigmoid_x.log() - (1-targets) * (1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class WeightedBCELoss(nn.Module):
    """ finished, checked,

    Reference (original source):
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    __name__ = "WeightedBCELoss"

    def __init__(self,
                 pos_weight:Tensor=1,
                 weight:Optional[Tensor]=None,
                 PosWeightIsDynamic:bool=False,
                 WeightIsDynamic:bool=False,
                 size_average:bool=True,
                 reduce:bool=True) -> NoReturn:
        """ checked,

        Parameters
        ----------
        pos_weight: Tensor, default 1,
            Weight for postive samples. Size [1,C]
        weight: Tensor, optional,
            Weight for Each class. Size [1,C]
        PosWeightIsDynamic: bool, default False,
            If True, the pos_weight is computed on each batch.
            If `pos_weight` is None, then it remains None.
        WeightIsDynamic: bool, default False,
            If True, the weight is computed on each batch.
            If `weight` is None, then it remains None.
        size_average: bool, default True,
        reduce: bool, default True,
        """
        super().__init__()

        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            the prediction tensor, of shape (batch_size, ...)
        target: Tensor,
            the target tensor, of shape (batch_size, ...)

        Returns
        -------
        loss: Tensor,
            the loss w.r.t. `input` and `target`
        """
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts) / (positive_counts + 1e-5)

        return weighted_binary_cross_entropy(input, target,
                                             pos_weight=self.pos_weight,
                                             weight=self.weight,
                                             size_average=self.size_average,
                                             reduce=self.reduce)


class BCEWithLogitsWithClassWeightLoss(nn.BCEWithLogitsLoss):
    """ finished, checked,
    """
    __name__ = "BCEWithLogitsWithClassWeightsLoss"

    def __init__(self, class_weight:Tensor) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        class_weight: Tensor,
            class weight, of shape (1, n_classes)
        """
        super().__init__(reduction="none")
        self.class_weight = class_weight

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            the prediction tensor, of shape (batch_size, ...)
        target: Tensor,
            the target tensor, of shape (batch_size, ...)

        Returns
        -------
        loss: Tensor,
            the loss (scalar tensor) w.r.t. `input` and `target`
        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * self.class_weight)
        return loss


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """ finished, checked,
    """
    __name__ = "MaskedBCEWithLogitsLoss"

    def __init__(self) -> NoReturn:
        """ finished, checked,
        """
        super().__init__(reduction="none")

    def forward(self, input:Tensor, target:Tensor, weight_mask:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            the prediction tensor, of shape (batch_size, ...)
        target: Tensor,
            the target tensor, of shape (batch_size, ...)
        weight_mask: Tensor,
            the weight mask tensor, of shape (batch_size, ...)

        Returns
        -------
        loss: Tensor,
            the loss (scalar tensor) w.r.t. `input` and `target`

        NOTE
        ----
        `input`, `target`, and `weight_mask` should be 3-D tensors of the same shape
        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * weight_mask)
        return loss


class FocalLoss(nn.modules.loss._WeightedLoss):
    r"""

    the focal loss is computed as follows:
    .. math::
        \operatorname{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \log(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    References
    ----------
    1. Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
    2. https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
    3. https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    4. https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327
    """
    __name__ = "FocalLoss"

    def __init__(self,
                 gamma:float=2.0,
                 weight:Optional[Tensor]=None,
                 class_weight:Optional[Tensor]=None,  # alpha
                 size_average:Optional[bool]=None,
                 reduce:Optional[bool]=None,
                 reduction:str="mean",
                 multi_label:bool=True,
                 **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        gamma: float, default 2.0,
            the gamma parameter of focal loss
        weight: Tensor, optional,
            if `multi_label` is True,
            is a manual rescaling weight given to the loss of each batch element, of size `batch_size`;
            if `multi_label` is False,
            is a weight for each class, of size `n_classes`
        class_weight: Tensor, optional,
            the class weight, of shape (1, n_classes)
        size_average: bool, optional,
            not used, to keep in accordance with PyTorch native loss
        reduce: bool, optional,
            not used, to keep in accordance with PyTorch native loss
        reduction: str, default "mean",
            the reduction to apply to the output, can be one of
            "none", "mean", "sum"
        """
        if multi_label or weight is not None:
            w = weight
        else:
            w = class_weight
        if not multi_label and w.ndim == 2:
            w = w.squeeze(0)
        super().__init__(weight=w, size_average=size_average, reduce=reduce, reduction=reduction)
        # In practice `alpha` may be set by inverse class frequency or treated as a hyperparameter
        # the `class_weight` are usually inverse class frequencies
        # self.alpha = alpha
        self.gamma = gamma
        if multi_label:
            self.entropy_func = F.binary_cross_entropy_with_logits
            # for `binary_cross_entropy_with_logits`,
            # its parameter `weight` is a manual rescaling weight given to the loss of each batch element
            self.class_weight = class_weight
        else:
            self.entropy_func = F.cross_entropy
            # for `cross_entropy`,
            # its parameter `weight` is a manual rescaling weight given to each class
            self.class_weight = None

    @property
    def alpha(self) -> Tensor:
        return self.class_weight

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            input tensor, of shape (batch_size, n_classes)
        target: Tensor,
            multi-label binarized vector of shape (batch_size, n_classes),
            or single label binarized vector of shape (batch_size,)

        Returns
        -------
        fl: Tensor,
            the focal loss w.r.t. `input` and `target`
        """
        entropy = self.entropy_func(
            input, target,
            weight=self.weight, reduction="none",
        )
        p_t = torch.exp(-entropy)
        fl = torch.pow(1 - p_t, self.gamma) * entropy
        if self.class_weight is not None:
            fl = fl * self.class_weight
        if self.reduction == "mean":
            fl = fl.mean()
        elif self.reduction == "sum":
            fl = fl.sum()
        return fl


class AsymmetricLoss(nn.Module):
    """ finished, checked,

    References
    ----------
    1. Ridnik, Tal, et al. "Asymmetric Loss for Multi-Label Classification." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    2. https://github.com/Alibaba-MIIL/ASL/
    """
    __name__ = "AsymmetricLoss"

    def __init__(self,
                 gamma_neg:Real=4,
                 gamma_pos:Real=1,
                 prob_shift:float=0.05, 
                 disable_torch_grad_focal_loss:bool=False,
                 reduction:str="mean") -> NoReturn:
        """ finished, checked,
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.prob_shift = prob_shift
        if self.prob_shift < 0:
            raise ValueError("`prob_shift` must be non-negative")
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = 1e-8
        self.reduction = reduction.lower()

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """" finished, checked,

        Parameters
        ----------
        input: Tensor,
            input tensor, of shape (batch_size, n_classes)
        target: Tensor,
            multi-label binarized vector, of shape (batch_size, n_classes)

        Returns
        -------
        loss: Tensor,
            the loss (scalar tensor) w.r.t. `input` and `target`
        """
        self.targets = target
        self.anti_targets = 1 - target

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(input)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.prob_shift > 0:
            self.xs_neg.add_(self.prob_shift).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(prev)
            self.loss *= self.asymmetric_w

        if self.reduction == "mean":
            loss = -self.loss.mean()
        elif self.reduction == "sum":
            loss = -self.loss.sum()
        else:
            loss = -self.loss
        return loss
