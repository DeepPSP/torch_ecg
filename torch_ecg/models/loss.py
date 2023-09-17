"""
torch_ecg.models.loss
=====================
Custom loss functions for ECG analysis, as a complement to
built-in loss functions in PyTorch.

.. contents:: torch_ecg.models
    :depth: 1
    :local:
    :backlinks: top

.. currentmodule:: torch_ecg.models.loss

.. autosummary::
    :toctree: generated/
    :recursive:

    WeightedBCELoss
    BCEWithLogitsWithClassWeightLoss
    MaskedBCEWithLogitsLoss
    FocalLoss
    AsymmetricLoss

"""

from numbers import Real
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "WeightedBCELoss",
    "BCEWithLogitsWithClassWeightLoss",
    "MaskedBCEWithLogitsLoss",
    "FocalLoss",
    "AsymmetricLoss",
]


def weighted_binary_cross_entropy(
    sigmoid_x: Tensor,
    targets: Tensor,
    pos_weight: Tensor,
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    reduce: bool = True,
) -> Tensor:
    """Weighted Binary Cross Entropy Loss function.

    This implementation is based on [#wbce]_.

    Parameters
    ----------
    sigmoid_x : torch.Tensor
        Predicted probability of size ``[N, C]``, N sample and C Class.
        Eg. Must be in range of ``[0, 1]``,
        i.e. output from :class:`~torch.nn.Sigmoid`.
    targets : torch.Tensor
        True value, one-hot-like vector of size ``[N, C]``.
    pos_weight : torch.Tensor
        Weight for postive sample.
    weight : torch.Tensor, optional
        Weight for each class, of size ``[1, C]``.
    size_average : bool, default True
        If True, the losses are averaged
        over each loss element in the batch.
        Valid only if `reduce` is True.
    reduce : bool, default True
        If True, the losses are averaged or summed
        over observations for each minibatch.

    Returns
    -------
    loss : torch.Tensor
        Weighted Binary Cross Entropy Loss.

    References
    ----------
    .. [#wbce] https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305

    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({sigmoid_x.size()})")

    loss = -pos_weight * targets * sigmoid_x.log() - (1 - targets) * (1 - sigmoid_x).log()
    # print(pos_weight, targets, sigmoid_x)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss class.

    This implementation is based on [#wbce]_.

    Parameters
    ----------
    pos_weight : torch.Tensor
        Weight for postive sample.
    weight : torch.Tensor, optional
        Weight for each class, of size ``[1, C]``.
    PosWeightIsDynamic : bool, default False
        If True, the pos_weight is computed on each batch.
        If `pos_weight` is None, then it remains None.
    WeightIsDynamic : bool, default False
        If True, the weight is computed on each batch.
        If `weight` is None, then it remains None.
    size_average : bool, default True
        If True, the losses are averaged
        over each loss element in the batch.
        Valid only if `reduce` is True.
    reduce : bool, default True
        If True, the losses are averaged or summed
        over observations for each minibatch.

    References
    ----------
    .. [#wbce] https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305

    """

    __name__ = "WeightedBCELoss"

    def __init__(
        self,
        pos_weight: Tensor,
        weight: Optional[Tensor] = None,
        PosWeightIsDynamic: bool = False,
        WeightIsDynamic: bool = False,
        size_average: bool = True,
        reduce: bool = True,
    ) -> None:
        super().__init__()

        self.register_buffer("pos_weight", pos_weight)
        if weight is None:
            weight = torch.ones_like(pos_weight)
        self.register_buffer("weight", weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            The predicted probability tensor,
            of shape ``(batch_size, ..., n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, ..., n_classes)``.

        Returns
        -------
        loss : torch.Tensor
            The weighted binary cross entropy loss.

        """
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0, keepdim=True)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts) / (positive_counts + 1e-7)

        return weighted_binary_cross_entropy(
            input,
            target,
            pos_weight=self.pos_weight,
            weight=self.weight,
            size_average=self.size_average,
            reduce=self.reduce,
        )


class BCEWithLogitsWithClassWeightLoss(nn.BCEWithLogitsLoss):
    """Class-weighted Binary Cross Entropy Loss class.

    Parameters
    ----------
    class_weight : torch.Tensor
        Class weight, of shape ``(1, n_classes)``.

    """

    __name__ = "BCEWithLogitsWithClassWeightLoss"

    def __init__(self, class_weight: Tensor) -> None:
        super().__init__(reduction="none")
        self.register_buffer("class_weight", class_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor (before sigmoid),
            of shape ``(batch_size, ..., n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, ..., n_classes)``.

        Returns
        -------
        torch.Tensor
            The class-weighted binary cross entropy loss.

        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * self.class_weight)
        return loss


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Masked Binary Cross Entropy Loss class.

    This loss is used mainly for the segmentation task, where
    there are some regions that are of much higher importance,
    for example, the onsets and offsets of some particular events
    (e.g. paroxysmal atrial fibrillation (AF) episodes).

    This loss is proposed in [#mbce]_, with a reference to the loss
    function used in the U-Net paper [#unet]_.


    References
    ----------
    .. [#mbce] Wen, Hao, and Jingsu Kang. "A comparative study on neural networks for
               paroxysmal atrial fibrillation events detection from electrocardiography."
               Journal of Electrocardiology 75 (2022): 19-27.
    .. [#unet] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional
               networks for biomedical image segmentation." International Conference on
               Medical image computing and computer-assisted intervention. Springer, Cham,
               2015.

    """

    __name__ = "MaskedBCEWithLogitsLoss"

    def __init__(self) -> None:
        super().__init__(reduction="none")

    def forward(self, input: Tensor, target: Tensor, weight_mask: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor (before sigmoid),
            of shape ``(batch_size, sig_len, n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, sig_len, n_classes)``.
        weight_mask: torch.Tensor
            The weight mask tensor,
            of shape ``(batch_size, sig_len, n_classes)``.

        Returns
        -------
        torch.Tensor
            The masked binary cross entropy loss.

        NOTE
        ----
        `input`, `target`, and `weight_mask` should be
        3-D tensors of the same shape.

        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * weight_mask)
        return loss


class FocalLoss(nn.modules.loss._WeightedLoss):
    """Focal loss class.

    The focal loss is proposed in [1]_, and this implementation is
    based on [2]_, [3]_, and [4]_. The focal loss is computed as follows:

    .. math::

        \\operatorname{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\log(p_t)

    Where:

       - :math:`p_t` is the model's estimated probability for each class.

    Parameters
    ----------
    gamma : float, default 2.0
        The gamma parameter of focal loss.
    weight : torch.Tensor, optional
        If `multi_label` is True,
        is a manual rescaling weight given to the loss of each batch element,
        of size ``batch_size``;
        if `multi_label` is False,
        is a weight for each class, of size ``n_classes``.
    class_weight : torch.Tensor, optional
        The class weight, of shape ``(1, n_classes)``.
    size_average : bool, optional
        Not used, to keep in accordance with PyTorch native loss.
    reduce : bool, optional
        Not used, to keep in accordance with PyTorch native loss.
    reduction: {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".
    multi_label : bool, default True
        If True, the loss is computed for multi-label classification.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection."
           Proceedings of the IEEE international conference on computer vision. 2017.
    .. [2] https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
    .. [3] https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    .. [4] https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327

    """

    __name__ = "FocalLoss"

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[Tensor] = None,
        class_weight: Optional[Tensor] = None,  # alpha
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        multi_label: bool = True,
        **kwargs: Any,
    ) -> None:
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
            self.register_buffer("class_weight", class_weight)
        else:
            self.entropy_func = F.cross_entropy
            # for `cross_entropy`,
            # its parameter `weight` is a manual rescaling weight given to each class
            self.class_weight = None

    @property
    def alpha(self) -> Tensor:
        return self.class_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor (before sigmoid),
            of shape ``(batch_size, n_classes)``.
        target : torch.Tensor
            Multi-label binarized vector of shape ``(batch_size, n_classes)``,
            or single label binarized vector of shape ``(batch_size,)``.

        Returns
        -------
        torch.Tensor
            The focal loss.

        """
        entropy = self.entropy_func(
            input,
            target,
            weight=self.weight,
            reduction="none",
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
    """Asymmetric loss class.

    The asymmetric loss is proposed in [#al]_, with official
    implementation in [#al_code]_. The asymmetric loss is defined as

    .. math::

        ASL = \\begin{cases}
            L_+ := (1-p)^{\\gamma_+} \\log(p) \\
            L_- := (p_m)^{\\gamma_-} \\log(1-p_m)
        \\end{cases}

    where :math:`p_m = \\max(p-m, 0)` is the shifted probability,
    with probability margin :math:`m`.
    The loss on one label of one sample is

    .. math::

        L = -yL_+ - (1-y)L_-

    Parameters
    ----------
    gamma_neg : numbers.Real, default 4
        Exponent of the multiplier to the negative loss.
    gamma_pos : numbers.Real, default 1
        Exponent of the multiplier to the positive loss.
    prob_margin : float, default 0.05
        The probability margin
    disable_torch_grad_focal_loss : bool, default False
        If True, disable :func:`torch.grad` for asymmetric focal loss computing.
    reduction : {"none", "mean", "sum"}, optional
        Specifies the reduction to apply to the output, by default "mean".
    implementation : {"alibaba-miil", "deep-psp"}, optional
        Implementation by Alibaba-MIIL, or by `DeepPSP`, case insensitive.

    NOTE
    ----
    Since :class:`AsymmetricLoss` aims at emphasizing the contribution of positive samples,
    `gamma_neg` is usually greater than `gamma_pos`.

    TODO
    ----
    1. Evaluate the settings that `gamma_neg`, `gamma_pos` are tensors,
       of shape ``(1, n_classes)``, in which case we would have one ratio
       of positive to negative for each class.

    References
    ----------
    .. [#al] Ridnik, Tal, et al. "Asymmetric Loss for Multi-Label Classification."
             Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
    .. [#al_code] https://github.com/Alibaba-MIIL/ASL/

    """

    __name__ = "AsymmetricLoss"

    def __init__(
        self,
        gamma_neg: Real = 4,
        gamma_pos: Real = 1,
        prob_margin: float = 0.05,
        disable_torch_grad_focal_loss: bool = False,
        reduction: str = "mean",
        implementation: str = "alibaba-miil",
    ) -> None:
        super().__init__()
        self.implementation = implementation.lower()
        assert self.implementation in [
            "alibaba-miil",
            "deep-psp",
            "deeppsp",
        ]
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.prob_margin = prob_margin
        if self.prob_margin < 0:
            raise ValueError("`prob_margin` must be non-negative")
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = 1e-8
        self.reduction = reduction.lower()

        if self.implementation == "alibaba-miil":
            self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None
        elif self.implementation in [
            "deep-psp",
            "deeppsp",
        ]:
            self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.loss = self.loss_pos = self.loss_neg = None

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor,
            of shape ``(batch_size, n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, n_classes)``.

        Returns
        -------
        torch.Tensor
            The asymmetric loss.

        """
        if self.implementation == "alibaba-miil":
            return self._forward_alibaba_miil(input, target)
        else:
            return self._forward_deep_psp(input, target)

    def _forward_deep_psp(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass of DeepPSP implementation.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor,
            of shape ``(batch_size, n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, n_classes)``.

        Returns
        -------
        torch.Tensor
            The asymmetric loss.

        """
        self.targets = target
        self.anti_targets = 1 - target

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(input)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.prob_margin > 0:
            self.xs_neg.add_(self.prob_margin).clamp_(max=1)

        # Basic CE calculation
        self.loss_pos = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss_neg = self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            self.loss_pos *= torch.pow(1 - self.xs_pos, self.gamma_pos)
            self.loss_neg *= torch.pow(self.xs_pos, self.gamma_neg)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(prev)
        self.loss = -self.loss_pos - self.loss_neg

        if self.reduction == "mean":
            self.loss = self.loss.mean()
        elif self.reduction == "sum":
            self.loss = self.loss.sum()
        return self.loss

    def _forward_alibaba_miil(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass of Alibaba MIIL implementation.

        Parameters
        ----------
        input : torch.Tensor
            The predicted value tensor,
            of shape ``(batch_size, n_classes)``.
        target : torch.Tensor
            The target tensor,
            of shape ``(batch_size, n_classes)``.

        Returns
        -------
        torch.Tensor
            The asymmetric loss.

        """
        self.targets = target
        self.anti_targets = 1 - target

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(input)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.prob_margin > 0:
            self.xs_neg.add_(self.prob_margin).clamp_(max=1)

        # Basic CE calculation
        # loss = y * log(p) + (1-y) * log(1-p)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets  # p * y
            self.xs_neg = self.xs_neg * self.anti_targets  # (1-p) * (1-y)
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(prev)
            self.loss *= self.asymmetric_w

        if self.reduction == "mean":
            self.loss = -self.loss.mean()
        elif self.reduction == "sum":
            self.loss = -self.loss.sum()
        else:
            self.loss = -self.loss
        return self.loss
