"""
"""

from random import choice
from typing import Any, NoReturn, Sequence, List, Tuple, Union, Optional
from numbers import Real

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.multiprocessing as tmp

from .base import Augmenter


__all__ = ["StretchCompress", "StretchCompressOffline",]


class StretchCompress(Augmenter):
    """
    stretch-or-compress augmenter on ECG tensors

    Example
    -------
    ```python
    sc = StretchCompress()
    sig = torch.ones((32, 12, 5000))
    lb = torch.ones((32, 5000, 3))
    mask = torch.ones((32, 5000, 1))
    sig, lb, mask = sc(sig, lb, mask)
    ```
    """
    __name__ = "StretchCompress"

    def __init__(self, ratio:Real=6, prob:float=0.5, inplace:bool=True, **kwargs: Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        ratio: real number, default 6,
            mean ratio of the stretch or compress,
            if is in [1, 100], will be transformed to [0, 1]
            the ratio of one batch element is sampled from a normal distribution
        prob: float, default 0.5,
            probability of the augmenter to be applied
        inplace: bool, default True,
            if True, the input ECGs will be modified inplace,
        kwargs: keyword arguments
        """
        super().__init__(**kwargs)
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace
        self.ratio = ratio
        if self.ratio > 1:
            self.ratio = self.ratio / 100
        assert 0<= self.ratio <= 1, "Ratio must be between 0 and 1, or between 0 and 100"

    def generate(self, sig:Tensor, *labels:Optional[Sequence[Tensor]]) -> Union[Tuple[Tensor,...],Tensor]:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be stretched or compressed, of shape (batch, lead, siglen)
        labels: sequence of Tensors, optional,
            label tensors of the ECGs,
            if set, each should be of ndim 3, of shape (batch, label_len, channels),
            siglen should be divisible by label_len

        Returns
        -------
        sig: Tensor,
            the stretched or compressed ECG tensors
        labels: sequence of Tensors, optional,
            the stretched or compressed label tensors
        """
        batch, lead, siglen = sig.shape
        if not self.inplace:
            sig = sig.clone()
        labels = [label.clone() for label in labels]
        if self.prob == 0:
            if label is not None:
                return (sig,) + tuple(labels)
            return sig
        label_len = []
        n_labels = len(labels)
        for idx in range(n_labels):
            labels[idx] = labels[idx].permute(0, 2, 1)  # (batch, label_len, n_classes) -> (batch, n_classes, label_len)
            ll = labels[idx].shape[-1]
            if ll != siglen:
                labels[idx] = F.interpolate(labels[idx], size=(siglen,), mode="linear", align_corners=True)
            label_len.append(ll)
        for batch_idx in self.get_indices(prob=self.prob, pop_size=batch):
            sign = choice([-1,1])
            ratio = np.clip(np.random.normal(self.ratio, 0.382*self.ratio), 0, 2*self.ratio)
            # print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
            new_len = int(round((1+sign*ratio) * siglen))
            diff_len = abs(new_len - siglen)
            half_diff_len = diff_len // 2
            if sign > 0:  # stretch and cut
                sig[batch_idx, ...] = F.interpolate(
                    sig[batch_idx, ...].unsqueeze(0),
                    size=new_len,
                    mode="linear",
                    align_corners=True,
                )[..., half_diff_len: siglen+half_diff_len].squeeze(0)
                for idx in range(n_labels):
                    labels[idx][batch_idx, ...] = F.interpolate(
                        labels[idx][batch_idx, ...].unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=True,
                    )[..., half_diff_len: siglen+half_diff_len].squeeze(0)
            else:  # compress and pad
                sig[batch_idx, ...] = F.pad(
                    F.interpolate(
                        sig[batch_idx, ...].unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=True,
                    ),
                    pad=(half_diff_len, diff_len-half_diff_len),
                    mode="constant",
                    value=0.0,
                ).squeeze(0)
                for idx in range(n_labels):
                    labels[idx][batch_idx, ...] = F.pad(
                        F.interpolate(
                            labels[idx][batch_idx, ...].unsqueeze(0),
                            size=new_len,
                            mode="linear",
                            align_corners=True,
                        ),
                        pad=(half_diff_len, diff_len-half_diff_len),
                        mode="constant",
                        value=0.0,
                    ).squeeze(0)
        for idx, (label, ll) in enumerate(zip(labels, label_len)):
            if ll != siglen:
                labels[idx] = F.interpolate(label, size=(ll,), mode="linear", align_corners=True)
            labels[idx] = labels[idx].permute(0, 2, 1)  # (batch, n_classes, label_len) -> (batch, label_len, n_classes)
        if len(labels) > 0:
            return (sig,) + tuple(labels)
        return sig

    def _generate(self, sig:Tensor, *labels:Optional[Sequence[Tensor]]) -> Union[Tuple[Tensor,...],Tensor]:
        """ NOT finished, NOT checked,

        parallel version of `self.generate`, NOT tested yet!

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be stretched or compressed, of shape (batch, lead, siglen)
        labels: sequence of Tensors, optional,
            label tensors of the ECGs,
            if set, should be of ndim 3, of shapes (batch, label_len, n_classes),
            siglen should be divisible by label_len

        Returns
        -------
        sig: Tensor,
            the stretched or compressed ECG tensors
        labels: sequence of Tensors, optional,
            the stretched or compressed label tensors
        """
        batch, lead, siglen = sig.shape
        if not self.inplace:
            sig = sig.clone()
        if self.prob == 0:
            return sig
        indices = self.get_indices(prob=self.prob, pop_size=batch)
        with tmp.Pool(processes=4) as pool:
            sig[indices, ...] = torch.as_tensor(
                pool.starmap(
                    func=_stretch_compress_one_batch_element,
                    iterable=[
                        (self.ratio, sig[batch_idx, ...].unsqueeze(0),) \
                            for batch_idx in indices
                    ],
                ),
                dtype=sig.dtype,
                device=sig.device,
            )
        return sig

    def __call__(self, sig:Tensor, *labels:Optional[Sequence[Tensor]]) -> Union[Tuple[Tensor,...],Tensor]:
        """
        alias of `self.generate`
        """
        return self.generate(sig, *labels)

    def extra_repr_keys(self) -> List[str]:
        """
        """
        return ["ratio", "prob", "inplace",] + super().extra_repr_keys()


def _stretch_compress_one_batch_element(ratio:Real, sig:Tensor, *labels:Sequence[Tensor]) -> Tensor:
    """ finished, NOT checked,

    Parameters
    ----------
    ratio: Real,
        ratio of the stretch/compress
    sig: Tensor,
        the ECGs to be stretched or compressed, of shape (1, lead, siglen)
    labels: sequence of Tensors, optional,
        label tensors of the ECGs,
        if set, each should be of ndim 3, of shape (1, label_len, channels),
        siglen should be divisible by label_len

    Returns
    -------
    sig: Tensor, of shape (lead, siglen)
        the stretched or compressed ECG tensor
    labels: Tensors, optional, of shapes (label_len, channels)
        the stretched or compressed label tensors
    """
    labels = list(labels)
    label_len = []
    n_labels = len(labels)
    for idx in range(n_labels):
        labels[idx] = labels[idx].permute(0, 2, 1)  # (1, label_len, n_classes) -> (1, n_classes, label_len)
        ll = labels[idx].shape[-1]
        if ll != siglen:
            labels[idx] = F.interpolate(labels[idx], size=(siglen,), mode="linear", align_corners=True)
        label_len.append(ll)
    sign = choice([-1,1])
    ratio = np.clip(np.random.normal(ratio, 0.382*ratio), 0, 2*ratio)
    # print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
    new_len = int(round((1+sign*ratio) * siglen))
    diff_len = abs(new_len - siglen)
    half_diff_len = diff_len // 2
    if sign > 0:  # stretch and cut
        sig = F.interpolate(
            sig,
            size=new_len,
            mode="linear",
            align_corners=True,
        )[..., half_diff_len: siglen+half_diff_len].squeeze(0)
        for idx in range(n_labels):
            labels[idx] = F.interpolate(
                labels[idx],
                size=new_len,
                mode="linear",
                align_corners=True,
            )[..., half_diff_len: siglen+half_diff_len].squeeze(0)
    else:  # compress and pad
        sig = F.pad(
            F.interpolate(
                sig,
                size=new_len,
                mode="linear",
                align_corners=True,
            ),
            pad=(half_diff_len, diff_len-half_diff_len),
            mode="constant",
            value=0.0,
        ).squeeze(0)
        for idx in range(n_labels):
            labels[idx] = F.pad(
                F.interpolate(
                    labels[idx],
                    size=new_len,
                    mode="linear",
                    align_corners=True,
                ),
                pad=(half_diff_len, diff_len-half_diff_len),
                mode="constant",
                value=0.0,
            ).squeeze(0)
    for idx, (label, ll) in enumerate(zip(labels, label_len)):
        if ll != siglen:
            labels[idx] = F.interpolate(label, size=(ll,), mode="linear", align_corners=True)
        labels[idx] = labels[idx].permute(1, 0)  # (n_classes, label_len) -> (label_len, n_classes)
    if len(labels) > 0:
        return (sig,) + tuple(labels)
    return sig


class StretchCompressOffline(object):
    """
    stretch-or-compress augmenter on orginal length-varying ECG signals (in the form of numpy arrays),
    for the purpose of offline data generation
    """
    __name__ = "StretchCompressOffline"

    def __init__(self,
                 ratio:Real=6,
                 prob:float=0.5,
                 overlap:float=0.5,
                 critical_overlap:float=0.85) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        ratio: real number, default 6,
            mean ratio of the stretch or compress,
            if is in [1, 100], will be transformed to [0, 1]
            the ratio of one batch element is sampled from a normal distribution
        prob: float, default 0.5,
            probability of the augmenter to be applied
        overlap: float, default 0.5,
            the overlap of offline generated data
        critical_overlap: float, default 0.85,
            the overlap of the critical region of the ECG
        """
        super().__init__(**kwargs)
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.ratio = ratio
        if self.ratio > 1:
            self.ratio = self.ratio / 100
        assert 0<= self.ratio <= 1, "Ratio must be between 0 and 1, or between 0 and 100"
        self.overlap = overlap
        assert 0<= self.overlap < 1, "Overlap ratio must be between 0 and 1 (1 not included)"
        self.critical_overlap = critical_overlap
        assert 0<= self.critical_overlap < 1, "Critical overlap ratio must be between 0 and 1 (1 not included)"

    def generate(self,
                 seg_len:int,
                 sig:np.ndarray,
                 label:Optional[np.ndarray]=None,
                 critical_points:Optional[Sequence[int]]=None) -> List[np.ndarray]:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        seg_len: int,
            the length of the ECG segments to be generated
        sig: ndarray,
            the ECGs to generate stretched or compressed segments, of shape (lead, siglen)
        label: ndarray, optional,
            the labels of the ECGs, of shape (label_len, channels),
            for example when doing segmentation,
            label_len should be divisible by siglen,
            channels should be the same as the number of classes
        critical_points: sequence of int, optional,
            indices of the critical points of the ECG,
            usually have larger overlap by `self.critical_overlap`
        """
        raise NotImplementedError

    def __call__(self, sig:np.ndarray) -> List[np.ndarray]:
        """
        alias of `self.generate`
        """
        return self.generate(sig)
