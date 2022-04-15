"""
"""

from numbers import Real
from random import choice, randint
from typing import Any, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.nn.functional as F
from scipy.signal import resample, resample_poly  # noqa: F401
from torch import Tensor

from ..cfg import DEFAULTS
from ..utils.misc import ReprMixin
from .base import Augmenter

__all__ = [
    "StretchCompress",
    "StretchCompressOffline",
]


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

    def __init__(
        self, ratio: Real = 6, prob: float = 0.5, inplace: bool = True, **kwargs: Any
    ) -> NoReturn:
        """

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
        super().__init__()
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.inplace = inplace
        self.ratio = ratio
        if self.ratio > 1:
            self.ratio = self.ratio / 100
        assert (
            0 <= self.ratio <= 1
        ), "Ratio must be between 0 and 1, or between 0 and 100"

    def forward(
        self, sig: Tensor, *labels: Optional[Sequence[Tensor]], **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be stretched or compressed, of shape (batch, lead, siglen)
        labels: sequence of Tensors, optional,
            label tensors of the ECGs,
            if set, labels of ndim = 3, of shape (batch, label_len, channels) will be stretched or compressed,
            siglen should be divisible by label_len
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

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
            return (sig, *labels)
        label_len = []
        n_labels = len(labels)
        for idx in range(n_labels):
            if labels[idx].ndim < 3:
                label_len.append(None)
                continue
            labels[idx] = labels[idx].permute(
                0, 2, 1
            )  # (batch, label_len, n_classes) -> (batch, n_classes, label_len)
            ll = labels[idx].shape[-1]
            if ll != siglen:
                labels[idx] = F.interpolate(
                    labels[idx], size=(siglen,), mode="linear", align_corners=True
                )
            label_len.append(ll)
        for batch_idx in self.get_indices(prob=self.prob, pop_size=batch):
            sign = choice([-1, 1])
            ratio = self._sample_ratio()
            # print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
            new_len = int(round((1 + sign * ratio) * siglen))
            diff_len = abs(new_len - siglen)
            half_diff_len = diff_len // 2
            if sign > 0:  # stretch and cut
                sig[batch_idx, ...] = F.interpolate(
                    sig[batch_idx, ...].unsqueeze(0),
                    size=new_len,
                    mode="linear",
                    align_corners=True,
                )[..., half_diff_len : siglen + half_diff_len].squeeze(0)
                for idx in range(n_labels):
                    if labels[idx].ndim < 3:
                        continue
                    labels[idx][batch_idx, ...] = F.interpolate(
                        labels[idx][batch_idx, ...].unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=True,
                    )[..., half_diff_len : siglen + half_diff_len].squeeze(0)
            else:  # compress and pad
                sig[batch_idx, ...] = F.pad(
                    F.interpolate(
                        sig[batch_idx, ...].unsqueeze(0),
                        size=new_len,
                        mode="linear",
                        align_corners=True,
                    ),
                    pad=(half_diff_len, diff_len - half_diff_len),
                    mode="constant",
                    value=0.0,
                ).squeeze(0)
                for idx in range(n_labels):
                    if labels[idx].ndim < 3:
                        continue
                    labels[idx][batch_idx, ...] = F.pad(
                        F.interpolate(
                            labels[idx][batch_idx, ...].unsqueeze(0),
                            size=new_len,
                            mode="linear",
                            align_corners=True,
                        ),
                        pad=(half_diff_len, diff_len - half_diff_len),
                        mode="constant",
                        value=0.0,
                    ).squeeze(0)
        for idx, (label, ll) in enumerate(zip(labels, label_len)):
            if labels[idx].ndim < 3:
                continue
            if ll != siglen:
                labels[idx] = F.interpolate(
                    label, size=(ll,), mode="linear", align_corners=True
                )
            labels[idx] = labels[idx].permute(
                0, 2, 1
            )  # (batch, n_classes, label_len) -> (batch, label_len, n_classes)
        return (sig, *labels)

    def _sample_ratio(self) -> float:
        """ """
        return np.clip(
            DEFAULTS.RNG.normal(self.ratio, 0.382 * self.ratio), 0, 2 * self.ratio
        )

    def _generate(
        self, sig: Tensor, *labels: Optional[Sequence[Tensor]]
    ) -> Union[Tuple[Tensor, ...], Tensor]:
        """NOT finished, NOT checked,

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
                        (
                            self.ratio,
                            sig[batch_idx, ...].unsqueeze(0),
                        )
                        for batch_idx in indices
                    ],
                ),
                dtype=sig.dtype,
                device=sig.device,
            )
        return sig

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "ratio",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()


def _stretch_compress_one_batch_element(
    ratio: Real, sig: Tensor, *labels: Sequence[Tensor]
) -> Tensor:
    """finished, NOT checked,

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
    siglen = sig.shape[-1]
    for idx in range(n_labels):
        labels[idx] = labels[idx].permute(
            0, 2, 1
        )  # (1, label_len, n_classes) -> (1, n_classes, label_len)
        ll = labels[idx].shape[-1]
        if ll != siglen:
            labels[idx] = F.interpolate(
                labels[idx], size=(siglen,), mode="linear", align_corners=True
            )
        label_len.append(ll)
    sign = choice([-1, 1])
    ratio = np.clip(DEFAULTS.RNG.normal(ratio, 0.382 * ratio), 0, 2 * ratio)
    # print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
    new_len = int(round((1 + sign * ratio) * siglen))
    diff_len = abs(new_len - siglen)
    half_diff_len = diff_len // 2
    if sign > 0:  # stretch and cut
        sig = F.interpolate(sig, size=new_len, mode="linear", align_corners=True,)[
            ..., half_diff_len : siglen + half_diff_len
        ].squeeze(0)
        for idx in range(n_labels):
            labels[idx] = F.interpolate(
                labels[idx],
                size=new_len,
                mode="linear",
                align_corners=True,
            )[..., half_diff_len : siglen + half_diff_len].squeeze(0)
    else:  # compress and pad
        sig = F.pad(
            F.interpolate(
                sig,
                size=new_len,
                mode="linear",
                align_corners=True,
            ),
            pad=(half_diff_len, diff_len - half_diff_len),
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
                pad=(half_diff_len, diff_len - half_diff_len),
                mode="constant",
                value=0.0,
            ).squeeze(0)
    for idx, (label, ll) in enumerate(zip(labels, label_len)):
        if ll != siglen:
            labels[idx] = F.interpolate(
                label, size=(ll,), mode="linear", align_corners=True
            )
        labels[idx] = labels[idx].permute(
            1, 0
        )  # (n_classes, label_len) -> (label_len, n_classes)
    if len(labels) > 0:
        return (sig,) + tuple(labels)
    return sig


class StretchCompressOffline(ReprMixin):
    """
    stretch-or-compress augmenter on orginal length-varying ECG signals (in the form of numpy arrays),
    for the purpose of offline data generation

    Example
    -------
    ```python
    sco = StretchCompressOffline()
    seglen = 600
    sig = torch.rand((12, 60000)).numpy()
    labels = torch.ones((60000, 3)).numpy().astype(int)
    masks = torch.ones((60000, 1)).numpy().astype(int)
    segments = sco(600, sig, labels, masks, critical_points=[10000,30000])
    ```
    """

    __name__ = "StretchCompressOffline"

    def __init__(
        self,
        ratio: Real = 6,
        prob: float = 0.5,
        overlap: float = 0.5,
        critical_overlap: float = 0.85,
    ) -> NoReturn:
        """

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
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.ratio = ratio
        if self.ratio > 1:
            self.ratio = self.ratio / 100
        assert (
            0 <= self.ratio <= 1
        ), "Ratio must be between 0 and 1, or between 0 and 100"
        self.overlap = overlap
        assert (
            0 <= self.overlap < 1
        ), "Overlap ratio must be between 0 and 1 (1 not included)"
        self.critical_overlap = critical_overlap
        assert (
            0 <= self.critical_overlap < 1
        ), "Critical overlap ratio must be between 0 and 1 (1 not included)"

    def generate(
        self,
        seglen: int,
        sig: np.ndarray,
        *labels: Sequence[np.ndarray],
        critical_points: Optional[Sequence[int]] = None,
    ) -> List[Tuple[Union[np.ndarray, int], ...]]:
        """

        Parameters
        ----------
        seglen: int,
            the length of the ECG segments to be generated
        sig: ndarray,
            the ECGs to generate stretched or compressed segments, of shape (lead, siglen)
        labels: ndarray, optional,
            the labels of the ECGs, of shape (label_len, channels),
            for example when doing segmentation,
            label_len should be divisible by siglen,
            channels should be the same as the number of classes
        critical_points: sequence of int, optional,
            indices of the critical points of the ECG,
            usually have larger overlap by `self.critical_overlap`

        Returns
        -------
        list of generated segments,
        with segments consists of (seg, label1, label2, ..., start_idx, end_idx)
        """
        siglen = sig.shape[1]
        forward_len = int(round(seglen - seglen * self.overlap))
        critical_forward_len = int(round(seglen - seglen * self.critical_overlap))
        critical_forward_len = [critical_forward_len // 4, critical_forward_len]
        # print(forward_len, critical_forward_len)

        # skip those records that are too short
        if siglen < seglen:
            return []

        segments = []

        # ordinary segments with constant forward_len
        for idx in range((siglen - seglen) // forward_len + 1):
            start_idx = idx * forward_len
            new_seg = self.__generate_segment(
                seglen,
                sig,
                *labels,
                start_idx=start_idx,
            )
            segments.append(new_seg)
        # the tail segment
        if (siglen - seglen) % forward_len != 0:
            new_seg = self.__generate_segment(
                seglen,
                sig,
                *labels,
                end_idx=siglen,
            )
            segments.append(new_seg)

        # special segments around critical_points with random forward_len in critical_forward_len
        for cp in critical_points or []:
            start_idx = max(
                0,
                cp - seglen + randint(critical_forward_len[0], critical_forward_len[1]),
            )
            while start_idx <= min(cp - critical_forward_len[1], siglen - seglen):
                new_seg = self.__generate_segment(
                    seglen,
                    sig,
                    *labels,
                    start_idx=start_idx,
                )
                segments.append(new_seg)
                start_idx += randint(critical_forward_len[0], critical_forward_len[1])
        return segments

    def __generate_segment(
        self,
        seglen: int,
        sig: np.ndarray,
        *labels: Sequence[np.ndarray],
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, int], ...]:
        """

        Parameters
        ----------
        seglen: int,
            the length of the ECG segments to be generated
        sig: ndarray,
            the ECGs to generate stretched or compressed segments, of shape (lead, siglen)
        labels: ndarray, optional,
            the labels of the ECGs, of shape (label_len, channels),
            for example when doing segmentation,
            label_len should be divisible by siglen,
            channels should be the same as the number of classes
        start_idx: int, optional,
            the start index of the segment in `sig`
        end_idx: int, optional,
            the end index of the segment in `sig`,
            if `start_idx` is set, `end_idx` is ignored,
            at least one of `start_idx` and `end_idx` should be set

        Returns
        -------
        tuple of generated segment,
        consists of (seg, label1, label2, ..., start_idx, end_idx)
        """
        assert not all(
            [start_idx is None, end_idx is None]
        ), "at least one of `start_idx` and `end_idx` should be set"

        siglen = sig.shape[1]
        ratio = self._sample_ratio()
        aug_labels = []
        if ratio != 0:
            sign = choice([-1, 1])
            new_len = int(round((1 + sign * ratio) * seglen))
            if start_idx is not None:
                end_idx = start_idx + new_len
            else:
                start_idx = end_idx - new_len
            if end_idx > siglen:
                end_idx = siglen
                start_idx = max(0, end_idx - new_len)
                ratio = (end_idx - start_idx) / seglen - 1
            aug_seg = sig[..., start_idx:end_idx]
            aug_seg = resample(x=aug_seg, num=seglen, axis=1)
            for lb in labels:
                dtype = lb.dtype
                aug_labels.append(
                    F.interpolate(
                        torch.from_numpy(
                            lb[start_idx:end_idx, ...].T.astype(np.float32)
                        ).unsqueeze(0),
                        size=seglen,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .numpy()
                    .T.astype(dtype)
                )
        else:
            if start_idx is not None:
                end_idx = start_idx + seglen
                if end_idx > siglen:
                    end_idx = siglen
                    start_idx = end_idx - seglen
            else:
                start_idx = end_idx - seglen
                if start_idx < 0:
                    start_idx = 0
                    end_idx = seglen
            aug_seg = sig[..., start_idx:end_idx]
            for lb in labels:
                aug_labels.append(lb[start_idx:end_idx, ...])
        return (aug_seg,) + tuple(aug_labels) + (start_idx, end_idx)

    def _sample_ratio(self) -> float:
        """ """
        if DEFAULTS.RNG.uniform() >= self.prob:
            return 0
        else:
            return np.clip(
                DEFAULTS.RNG.normal(self.ratio, 0.382 * self.ratio),
                0.01 * self.ratio,
                2 * self.ratio,
            )

    def __call__(
        self,
        seglen: int,
        sig: np.ndarray,
        *labels: Sequence[np.ndarray],
        critical_points: Optional[Sequence[int]] = None,
    ) -> List[Tuple[np.ndarray, ...]]:
        """
        alias of `self.generate`
        """
        return self.generate(seglen, sig, *labels, critical_points=critical_points)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return super().extra_repr_keys() + [
            "ratio",
            "prob",
            "overlap",
            "critical_overlap",
        ]
