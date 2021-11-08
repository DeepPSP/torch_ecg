"""
"""

from random import choice
from typing import Any, NoReturn, Sequence, List, Union, Optional
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

    def generate(self, sig:Tensor, label:Optional[Tensor]=None) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be stretched or compressed, of shape (batch, lead, siglen)
        label: Tensor, optional,
            label tensor of the ECGs, not used

        Returns
        -------
        sig: Tensor,
            the stretched or compressed ECGs
        """
        batch, lead, siglen = sig.shape
        if not self.inplace:
            sig = sig.clone()
        if self.prob == 0:
            return sig
        for batch_idx in self.get_indices(prob=self.prob, pop_size=batch):
            sign = choice([-1,1])
            ratio = np.clip(np.random.normal(self.ratio, 0.382*self.ratio), 0, 2*self.ratio)
            print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
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
        return sig

    def _generate(self, sig:Tensor, label:Optional[Tensor]=None) -> Tensor:
        """ finished, NOT checked,

        parallel version of `self.generate`, NOT tested yet!

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be stretched or compressed, of shape (batch, lead, siglen)
        label: Tensor, optional,
            label tensor of the ECGs, not used

        Returns
        -------
        sig: Tensor,
            the stretched or compressed ECGs
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
                        (sig[batch_idx, ...].unsqueeze(0), self.ratio,) \
                            for batch_idx in indices
                    ],
                ),
                dtype=sig.dtype,
                device=sig.device,
            )
        return sig


def _stretch_compress_one_batch_element(sig:Tensor, ratio:Real) -> Tensor:
    """ finished, checked,

    Parameters
    ----------
    sig: Tensor,
        the ECG to be stretched or compressed, of shape (1, lead, siglen)
    ratio: Real,
        ratio of the stretch/compress

    Returns
    -------
    Tensor, of shape (lead, siglen)
        the stretched or compressed ECG
    """
    sign = choice([-1,1])
    ratio = np.clip(np.random.normal(ratio, 0.382*ratio), 0, 2*ratio)
    # print(f"batch_idx = {batch_idx}, sign = {sign}, ratio = {ratio}")
    new_len = int(round((1+sign*ratio) * siglen))
    diff_len = abs(new_len - siglen)
    half_diff_len = diff_len // 2
    if sign > 0:  # stretch and cut
        return F.interpolate(
            sig,
            size=new_len,
            mode="linear",
            align_corners=True,
        )[..., half_diff_len: siglen+half_diff_len].squeeze(0)
    else:  # compress and pad
        return F.pad(
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

    def generate(self, sig:np.ndarray, seg_len:int, critical_points:Optional[Sequence[int]]=None) -> List[np.ndarray]:
        """ NOT finished, NOT checked,

        Parameters
        ----------
        sig: ndarray,
            the ECGs to generate stretched or compressed segments, of shape (lead, siglen)
        seg_len: int,
            the length of the ECG segments to be generated
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
