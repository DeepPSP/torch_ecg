"""
add baseline wander composed of sinusoidal and Gaussian noise to the ECGs
"""

import multiprocessing as mp
from itertools import repeat
from numbers import Real
from random import randint
from typing import Any, List, NoReturn, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..cfg import DEFAULTS
from ..utils.utils_signal import get_ampl
from .base import Augmenter

__all__ = [
    "BaselineWanderAugmenter",
]


class BaselineWanderAugmenter(Augmenter):
    """ """

    __name__ = "BaselineWanderAugmenter"

    def __init__(
        self,
        fs: int,
        bw_fs: Optional[np.ndarray] = None,
        ampl_ratio: Optional[np.ndarray] = None,
        gaussian: Optional[np.ndarray] = None,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any
    ) -> NoReturn:
        """finished, checked

        Parameters
        ----------
        fs: int,
            sampling frequency of the ECGs to be augmented
        bw_fs: ndarray, optional,
            frequencies of the sinusoidal noises,
            of shape (n,)
        ampl_ratio: ndarray, optional,
            candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
            of shape (m,n)
        gaussian: ndarray, optional,
            candidate mean and std of the Gaussian noises,
            of shape (k, 2)
        prob: float, default 0.5,
            probability of performing the augmentation
        inplace: bool, default True,
            if True, ECG signal tensors will be modified inplace
        kwargs: Keyword arguments.

        """
        super().__init__()
        self.fs = fs
        self.bw_fs = bw_fs if bw_fs is not None else np.array([0.33, 0.1, 0.05, 0.01])
        self.prob = prob
        assert 0 <= self.prob <= 1, "Probability must be between 0 and 1"
        self.ampl_ratio = (
            ampl_ratio
            if ampl_ratio is not None
            else np.array(
                [  # default ampl_ratio
                    [0.01, 0.01, 0.02, 0.03],  # low
                    [0.01, 0.02, 0.04, 0.05],  # low
                    [0.1, 0.06, 0.04, 0.02],  # low
                    [0.02, 0.04, 0.07, 0.1],  # low
                    [0.05, 0.1, 0.16, 0.25],  # medium
                    [0.1, 0.15, 0.25, 0.3],  # high
                    [0.25, 0.25, 0.3, 0.35],  # extremely high
                ]
            )
        )
        if self.prob > 0:
            self.ampl_ratio = np.concatenate(
                (
                    np.zeros(
                        (
                            int((1 - self.prob) * self.ampl_ratio.shape[0] / self.prob),
                            self.ampl_ratio.shape[1],
                        )
                    ),
                    self.ampl_ratio,
                )
            )
        self.gaussian = (
            gaussian
            if gaussian is not None
            else np.array(
                [  # default gaussian, mean and std, in terms of ratio
                    [0.0, 0.001],
                    [0.0, 0.003],
                    [0.0, 0.01],
                ]
            )
        )
        if self.prob > 0:
            self.gaussian = np.concatenate(
                (
                    np.zeros(
                        (
                            int((1 - self.prob) * self.gaussian.shape[0] / self.prob),
                            self.gaussian.shape[1],
                        )
                    ),
                    self.gaussian,
                )
            )
        assert (
            self.bw_fs.ndim == 1
            and self.ampl_ratio.ndim == 2
            and self.bw_fs.shape[0] == self.ampl_ratio.shape[1]
        )
        self.inplace = inplace

        self._n_bw_choices = len(self.ampl_ratio)
        self._n_gn_choices = len(self.gaussian)

    def forward(
        self,
        sig: Tensor,
        label: Optional[Tensor],
        *extra_tensors: Sequence[Tensor],
        **kwargs: Any
    ) -> Tuple[Tensor, ...]:
        """

        Parameters
        ----------
        sig: Tensor,
            the ECGs to be augmented, of shape (batch, lead, siglen)
        label: Tensor, optional,
            label tensor of the ECGs,
            not used, but kept for consistency with other augmenters
        extra_tensors: sequence of Tensors, optional,
            not used, but kept for consistency with other augmenters
        kwargs: keyword arguments,
            not used, but kept for consistency with other augmenters

        Returns
        -------
        sig: Tensor,
            the augmented ECGs
        label: Tensor,
            label tensor of the augmented ECGs, unchanged
        extra_tensors: sequence of Tensors, optional,
            if set in the input arguments, unchanged

        """
        if not self.inplace:
            sig = sig.clone()
        if self.prob > 0:
            sig.add_(
                gen_baseline_wander(
                    sig, self.fs, self.bw_fs, self.ampl_ratio, self.gaussian
                )
            )
        return (sig, label, *extra_tensors)

    def extra_repr_keys(self) -> List[str]:
        """ """
        return [
            "fs",
            "bw_fs",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()


def _get_ampl(sig: Tensor, fs: int) -> Tensor:
    """finished, checked

    Parameters
    ----------
    sig: Tensor,
        the ECG signal tensor, of shape (batch, lead, siglen)
    fs: int,
        sampling frequency of the ECGs

    Returns
    -------
    ampl: Tensor,
        amplitude of each lead, of shape (batch, lead, 1)

    """
    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        ampl = pool.starmap(
            get_ampl,
            iterable=[(sig[i].cpu().numpy(), fs) for i in range(sig.shape[0])],
        )
    ampl = torch.as_tensor(ampl, dtype=sig.dtype, device=sig.device).unsqueeze(-1)
    return ampl


def _gen_gaussian_noise(siglen: int, mean: Real = 0, std: Real = 0) -> np.ndarray:
    """

    generate 1d Gaussian noise of given length, mean, and standard deviation

    Parameters
    ----------
    siglen: int,
        length of the noise signal
    mean: real number, default 0,
        mean of the noise
    std: real number, default 0,
        standard deviation of the noise

    Returns
    -------
    gn: ndarray,
        the gaussian noise of given length, mean, and standard deviation

    """
    gn = DEFAULTS.RNG.normal(mean, std, siglen)
    return gn


def _gen_sinusoidal_noise(
    siglen: int,
    start_phase: Real,
    end_phase: Real,
    amplitude: Real,
    amplitude_mean: Real = 0,
    amplitude_std: Real = 0,
) -> np.ndarray:
    """

    generate 1d sinusoidal noise of given length, amplitude, start phase, and end phase

    Parameters
    ----------
    siglen: int,
        length of the (noise) signal
    start_phase: real number,
        start phase, with units in degrees
    end_phase: real number,
        end phase, with units in degrees
    amplitude: real number,
        amplitude of the sinusoidal curve
    amplitude_mean: real number,
        mean amplitude of an extra Gaussian noise
    amplitude_std: real number, default 0,
        standard deviation of an extra Gaussian noise

    Returns
    -------
    sn: ndarray,
        the sinusoidal noise of given length, amplitude, start phase, and end phase

    """
    sn = np.linspace(start_phase, end_phase, siglen)
    sn = amplitude * np.sin(np.pi * sn / 180)
    sn += _gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    return sn


def _gen_baseline_wander(
    siglen: int,
    fs: Real,
    bw_fs: Union[Real, Sequence[Real]],
    amplitude: Union[Real, Sequence[Real]],
    amplitude_gaussian: Sequence[Real] = [0, 0],
) -> np.ndarray:
    """

    generate 1d baseline wander of given length, amplitude, and frequency

    Parameters
    ----------
    siglen: int,
        length of the (noise) signal
    fs: real number,
        sampling frequency of the original signal
    bw_fs: real number, or list of real numbers,
        frequency (frequencies) of the baseline wander
    amplitude: real number, or list of real numbers,
        amplitude of the baseline wander (corr. to each frequency band)
    amplitude_gaussian: 2-tuple of real number, default [0,0],
        mean and std of amplitude of an extra Gaussian noise

    Returns
    -------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency

    Example
    -------
    >>> _gen_baseline_wander(4000, 400, [0.4,0.1,0.05], [0.1,0.2,0.4])

    """
    bw = _gen_gaussian_noise(siglen, amplitude_gaussian[0], amplitude_gaussian[1])
    if isinstance(bw_fs, Real):
        _bw_fs = [bw_fs]
    else:
        _bw_fs = bw_fs
    if isinstance(amplitude, Real):
        _amplitude = list(repeat(amplitude, len(_bw_fs)))
    else:
        _amplitude = amplitude
    assert len(_bw_fs) == len(_amplitude)
    duration = siglen / fs
    for bf, a in zip(_bw_fs, _amplitude):
        start_phase = DEFAULTS.RNG_randint(0, 360)
        end_phase = duration * bf * 360 + start_phase
        bw += _gen_sinusoidal_noise(siglen, start_phase, end_phase, a, 0, 0)
    return bw


def gen_baseline_wander(
    sig: Tensor,
    fs: Real,
    bw_fs: Union[Real, Sequence[Real]],
    ampl_ratio: np.ndarray,
    gaussian: np.ndarray,
) -> np.ndarray:
    """

    generate 1d baseline wander of given length, amplitude, and frequency

    Parameters
    ----------
    sig: Tensor,
        the ECGs to be augmented, of shape (batch, lead, siglen)
    fs: real number,
        sampling frequency of the original signal
    bw_fs: real number, or list of real numbers,
        frequency (frequencies) of the baseline wander
    ampl_ratio: ndarray, optional,
        candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
        of shape (m,n)
    gaussian: ndarray, optional,
        candidate mean and std of the Gaussian noises,
        of shape (k, 2)

    Returns
    -------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency,
        of shape (batch, lead, siglen)

    """
    batch, lead, siglen = sig.shape
    sig_ampl = _get_ampl(sig, fs)
    _n_bw_choices = len(ampl_ratio)
    _n_gn_choices = len(gaussian)

    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        bw = pool.starmap(
            _gen_baseline_wander,
            iterable=[
                (
                    siglen,
                    fs,
                    bw_fs,
                    ampl_ratio[randint(0, _n_bw_choices - 1)],
                    gaussian[randint(0, _n_gn_choices - 1)],
                )
                for i in range(sig.shape[0])
                for j in range(sig.shape[1])
            ],
        )
    bw = torch.as_tensor(bw, dtype=sig.dtype, device=sig.device).reshape(
        batch, lead, siglen
    )
    return bw
