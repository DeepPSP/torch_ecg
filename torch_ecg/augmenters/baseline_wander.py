"""Add baseline wander composed of sinusoidal and Gaussian noise to the ECGs."""

import multiprocessing as mp
from itertools import repeat
from numbers import Real
from random import randint
from typing import Any, List, Optional, Sequence, Tuple, Union

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
    """Generate baseline wander composed of
    sinusoidal and Gaussian noise.

    Parameters
    ----------
    fs : int
        Sampling frequency of the ECGs to be augmented
    bw_fs : numpy.ndarray, optional
        Frequencies of the sinusoidal noises,
        of shape ``(n,)``,
        defaults to ``[0.33, 0.1, 0.05, 0.01]``.
    ampl_ratio : numpy.ndarray, optional
        Candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
        of shape ``(m, n)``,
        defaults to

        .. code-block:: python

            np.array(
                [
                    [0.01, 0.01, 0.02, 0.03],  # low
                    [0.01, 0.02, 0.04, 0.05],  # low
                    [0.1, 0.06, 0.04, 0.02],  # low
                    [0.02, 0.04, 0.07, 0.1],  # low
                    [0.05, 0.1, 0.16, 0.25],  # medium
                    [0.1, 0.15, 0.25, 0.3],  # high
                    [0.25, 0.25, 0.3, 0.35],  # extremely high
                ]
            )

    gaussian : numpy.ndarray, optional
        Candidate mean and std of the Gaussian noises,
        of shape ``(k, 2)``,
        defaults to

        .. code-block:: python

            np.array(
                [  # mean and std, in terms of ratio
                    [0.0, 0.001],
                    [0.0, 0.003],
                    [0.0, 0.01],
                ]
            )

    prob : float, default 0.5
        Probability of performing the augmentation.
    inplace : bool, default True
        If True, ECG signal tensors will be modified inplace.
    kwargs : dict, optional
        Additional keyword arguments.

    Examples
    --------
    .. code-block:: python

        blw = BaselineWanderAugmenter(300, prob=0.7)
        sig = torch.randn(32, 12, 5000)
        label = torch.ones((32, 20))
        sig, _ = blw(sig, label)

    """

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
    ) -> None:
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
        """Forward function of the :class:`BaselineWanderAugmenter`.

        Parameters
        ----------
        sig : torch.Tensor
            Batched ECGs to be augmented, of shape ``(batch, lead, siglen)``.
        label : torch.Tensor, optional
            Batched label tensor of the ECGs.
            Not used, but kept for consistency with other augmenters.
        extra_tensors : Sequence[torch.Tensor], optional,
            Not used, but kept for consistency with other augmenters.
        **kwargs : dict, optional
            Not used, but kept for consistency with other augmenters.

        Returns
        -------
        sig : torch.Tensor
            The augmented ECGs.
        label : torch.Tensor
            Label tensor of the augmented ECGs, unchanged.
        extra_tensors : Sequence[torch.Tensor], optional
            Unchanged extra tensors.

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
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return [
            "fs",
            "bw_fs",
            "prob",
            "inplace",
        ] + super().extra_repr_keys()


def _get_ampl(sig: Tensor, fs: int) -> Tensor:
    """Get the amplitude of each lead.

    Parameters
    ----------
    sig : torch.Tensor
        Batched ECG signal tensor, of shape ``(batch, lead, siglen)``.
    fs : int
        Sampling frequency of the ECGs.

    Returns
    -------
    ampl : torch.Tensor
        Amplitude of each lead, of shape ``(batch, lead, 1)``.

    """
    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        ampl = pool.starmap(
            get_ampl,
            iterable=[(sig[i].cpu().numpy(), fs) for i in range(sig.shape[0])],
        )
    ampl = torch.as_tensor(
        np.array(ampl), dtype=sig.dtype, device=sig.device
    ).unsqueeze(-1)
    return ampl


def _gen_gaussian_noise(siglen: int, mean: Real = 0, std: Real = 0) -> np.ndarray:
    """Generate 1d Gaussian noise of given
    length, mean, and standard deviation.

    Parameters
    ----------
    siglen : int
        Length of the noise signal.
    mean : numbers.Real, default 0
        Mean value of the noise.
    std : numbers.Real, default 0
        Standard deviation of the noise.

    Returns
    -------
    gn : numpy.ndarray
        Gaussian noise of given length, mean, and standard deviation.

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
    """Generate 1d sinusoidal noise of given
    length, amplitude, start phase, and end phase.

    Parameters
    ----------
    siglen : int
        Length of the (noise) signal.
    start_phase : numbers.Real
        Start phase, with units in degrees.
    end_phase : numbers.Real
        End phase, with units in degrees.
    amplitude : numbers.Real
        Amplitude of the sinusoidal curve.
    amplitude_mean : numbers.Real
        Mean amplitude of an extra Gaussian noise.
    amplitude_std : numbers.Real, default 0
        Standard deviation of an extra Gaussian noise

    Returns
    -------
    sn : numpy.ndarray,
        Sinusoidal noise of given length, amplitude, start phase, and end phase.

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
    """Generate 1d baseline wander of given
    length, amplitude, and frequency.

    Parameters
    ----------
    siglen : int
        Length of the (noise) signal.
    fs : numbers.Real
        Sampling frequency of the original signal.
    bw_fs : numbers.Real, or list of numbers.Real
        Frequency (Frequencies) of the baseline wander.
    amplitude : numbers.Real, or list of numbers.Real
        Amplitude of the baseline wander (corr. to each frequency band).
    amplitude_gaussian : Tuple[numbers.Real], default [0,0]
        2-tuple of :class:`~numbers.Real`.
        Mean and std of amplitude of an extra Gaussian noise.

    Returns
    -------
    bw : numpy.ndarray
        Baseline wander of given length, amplitude, frequency.

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
    """Generate 1d baseline wander of given
    length, amplitude, and frequency.

    Parameters
    ----------
    sig : torch.Tensor
        Batched ECGs to be augmented, of shape (batch, lead, siglen).
    fs : numbers.Real
        Sampling frequency of the original signal.
    bw_fs : numbers.Real, or list of numbers.Real,
        Frequency (Frequencies) of the baseline wander.
    ampl_ratio : numpy.ndarray, optional
        Candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
        of shape ``(m, n)``.
    gaussian : numpy.ndarray, optional
        Candidate mean and std of the Gaussian noises,
        of shape ``(k, 2)``.

    Returns
    -------
    bw : numpy.ndarray
        Baseline wander of given length, amplitude, frequency,
        of shape ``(batch, lead, siglen)``.

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
    bw = torch.as_tensor(np.array(bw), dtype=sig.dtype, device=sig.device).reshape(
        batch, lead, siglen
    )
    return bw
