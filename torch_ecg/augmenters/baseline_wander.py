"""Add baseline wander composed of sinusoidal and Gaussian noise to the ECGs."""

from itertools import repeat
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..cfg import DEFAULTS
from ..utils.utils_signal import get_ampl
from .base import Augmenter
from .registry import AUGMENTERS

__all__ = [
    "BaselineWanderAugmenter",
]


@AUGMENTERS.register(name="baseline_wander")
@AUGMENTERS.register()
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
        bw_fs: Optional[NDArray] = None,
        ampl_ratio: Optional[NDArray] = None,
        gaussian: Optional[NDArray] = None,
        prob: float = 0.5,
        inplace: bool = True,
        **kwargs: Any,
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
        assert self.bw_fs.ndim == 1 and self.ampl_ratio.ndim == 2 and self.bw_fs.shape[0] == self.ampl_ratio.shape[1]
        self.inplace = inplace

        self._n_bw_choices = len(self.ampl_ratio)
        self._n_gn_choices = len(self.gaussian)

    def forward(
        self, sig: Tensor, label: Optional[Tensor], *extra_tensors: Sequence[Tensor], **kwargs: Any
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
            sig.add_(gen_baseline_wander(sig, self.fs, self.bw_fs, self.ampl_ratio, self.gaussian))  # type: ignore
        return (sig, label, *extra_tensors)  # type: ignore

    def extra_repr_keys(self) -> List[str]:
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
        Amplitude of each lead, of shape ``(batch * lead, 1)``.

    """
    # `get_ampl` is natively vectorized over any leading dimensions,
    # hence no loop (nor multiprocessing pool) is needed
    ampl = get_ampl(sig.detach().cpu().numpy(), fs)
    ampl = torch.as_tensor(np.asarray(ampl), dtype=sig.dtype, device=sig.device).reshape(-1, 1)
    return ampl


def _gen_gaussian_noise(siglen: int, mean: Union[float, int] = 0, std: Union[float, int] = 0) -> NDArray:
    """Generate 1d Gaussian noise of given
    length, mean, and standard deviation.

    Parameters
    ----------
    siglen : int
        Length of the noise signal.
    mean : float or int, default 0
        Mean value of the noise.
    std : float or int, default 0
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
    start_phase: Union[float, int],
    end_phase: Union[float, int],
    amplitude: Union[float, int],
    amplitude_mean: Union[float, int] = 0,
    amplitude_std: Union[float, int] = 0,
) -> NDArray:
    """Generate 1d sinusoidal noise of given
    length, amplitude, start phase, and end phase.

    Parameters
    ----------
    siglen : int
        Length of the (noise) signal.
    start_phase : float or int
        Start phase, with units in degrees.
    end_phase : float or int
        End phase, with units in degrees.
    amplitude : float or int
        Amplitude of the sinusoidal curve.
    amplitude_mean : float or int
        Mean amplitude of an extra Gaussian noise.
    amplitude_std : float or int, default 0
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
    fs: Union[float, int],
    bw_fs: Union[float, int, Sequence[Union[float, int]]],
    amplitude: Union[float, int, Sequence[Union[float, int]]],
    amplitude_gaussian: Sequence[Union[float, int]] = [0, 0],
) -> NDArray:
    """Generate 1d baseline wander of given
    length, amplitude, and frequency.

    Parameters
    ----------
    siglen : int
        Length of the (noise) signal.
    fs : float or int
        Sampling frequency of the original signal.
    bw_fs : float or int, or list of float or int
        Frequency (Frequencies) of the baseline wander.
    amplitude : float or int, or list of float or int
        Amplitude of the baseline wander (corr. to each frequency band).
    amplitude_gaussian : Tuple[float or int], default [0,0]
        2-tuple of :class:`~float or int`.
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
    if isinstance(bw_fs, (int, float)):
        _bw_fs = [bw_fs]
    else:
        _bw_fs = bw_fs
    if isinstance(amplitude, (int, float)):
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
    fs: Union[float, int],
    bw_fs: Union[float, int, Sequence[Union[float, int]]],
    ampl_ratio: NDArray,
    gaussian: NDArray,
) -> Tensor:
    """Generate baseline wander for a batch of ECGs,
    one independent wander curve per (batch, lead) item.

    Fully vectorized over ``(batch, lead)``.

    Parameters
    ----------
    sig : torch.Tensor
        Batched ECGs to be augmented, of shape (batch, lead, siglen).
    fs : float or int
        Sampling frequency of the original signal.
    bw_fs : float or int, or list of float or int,
        Frequency (Frequencies) of the baseline wander.
    ampl_ratio : numpy.ndarray, optional
        Candidate ratios of noise amplitudes compared to the original
        ECGs for each `fs`, of shape ``(m, n)``.
    gaussian : numpy.ndarray, optional
        Candidate mean and std of the Gaussian noises
        (in terms of ratios of the ECG amplitudes),
        of shape ``(k, 2)``.

    Returns
    -------
    bw : torch.Tensor
        Baseline wander of given length, amplitude, frequency,
        of shape ``(batch, lead, siglen)``.

    """
    batch, lead, siglen = sig.shape
    n_items = batch * lead
    # amplitude of each (batch, lead) item, of shape (n_items, 1)
    sig_ampl = _get_ampl(sig, fs).cpu().numpy()
    _n_bw_choices = len(ampl_ratio)
    _n_gn_choices = len(gaussian)
    _bw_fs = np.atleast_1d(np.asarray(bw_fs, dtype=float))

    # one independent choice of (ampl_ratio row, gaussian row) per item
    ampl_idx = DEFAULTS.RNG.integers(0, _n_bw_choices, n_items)
    gn_idx = DEFAULTS.RNG.integers(0, _n_gn_choices, n_items)
    # of shape (n_items, n_bw_fs)
    amplitude = sig_ampl * ampl_ratio[ampl_idx]
    # of shape (n_items, 2), ratios scaled by the ECG amplitude
    amplitude_gaussian = sig_ampl * gaussian[gn_idx]

    # Gaussian noise part, of shape (n_items, siglen)
    bw = DEFAULTS.RNG.normal(
        amplitude_gaussian[:, 0:1],
        amplitude_gaussian[:, 1:2],
        (n_items, siglen),
    )

    # sinusoidal part, vectorized over (n_items, n_bw_fs, siglen)
    duration = siglen / fs
    # start phase in degrees, of shape (n_items, n_bw_fs)
    start_phase = DEFAULTS.RNG.integers(0, 361, (n_items, _bw_fs.shape[0]))
    end_phase = duration * _bw_fs[None, :] * 360 + start_phase
    # linear interpolation of the phases from start to end
    t = np.linspace(0, 1, siglen)
    phase = start_phase[..., None] + t * (end_phase - start_phase)[..., None]
    bw += np.sum(amplitude[..., None] * np.sin(np.pi * phase / 180), axis=1)

    bw = torch.as_tensor(bw, dtype=sig.dtype, device=sig.device).reshape(batch, lead, siglen)
    return bw
