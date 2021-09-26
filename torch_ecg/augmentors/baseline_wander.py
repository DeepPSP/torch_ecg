"""
add baseline wander composed of sinusoidal and Gaussian noise to the ECGs
"""

from random import randint, uniform
import multiprocessing as mp
from typing import Any, NoReturn, Sequence, Union
from numbers import Real

from .base import Augmentor
from ..utils.utils_signal import get_ampl


__all__ = ["BaselineWanderAugmentor",]


class BaselineWanderAugmentor(Augmentor):
    """
    """
    __name__ = "BaselineWanderAugmentor"

    def __init__(self,
                 fs:Optional[np.ndarray]=None,
                 ampl_ratio:Optional[np.ndarray]=None,
                 gaussian:Optional[np.ndarray]=None,
                 inplace:bool=False,) -> NoReturn:
        """ finished, NOT checked

        Parameters
        ----------
        fs: ndarray, optional,
            frequencies of the sinusoidal noises,
            of shape (n,)
        ampl_ratio: ndarray, optional,
            candidate ratios of noise amplitdes compared to the original ECGs for each `fs`,
            of shape (m,n)
        gaussian: ndarray, optional,
            candidate mean and std of the Gaussian noises,
            of shape (k, 2)
        inplace: bool, default False,
            currently not used
        """
        self.fs = fs if fs is not None else np.array([0.33, 0.1, 0.05, 0.01])
        self.ampl_ratio = ampl_ratio if ampl_ratio is not None else np.array([
            [0.01, 0.01, 0.02, 0.03],  # low
            [0.01, 0.02, 0.04, 0.05],  # low
            [0.1, 0.06, 0.04, 0.02],  # low
            [0.02, 0.04, 0.07, 0.1],  # low
            [0.05, 0.1, 0.16, 0.25],  # medium
            [0.1, 0.15, 0.25, 0.3],  # high
            [0.25, 0.25, 0.3, 0.35],  # extremely high
        ])
        self.gaussian = gaussian if gaussian is not None else np.array([  # mean and std, ratio
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],  # ensure at least one with no gaussian noise
            [0.0, 0.003],
            [0.0, 0.01],
        ])
        assert self.fs.ndim == 1 and self.ampl_ratio.ndim == 2 and self.fs.shape[0] == self.ampl_ratio.shape[1]
        self.inplace = inplace

        self._n_bw_choices = len(self.ampl_ratio)
        self._n_gn_choices = len(self.gaussian)

    def generate(self, sig:Tensor, label:Tensor, fs:int) -> Tensor:
        """ finished, NOT checked,

        Parameters
        ----------
        to write

        Returns
        -------
        to write
        """
        batch, lead, siglen = sig.shape
        _sig = sig.cpu().numpy()
        with mp.Pool(processes=max(1, mp.cpu_count()-2)) as pool:
            bw_sig = pool.starmap(
                self._generate_single,
                iterable=[(_sig[i][j], fs) for i in range(batch) for j in range(lead)],
            )
        bw_sig = np.array(bw_sig, dtype=_sig.dtype).reshape((batch, lead, siglen))
        bw_sig = torch.from_numpy(bw_sig).to(sig.device)
        # if self.inplace:
        return bw_sig

    def _generate_single(self, sig:np.ndarray, fs:int, siglen:int) -> ndarray:
        """ finished, NOT checked

        Parameters
        ----------
        to write

        Returns
        -------
        to write
        """
        sig_ampl = get_ampl(sig, fs)
        ar = self.ampl_ratio[randint(0, self._n_bw_choices-1)]
        gm, gs = self.aussian[randint(0, self._n_gn_choices-1)]
        bw_ampl = ar * seg_ampl
        g_ampl = gm * seg_ampl
        bw = gen_baseline_wander(
            siglen=siglen,
            fs=fs,
            bw_fs=self.fs,
            amplitude=bw_ampl,
            amplitude_mean=gm,
            amplitude_std=gs,
        )
        bw_sig = sig + bw
        return bw_sig


def gen_gaussian_noise(siglen:int, mean:Real=0, std:Real=0) -> np.ndarray:
    """ finished, checked,

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
    gn = np.random.normal(mean, std, siglen)
    return gn


def gen_sinusoidal_noise(siglen:int,
                         start_phase:Real,
                         end_phase:Real,
                         amplitude:Real,
                         amplitude_mean:Real=0,
                         amplitude_std:Real=0) -> np.ndarray:
    """ finished, checked,

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
    sn += gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    return sn


def gen_baseline_wander(siglen:int,
                        fs:Real,
                        bw_fs:Union[Real,Sequence[Real]],
                        amplitude:Union[Real,Sequence[Real]],
                        amplitude_mean:Real=0,
                        amplitude_std:Real=0) -> np.ndarray:
    """ finished, checked,

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
    amplitude_mean: real number, default 0,
        mean amplitude of an extra Gaussian noise
    amplitude_std: real number, default 0,
        standard deviation of an extra Gaussian noise

    Returns
    -------
    bw: ndarray,
        the baseline wander of given length, amplitude, frequency

    Example
    -------
    >>> gen_baseline_wander(4000, 400, [0.4,0.1,0.05], [0.1,0.2,0.4])
    """
    bw = gen_gaussian_noise(siglen, amplitude_mean, amplitude_std)
    if isinstance(bw_fs, Real):
        _bw_fs = [bw_fs]
    else:
        _bw_fs = bw_fs
    if isinstance(amplitude, Real):
        _amplitude = list(repeat(amplitude, len(_bw_fs)))
    else:
        _amplitude = amplitude
    assert len(_bw_fs) == len(_amplitude)
    duration = (siglen / fs)
    for bf, a in zip(_bw_fs, _amplitude):
        start_phase = np.random.randint(0,360)
        end_phase = duration * bf * 360 + start_phase
        bw += gen_sinusoidal_noise(siglen, start_phase, end_phase, a, 0, 0)
    return bw
