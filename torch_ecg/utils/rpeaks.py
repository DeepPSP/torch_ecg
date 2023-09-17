"""
algorithms detecting R peaks from (filtered) single-lead ECG signal

Exists algorithms from wfdb and biosppy,

TODO: algorithms compared in [1]_

NOTE:
detectors (xqrs tested) from `wfdb` better fed with signal units in mV,
if in μV, more false r peaks might be detected,
currently not tested using detectors from `biosppy`

References:
-----------
.. [1] Liu, Feifei, et al. "Performance analysis of ten common QRS detectors on different ECG application cases." Journal of healthcare engineering 2018 (2018).

"""

from numbers import Real

import biosppy.signals.ecg as BSE
import numpy as np
from wfdb.processing.qrs import gqrs_detect as _gqrs_detect
from wfdb.processing.qrs import xqrs_detect as _xqrs_detect

__all__ = [
    "xqrs_detect",
    "gqrs_detect",
    "hamilton_detect",
    "ssf_detect",
    "christov_detect",
    "engzee_detect",
    "gamboa_detect",
]


# ---------------------------------------------------------------------
# algorithms from wfdb


def xqrs_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """XQRS algorithm.

    default kwargs:

        sampfrom=0, sampto='end', conf=None, learn=True, verbose=True

    """
    sig = sig.copy()  # to avoid changing the original signal
    kw = dict(sampfrom=0, sampto="end", conf=None, learn=True, verbose=False)
    kw = {k: kwargs.get(k, v) for k, v in kw.items()}
    rpeaks = _xqrs_detect(sig, fs, **kw)
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


def gqrs_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """GQRS algorithm.

    default kwargs:

        d_sig=None, adc_gain=None, adc_zero=None,
        threshold=1.0, hr=75, RRdelta=0.2, RRmin=0.28, RRmax=2.4,
        QS=0.07, QT=0.35, RTmin=0.25, RTmax=0.33,
        QRSa=750, QRSamin=130

    """
    sig = sig.copy()  # to avoid changing the original signal
    kw = dict(
        d_sig=None,
        adc_gain=None,
        adc_zero=None,
        threshold=1.0,
        hr=75,
        RRdelta=0.2,
        RRmin=0.28,
        RRmax=2.4,
        QS=0.07,
        QT=0.35,
        RTmin=0.25,
        RTmax=0.33,
        QRSa=750,
        QRSamin=130,
    )
    kw = {k: kwargs.get(k, v) for k, v in kw.items()}
    rpeaks = _gqrs_detect(sig, fs, **kw)
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


# ---------------------------------------------------------------------
# algorithms from biosppy
def hamilton_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """The default detector used by `biosppy`.

    This algorithm is based on [#ham]_.

    References
    ----------
    .. [#ham] Hamilton, Pat. "Open source ECG analysis." Computers in cardiology. IEEE, 2002.

    """
    sig = sig.copy()  # to avoid changing the original signal
    # segment
    (rpeaks,) = BSE.hamilton_segmenter(signal=sig, sampling_rate=fs)

    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


def ssf_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """Slope Sum Function (SSF)

    This algorithm is originally proposed for blood pressure (BP)
    pulse detection [#ssf]_.

    References
    ----------
    .. [#ssf] Zong, W., et al. "An open-source algorithm to
              detect onset of arterial blood pressure pulses."
              Computers in Cardiology, 2003. IEEE, 2003.

    """
    sig = sig.copy()  # to avoid changing the original signal
    (rpeaks,) = BSE.ssf_segmenter(
        signal=sig,
        sampling_rate=fs,
        threshold=kwargs.get("threshold", 20),
        before=kwargs.get("before", 0.03),
        after=kwargs.get("after", 0.01),
    )
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


def christov_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """Christov detector.

    Detector proposed in [#chr]_.

    References
    ----------
    .. [#chr] Ivaylo I. Christov, "Real time electrocardiogram QRS detection
              using combined adaptive threshold", BioMedical Engineering OnLine 2004,
              vol. 3:28, 2004

    """
    sig = sig.copy()  # to avoid changing the original signal
    (rpeaks,) = BSE.christov_segmenter(signal=sig, sampling_rate=fs)
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


def engzee_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """Detector proposed by Engelse and Zeelenberg.

    This algorithm is originally proposed in [#ez1]_,
    and improved in [#ez2]_.

    References
    ----------
    .. [#ez1] W. Engelse and C. Zeelenberg, "A single scan algorithm for
              QRS detection and feature extraction", IEEE Comp. in Cardiology,
              vol. 6, pp. 37-42, 1979
    .. [#ez2] A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
              "Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics",
              BIOSIGNALS 2012, pp. 49-54, 2012

    """
    sig = sig.copy()  # to avoid changing the original signal
    (rpeaks,) = BSE.engzee_segmenter(
        signal=sig,
        sampling_rate=fs,
        threshold=kwargs.get("threshold", 0.48),
    )
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks


def gamboa_detect(sig: np.ndarray, fs: Real, **kwargs) -> np.ndarray:
    """Detector proposed by Gamboa.

    This algorithm is proposed in a PhD thesis [#gam]_.

    References
    ----------
    .. [#gam] Gamboa, Hugo. "Multi-modal behavioral biometrics based on
              HCI and electrophysiology." PhD ThesisUniversidade (2008).

    """
    sig = sig.copy()  # to avoid changing the original signal
    (rpeaks,) = BSE.gamboa_segmenter(
        signal=sig,
        sampling_rate=fs,
        tol=kwargs.get("tol", 0.48),
    )
    # correct R-peak locations
    (rpeaks,) = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=kwargs.get("correct_tol", 0.05),
    )
    return rpeaks
