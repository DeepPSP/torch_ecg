"""
"""

from typing import Union, Optional

import pywt
import numpy as np
from easydict import EasyDict as ED


__all__ = [
    "get_dwt_features",
    "get_full_dwt_features",
]


def get_dwt_features(
    signal: np.ndarray, fs: int, config: Optional[dict] = None
) -> np.ndarray:
    """

    compute the discrete wavelet transform (DWT) features using Springer's algorithm

    Parameters
    ----------
    signal : np.ndarray,
        the (ECG) signal, of shape (nsamples,)
    fs : int,
        the sampling frequency
    config : dict, optional,
        the configuration, with the following keys:
        - ``'wavelet_level'``: int,
            the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'``: str,
            the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : np.ndarray,
        the DWT features, of shape (nsamples,)

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.downcoef(
        "d", signal, wavelet=cfg.wavelet_name, level=cfg.wavelet_level
    )
    dwt_features = _wkeep1(np.repeat(detail_coefs, 2**cfg.wavelet_level), siglen)
    return dwt_features


def get_full_dwt_features(
    signal: np.ndarray, fs: int, config: Optional[dict] = None
) -> np.ndarray:
    """

    compute the full DWT features using Springer's algorithm

    Parameters
    ----------
    signal : np.ndarray,
        the (ECG) signal, of shape (nsamples,)
    fs : int,
        the sampling frequency
    config : dict, optional,
        the configuration, with the following keys:
        - ``'wavelet_level'``: int,
            the level of the wavelet decomposition, default: 3
        - ``'wavelet_name'``: str,
            the name of the wavelet, default: "db7"

    Returns
    -------
    dwt_features : np.ndarray,
        the full DWT features, of shape (``'wavelet_level'``, nsamples)

    """
    cfg = ED(
        wavelet_level=3,
        wavelet_name="db7",
    )
    cfg.update(config or {})
    siglen = len(signal)

    detail_coefs = pywt.wavedec(signal, cfg.wavelet_name, level=cfg.wavelet_level)[
        :0:-1
    ]
    dwt_features = np.zeros((cfg.wavelet_level, siglen), dtype=signal.dtype)
    for i, detail_coef in enumerate(detail_coefs):
        dwt_features[i] = _wkeep1(np.repeat(detail_coef, 2 ** (i + 1)), siglen)
    return dwt_features


def _wkeep1(x: np.ndarray, k: int, opt: Union[str, int] = "c") -> np.ndarray:
    """

    modified from the matlab function ``wkeep1``

    References
    ----------
    wkeep1.m of the matlab wavelet toolbox

    """
    x_len = len(x)
    if x_len <= k:
        return x
    if isinstance(opt, int):
        first = opt
    elif opt.lower() in ["c", "center", "centre"]:
        first = (x_len - k) // 2
    elif opt.lower() in ["l", "left"]:
        first = 0
    elif opt.lower() in ["r", "right"]:
        first = x_len - k
    else:
        raise ValueError(f"Unknown option: {opt}")
    assert 0 <= first <= x_len - k
    return x[first : first + k]
