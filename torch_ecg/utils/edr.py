"""
"""

from numbers import Real
from typing import Sequence

import numpy as np

__all__ = [
    "phs_edr",
]


def phs_edr(
    sig: Sequence,
    fs: int,
    rpeaks: Sequence,
    winL_t: Real = 40,
    winR_t: Real = 40,
    return_with_time: bool = True,
    mode: str = "complex",
    verbose: int = 0,
) -> np.ndarray:
    """

    computes the respiratory rate from single-lead ECG signals.

    ref. the `main` function and the `getxy`, `edr` function of physionet edr.c

    Parameters
    ----------
    sig: array-like,
        the (single lead ECG) signal
    fs: int,
        sampling frequency of the signal
    rpeaks: array-like,
        indices of R peaks in the signal
    winL_t: real number, default 40,
        left length of the window at R peaks for the computation of the area of a QRS complex, with units in ms
    winR_t: real number, default 40,
        right length of the window at R peaks for the computation of the area of a QRS complex, with units in ms
    return_with_time: bool, default True,
        if True, returns the time along with the EDR values at which they are computed
    mode: str, default "complex", can also be "simple",
        apply a filtering process (the `edr` function of physionet edr.c) or simply use the `_getxy` function to compute EDR
    verbose: int, default 0,
        for printing the computation details

    Returns
    -------
    np.ndarray,
        1d, if `return_with_time` is set False,
        2d in the form of [idx,val], if `return_with_time` is set True

    """
    ts = np.array(rpeaks) * 1000 // fs
    winL, winR = int(winL_t * fs / 1000), int(winR_t * fs / 1000)

    if mode == "simple":
        ecg_der_rsp = np.vectorize(lambda idx: _getxy(sig, idx - winL, idx + winR))(
            np.array(rpeaks)
        )
    elif mode == "complex":
        ecg_der_rsp = []
        xm, xc, xd, xdmax = 0, 0, 0, 0
        for idx in rpeaks:
            if verbose == -1:
                print("-" * 80)
                print(f"idx = {idx}, winL = {winL}, winR = {winR}")
            x = _getxy(sig, idx - winL, idx + winR)
            if verbose == -1:
                print(f"x = {x}")

            # calculate instantaneous EDR
            if x == 0:
                ecg_der_rsp.append(0)
                continue

            d = x - xm
            if verbose == -1:
                print(f"before: d = {d}, xc = {xc}, xdmax = {xdmax}")

            if xc < 500:  # why 500?
                xc += 1
                dn = d / xc
            else:
                dn = d / xc
                if dn > xdmax:
                    dn = xdmax
                elif dn < -xdmax:
                    dn = -xdmax
            if verbose == -1:
                print(f"after: d = {d}, xc = {xc}, dn = {dn}, xdmax = {xdmax}")

            xm += dn
            xd += abs(dn) - xd / xc

            if xd < 1.0:
                xd = 1.0
            xdmax = 3.0 * xd / xc
            r = d / xd
            if verbose == -1:
                print(f"xm = {xm}, xc = {xc}, xd = {xd}, xdmax = {xdmax}, r = {r}")
            ecg_der_rsp.append(int(r * 50))
            # end of calculation of instantaneous EDR
    else:
        raise ValueError(f"No mode named {mode}!")

    ecg_der_rsp = np.array(ecg_der_rsp)

    if verbose >= 2:
        pass  # TODO: some plot

    if return_with_time:
        return np.column_stack((ts, ecg_der_rsp))
    else:
        return ecg_der_rsp


def _getxy(sig: Sequence, von: int, bis: int) -> Real:
    """
    compute the integrand from `von` to `bis` of the signals with baseline removed

    """
    return (np.array(sig)[von : bis + 1]).sum()
