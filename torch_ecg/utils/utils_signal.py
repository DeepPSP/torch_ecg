"""
utilities for signal processing,
including spatial, temporal, spatio-temporal domains

"""

from copy import deepcopy
from numbers import Real
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import interpolate
from scipy.signal import butter, filtfilt, peak_prominences

from .utils_data import ensure_siglen


__all__ = [
    "smooth",
    "resample_irregular_timeseries",
    "detect_peaks",
    "remove_spikes_naive",
    "butter_bandpass_filter",
    "get_ampl",
    "normalize",
]


def smooth(
    x: np.ndarray,
    window_len: int = 11,
    window: str = "hanning",
    mode: str = "valid",
    keep_dtype: bool = True,
) -> np.ndarray:
    """finished, checked

    smooth the 1d data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x: ndarray,
        the input signal
    window_len: int, default 11,
        the length of the smoothing window,
        (previously should be an odd integer, currently can be any (positive) integer)
    window: str, default 'hanning',
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
        flat window will produce a moving average smoothing
    mode: str, default 'valid',
        ref. np.convolve
    keep_dtype: bool, default True,
        dtype of the returned value keeps the same with that of `x` or not

    Returns
    -------
    y: ndarray,
        the smoothed signal

    Examples
    --------
    >>> t = linspace(-2, 2, 0.1)
    >>> x = sin(t) + randn(len(t)) * 0.1
    >>> y = smooth(x)

    See also:
    ---------
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    References
    ----------
    [1] https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    """
    radius = min(len(x), window_len)
    radius = radius if radius % 2 == 1 else radius - 1

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    # if x.size < radius:
    #     raise ValueError("Input vector needs to be bigger than window size.")

    if radius < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[radius - 1 : 0 : -1], x, x[-2 : -radius - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(radius, "d")
    else:
        w = eval("np." + window + "(radius)")

    y = np.convolve(w / w.sum(), s, mode=mode)
    y = y[(radius // 2 - 1) : -(radius // 2) - 1]
    assert len(x) == len(y)

    if keep_dtype:
        y = y.astype(x.dtype)

    return y


def resample_irregular_timeseries(
    s: np.ndarray,
    output_fs: Real = 2,
    method: str = "spline",
    return_with_time: bool = False,
    tnew: Optional[np.ndarray] = None,
    interp_kw: dict = {},
    verbose: int = 0,
) -> np.ndarray:
    """

    resample the 2d irregular timeseries `s` into a 1d or 2d regular time series with frequency `output_fs`,
    elements of `s` are in the form [time, value], where the unit of `time` is ms

    Parameters
    ----------
    s: array_like,
        the 2d irregular timeseries
    output_fs: Real, default 2,
        the frequency of the output 1d regular timeseries
    method: str, default "spline"
        interpolation method, can be 'spline' or 'interp1d'
    return_with_time: bool, default False,
        return a 2d array, with the 0-th coordinate being time
    tnew: array_like, optional,
        the array of time of the output array
    interp_kw: dict, default {},
        additional options for the corresponding methods in scipy.interpolate

    Returns
    -------
    np.ndarray, a 1d or 2d regular time series with frequency `output_freq`

    NOTE:
    pandas also has the function to regularly resample irregular timeseries

    """
    if method.lower() not in ["spline", "interp1d"]:
        raise ValueError("method {} not implemented".format(method))

    if verbose >= 1:
        print("len(s) = {}".format(len(s)))

    if len(s) == 0:
        return np.array([])

    time_series = np.atleast_2d(s)
    step_ts = 1000 / output_fs
    tot_len = int((time_series[-1][0] - time_series[0][0]) / step_ts) + 1
    if tnew is None:
        xnew = time_series[0][0] + np.arange(0, tot_len * step_ts, step_ts)
    else:
        xnew = np.array(tnew)

    if verbose >= 1:
        print(
            f"time_series start ts = {time_series[0][0]}, end ts = {time_series[-1][0]}"
        )
        print(f"tot_len = {tot_len}")
        print(f"xnew start = {xnew[0]}, end = {xnew[-1]}")

    if method.lower() == "spline":
        m = len(time_series)
        w = interp_kw.get("w", np.ones(shape=(m,)))
        # s = interp_kw.get("s", np.random.uniform(m-np.sqrt(2*m),m+np.sqrt(2*m)))
        s = interp_kw.get("s", m - np.sqrt(2 * m))
        interp_kw.update(w=w, s=s)

        tck = interpolate.splrep(time_series[:, 0], time_series[:, 1], **interp_kw)

        regular_timeseries = interpolate.splev(xnew, tck)
    elif method.lower() == "interp1d":
        f = interpolate.interp1d(time_series[:, 0], time_series[:, 1], **interp_kw)

        regular_timeseries = f(xnew)

    if return_with_time:
        return np.column_stack((xnew, regular_timeseries))
    else:
        return regular_timeseries


def detect_peaks(
    x: Sequence,
    mph: Optional[Real] = None,
    mpd: int = 1,
    threshold: Real = 0,
    left_threshold: Real = 0,
    right_threshold: Real = 0,
    prominence: Optional[Real] = None,
    prominence_wlen: Optional[int] = None,
    edge: Union[str, None] = "rising",
    kpsh: bool = False,
    valley: bool = False,
    show: bool = False,
    ax=None,
    verbose: int = 0,
) -> np.ndarray:
    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x: 1D array_like,
        data
    mph: positive number, optional,
        abbr. for maximum (minimum) peak height,
        detect peaks that are greater than minimum peak height (if parameter `valley` is False),
        or peaks that are smaller than maximum peak height (if parameter `valley` is True)
    mpd: positive integer, default 1,
        abbr. for minimum peak distance,
        detect peaks that are at least separated by minimum peak distance (in number of samples)
    threshold: positive number, default 0,
        detect peaks (valleys) that are greater (smaller) than `threshold`,
        in relation to their neighbors within the range of `mpd`
    left_threshold: positive number, default 0,
        `threshold` that is restricted to the left
    right_threshold: positive number, default 0,
        `threshold` that is restricted to the left
    prominence: positive number, optional,
        threshold of prominence of the detected peaks (valleys)
    prominence_wlen: positive int, optional,
        the `wlen` parameter of the function `scipy.signal.peak_prominences`
    edge: str or None, default 'rising',
        can also be 'falling', 'both',
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'),
        both edges ('both'), or don't detect a flat peak (None)
    kpsh: bool, default False,
        keep peaks with same height even if they are closer than `mpd`
    valley: bool, default False,
        if True (1), detect valleys (local minima) instead of peaks
    show: bool, default False,
        if True (1), plot data in matplotlib figure
    ax: a matplotlib.axes.Axes instance, optional,

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    NOTE
    ----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """
    data = deepcopy(x)
    data = np.atleast_1d(data).astype("float64")
    if data.size < 3:
        return np.array([], dtype=int)

    if valley:
        data = -data
        if mph is not None:
            mph = -mph

    # find indices of all peaks
    dx = data[1:] - data[:-1]  # equiv to np.diff()

    # handle NaN's
    indnan = np.where(np.isnan(data))[0]
    if indnan.size:
        data[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ["rising", "both"]:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ["falling", "both"]:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))

    if verbose >= 1:
        print(
            f"before filtering by mpd = {mpd}, and threshold = {threshold}, ind = {ind.tolist()}"
        )
        print(
            f"additionally, left_threshold = {left_threshold}, right_threshold = {right_threshold}, length of data = {len(data)}"
        )

    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            np.in1d(
                ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True
            )
        ]

    if verbose >= 1:
        print(f"after handling nan values, ind = {ind.tolist()}")

    # peaks are only valid within [mpb, len(data)-mpb[
    ind = np.array([pos for pos in ind if mpd <= pos < len(data) - mpd])

    if verbose >= 1:
        print(
            f"after fitering out elements too close to border by mpd = {mpd}, ind = {ind.tolist()}"
        )

    # first and last values of data cannot be peaks
    # if ind.size and ind[0] == 0:
    #     ind = ind[1:]
    # if ind.size and ind[-1] == data.size-1:
    #     ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[data[ind] >= mph]

    if verbose >= 1:
        print(f"after filtering by mph = {mph}, ind = {ind.tolist()}")

    # remove peaks - neighbors < threshold
    _left_threshold = left_threshold if left_threshold > 0 else threshold
    _right_threshold = right_threshold if right_threshold > 0 else threshold
    if ind.size and (_left_threshold > 0 and _right_threshold > 0):
        # dx = np.min(np.vstack([data[ind]-data[ind-1], data[ind]-data[ind+1]]), axis=0)
        dx = np.max(
            np.vstack([data[ind] - data[ind + idx] for idx in range(-mpd, 0)]), axis=0
        )
        ind = np.delete(ind, np.where(dx < _left_threshold)[0])
        if verbose >= 2:
            print(f"from left, dx = {dx.tolist()}")
            print(
                f"after deleting those dx < _left_threshold = {_left_threshold}, ind = {ind.tolist()}"
            )
        dx = np.max(
            np.vstack([data[ind] - data[ind + idx] for idx in range(1, mpd + 1)]),
            axis=0,
        )
        ind = np.delete(ind, np.where(dx < _right_threshold)[0])
        if verbose >= 2:
            print(f"from right, dx = {dx.tolist()}")
            print(
                f"after deleting those dx < _right_threshold = {_right_threshold}, ind = {ind.tolist()}"
            )
    if verbose >= 1:
        print(f"after filtering by threshold, ind = {ind.tolist()}")
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(data[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    data[ind[i]] > data[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    ind = np.array(
        [
            item
            for item in ind
            if data[item] == np.max(data[item - mpd : item + mpd + 1])
        ]
    )

    if verbose >= 1:
        print(f"after filtering by mpd, ind = {ind.tolist()}")

    if prominence:
        _p = peak_prominences(data, ind, prominence_wlen)[0]
        ind = ind[np.where(_p >= prominence)[0]]
        if verbose >= 1:
            print(f"after filtering by prominence, ind = {ind.tolist()}")
            if verbose >= 2:
                print(f"with detailed prominence = {_p.tolist()}")

    if show:
        if indnan.size:
            data[indnan] = np.nan
        if valley:
            data = -data
            if mph is not None:
                mph = -mph
        _plot(data, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """
    Plot results of the detect_peaks function, see its help.

    Parameters: ref. the function `detect_peaks`

    """
    if "plt" not in dir():
        import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x, "b", lw=1)
    if ind.size:
        label = "valley" if valley else "peak"
        label = label + "s" if ind.size > 1 else label
        ax.plot(
            ind,
            x[ind],
            "+",
            mfc=None,
            mec="r",
            mew=2,
            ms=8,
            label="%d %s" % (ind.size, label),
        )
        ax.legend(loc="best", framealpha=0.5, numpoints=1)
    ax.set_xlim(-0.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel("Data #", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    mode = "Valley detection" if valley else "Peak detection"
    ax.set_title(
        "%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
        % (mode, str(mph), mpd, str(threshold), edge)
    )
    # plt.grid()
    plt.show()


def remove_spikes_naive(
    sig: np.ndarray, threshold: Real = 20, inplace: bool = True
) -> np.ndarray:
    """

    remove `spikes` from `sig` using a naive method proposed in entry 0416 of CPSC2019

    `spikes` here refers to abrupt large bumps with (abs) value larger than the given threshold,
    or nan values (read by `wfdb`),
    do NOT confuse with `spikes` in paced rhythm

    Parameters
    ----------
    sig: ndarray,
        1d signal with potential spikes
    threshold: real number,
        values of `sig` that are larger than `threshold` will be removed

    Returns
    -------
    sig: ndarray,
        signal with `spikes` removed

    """
    b = list(
        filter(
            lambda k: k > 0,
            np.argwhere(np.logical_or(np.abs(sig) > threshold, np.isnan(sig))).squeeze(
                -1
            ),
        )
    )
    if not inplace:
        sig = sig.copy()
    for k in b:
        sig[k] = sig[k - 1]
    return sig


def butter_bandpass(
    lowcut: Real, highcut: Real, fs: Real, order: int, verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Butterworth Bandpass Filter Design

    Parameters
    ----------
    lowcut: real,
        low cutoff frequency
    highcut: real,
        high cutoff frequency
    fs: real,
        frequency of `data`
    order: int,
        order of the filter
    verbose: int, default 0

    Returns
    -------
    b, a: tuple of ndarray,
        coefficients of numerator and denominator of the filter

    NOTE
    ----
    according to `lowcut` and `highcut`, the filter type might fall to lowpass or highpass filter

    References
    ----------
    [1] scipy.signal.butter
    [2] https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    if low >= 1:
        raise ValueError("frequency out of range!")
    high = highcut / nyq

    if low <= 0 and high >= 1:
        b, a = [1], [1]
        return b, a

    if low <= 0:
        Wn = high
        btype = "low"
    elif high >= 1:
        Wn = low
        btype = "high"
    elif lowcut == highcut:
        Wn = high
        btype = "low"
    else:
        Wn = [low, high]
        btype = "band"

    if verbose >= 1:
        print(
            f"by the setup of lowcut and highcut, the filter type falls to {btype}, with Wn = {Wn}"
        )

    b, a = butter(order, Wn, btype=btype)
    return b, a


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: Real,
    highcut: Real,
    fs: Real,
    order: int,
    btype: Optional[str] = None,
    verbose: int = 0,
) -> np.ndarray:
    """
    Butterworth Bandpass

    Parameters
    ----------
    data: ndarray,
        data to be filtered
    lowcut: real,
        low cutoff frequency
    highcut: real,
        high cutoff frequency
    fs: real,
        frequency of `data`
    order: int,
        order of the filter
    btype: str, optional,
        (special) type of the filter, can be "lohi", "hilo",
        ignored for lowpass and highpass filters (as given by `lowcut` and `highcut`)
    verbose: int, default 0

    Returns
    -------
    y, ndarray,
        the filtered signal

    References
    ----------
    [1] https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    [2] https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt

    """
    if btype is None:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order, verbose=verbose)
        y = filtfilt(b, a, data)
        return y
    if btype.lower() == "lohi":
        b, a = butter_bandpass(0, highcut, fs, order=order, verbose=verbose)
        y = filtfilt(b, a, data)
        b, a = butter_bandpass(lowcut, fs, fs, order=order, verbose=verbose)
        y = filtfilt(b, a, y)
    elif btype.lower() == "hilo":
        b, a = butter_bandpass(lowcut, fs, fs, order=order, verbose=verbose)
        y = filtfilt(b, a, data)
        b, a = butter_bandpass(0, highcut, fs, order=order, verbose=verbose)
        y = filtfilt(b, a, y)
    else:
        raise ValueError(f"special btype {btype} is not supported")
    return y


def get_ampl(
    sig: np.ndarray,
    fs: Real,
    fmt: str = "lead_first",
    window: Real = 0.2,
    critical_points: Optional[Sequence] = None,
) -> Union[float, np.ndarray]:
    """

    get amplitude of a signal (near critical points if given)

    Parameters
    ----------
    sig: ndarray,
        (ecg) signal
    fs: real number,
        sampling frequency of the signal
    fmt: str, default "lead_first",
        format of the signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first"),
        ignored if sig is 1d array (single-lead)
    window: int, default 0.2s,
        window length of a window for computing amplitude, with units in seconds
    critical_points: ndarray, optional,
        positions of critical points near which to compute amplitude,
        e.g. can be rpeaks, t peaks, etc.

    Returns
    -------
    ampl: float, or ndarray,
        amplitude of the signal

    """
    if fmt.lower() in ["channel_last", "lead_last"]:
        _sig = sig.T
    else:
        _sig = sig
    _window = int(round(window * fs))
    half_window = _window // 2
    _window = half_window * 2
    if _sig.ndim == 1:
        ampl = 0
    else:
        ampl = np.zeros((_sig.shape[0],))
    if critical_points is not None:
        s = np.stack(
            [
                ensure_siglen(
                    _sig[
                        ...,
                        max(0, p - half_window) : min(_sig.shape[-1], p + half_window),
                    ],
                    siglen=_window,
                    fmt="lead_first",
                )
                for p in critical_points
            ],
            axis=-1,
        )
        # the following is much slower
        # for p in critical_points:
        #     s = _sig[...,max(0,p-half_window):min(_sig.shape[-1],p+half_window)]
        #     ampl = np.max(np.array([ampl, np.max(s,axis=-1) - np.min(s,axis=-1)]), axis=0)
    else:
        s = np.stack(
            [
                _sig[..., idx * half_window : idx * half_window + _window]
                for idx in range(_sig.shape[-1] // half_window - 1)
            ],
            axis=-1,
        )
        # the following is much slower
        # for idx in range(_sig.shape[-1]//half_window-1):
        #     s = _sig[..., idx*half_window: idx*half_window+_window]
        #     ampl = np.max(np.array([ampl, np.max(s,axis=-1) - np.min(s,axis=-1)]), axis=0)
    ampl = np.max(np.max(s, axis=-2) - np.min(s, axis=-2), axis=-1)
    return ampl


def normalize(
    sig: np.ndarray,
    method: str,
    mean: Union[Real, Iterable[Real]] = 0.0,
    std: Union[Real, Iterable[Real]] = 1.0,
    sig_fmt: str = "channel_first",
    per_channel: bool = False,
) -> np.ndarray:
    r"""

    perform z-score normalization on `sig`,
    to make it has fixed mean and standard deviation,
    or perform min-max normalization on `sig`,
    or normalize `sig` using `mean` and `std` via (sig - mean) / std.
    More precisely,

        .. math::
            \begin{align*}
            \text{Min-Max normalization:} & \frac{sig - \min(sig)}{\max(sig) - \min(sig)} \\
            \text{Naive normalization:} & \frac{sig - m}{s} \\
            \text{Z-score normalization:} & \left(\frac{sig - mean(sig)}{std(sig)}\right) \cdot s + m
            \end{align*}

    Parameters
    ----------
    sig: ndarray,
        signal to be normalized
    method: str,
        normalization method, case insensitive, can be one of
        "naive", "min-max", "z-score",
    mean: real number or array_like, default 0.0,
        mean value of the normalized signal,
        or mean values for each lead of the normalized signal,
        useless if `method` is "min-max"
    std: real number or array_like, default 1.0,
        standard deviation of the normalized signal,
        or standard deviations for each lead of the normalized signal,
        useless if `method` is "min-max"
    sig_fmt: str, default "channel_first",
        format of the signal, can be of one of
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first")
    per_channel: bool, default False,
        if True, normalization will be done per channel

    Returns
    -------
    nm_sig: ndarray,
        the normalized signal

    NOTE
    ----
    in cases where normalization is infeasible (std = 0),
    only the mean value will be shifted

    """
    _method = method.lower()
    assert _method in [
        "z-score",
        "naive",
        "min-max",
    ]
    if isinstance(std, Real):
        assert std > 0, "standard deviation should be positive"
    else:
        assert (np.array(std) > 0).all(), "standard deviations should all be positive"
    if not per_channel:
        assert isinstance(mean, Real) and isinstance(
            std, Real
        ), "mean and std should be real numbers in the non per-channel setting"
    assert sig_fmt.lower() in [
        "channel_first",
        "lead_first",
        "channel_last",
        "lead_last",
    ], f"format {sig_fmt} of the signal not supported!"

    if isinstance(mean, Iterable):
        if sig_fmt.lower() in [
            "channel_first",
            "lead_first",
        ]:
            _mean = np.array(mean)[..., np.newaxis]
        else:
            _mean = np.array(mean)[np.newaxis, ...]
    else:
        _mean = mean
    if isinstance(std, Iterable):
        if sig_fmt.lower() in [
            "channel_first",
            "lead_first",
        ]:
            _std = np.array(std)[..., np.newaxis]
        else:
            _std = np.array(std)[np.newaxis, ...]
    else:
        _std = std

    if _method == "naive":
        nm_sig = (sig - _mean) / _std
        return nm_sig

    eps = 1e-7  # to avoid dividing by zero
    if sig.ndim == 3:  # the first dimension is the batch dimension
        if not per_channel:
            options = dict(axis=(1, 2), keepdims=True)
        elif sig_fmt.lower() in [
            "channel_first",
            "lead_first",
        ]:
            options = dict(axis=2, keepdims=True)
        else:
            options = dict(axis=1, keepdims=True)
    else:
        if not per_channel:
            options = dict(axis=None)
        elif sig_fmt.lower() in [
            "channel_first",
            "lead_first",
        ]:
            options = dict(axis=1, keepdims=True)
        else:
            options = dict(axis=0, keepdims=True)

    if _method == "z-score":
        nm_sig = (
            (sig - np.mean(sig, **options)) / (np.std(sig, **options) + eps)
        ) * _std + _mean
    elif _method == "min-max":
        nm_sig = (sig - np.amin(sig, **options)) / (
            np.amax(sig, **options) - np.amin(sig, **options) + eps
        )
    return nm_sig
