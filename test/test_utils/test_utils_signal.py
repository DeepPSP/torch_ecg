"""
"""

from pathlib import Path

import numpy as np
import pytest
import wfdb

from torch_ecg.cfg import DEFAULTS
from torch_ecg.utils.utils_signal import (
    butter_bandpass_filter,
    detect_peaks,
    get_ampl,
    normalize,
    remove_spikes_naive,
    resample_irregular_timeseries,
    smooth,
)

sample_path = list((Path(__file__).parents[2] / "sample-data" / "cinc2021").resolve().rglob("*.mat"))[0]
sample_rec = wfdb.rdrecord(str(sample_path).replace(".mat", ""))


def test_smooth():
    t = np.linspace(-2, 2, 50)
    x = np.sin(t) + DEFAULTS.RNG.normal(size=(len(t),)) * 0.1
    x = x.astype("float32")
    y = smooth(x, window_len=7, keep_dtype=False)
    assert y.shape == x.shape
    assert y.dtype == np.float64
    y = smooth(x, window_len=16, window="flat")
    assert y.shape == x.shape
    assert y.dtype == np.float32

    assert (x == smooth(x, window_len=2)).all()

    with pytest.raises(ValueError, match="function `smooth` only accepts 1 dimension arrays"):
        smooth(x.reshape(-1, 1))
    with pytest.raises(
        ValueError,
        match=""" `window` should be of "flat", "hanning", "hamming", "bartlett", "blackman" """,
    ):
        smooth(x, window="not-supported")


def test_resample_irregular_timeseries():
    fs = 100
    t_irr = np.sort(DEFAULTS.RNG.uniform(size=(fs,))) * 1000
    vals = DEFAULTS.RNG.normal(size=(fs,))
    sig = np.stack([t_irr, vals], axis=1)
    sig_reg = resample_irregular_timeseries(sig, output_fs=fs * 2, verbose=2)
    assert sig_reg.ndim == 1
    sig_reg = resample_irregular_timeseries(sig, output_fs=fs, method="interp1d", return_with_time=True)
    assert sig_reg.ndim == 2
    assert np.allclose(np.diff(sig_reg[:, 0], n=2), 0)
    t_irr_2 = np.sort(DEFAULTS.RNG.uniform(size=(2 * fs,))) * 1000
    sig_reg = resample_irregular_timeseries(sig, tnew=t_irr_2, return_with_time=True)
    assert sig_reg.shape == (2 * fs, 2)

    assert resample_irregular_timeseries(DEFAULTS.RNG.normal(size=(0, 2)), output_fs=fs).shape == (0,)

    with pytest.raises(AssertionError, match="`sig` should be a 2D array"):
        resample_irregular_timeseries(sig[:, 1], output_fs=fs * 2)
    with pytest.raises(AssertionError, match="method `not-supported` not supported"):
        resample_irregular_timeseries(sig, output_fs=fs * 2, method="not-supported")
    with pytest.raises(
        AssertionError,
        match="one and only one of `output_fs` and `tnew` should be specified",
    ):
        resample_irregular_timeseries(sig, output_fs=fs * 2, tnew=t_irr_2)
    with pytest.raises(AssertionError, match="`tnew` should be a 1D array"):
        resample_irregular_timeseries(sig, tnew=t_irr_2.reshape(-1, 1))


def test_detect_peaks():
    # TODO: it's quite weird that
    # very rarely, the detected peaks are empty
    x = DEFAULTS.RNG.normal(size=(100,))
    x[60:81] = np.nan
    ind = detect_peaks(x, verbose=2)
    # assert ind.ndim == 1 and len(ind) > 0
    assert ind.ndim == 1 and "int" in str(ind.dtype)

    x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200)) + DEFAULTS.RNG.normal(size=(200,)) / 5
    # set minimum peak height = 0 and minimum peak distance = 20
    ind = detect_peaks(x, mph=0, mpd=20, verbose=2)
    # assert ind.ndim == 1 and len(ind) > 0
    assert ind.ndim == 1 and "int" in str(ind.dtype)

    x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    # set minimum peak distance = 2
    ind = detect_peaks(x, mpd=2, verbose=2)
    # assert ind.ndim == 1 and len(ind) > 0
    assert ind.ndim == 1 and "int" in str(ind.dtype)

    x = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 200)) + DEFAULTS.RNG.normal(size=(200,)) / 5
    # detection of valleys instead of peaks
    ind = detect_peaks(x, mph=-1.2, mpd=20, valley=True, verbose=2)
    assert ind.ndim == 1 and len(ind) > 0

    x = [0, 1, 3, 0, 1, -1, 0]
    # detect both edges
    ind = detect_peaks(x, edge="both", verbose=2)
    # assert ind.ndim == 1 and len(ind) > 0
    assert ind.ndim == 1 and "int" in str(ind.dtype)
    ind = detect_peaks(x, edge=None, prominence=0.4, verbose=2)
    assert ind.ndim == 1 and "int" in str(ind.dtype)

    x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    ind = detect_peaks(x, threshold=2, verbose=2)
    # assert ind.ndim == 1 and len(ind) > 0
    assert ind.ndim == 1 and "int" in str(ind.dtype)

    assert len(detect_peaks([0, 1], verbose=2)) == 0


def test_remove_spikes_naive():
    siglen = 1000
    sig = DEFAULTS.RNG.normal(size=(siglen,))
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = 100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = -100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = np.nan
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = -np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = 15
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[pos] = -15
    sig[0] = np.nan
    new_sig = remove_spikes_naive(sig, threshold=50, inplace=False)
    assert (new_sig <= 50).all() and (new_sig >= -50).all()
    assert (not (new_sig <= 10).all()) and (not (new_sig >= -10).all())
    new_sig = remove_spikes_naive(sig, threshold=12, inplace=False)
    assert (new_sig <= 12).all() and (new_sig >= -12).all()

    pos = DEFAULTS.RNG_randint(0, siglen - 1, 1)
    sig[pos] = np.nan
    sig = remove_spikes_naive(sig, inplace=True)
    assert not np.isnan(sig).any()

    sig = DEFAULTS.RNG.uniform(size=(siglen,))
    new_sig = remove_spikes_naive(sig)
    assert (sig == new_sig).all()

    # test 2D signal
    sig = DEFAULTS.RNG.normal(size=(2, siglen))
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, pos] = 100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, pos] = -100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, pos] = np.nan
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, pos] = np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, pos] = -np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, pos] = 15
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, pos] = -15
    sig[0, 0] = np.nan

    new_sig = remove_spikes_naive(sig, threshold=50, inplace=False)
    assert (new_sig <= 50).all() and (new_sig >= -50).all()
    assert (not (new_sig <= 10).all()) and (not (new_sig >= -10).all())
    new_sig = remove_spikes_naive(sig, threshold=12, inplace=False)
    assert (new_sig <= 12).all() and (new_sig >= -12).all()

    # test 3D signal
    sig = DEFAULTS.RNG.normal(size=(2, 2, siglen))
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, 0, pos] = 100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, 0, pos] = -100
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, 1, pos] = np.nan
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, 1, pos] = np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, 0, pos] = -np.inf
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[1, 0, pos] = 15
    pos = DEFAULTS.RNG_randint(0, siglen - 1, 10)
    sig[0, 1, pos] = -15
    sig[0, 1, 0] = np.nan

    new_sig = remove_spikes_naive(sig, threshold=50, inplace=False)
    assert (new_sig <= 50).all() and (new_sig >= -50).all()
    assert (not (new_sig <= 10).all()) and (not (new_sig >= -10).all())
    new_sig = remove_spikes_naive(sig, threshold=12, inplace=False)
    assert (new_sig <= 12).all() and (new_sig >= -12).all()

    with pytest.raises(AssertionError, match="Only 1D, 2D or 3D signal is supported"):
        sig = DEFAULTS.RNG.normal(size=(1, 2, 2, siglen))
        remove_spikes_naive(sig)


def test_butter_bandpass_filter():
    data = sample_rec.p_signal.T  # (n_channels, n_samples)
    fs = sample_rec.fs
    filtered_data = butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=fs, order=5, verbose=2)
    assert filtered_data.shape == data.shape
    filtered_data_1 = butter_bandpass_filter(data[0], lowcut=0.5, highcut=40, fs=fs, order=5)
    assert filtered_data_1.shape == data[0].shape
    assert np.allclose(filtered_data[0], filtered_data_1)
    filtered_data = butter_bandpass_filter(data, lowcut=0, highcut=40, fs=fs, order=5)
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=-np.inf, highcut=40, fs=fs, order=5)
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=40, highcut=np.inf, fs=fs, order=5)
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=40, highcut=fs, fs=fs, order=5)
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=fs, order=5, btype="lohi")
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=fs, order=5, btype="hilo")
    assert filtered_data.shape == data.shape
    filtered_data = butter_bandpass_filter(data, lowcut=2, highcut=2, fs=fs, order=5)
    assert filtered_data.shape == data.shape

    with pytest.raises(ValueError, match="frequency out of range!"):
        butter_bandpass_filter(data, lowcut=fs, highcut=np.inf, fs=fs, order=5)
    with pytest.raises(ValueError, match="frequency out of range!"):
        butter_bandpass_filter(data, lowcut=0, highcut=fs / 2, fs=fs, order=5)
    with pytest.raises(ValueError, match="special btype `lolo` is not supported"):
        butter_bandpass_filter(data, lowcut=0.5, highcut=40, fs=fs, order=5, btype="lolo")


def test_get_ampl():
    data = sample_rec.p_signal.T  # (n_channels, n_samples)
    fs = sample_rec.fs
    ampl = get_ampl(data, fs=fs, critical_points=[data.shape[1] // 3, data.shape[1] // 3])
    assert ampl.shape == (data.shape[0],)
    ampl = get_ampl(data.T, fs=fs, fmt="channel_last")
    assert ampl.shape == (data.shape[0],)
    ampl = get_ampl(data[0], fs=fs)
    assert isinstance(ampl, float)

    with pytest.raises(ValueError, match="unknown format `channel_first_last`"):
        get_ampl(data, fs=fs, fmt="channel_first_last")


def test_normalize():
    # 2d signal, channel_first and channel_last
    data = sample_rec.p_signal.T  # (n_channels, n_samples)
    nm_data = normalize(data, method="min-max")
    assert nm_data.shape == data.shape
    nm_data = normalize(data.T, method="naive", sig_fmt="channel_last")
    assert nm_data.shape == data.T.shape
    nm_data = normalize(data[0], method="z-score")
    assert nm_data.shape == data[0].shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        std=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        per_channel=True,
    )
    assert nm_data.shape == data.shape

    # 1d signal
    nm_data = normalize(data[0], method="min-max")
    assert nm_data.shape == data[0].shape
    nm_data = normalize(data[0], method="z-score")
    assert nm_data.shape == data[0].shape

    # 3d signal, channel_first
    data = np.expand_dims(data, axis=0)
    nm_data = normalize(data, method="min-max")
    assert nm_data.shape == data.shape
    nm_data = normalize(data, method="z-score")
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        std=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        per_channel=True,
    )
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[1],)),
        std=DEFAULTS.RNG.uniform(size=data.shape[:2]),
        per_channel=True,
    )
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=data.shape[:2]),
        std=DEFAULTS.RNG.uniform(size=(data.shape[1],)),
        per_channel=True,
    )
    assert nm_data.shape == data.shape

    # 3d signal, channel_last
    data = np.expand_dims(sample_rec.p_signal, axis=0)
    nm_data = normalize(data, method="min-max", sig_fmt="channel_last")
    assert nm_data.shape == data.shape
    nm_data = normalize(data, method="z-score", sig_fmt="channel_last")
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        std=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        per_channel=True,
        sig_fmt="channel_last",
    )
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[-1],)),
        std=DEFAULTS.RNG.uniform(size=(data.shape[0], data.shape[-1])),
        per_channel=True,
        sig_fmt="channel_last",
    )
    assert nm_data.shape == data.shape
    nm_data = normalize(
        data,
        method="z-score",
        mean=DEFAULTS.RNG.uniform(size=(data.shape[0], data.shape[-1])),
        std=DEFAULTS.RNG.uniform(size=(data.shape[-1],)),
        per_channel=True,
        sig_fmt="channel_last",
    )
    assert nm_data.shape == data.shape

    data = sample_rec.p_signal.T
    with pytest.raises(AssertionError, match="signal `sig` should be 1d or 2d or 3d array"):
        normalize(np.expand_dims(data, axis=(0, 1)), method="z-score")
    with pytest.raises(AssertionError, match="unknown normalization method `unknown`"):
        normalize(data, method="unknown")
    with pytest.raises(AssertionError, match="standard deviation should be positive"):
        normalize(data, method="z-score", std=-1)
    with pytest.raises(AssertionError, match="standard deviations should all be positive"):
        normalize(
            data,
            method="z-score",
            std=DEFAULTS.RNG.uniform(size=(data.shape[0],)) - 1,
            per_channel=True,
        )
    with pytest.raises(
        AssertionError,
        match="`mean` and `std` should be real numbers in the non per-channel setting for 2d signal",
    ):
        normalize(data, method="z-score", mean=DEFAULTS.RNG.uniform(size=(data.shape[0],)))
    with pytest.raises(
        AssertionError,
        match="`mean` and `std` should be real numbers in the non per-channel setting for 2d signal",
    ):
        normalize(data, method="z-score", std=DEFAULTS.RNG.uniform(size=(data.shape[0],)))
    with pytest.raises(
        AssertionError,
        match="`mean` and `std` should be real numbers or have shape \\(\\d+,\\) in the non per-channel setting for 3d signal",
    ):
        normalize(
            np.expand_dims(data, axis=0),
            method="z-score",
            mean=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        )
    with pytest.raises(
        AssertionError,
        match="`mean` and `std` should be real numbers or have shape \\(\\d+,\\) in the non per-channel setting for 3d signal",
    ):
        normalize(
            np.expand_dims(data, axis=0),
            method="z-score",
            std=DEFAULTS.RNG.uniform(size=(data.shape[0],)),
        )
    with pytest.raises(AssertionError, match="format `channel_first_last` of the signal not supported!"):
        normalize(data, method="z-score", sig_fmt="channel_first_last")
    with pytest.raises(AssertionError, match="shape of `mean` = .+ not compatible with the `sig` = .+"):
        normalize(
            data,
            method="z-score",
            mean=DEFAULTS.RNG.uniform(size=(data.shape[0] + 1,)),
            per_channel=True,
        )
    with pytest.raises(AssertionError, match="shape of `std` = .+ not compatible with the `sig` = .+"):
        normalize(
            data,
            method="z-score",
            std=DEFAULTS.RNG.uniform(size=(data.shape[0] + 1,)),
            per_channel=True,
        )
    with pytest.raises(AssertionError, match="shape of `mean` = .+ not compatible with the `sig` = .+"):
        normalize(
            np.expand_dims(data, axis=0),
            method="z-score",
            mean=DEFAULTS.RNG.uniform(
                size=(
                    2,
                    data.shape[0],
                )
            ),
            per_channel=True,
        )
    with pytest.raises(AssertionError, match="shape of `std` = .+ not compatible with the `sig` = .+"):
        normalize(
            np.expand_dims(data, axis=0),
            method="z-score",
            std=DEFAULTS.RNG.uniform(
                size=(
                    2,
                    data.shape[0],
                )
            ),
            per_channel=True,
        )
    with pytest.raises(AssertionError, match="shape of `mean` = .+ not compatible with the `sig` = .+"):
        normalize(
            np.expand_dims(data.T, axis=0),
            method="z-score",
            mean=DEFAULTS.RNG.uniform(
                size=(
                    2,
                    data.shape[0],
                )
            ),
            per_channel=True,
            sig_fmt="channel_last",
        )
    with pytest.raises(AssertionError, match="shape of `std` = .+ not compatible with the `sig` = .+"):
        normalize(
            np.expand_dims(data.T, axis=0),
            method="z-score",
            std=DEFAULTS.RNG.uniform(
                size=(
                    2,
                    data.shape[0],
                )
            ),
            per_channel=True,
            sig_fmt="channel_last",
        )

    with pytest.warns(RuntimeWarning, match="per-channel normalization is not supported for 1d signal"):
        normalize(data[0], method="z-score", per_channel=True)
