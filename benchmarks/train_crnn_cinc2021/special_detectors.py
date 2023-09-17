"""
special detectors using rules,
for (perhaps auxiliarily) detecting PR, Brady (including SB), LQRSV, RAD, LAD, STach

pending arrhythmia classes: LPR, LQT

NOTE:
-----
1. ALL signals are assumed to have units in mV
2. almost all the rules can be found in `utils.ecg_arrhythmia_knowledge`
3. "PR" is superior to electrical axis deviation, which should be considered in the final decision.
the co-occurrence of "PR" and "LAD" is 7; the co-occurrence of "PR" and "RAD" is 3, whose probabilities are both relatively low

TODO:
-----
currently all are binary detectors, --> detectors producing a probability?
"""

from itertools import repeat
from numbers import Real
from typing import Any, Optional, Sequence

import numpy as np
from biosppy.signals.tools import filter_signal
from deprecated import deprecated
from scipy.signal import peak_prominences

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from cfg import SpecialDetectorCfg

from torch_ecg.cfg import CFG
from torch_ecg.utils._preproc import preprocess_multi_lead_signal
from torch_ecg.utils.ecg_arrhythmia_knowledge import LimbLeads, PrecordialLeads, Standard12Leads
from torch_ecg.utils.misc import ms2samples, samples2ms
from torch_ecg.utils.utils_data import get_mask
from torch_ecg.utils.utils_signal import detect_peaks, get_ampl

__all__ = [
    "special_detectors",
    "pacing_rhythm_detector",
    "electrical_axis_detector",
    "brady_tachy_detector",
    "LQRSV_detector",
    "PRWP_detector",
]


def special_detectors(
    raw_sig: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    verbose: int = 0,
    **kwargs: Any,
) -> dict:
    """

    Parameters
    ----------
    raw_sig: ndarray,
        the raw multi-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    verbose: int, default 0,
        print verbosity
    kwargs: dict,
        keyword arguments, including:
        "rpeak_fn": rpeak detection method, can be one of
           "seq_lab", "xqrs", "gqrs", "hamilton", "ssf", "christov", "engzee", gamboa"
        the default method is "xqrs",
        which has less environment issues compared to the deep learning method "seq_lab"
        "axis_method": electrical axis detection method, can be one of
            "2-lead", "3-lead"
        the default method is "2-lead"

    Returns
    -------
    conclusion: dict,
        probability or binary conclusion for each arrhythm
    """
    preprocess = preprocess_multi_lead_signal(
        raw_sig,
        fs,
        sig_fmt,
        rpeak_fn=kwargs.get("rpeak_fn", "xqrs"),
        # rpeak_fn=kwargs.get("rpeak_fn", "seq_lab"),
        verbose=verbose,
    )
    filtered_sig = preprocess["filtered_ecg"]
    rpeaks = preprocess["rpeaks"]
    is_PR = pacing_rhythm_detector(raw_sig, fs, sig_fmt, leads, ret_prob=False, verbose=verbose)
    axis = electrical_axis_detector(
        filtered_sig,
        rpeaks,
        fs,
        sig_fmt,
        leads,
        method=kwargs.get("axis_method", "2-lead"),
        verbose=verbose,
    )
    brady_tachy = brady_tachy_detector(rpeaks, fs, verbose=verbose)
    is_LQRSV = LQRSV_detector(filtered_sig, rpeaks, fs, sig_fmt, leads, verbose=verbose)
    is_PRWP = PRWP_detector(filtered_sig, rpeaks, fs, sig_fmt, leads, verbose=verbose)

    is_LAD = axis == "LAD"
    is_RAD = axis == "RAD"
    is_brady = brady_tachy == "B"
    is_tachy = brady_tachy == "T"
    conclusion = CFG(
        is_brady=is_brady,
        is_tachy=is_tachy,
        is_LAD=is_LAD,
        is_RAD=is_RAD,
        is_PR=is_PR,
        is_LQRSV=is_LQRSV,
        is_PRWP=is_PRWP,
    )
    return conclusion


def pacing_rhythm_detector(
    raw_sig: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    ret_prob: bool = True,
    verbose: int = 0,
) -> Real:
    """to be improved (fine-tuning hyper-parameters in cfg.py),

    Parameters
    ----------
    raw_sig: ndarray,
        the raw multi-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    ret_prob: bool, default True,
        if True, a probability will be returned,
        otherwise, a binary prediction will be returned
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    is_PR: real number,
        probability for the ecg signal to be of pacing rhythm,
        or a binary disicion
    """
    if sig_fmt.lower() in ["channel_first", "lead_first"]:
        s = raw_sig.copy()
    else:
        s = raw_sig.T

    data_hp = np.array(
        [
            filter_signal(
                s[lead, ...],
                ftype="butter",
                band="highpass",
                order=20,
                frequency=SpecialDetectorCfg.pr_fs_lower_bound,
                sampling_rate=fs,
            )["signal"]
            for lead in range(s.shape[0])
        ]
    )

    potential_spikes = []
    # sig_len = data_hp.shape[-1]
    n_leads, sig_len = data_hp.shape
    assert n_leads == len(leads)

    for ld in range(n_leads):
        lead_hp = np.abs(data_hp[ld, ...])
        mph = SpecialDetectorCfg.pr_spike_mph_ratio * np.sum(lead_hp) / sig_len
        lead_spikes = detect_peaks(
            x=lead_hp,
            mph=mph,
            mpd=ms2samples(SpecialDetectorCfg.pr_spike_mpd, fs),
            prominence=SpecialDetectorCfg.pr_spike_prominence,
            prominence_wlen=ms2samples(SpecialDetectorCfg.pr_spike_prominence_wlen, fs),
            verbose=0,
        )
        if verbose >= 2:
            print(f"for the {ld}-th lead, its spike detecting mph = {mph:.4f} mV")
            print(f"lead_spikes = {lead_spikes.tolist()}")
            print(
                f"with prominences = {np.round(peak_prominences(lead_hp, lead_spikes, wlen=ms2samples(SpecialDetectorCfg.pr_spike_prominence_wlen, fs))[0], 5).tolist()}"
            )
        potential_spikes.append(lead_spikes)

    # make decision using `potential_spikes`
    sig_duration_ms = samples2ms(sig_len, fs)
    # lead_has_enough_spikes = [False if len(potential_spikes[ld]) ==0 else sig_duration_ms / len(potential_spikes[ld]) < SpecialDetectorCfg.pr_spike_inv_density_threshold for ld in range(n_leads)]
    lead_has_enough_spikes = list(repeat(0, n_leads))
    for ld in range(n_leads):
        if len(potential_spikes[ld]) > 0:
            relative_inv_density = SpecialDetectorCfg.pr_spike_inv_density_threshold - sig_duration_ms / len(
                potential_spikes[ld]
            )
            # sigmoid
            lead_has_enough_spikes[ld] = 1 / (1 + np.exp(-relative_inv_density / 100))
            if not ret_prob:
                lead_has_enough_spikes[ld] = int(lead_has_enough_spikes[ld] >= 0.5)
    if verbose >= 1:
        print(f"lead_has_enough_spikes = {lead_has_enough_spikes}")
        print(f"leads spikes density (units in ms) = {[len(potential_spikes[ld]) / sig_duration_ms for ld in range(n_leads)]}")

    _threshold = int(round(SpecialDetectorCfg.pr_spike_leads_threshold * n_leads))
    if ret_prob:
        # pooling (max, or avg)
        is_PR = sorted(lead_has_enough_spikes, reverse=True)[:_threshold]
        is_PR = np.mean(is_PR)
    else:
        is_PR = sum(lead_has_enough_spikes) >= _threshold
    return is_PR


def electrical_axis_detector(
    filtered_sig: np.ndarray,
    rpeaks: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    method: Optional[str] = None,
    verbose: int = 0,
) -> str:
    """to be improved (fine-tuning hyper-parameters in cfg.py),

    detector of the heart electrical axis by means of "2-lead" method or "3-lead" method,
    NOTE that the extreme axis is not checked and treated as "normal"

    Parameters
    ----------
    filtered_sig: ndarray,
        the filtered multi-lead ecg signal, with units in mV
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    method: str, optional,
        method for detecting electrical axis, can be "2-lead", "3-lead",
        if not specified, `SpecialDetectorCfg.axis_method` will be used
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    axis: str,
        one of "normal", "LAD", "RAD",
        the heart electrical axis
    """
    decision_method = method or SpecialDetectorCfg.axis_method
    decision_method = decision_method.lower()
    assert decision_method in [
        "2-lead",
        "3-lead",
    ], f"Method `{decision_method}` not supported!"

    if sig_fmt.lower() in ["channel_first", "lead_first"]:
        s = filtered_sig.copy()
    else:
        s = filtered_sig.T

    if len(set(["I", "aVF"]).intersection(leads)) < 2:
        # impossible to make decision
        # return "normal" by default
        axis = "normal"
        return axis

    lead_I = s[list(leads).index("I")]
    lead_aVF = s[list(leads).index("aVF")]
    try:
        lead_II = s[list(leads).index("II")]
    except Exception:
        # no lead II, degenerates to the "2-lead" method
        method = "2-lead"

    if len(rpeaks == 0):
        # degenerate case
        # voltage might be too low to detect rpeaks
        lead_I_positive = np.max(lead_I) > np.abs(np.min(lead_I))
        lead_II_positive = np.max(lead_II) > np.abs(np.min(lead_II))
        lead_aVF_positive = np.max(lead_aVF) > np.abs(np.min(lead_aVF))
        # decision making
        if decision_method == "2-lead":
            if lead_I_positive and not lead_aVF_positive:
                axis = "LAD"
            elif not lead_I_positive and lead_aVF_positive:
                axis = "RAD"
            else:  # if `rpeaks` is empty, all conditions are False
                axis = "normal"  # might also include extreme axis
        elif decision_method == "3-lead":
            if lead_I_positive and not lead_II_positive and not lead_aVF_positive:
                axis = "LAD"
            elif not lead_I_positive and lead_aVF_positive:
                axis = "RAD"
            else:
                axis = "normal"  # might also include extreme axis
        return axis

    sig_len = s.shape[1]
    radius = ms2samples(SpecialDetectorCfg.axis_qrs_mask_radius, fs)
    l_qrs = []
    for r in rpeaks:
        l_qrs.append([max(0, r - radius), min(sig_len - 1, r + radius)])

    if verbose >= 1:
        print(f"qrs mask radius = {radius}, sig_len = {sig_len}")
        print(f"l_qrs = {l_qrs}")

    # lead I
    lead_I_positive = (
        sum([np.max(lead_I[qrs_itv[0] : qrs_itv[1]]) > np.abs(np.min(lead_I[qrs_itv[0] : qrs_itv[1]])) for qrs_itv in l_qrs])
        >= len(l_qrs) // 2 + 1
    )

    # lead aVF
    lead_aVF_positive = (
        sum(
            [np.max(lead_aVF[qrs_itv[0] : qrs_itv[1]]) > np.abs(np.min(lead_aVF[qrs_itv[0] : qrs_itv[1]])) for qrs_itv in l_qrs]
        )
        >= len(l_qrs) // 2 + 1
    )

    # lead II
    lead_II_positive = (
        sum([np.max(lead_II[qrs_itv[0] : qrs_itv[1]]) > np.abs(np.min(lead_II[qrs_itv[0] : qrs_itv[1]])) for qrs_itv in l_qrs])
        >= len(l_qrs) // 2 + 1
    )

    # decision making
    if decision_method == "2-lead":
        if lead_I_positive and not lead_aVF_positive:
            axis = "LAD"
        elif not lead_I_positive and lead_aVF_positive:
            axis = "RAD"
        else:  # if `rpeaks` is empty, all conditions are False
            axis = "normal"  # might also include extreme axis
    elif decision_method == "3-lead":
        if lead_I_positive and not lead_II_positive and not lead_aVF_positive:
            axis = "LAD"
        elif not lead_I_positive and lead_aVF_positive:
            axis = "RAD"
        else:
            axis = "normal"  # might also include extreme axis

    return axis


def brady_tachy_detector(
    rpeaks: np.ndarray,
    fs: Real,
    normal_rr_range: Optional[Sequence[Real]] = None,
    verbose: int = 0,
) -> str:
    """to be improved (fine-tuning hyper-parameters in cfg.py),

    detemine if the ecg is bradycadia or tachycardia or normal,
    only by the mean rr interval.

    this detector can be used alone (e.g. for the arrhythmia `Brady`),
    or combined with other detectors (e.g. for the arrhythmia `STach`)

    Parameters
    ----------
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    normal_rr_range: sequence of int, optional,
        the range of normal rr interval, with units in ms;
        if not given, default values from `SpecialDetectorCfg` will be used
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    conclusion: str,
        one of "T" (tachycardia), "B" (bradycardia), "N" (normal)
    """
    if len(rpeaks) <= 1:
        # unable to make predictions
        # TODO: try using spectral method
        conclusion = "N"
        return conclusion
    rr_intervals = np.diff(rpeaks)
    mean_rr = np.mean(rr_intervals)
    if verbose >= 1:
        if len(rr_intervals) > 0:
            print(
                f"mean_rr = {round(samples2ms(mean_rr, fs), 1)} ms, with detailed rr_intervals (with units in ms) = {(np.vectorize(lambda item:samples2ms(item, fs))(rr_intervals)).tolist()}"
            )
        else:
            print("not enough r peaks for computing rr intervals")
    nrr = normal_rr_range or [
        SpecialDetectorCfg.tachy_threshold,
        SpecialDetectorCfg.brady_threshold,
    ]
    nrr = sorted(nrr)
    assert len(nrr) >= 2
    nrr = [ms2samples(nrr[0], fs), ms2samples(nrr[-1], fs)]
    # if mean_rr is nan, then all conditions are False, hence the `else` branch is entered
    if mean_rr < nrr[0]:
        conclusion = "T"
    elif mean_rr > nrr[1]:
        conclusion = "B"
    else:
        conclusion = "N"
    return conclusion


def LQRSV_detector(
    filtered_sig: np.ndarray,
    rpeaks: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    verbose: int = 0,
) -> bool:
    """to be improved (fine-tuning hyper-parameters in cfg.py),

    Parameters
    ----------
    filtered_sig: ndarray,
        the filtered multi-lead ecg signal, with units in mV
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    is_LQRSV: bool,
        the ecg signal is of arrhythmia `LQRSV` or not
    """
    sig_ampl = get_ampl(
        sig=filtered_sig,
        fs=fs,
        fmt=sig_fmt,
        window=2 * SpecialDetectorCfg.lqrsv_qrs_mask_radius / 1000,  # ms to s
        critical_points=rpeaks,
    )

    limb_leads = [ld for ld in leads if ld in LimbLeads]
    limb_lead_inds = [list(leads).index(ld) for ld in limb_leads]
    precordial_leads = [ld for ld in leads if ld in PrecordialLeads]
    precordial_lead_inds = [list(leads).index(ld) for ld in precordial_leads]

    if verbose >= 1:
        print(f"limb_lead_inds = {limb_lead_inds}, precordial_lead_inds = {precordial_lead_inds}")

    low_qrs_limb_leads = [sig_ampl[idx] <= 0.5 + SpecialDetectorCfg.lqrsv_ampl_bias for idx in limb_lead_inds]
    if len(low_qrs_limb_leads) > 0:
        low_qrs_limb_leads = sum(low_qrs_limb_leads) / len(low_qrs_limb_leads)  # to ratio
    else:  # no limb leads
        # determining LQRSV using limb leads and precordial leads, its relation is OR
        # hence default values are set 0 if no limb leads or precordial leads
        low_qrs_limb_leads = 0
    low_qrs_precordial_leads = [sig_ampl[idx] <= 1 + SpecialDetectorCfg.lqrsv_ampl_bias for idx in precordial_lead_inds]
    if len(low_qrs_precordial_leads) > 0:
        low_qrs_precordial_leads = sum(low_qrs_precordial_leads) / len(low_qrs_precordial_leads)
    else:
        low_qrs_precordial_leads = 0

    if verbose >= 2:
        print(f"ratio of low qrs in limb leads = {low_qrs_limb_leads}")
        print(f"ratio of low qrs in precordial leads = {low_qrs_precordial_leads}")

    is_LQRSV = (low_qrs_limb_leads >= SpecialDetectorCfg.lqrsv_ratio_threshold) or (
        low_qrs_precordial_leads >= SpecialDetectorCfg.lqrsv_ratio_threshold
    )

    return is_LQRSV


@deprecated
def LQRSV_detector_backup(
    filtered_sig: np.ndarray,
    rpeaks: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    verbose: int = 0,
) -> bool:
    """to be improved (fine-tuning hyper-parameters in cfg.py),

    Parameters
    ----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal, with units in mV
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    is_LQRSV: bool,
        the ecg signal is of arrhythmia `LQRSV` or not
    """
    if sig_fmt.lower() in ["channel_first", "lead_first"]:
        sig_ampl = filtered_sig.copy()
    else:
        sig_ampl = filtered_sig.T
    qrs_mask_radius = ms2samples(SpecialDetectorCfg.lqrsv_qrs_mask_radius, fs)
    l_qrs = get_mask(
        shape=sig_ampl.shape,
        critical_points=rpeaks,
        left_bias=qrs_mask_radius,
        right_bias=qrs_mask_radius,
        return_fmt="intervals",
    )
    if verbose >= 2:
        print(f"qrs intervals = {l_qrs}")

    limb_leads = [ld for ld in leads if ld in LimbLeads]
    limb_lead_inds = [list(leads).index(ld) for ld in limb_leads]
    precordial_leads = [ld for ld in leads if ld in PrecordialLeads]
    precordial_lead_inds = [list(leads).index(ld) for ld in precordial_leads]

    l_qrs_limb_leads = []
    l_qrs_precordial_leads = []

    if len(l_qrs) == 0:
        # no rpeaks detected
        low_qrs_limb_leads = [np.max(sig_ampl[idx]) <= 0.5 + SpecialDetectorCfg.lqrsv_ampl_bias for idx in limb_lead_inds]
        low_qrs_limb_leads = sum(low_qrs_limb_leads) / len(low_qrs_limb_leads)  # to ratio
        low_qrs_precordial_leads = [
            np.max(sig_ampl[idx]) <= 1 + SpecialDetectorCfg.lqrsv_ampl_bias for idx in precordial_lead_inds
        ]
        low_qrs_precordial_leads = sum(low_qrs_precordial_leads) / len(low_qrs_precordial_leads)
    else:
        for itv in l_qrs:
            for idx in limb_lead_inds:
                l_qrs_limb_leads.append(sig_ampl[idx, itv[0] : itv[1]].flatten())
            for idx in precordial_lead_inds:
                l_qrs_precordial_leads.append(sig_ampl[idx, itv[0] : itv[1]].flatten())

        if verbose >= 2:
            print("for limb leads, the qrs amplitudes are as follows:")
            for idx, lead_name in enumerate(limb_leads):
                print(
                    f"for limb lead {lead_name}, the qrs amplitudes are {[np.max(item) for item in l_qrs_limb_leads[idx*len(l_qrs): (idx+1)*len(l_qrs)]]}"
                )
            for idx, lead_name in enumerate(precordial_leads):
                print(
                    f"for precordial lead {lead_name}, the qrs amplitudes are {[np.max(item) for item in l_qrs_limb_leads[idx*len(l_qrs): (idx+1)*len(l_qrs)]]}"
                )

        low_qrs_limb_leads = [np.max(item) <= 0.5 + SpecialDetectorCfg.lqrsv_ampl_bias for item in l_qrs_limb_leads]
        low_qrs_limb_leads = sum(low_qrs_limb_leads) / len(low_qrs_limb_leads)  # to ratio
        low_qrs_precordial_leads = [np.max(item) <= 1 + SpecialDetectorCfg.lqrsv_ampl_bias for item in l_qrs_precordial_leads]
        low_qrs_precordial_leads = sum(low_qrs_precordial_leads) / len(low_qrs_precordial_leads)

    if verbose >= 2:
        print(f"ratio of low qrs in limb leads = {low_qrs_limb_leads}")
        print(f"ratio of low qrs in precordial leads = {low_qrs_precordial_leads}")

    is_LQRSV = (low_qrs_limb_leads >= SpecialDetectorCfg.lqrsv_ratio_threshold) or (
        low_qrs_precordial_leads >= SpecialDetectorCfg.lqrsv_ratio_threshold
    )

    return is_LQRSV


def PRWP_detector(
    filtered_sig: np.ndarray,
    rpeaks: np.ndarray,
    fs: Real,
    sig_fmt: str = "channel_first",
    leads: Sequence[str] = Standard12Leads,
    verbose: int = 0,
) -> bool:
    """to be improved

    Parameters
    ----------
    filtered_sig: ndarray,
        the filtered multi-lead ecg signal, with units in mV
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        "channel_last" (alias "lead_last"), or
        "channel_first" (alias "lead_first", original)
    leads: sequence of str,
        names of the leads in the input signal
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    is_PRWP: bool,
        the ecg signal is of arrhythmia `PRWP` or not
    """
    if sig_fmt.lower() in ["channel_first", "lead_first"]:
        r_ampl = filtered_sig[..., rpeaks]
    else:
        # all change to lead_first
        r_ampl = filtered_sig[rpeaks, ...].T

    if len(set([f"V{n}" for n in range(1, 5)]).intersection(leads)) < 2 and "V3" not in leads:
        # leads insufficient to make decision
        is_PRWP = False
        return is_PRWP

    limb_leads = [ld for ld in leads if ld in LimbLeads]
    limb_lead_inds = [list(leads).index(ld) for ld in limb_leads]

    try:
        lead_V3_ind = list(leads).index("V3")
    except Exception:
        lead_V3_ind = None

    leads_V1_4 = [ld for ld in leads if ld in ["V1", "V2", "V3", "V4"]]
    leads_V1_4_inds = [list(leads).index(ld) for ld in leads_V1_4]

    # condition 1: R<3mm in V3
    if lead_V3_ind is not None:
        cond1 = np.mean(r_ampl[lead_V3_ind, ...]) < SpecialDetectorCfg.prwp_v3_thr
        if verbose >= 1:
            print(f"PRWP condition 1: R amplitude in lead V3 = {np.mean(r_ampl[lead_V3_ind, ...])}")
    else:
        cond1 = False

    # condition 2: reversed R wave progression, which is defined as R in V4 < R in V3 or R in V3 < R in V2 or R in V2 < R in V1
    cond2 = (np.diff(np.mean(r_ampl[leads_V1_4_inds, ...], axis=-1)) < 0).any()
    if verbose >= 1:
        diff = np.diff(np.mean(r_ampl[leads_V1_4_inds, ...], axis=-1))
        print(f"PRWP condition 2: reversed R wave progression, diff of mean R amplitude in V1-4 = {diff}")

    # condition 3: delayed transition beyond V4
    # currently, exact meaning of condition 3 is not clear
    cond3 = False

    # the or rule
    is_PRWP = cond1 or cond2 or cond3

    return is_PRWP
