"""
Auxiliary metrics for the CPSC2021 benchmark.

`compute_rr_metric` and `compute_main_task_metric` imitate the challenge
metric of CPSC2021 for the rr_lstm and the main (seq_lab / unet) tasks
respectively. The official challenge score itself lives in the library at
``torch_ecg.databases.cpsc_databases.cpsc2021`` (``compute_metrics``), and
the R-peak (QRS detection) score is identical to the CPSC2019 challenge
metric, also available from the library.
"""

import multiprocessing as mp
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from torch_ecg.models.loss import MaskedBCEWithLogitsLoss
from torch_ecg.utils.utils_data import mask_to_intervals

__all__ = [
    "compute_rr_metric",
    "compute_main_task_metric",
]


_MBCE = MaskedBCEWithLogitsLoss()


def compute_rr_metric(
    rr_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rr_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    weight_masks: Optional[Sequence[Union[np.ndarray, Sequence[int]]]] = None,
    verbose: int = 0,
) -> Dict[str, float]:
    """

    this metric for evaluating the RR_LSTM model,
    which imitates the metric provided by the organizers of CPSC2021

    Parameters
    ----------
    rr_truths: array_like,
        sequences of AF labels on rr intervals, of shape (n_samples, seq_len)
    rr_preds: array_like,
        sequences of AF predictions on rr intervals, of shape (n_samples, seq_len)

    Returns
    -------
    rr_score: float,
        the score computed from predicts from rr sequences,
        similar to CPSC2021 challenge metric
    neg_masked_bce: float,
        negative masked BCE loss
    """
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_truths = pool.starmap(func=mask_to_intervals, iterable=[(row, 1, True) for row in rr_truths])
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(func=mask_to_intervals, iterable=[(row, 1, True) for row in rr_preds])
    scoring_mask = np.zeros_like(np.array(rr_truths))
    n_samples, seq_len = scoring_mask.shape
    for idx, sample in enumerate(af_episode_truths):
        for itv in sample:
            scoring_mask[idx][max(0, itv[0] - 2) : min(seq_len, itv[0] + 3)] = 0.5
            scoring_mask[idx][max(0, itv[1] - 2) : min(seq_len, itv[1] + 3)] = 0.5
            scoring_mask[idx][max(0, itv[0] - 1) : min(seq_len, itv[0] + 2)] = 1
            scoring_mask[idx][max(0, itv[1] - 1) : min(seq_len, itv[1] + 2)] = 1
    rr_score = sum(
        [
            scoring_mask[idx][itv].sum() / max(1, len(af_episode_truths[idx]))
            for idx in range(n_samples)
            for itv in af_episode_preds[idx]
        ]
    )
    rr_score += sum([0 == len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)])
    neg_masked_bce = -_MBCE(
        torch.as_tensor(rr_preds, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(rr_truths, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(weight_masks, dtype=torch.float32, device=torch.device("cpu")),
    ).item()
    metrics = {
        "rr_score": rr_score,
        "neg_masked_bce": neg_masked_bce,
    }
    return metrics


def compute_main_task_metric(
    mask_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    mask_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: int,
    reduction: int,
    weight_masks: Optional[Sequence[Union[np.ndarray, Sequence[int]]]] = None,
    rpeaks: Optional[Sequence[Sequence[int]]] = None,
    verbose: int = 0,
) -> Dict[str, float]:
    """

    this metric for evaluating the main task model (seq_lab or unet),
    which imitates the metric provided by the organizers of CPSC2021

    Parameters
    ----------
    mask_truths: array_like,
        sequences of AF labels on rr intervals, of shape (n_samples, seq_len)
    mask_preds: array_like,
        sequences of AF predictions on rr intervals, of shape (n_samples, seq_len)
    fs: int,
        sampling frequency of the model input ECGs,
        used when (indices of) `rpeaks` not privided
    reduction: int,
        reduction ratio of the main task model
    rpeaks: array_like, optional,
        indices of rpeaks in the model input ECGs,
        if set, more precise scores can be computed

    Returns
    -------
    main_score: float,
        the score computed from predicts from the main task model,
        similar to CPSC2021 challenge metric
    neg_masked_bce: float,
        negative masked BCE loss
    """
    default_rr = int(fs * 0.8 / reduction)
    if rpeaks is not None:
        assert len(rpeaks) == len(mask_truths)
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_truths = pool.starmap(func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_truths])
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_preds])
    af_episode_truths = [[[itv[0] * reduction, itv[1] * reduction] for itv in sample] for sample in af_episode_truths]
    af_episode_preds = [[[itv[0] * reduction, itv[1] * reduction] for itv in sample] for sample in af_episode_preds]
    n_samples, seq_len = np.array(mask_truths).shape
    scoring_mask = np.zeros((n_samples, seq_len * reduction))
    for idx, sample in enumerate(af_episode_truths):
        for itv in sample:
            if rpeaks is not None:
                itv_rpeaks = [i for i, r in enumerate(rpeaks[idx]) if itv[0] <= r < itv[1]]
                start = rpeaks[idx][max(0, itv_rpeaks[0] - 2)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[0] + 2)] + 1
                scoring_mask[idx][start:end] = 0.5
                start = rpeaks[idx][max(0, itv_rpeaks[-1] - 2)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[-1] + 2)] + 1
                scoring_mask[idx][start:end] = 0.5
                start = rpeaks[idx][max(0, itv_rpeaks[0] - 1)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[0] + 1)] + 1
                scoring_mask[idx][start:end] = 1
                start = rpeaks[idx][max(0, itv_rpeaks[-1] - 1)]
                end = rpeaks[idx][min(len(rpeaks[idx]) - 1, itv_rpeaks[-1] + 1)] + 1
                scoring_mask[idx][start:end] = 1
            else:
                scoring_mask[idx][max(0, itv[0] - 2 * default_rr) : min(seq_len, itv[0] + 2 * default_rr + 1)] = 0.5
                scoring_mask[idx][max(0, itv[1] - 2 * default_rr) : min(seq_len, itv[1] + 2 * default_rr + 1)] = 0.5
                scoring_mask[idx][max(0, itv[0] - 1 * default_rr) : min(seq_len, itv[0] + 1 * default_rr + 1)] = 1
                scoring_mask[idx][max(0, itv[1] - 1 * default_rr) : min(seq_len, itv[1] + 1 * default_rr + 1)] = 1
    main_score = sum(
        [
            scoring_mask[idx][itv].sum() / max(1, len(af_episode_truths[idx]))
            for idx in range(n_samples)
            for itv in af_episode_preds[idx]
        ]
    )
    main_score += sum([0 == len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)])
    neg_masked_bce = -_MBCE(
        torch.as_tensor(mask_preds, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(mask_truths, dtype=torch.float32, device=torch.device("cpu")),
        torch.as_tensor(weight_masks, dtype=torch.float32, device=torch.device("cpu")),
    ).item()
    metrics = {
        "main_score": main_score,
        "neg_masked_bce": neg_masked_bce,
    }
    return metrics
