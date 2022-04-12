"""
auxiliary metrics for the task of qrs detection

References
----------
[1] http://2019.icbeb.org/Challenge.html
"""

import multiprocessing as mp
from numbers import Real
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.models.loss import MaskedBCEWithLogitsLoss
from torch_ecg.utils.utils_interval import mask_to_intervals

__all__ = [
    "compute_rpeak_metric",
    "compute_rr_metric",
    "compute_main_task_metric",
]


_MBCE = MaskedBCEWithLogitsLoss()


def compute_rpeak_metric(
    rpeaks_truths: Sequence[Union[np.ndarray, Sequence[int]]],
    rpeaks_preds: Sequence[Union[np.ndarray, Sequence[int]]],
    fs: Real,
    thr: float = 0.075,
    verbose: int = 0,
) -> Dict[str, float]:
    """

    Parameters
    ----------
    rpeaks_truths: sequence,
        sequence of ground truths of rpeaks locations (indices) from multiple records
    rpeaks_preds: sequence,
        predictions of ground truths of rpeaks locations (indices) for multiple records
    fs: real number,
        sampling frequency of ECG signal
    thr: float, default 0.075,
        threshold for a prediction to be truth positive,
        with units in seconds,
    verbose: int, default 0,
        print verbosity

    Returns
    -------
    rec_acc: float,
        accuracy of predictions
    """
    assert len(rpeaks_truths) == len(
        rpeaks_preds
    ), f"number of records does not match, truth indicates {len(rpeaks_truths)}, while pred indicates {len(rpeaks_preds)}"
    n_records = len(rpeaks_truths)
    record_flags = np.ones((len(rpeaks_truths),), dtype=float)
    thr_ = thr * fs
    if verbose >= 1:
        print(f"number of records = {n_records}")
        print(f"threshold in number of sample points = {thr_}")
    for idx, (truth_arr, pred_arr) in enumerate(zip(rpeaks_truths, rpeaks_preds)):
        false_negative = 0
        false_positive = 0
        true_positive = 0
        extended_truth_arr = np.concatenate((truth_arr.astype(int), [int(9.5 * fs)]))
        for j, t_ind in enumerate(extended_truth_arr[:-1]):
            next_t_ind = extended_truth_arr[j + 1]
            loc = np.where(np.abs(pred_arr - t_ind) <= thr_)[0]
            if j == 0:
                err = np.where(
                    (pred_arr >= 0.5 * fs + thr_) & (pred_arr <= t_ind - thr_)
                )[0]
            else:
                err = np.array([], dtype=int)
            err = np.append(
                err,
                np.where((pred_arr >= t_ind + thr_) & (pred_arr <= next_t_ind - thr_))[
                    0
                ],
            )

            false_positive += len(err)
            if len(loc) >= 1:
                true_positive += 1
                false_positive += len(loc) - 1
            elif len(loc) == 0:
                false_negative += 1

        if false_negative + false_positive > 1:
            record_flags[idx] = 0
        elif false_negative == 1 and false_positive == 0:
            record_flags[idx] = 0.3
        elif false_negative == 0 and false_positive == 1:
            record_flags[idx] = 0.7

        if verbose >= 2:
            print(
                f"for the {idx}-th record,\ntrue positive = {true_positive}\nfalse positive = {false_positive}\nfalse negative = {false_negative}"
            )

    rec_acc = round(np.sum(record_flags) / n_records, 4)

    if verbose >= 1:
        print(f"QRS_acc: {rec_acc}")
        print("Scoring complete.")

    metrics = {"qrs_score": rec_acc}

    return metrics


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
        af_episode_truths = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in rr_truths]
        )
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in rr_preds]
        )
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
    rr_score += sum(
        [0 == len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)]
    )
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
    fs: Real,
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
    fs: Real,
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
        af_episode_truths = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_truths]
        )
    with mp.Pool(processes=max(1, mp.cpu_count())) as pool:
        af_episode_preds = pool.starmap(
            func=mask_to_intervals, iterable=[(row, 1, True) for row in mask_preds]
        )
    af_episode_truths = [
        [[itv[0] * reduction, itv[1] * reduction] for itv in sample]
        for sample in af_episode_truths
    ]
    af_episode_preds = [
        [[itv[0] * reduction, itv[1] * reduction] for itv in sample]
        for sample in af_episode_preds
    ]
    n_samples, seq_len = np.array(mask_truths).shape
    scoring_mask = np.zeros((n_samples, seq_len * reduction))
    for idx, sample in enumerate(af_episode_truths):
        for itv in sample:
            if rpeaks is not None:
                itv_rpeaks = [
                    i for i, r in enumerate(rpeaks[idx]) if itv[0] <= r < itv[1]
                ]
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
                scoring_mask[idx][
                    max(0, itv[0] - 2 * default_rr) : min(
                        seq_len, itv[0] + 2 * default_rr + 1
                    )
                ] = 0.5
                scoring_mask[idx][
                    max(0, itv[1] - 2 * default_rr) : min(
                        seq_len, itv[1] + 2 * default_rr + 1
                    )
                ] = 0.5
                scoring_mask[idx][
                    max(0, itv[0] - 1 * default_rr) : min(
                        seq_len, itv[0] + 1 * default_rr + 1
                    )
                ] = 1
                scoring_mask[idx][
                    max(0, itv[1] - 1 * default_rr) : min(
                        seq_len, itv[1] + 1 * default_rr + 1
                    )
                ] = 1
    main_score = sum(
        [
            scoring_mask[idx][itv].sum() / max(1, len(af_episode_truths[idx]))
            for idx in range(n_samples)
            for itv in af_episode_preds[idx]
        ]
    )
    main_score += sum(
        [0 == len(t) == len(p) for t, p in zip(af_episode_truths, af_episode_preds)]
    )
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


# class WeightedBoundaryLoss(nn.Module):
#     """
#     """
#     __name__ = "WeightedBoundaryLoss"

#     def __init__(self, weight_map:Dict[int,Real], sigma:Real, w:Real) -> NoReturn:
#         """
#         """
#         self.weight_map = weight_map
#         self.sigma = sigma
#         self.w = w

#     def forward(self, input:Tensor, target:Tensor) -> Tensor:
#         """
#         """
#         _device = input.device
#         _dtype = input.dtype
#         weight_mask = torch.zeros_like(input, dtype=_dtype, device=_device)
#         if target.shape[-1] == 1:
#             w = torch.full_like(input, self.weight_map[0], dtype=_dtype, device=_device)
#             weight_mask.add_((target < 0.5)*w)
#             w = torch.full_like(input, self.weight_map[1], dtype=_dtype, device=_device)
#             weight_mask.add_((target > 0.5)*w)
#         else:
#             for i in range(input.shape[-1]):
#                 w = torch.full(input.shape[:-1], self.weight_map[i], dtype=_dtype, device=_device)
#                 weight_mask[...,i].add_((target[...,i] > 0.5)*w)
