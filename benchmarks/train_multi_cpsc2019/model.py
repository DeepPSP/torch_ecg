"""
"""

from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import biosppy.signals.ecg as BSE
import numpy as np
import torch
from torch import Tensor

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from cfg import ModelCfg

from torch_ecg.cfg import CFG
from torch_ecg.components.outputs import RPeaksDetectionOutput
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET  # _ECG_SEQ_LAB_NET,
from torch_ecg.models.unets.ecg_subtract_unet import ECG_SUBTRACT_UNET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_data import mask_to_intervals

__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2019",
]


# class ECG_SEQ_LAB_NET_CPSC2019(_ECG_SEQ_LAB_NET):
class ECG_SEQ_LAB_NET_CPSC2019(ECG_SEQ_LAB_NET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2019"

    def __init__(self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        model_config = deepcopy(ModelCfg.seq_lab_crnn)
        model_config.update(deepcopy(config) or {})
        # print(f"model_config = {model_config}")
        super().__init__(model_config.classes, n_leads, model_config, **kwargs)

    @torch.no_grad()
    def inference(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> RPeaksDetectionOutput:
        """
        auxiliary function to `forward`, for CPSC2019,

        NOTE: each segment of input be better filtered using `remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).
        correction: bool, default False,
            if True, correct rpeaks to local maximum in a small nbh
            of rpeaks detected by DL model using `BSE.correct_rpeaks`

        Returns
        -------
        output: RPeaksDetectionOutput, with items:
            - rpeak_indices: list of ndarray,
                list of ndarray of rpeak indices for each batch element
            - prob: array_like,
                the probability array of the input sequence of signals

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        if prob.shape[1] != _input.shape[-1]:
            prob = self._recover_length(prob, _input.shape[-1])
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _inference_post_process(
            prob=prob,
            fs=self.config.fs,
            skip_dist=self.config.skip_dist,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr,
        )

        if correction:
            rpeaks = [
                BSE.correct_rpeaks(
                    signal=b_input,
                    rpeaks=b_rpeaks,
                    sampling_rate=self.config.fs,
                    tol=0.05,
                )[0]
                for b_input, b_rpeaks in zip(_input.detach().numpy().squeeze(1), rpeaks)
            ]

        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks,
            prob=prob,
        )

    @add_docstring(inference.__doc__)
    def inference_CPSC2019(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)


class ECG_SUBTRACT_UNET_CPSC2019(ECG_SUBTRACT_UNET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_SUBTRACT_UNET_CPSC2019"

    def __init__(self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        model_config = deepcopy(ModelCfg.subtract_unet)
        model_config.update(deepcopy(config) or {})
        super().__init__(model_config.classes, n_leads, model_config, **kwargs)

    @torch.no_grad()
    def inference(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> RPeaksDetectionOutput:
        """
        auxiliary function to `forward`, for CPSC2019,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).
        correction: bool, default False,
            if True, correct rpeaks to local maximum in a small nbh
            of rpeaks detected by DL model using `BSE.correct_rpeaks`

        Returns
        -------
        output: RPeaksDetectionOutput, with items:
            - rpeak_indices: list of ndarray,
                list of ndarray of rpeak indices for each batch element
            - prob: array_like,
                the probability array of the input sequence of signals

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _inference_post_process(
            prob=prob,
            fs=self.config.fs,
            skip_dist=self.config.skip_dist,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr,
        )

        if correction:
            rpeaks = [
                BSE.correct_rpeaks(
                    signal=b_input,
                    rpeaks=b_rpeaks,
                    sampling_rate=self.config.fs,
                    tol=0.05,
                )[0]
                for b_input, b_rpeaks in zip(_input.detach().numpy().squeeze(1), rpeaks)
            ]

        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks,
            prob=prob,
        )

    @add_docstring(inference.__doc__)
    def inference_CPSC2019(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> RPeaksDetectionOutput:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)


class ECG_UNET_CPSC2019(ECG_UNET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_UNET_CPSC2019"

    def __init__(self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        model_config = deepcopy(ModelCfg.unet)
        model_config.update(deepcopy(config) or {})
        super().__init__(model_config.classes, n_leads, model_config)

    @torch.no_grad()
    def inference(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> RPeaksDetectionOutput:
        """
        auxiliary function to `forward`, for CPSC2019,

        NOTE: each segment of input be better filtered using `_remove_spikes_naive`,
        and normalized to a suitable mean and std

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        duration_thr: int, default 4*16,
            minimum duration for a "true" qrs complex, units in ms
        dist_thr: int or sequence of int, default 200,
            if is sequence of int,
            (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
            (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
            e.g. [200, 1200]
            if is int, then is the case of (0-th element).
        correction: bool, default False,
            if True, correct rpeaks to local maximum in a small nbh
            of rpeaks detected by DL model using `BSE.correct_rpeaks`

        Returns
        -------
        output: RPeaksDetectionOutput, with items:
            - rpeak_indices: list of ndarray,
                list of ndarray of rpeak indices for each batch element
            - prob: array_like,
                the probability array of the input sequence of signals

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _inference_post_process(
            prob=prob,
            fs=self.config.fs,
            skip_dist=self.config.skip_dist,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr,
        )

        if correction:
            rpeaks = [
                BSE.correct_rpeaks(
                    signal=b_input,
                    rpeaks=b_rpeaks,
                    sampling_rate=self.config.fs,
                    tol=0.05,
                )[0]
                for b_input, b_rpeaks in zip(_input.detach().numpy().squeeze(1), rpeaks)
            ]

        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks,
            prob=prob,
        )

    @add_docstring(inference.__doc__)
    def inference_CPSC2019(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
        duration_thr: int = 4 * 16,
        dist_thr: Union[int, Sequence[int]] = 200,
        correction: bool = False,
    ) -> RPeaksDetectionOutput:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)


def _inference_post_process(
    prob: np.ndarray,
    fs: int,
    skip_dist: int,
    bin_pred_thr: float = 0.5,
    duration_thr: int = 4 * 16,
    dist_thr: Union[int, Sequence[int]] = 200,
) -> List[np.ndarray]:
    """
    prob --> qrs mask --> qrs intervals --> rpeaks

    Parameters: ref. `inference` method of the models
    """
    batch_size, prob_arr_len = prob.shape
    input_len = prob_arr_len
    model_spacing = 1000 / fs  # units in ms
    _duration_thr = duration_thr / model_spacing
    _dist_thr = [dist_thr] if isinstance(dist_thr, int) else dist_thr
    assert len(_dist_thr) <= 2

    # mask = (prob > bin_pred_thr).astype(int)
    rpeaks = []
    for b_idx in range(batch_size):
        b_prob = prob[b_idx, ...]
        b_mask = (b_prob > bin_pred_thr).astype(int)
        b_qrs_intervals = mask_to_intervals(b_mask, 1)
        b_rpeaks = np.array([(itv[0] + itv[1]) // 2 for itv in b_qrs_intervals if itv[1] - itv[0] >= _duration_thr])
        # print(f"before post-process, b_qrs_intervals = {b_qrs_intervals}")
        # print(f"before post-process, b_rpeaks = {b_rpeaks}")

        check = True
        dist_thr_inds = _dist_thr[0] / model_spacing
        while check:
            check = False
            b_rpeaks_diff = np.diff(b_rpeaks)
            for r in range(len(b_rpeaks_diff)):
                if b_rpeaks_diff[r] < dist_thr_inds:  # 200 ms
                    prev_r_ind = b_rpeaks[r]
                    next_r_ind = b_rpeaks[r + 1]
                    if b_prob[prev_r_ind] > b_prob[next_r_ind]:
                        del_ind = r + 1
                    else:
                        del_ind = r
                    b_rpeaks = np.delete(b_rpeaks, del_ind)
                    check = True
                    break
        if len(_dist_thr) == 1:
            b_rpeaks = b_rpeaks[np.where((b_rpeaks >= skip_dist) & (b_rpeaks < input_len - skip_dist))[0]]
            rpeaks.append(b_rpeaks)
            continue
        check = True
        # TODO: parallel the following block
        # CAUTION !!!
        # this part is extremely slow in some cases (long duration and low SNR)
        dist_thr_inds = _dist_thr[1] / model_spacing
        while check:
            check = False
            b_rpeaks_diff = np.diff(b_rpeaks)
            for r in range(len(b_rpeaks_diff)):
                if b_rpeaks_diff[r] >= dist_thr_inds:  # 1200 ms
                    prev_r_ind = b_rpeaks[r]
                    next_r_ind = b_rpeaks[r + 1]
                    prev_qrs = [itv for itv in b_qrs_intervals if itv[0] <= prev_r_ind <= itv[1]][0]
                    next_qrs = [itv for itv in b_qrs_intervals if itv[0] <= next_r_ind <= itv[1]][0]
                    check_itv = [prev_qrs[1], next_qrs[0]]
                    l_new_itv = mask_to_intervals(b_mask[check_itv[0] : check_itv[1]], 1)
                    if len(l_new_itv) == 0:
                        continue
                    l_new_itv = [[itv[0] + check_itv[0], itv[1] + check_itv[0]] for itv in l_new_itv]
                    new_itv = max(l_new_itv, key=lambda itv: itv[1] - itv[0])
                    new_max_prob = (b_prob[new_itv[0] : new_itv[1]]).max()
                    for itv in l_new_itv:
                        itv_prob = (b_prob[itv[0] : itv[1]]).max()
                        if itv[1] - itv[0] == new_itv[1] - new_itv[0] and itv_prob > new_max_prob:
                            new_itv = itv
                            new_max_prob = itv_prob
                    b_rpeaks = np.insert(b_rpeaks, r + 1, 4 * (new_itv[0] + new_itv[1]))
                    check = True
                    break
        b_rpeaks = b_rpeaks[np.where((b_rpeaks >= skip_dist) & (b_rpeaks < input_len - skip_dist))[0]]
        rpeaks.append(b_rpeaks)
    return rpeaks
