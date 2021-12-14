"""
"""
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, Sequence, Tuple, List, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from easydict import EasyDict as ED
import biosppy.signals.ecg as BSE

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.utils.misc import mask_to_intervals
from torch_ecg.utils.utils_signal import remove_spikes_naive

from cfg import ModelCfg


__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2019",
]


class ECG_SEQ_LAB_NET_CPSC2019(ECG_SEQ_LAB_NET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2019"
    
    def __init__(self, n_leads:int, config:Optional[ED]=None, **kwargs:Any) -> NoReturn:
        """ finished, checked,

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
        print(f"model_config = {model_config}")
        super().__init__(model_config.classes, n_leads, model_config, **kwargs)

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  duration_thr:int=4*16,
                  dist_thr:Union[int,Sequence[int]]=200,
                  correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """ finished, checked,

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
        pred: ndarray,
            the array of scalar predictions
        rpeaks: list of ndarray,
            list of rpeak indices for each batch element
        """
        self.eval()
        _device = next(self.parameters()).device
        _dtype = next(self.parameters()).dtype
        _input = torch.as_tensor(input, dtype=_dtype, device=_device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        pred = self.forward(_input)
        pred = self.sigmoid(pred)
        pred = pred.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = self._inference_post_process(
            pred=pred,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )

        if correction:
            rpeaks = [
                BSE.correct_rpeaks(
                    signal=b_input,
                    rpeaks=b_rpeaks,
                    sampling_rate=self.config.fs,
                    tol=0.05,
                )[0] for b_input, b_rpeaks in zip(_input.detach().numpy().squeeze(1), rpeaks)
            ]

        return pred, rpeaks

    def _inference_post_process(self,
                                pred:np.ndarray,
                                bin_pred_thr:float=0.5,
                                duration_thr:int=4*16,
                                dist_thr:Union[int,Sequence[int]]=200) -> List[np.ndarray]:
        """ finished, checked,

        prob --> qrs mask --> qrs intervals --> rpeaks

        Parameters: ref. `self.inference`
        """
        batch_size, prob_arr_len = pred.shape
        model_spacing = 1000 / self.config.fs  # units in ms
        model_granularity = 8  # TODO: automatic adjust via model config
        input_len = model_granularity * prob_arr_len
        _duration_thr = duration_thr / model_spacing / model_granularity
        _dist_thr = [dist_thr] if isinstance(dist_thr, int) else dist_thr
        assert len(_dist_thr) <= 2

        # mask = (pred > bin_pred_thr).astype(int)
        rpeaks = []
        for b_idx in range(batch_size):
            b_prob = pred[b_idx,...]
            b_mask = (b_prob > bin_pred_thr).astype(int)
            b_qrs_intervals = mask_to_intervals(b_mask, 1)
            b_rpeaks = np.array([itv[0]+itv[1] for itv in b_qrs_intervals if itv[1]-itv[0] >= _duration_thr])
            b_rpeaks = (model_granularity//2) * b_rpeaks
            # print(f"before post-process, b_qrs_intervals = {b_qrs_intervals}")
            # print(f"before post-process, b_rpeaks = {b_rpeaks}")

            check = True
            dist_thr_inds = _dist_thr[0] / model_spacing
            while check:
                check = False
                b_rpeaks_diff = np.diff(b_rpeaks)
                for r in range(len(b_rpeaks_diff)):
                    if b_rpeaks_diff[r] < dist_thr_inds:  # 200 ms
                        prev_r_ind = int(b_rpeaks[r]/model_granularity)  # ind in _prob
                        next_r_ind = int(b_rpeaks[r+1]/model_granularity)  # ind in _prob
                        if b_prob[prev_r_ind] > b_prob[next_r_ind]:
                            del_ind = r+1
                        else:
                            del_ind = r
                        b_rpeaks = np.delete(b_rpeaks, del_ind)
                        check = True
                        break
            if len(_dist_thr) == 1:
                b_rpeaks = b_rpeaks[np.where((b_rpeaks>=self.config.skip_dist) & (b_rpeaks<input_len-self.config.skip_dist))[0]]
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
                        prev_r_ind = int(b_rpeaks[r]/model_granularity)  # ind in _prob
                        next_r_ind = int(b_rpeaks[r+1]/model_granularity)  # ind in _prob
                        prev_qrs = [itv for itv in b_qrs_intervals if itv[0]<=prev_r_ind<=itv[1]][0]
                        next_qrs = [itv for itv in b_qrs_intervals if itv[0]<=next_r_ind<=itv[1]][0]
                        check_itv = [prev_qrs[1], next_qrs[0]]
                        l_new_itv = mask_to_intervals(b_mask[check_itv[0]: check_itv[1]], 1)
                        if len(l_new_itv) == 0:
                            continue
                        l_new_itv = [[itv[0]+check_itv[0], itv[1]+check_itv[0]] for itv in l_new_itv]
                        new_itv = max(l_new_itv, key=lambda itv: itv[1]-itv[0])
                        new_max_prob = (b_prob[new_itv[0]:new_itv[1]]).max()
                        for itv in l_new_itv:
                            itv_prob = (b_prob[itv[0]:itv[1]]).max()
                            if itv[1] - itv[0] == new_itv[1] - new_itv[0] and itv_prob > new_max_prob:
                                new_itv = itv
                                new_max_prob = itv_prob
                        b_rpeaks = np.insert(b_rpeaks, r+1, 4*(new_itv[0]+new_itv[1]))
                        check = True
                        break
            b_rpeaks = b_rpeaks[np.where((b_rpeaks>=self.config.skip_dist) & (b_rpeaks<input_len-self.config.skip_dist))[0]]
            rpeaks.append(b_rpeaks)
        return rpeaks

    @torch.no_grad()
    def inference_CPSC2019(self,
                           input:Union[np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           duration_thr:int=4*16,
                           dist_thr:Union[int,Sequence[int]]=200,
                           correction:bool=False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr, duration_thr, dist_thr, correction)
