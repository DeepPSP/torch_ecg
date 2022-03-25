"""

Possible Solutions
------------------
1. segmentation (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
2. sequence labelling (AF, non-AF) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
3. per-beat (R peak detection first) classification (CNN, etc. + RR LSTM) -> postprocess (merge too-close intervals, etc) -> onsets & offsets
4. object detection (? onsets and offsets)
"""

from itertools import repeat
from copy import deepcopy
from numbers import Real
from typing import Union, Optional, Sequence, Tuple, List, NoReturn, Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from torch_ecg.cfg import CFG
# models from torch_ecg
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.models.unets import ECG_UNET, ECG_SUBTRACT_UNET
from torch_ecg.models.rr_lstm import RR_LSTM
from torch_ecg.utils.misc import mask_to_intervals
from torch_ecg.utils.utils_interval import intervals_union
from torch_ecg.utils.preproc import merge_rpeaks
from torch_ecg.utils.outputs import SequenceTaggingOutput, RPeaksDetectionOutput

from cfg import ModelCfg


__all__ = [
    "ECG_SEQ_LAB_NET_CPSC2021",
    "ECG_UNET_CPSC2021",
    "ECG_SUBTRACT_UNET_CPSC2021",
    "RR_LSTM_CPSC2021",
]


class ECG_SEQ_LAB_NET_CPSC2021(ECG_SEQ_LAB_NET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2021"

    def __init__(self, config:CFG, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        ```python
        from cfg import ModelCfg
        task = "qrs_detection"  # or "main"
        model_cfg = deepcopy(ModelCfg[task])
        model_cfg.model_name = "seq_lab"
        model = ECG_SEQ_LAB_NET_CPSC2021(model_cfg)
        ````
        """
        if config[config.model_name].reduction == 1:
            config[config.model_name].recover_length = True
        super().__init__(config.classes, config.n_leads, config[config.model_name])
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        kwargs: task specific key word arguments

        Returns
        -------
        output: SequenceTaggingOutput or RPeaksDetectionOutput,
            the output of the model
            for qrs_detection task, the output is a RPeaksDetectionOutput instance, with items:
                - rpeak_indices: list of ndarray,
                    list of ndarray of rpeak indices for each batch element
                - prob: array_like,
                    the probability array of the input sequence of signals
            for main task, the output is a SequenceTaggingOutput instance, with items:
                - classes: list,
                    the list of classes
                - prob: array_like,
                    the probability array of the input sequence of signals
                - pred: array_like,
                    the binary prediction array of the input sequence of signals
                - af_episodes: list of list of intervals,
                    af episodes, in the form of intervals of [start, end], right inclusive
                - af_mask: alias of pred
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[Sequence[float],np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[Sequence[float],np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> RPeaksDetectionOutput:
        """ finished, checked,

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
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )
        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks, prob=prob,
        )

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[Sequence[float],np.ndarray,Tensor],
                             bin_pred_thr:float=0.5,
                             rpeaks:Optional[Union[Sequence[int],Sequence[Sequence[int]]]]=None,
                             episode_len_thr:int=5,) -> SequenceTaggingOutput:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeaks: sequence of sequence of int, optional,
            sequences of r peak indices
        episode_len_thr: int, default 5,
            minimal length of (both af and normal) episodes,
            with units in number of beats (rpeaks)

        Returns
        -------
        output: SequenceTaggingOutput, with items:
            - classes: list,
                the list of classes
            - prob: array_like,
                the probability array of the input sequence of signals
            - pred: array_like,
                the binary prediction array of the input sequence of signals
            - af_episodes: list of list of intervals,
                af episodes, in the form of intervals of [start, end], right inclusive
            - af_mask: alias of pred
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, n_leads, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)
        
        af_episodes, af_mask = _main_task_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            rpeaks=rpeaks,
            siglens=list(repeat(seq_len, batch_size)),
            episode_len_thr=episode_len_thr,
        )
        return SequenceTaggingOutput(
            classes=self.class_names, prob=prob, pred=af_mask,
            af_episodes=af_episodes,
            af_mask=af_mask,  # alias of pred
        )


class ECG_UNET_CPSC2021(ECG_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_UNET_CPSC2021"
    
    def __init__(self, config:CFG, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        from cfg import ModelCfg
        task = "qrs_detection"  # or "main"
        model_cfg = deepcopy(ModelCfg[task])
        model_cfg.model_name = "unet"
        model = ECG_SEQ_LAB_NET_CPSC2021(model_cfg)
        """
        super().__init__(config.classes, config.n_leads, config[config.model_name], **kwargs)
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        kwargs: task specific key word arguments

        Returns
        -------
        output: SequenceTaggingOutput or RPeaksDetectionOutput,
            the output of the model
            for qrs_detection task, the output is a RPeaksDetectionOutput instance, with items:
                - rpeak_indices: list of ndarray,
                    list of ndarray of rpeak indices for each batch element
                - prob: array_like,
                    the probability array of the input sequence of signals
            for main task, the output is a SequenceTaggingOutput instance, with items:
                - classes: list,
                    the list of classes
                - prob: array_like,
                    the probability array of the input sequence of signals
                - pred: array_like,
                    the binary prediction array of the input sequence of signals
                - af_episodes: list of list of intervals,
                    af episodes, in the form of intervals of [start, end], right inclusive
                - af_mask: alias of pred
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[Sequence[float],np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[Sequence[float],np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> RPeaksDetectionOutput:
        """ finished, checked,

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
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=1,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )
        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks, prob=prob,
        )

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[Sequence[float],np.ndarray,Tensor],
                             bin_pred_thr:float=0.5,
                             rpeaks:Optional[Union[Sequence[int],Sequence[Sequence[int]]]]=None,
                             episode_len_thr:int=5,) -> SequenceTaggingOutput:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeaks: sequence of sequence of int, optional,
            sequences of r peak indices
        episode_len_thr: int, default 5,
            minimal length of (both af and normal) episodes,
            with units in number of beats (rpeaks)

        Returns
        -------
        output: SequenceTaggingOutput, with items:
            - classes: list,
                the list of classes
            - prob: array_like,
                the probability array of the input sequence of signals
            - pred: array_like,
                the binary prediction array of the input sequence of signals
            - af_episodes: list of list of intervals,
                af episodes, in the form of intervals of [start, end], right inclusive
            - af_mask: alias of pred
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, n_leads, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)
        
        af_episodes, af_mask = _main_task_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            rpeaks=rpeaks,
            siglens=list(repeat(seq_len, batch_size)),
            episode_len_thr=episode_len_thr,
        )
        return SequenceTaggingOutput(
            classes=self.classes, prob=prob, pred=af_mask,
            af_episodes=af_episodes,
            af_mask=af_mask,  # alias of pred
        )


class ECG_SUBTRACT_UNET_CPSC2021(ECG_SUBTRACT_UNET):
    """
    """
    __DEBUG__ = True
    __name__ = "ECG_SUBTRACT_UNET_CPSC2021"

    def __init__(self, config:CFG, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        from cfg import ModelCfg
        task = "qrs_detection"  # or "main"
        model_cfg = deepcopy(ModelCfg[task])
        model_cfg.model_name = "unet"
        model = ECG_SEQ_LAB_NET_CPSC2021(model_cfg)
        """
        super().__init__(config.classes, config.n_leads, config[config.model_name], **kwargs)
        self.task = config.task

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  **kwargs:Any) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        kwargs: task specific key word arguments

        Returns
        -------
        output: SequenceTaggingOutput or RPeaksDetectionOutput,
            the output of the model
            for qrs_detection task, the output is a RPeaksDetectionOutput instance, with items:
                - rpeak_indices: list of ndarray,
                    list of ndarray of rpeak indices for each batch element
                - prob: array_like,
                    the probability array of the input sequence of signals
            for main task, the output is a SequenceTaggingOutput instance, with items:
                - classes: list,
                    the list of classes
                - prob: array_like,
                    the probability array of the input sequence of signals
                - pred: array_like,
                    the binary prediction array of the input sequence of signals
                - af_episodes: list of list of intervals,
                    af episodes, in the form of intervals of [start, end], right inclusive
                - af_mask: alias of pred
        """
        if self.task == "qrs_detection":
            return self._inference_qrs_detection(input, bin_pred_thr, **kwargs)
        elif self.task == "main":
            return self._inference_main_task(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[Sequence[float],np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,
                           **kwargs:Any,) -> Union[SequenceTaggingOutput, RPeaksDetectionOutput]:
        """
        alias for `self.inference`
        """
        return self.inference(input, bin_pred_thr, **kwargs)

    @torch.no_grad()
    def _inference_qrs_detection(self,
                                 input:Union[Sequence[float],np.ndarray,Tensor],
                                 bin_pred_thr:float=0.5,
                                 duration_thr:int=4*16,
                                 dist_thr:Union[int,Sequence[int]]=200,) -> RPeaksDetectionOutput:
        """ finished, checked,

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
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)

        # prob --> qrs mask --> qrs intervals --> rpeaks
        rpeaks = _qrs_detection_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=1,
            bin_pred_thr=bin_pred_thr,
            duration_thr=duration_thr,
            dist_thr=dist_thr
        )
        return RPeaksDetectionOutput(
            rpeak_indices=rpeaks, prob=prob,
        )

    @torch.no_grad()
    def _inference_main_task(self,
                             input:Union[Sequence[float],np.ndarray,Tensor],
                             bin_pred_thr:float=0.5,
                             rpeaks:Optional[Union[Sequence[int],Sequence[Sequence[int]]]]=None,
                             episode_len_thr:int=5,) -> SequenceTaggingOutput:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., channels, seq_len)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeaks: sequence of sequence of int, optional,
            sequences of r peak indices
        episode_len_thr: int, default 5,
            minimal length of (both af and normal) episodes,
            with units in number of beats (rpeaks)

        Returns
        -------
        output: SequenceTaggingOutput, with items:
            - classes: list,
                the list of classes
            - prob: array_like,
                the probability array of the input sequence of signals
            - pred: array_like,
                the binary prediction array of the input sequence of signals
            - af_episodes: list of list of intervals,
                af episodes, in the form of intervals of [start, end], right inclusive
            - af_mask: alias of pred
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, n_leads, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        prob = prob.cpu().detach().numpy().squeeze(-1)
        
        af_episodes, af_mask = _main_task_post_process(
            prob=prob,
            fs=self.config.fs,
            reduction=self.config.reduction,
            bin_pred_thr=bin_pred_thr,
            rpeaks=rpeaks,
            siglens=list(repeat(seq_len, batch_size)),
            episode_len_thr=episode_len_thr,
        )
        return SequenceTaggingOutput(
            classes=self.classes, prob=prob, pred=af_mask,
            af_episodes=af_episodes,
            af_mask=af_mask,  # alias of pred
        )


class RR_LSTM_CPSC2021(RR_LSTM):
    """
    """
    __DEBUG__ = True
    __name__ = "RR_LSTM_CPSC2021"

    def __init__(self, config:CFG, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        config: dict,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        Usage
        -----
        from cfg import ModelCfg
        task = "rr_lstm"
        model_cfg = deepcopy(ModelCfg[task])
        model_cfg.model_name = "rr_lstm"
        model = ECG_SEQ_LAB_NET_CPSC2021(model_cfg)
        """
        super().__init__(config.classes, config[config.model_name], **kwargs)

    @torch.no_grad()
    def inference(self,
                  input:Union[Sequence[float],np.ndarray,Tensor],
                  bin_pred_thr:float=0.5,
                  rpeaks:Optional[Union[Sequence[int],Sequence[Sequence[int]]]]=None,
                  episode_len_thr:int=5,) -> SequenceTaggingOutput:
        """ finished, checked,

        Parameters
        ----------
        input: array_like,
            input tensor, of shape (..., seq_len, ...)
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions
        rpeaks: sequence of sequence of int, optional,
            sequences of r peak indices
        episode_len_thr: int, default 5,
            minimal length of (both af and normal) episodes,
            with units in number of beats (rpeaks)

        Returns
        -------
        output: SequenceTaggingOutput, with items:
            - classes: list,
                the list of classes
            - prob: array_like,
                the probability array of the input sequence of signals
            - pred: array_like,
                the binary prediction array of the input sequence of signals
            - af_episodes: list of list of intervals,
                af episodes, in the form of intervals of [start, end], right inclusive
            - af_mask: alias of pred

        WARNING
        -------
        for AFf, further processing is needed to move the start and end
        to the first and last indices of the signal,
        rather than the indices of the first and the last rpeak
        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        elif _input.ndim == 1:
            _input = _input.unsqueeze(0).unsqueeze(-1)  # add a batch dimension and a channel dimension
        # (batch_size, seq_len, n_channels) -> (seq_len, batch_size, n_channels)
        _input = _input.permute(1,0,2)
        prob = self.forward(_input)
        if self.config.clf.name != "crf":
            prob = self.sigmoid(prob)
        prob = prob.cpu().detach().numpy().squeeze(-1)

        af_episodes, af_mask = _main_task_post_process(
            prob=prob,
            fs=1/0.8,
            reduction=1,
            bin_pred_thr=bin_pred_thr,
            rpeaks=None,
            siglens=None,
            episode_len_thr=episode_len_thr,
        )
        if rpeaks is not None:
            if isinstance((rpeaks[0]), Real):
                _rpeaks = [rpeaks]
            else:
                _rpeaks = rpeaks
            # WARNING: need further processing to move start and end for the case of AFf
            # NOTE that the next rpeak to the interval (of rr sequences) ends are added
            af_episodes = [[[r[itv[0]], r[itv[1]+1]] for itv in a] for a,r in zip(af_episodes, _rpeaks)]
        return SequenceTaggingOutput(
            classes=self.classes, prob=prob, pred=af_mask,
            af_episodes=af_episodes,
            af_mask=af_mask,  # alias of pred
        )

    @torch.no_grad()
    def inference_CPSC2021(self,
                           input:Union[Sequence[float],np.ndarray,Tensor],
                           bin_pred_thr:float=0.5,) -> SequenceTaggingOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)

    @staticmethod
    def from_checkpoint(path:str, device:Optional[torch.device]=None) -> Tuple[torch.nn.Module, dict]:
        """ finished, checked,

        Parameters
        ----------
        path: str,
            path of the checkpoint
        device: torch.device, optional,
            map location of the model parameters,
            defaults "cuda" if available, otherwise "cpu"

        Returns
        -------
        model: Module,
            the model loaded from a checkpoint
        aux_config: dict,
            auxiliary configs that are needed for data preprocessing, etc.
        """
        _device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        model = RR_LSTM_CPSC2021(config=ckpt["model_config"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model, aux_config


def _qrs_detection_post_process(prob:np.ndarray,
                                fs:Real,
                                reduction:int,
                                bin_pred_thr:float=0.5,
                                skip_dist:int=500,
                                duration_thr:int=4*16,
                                dist_thr:Union[int,Sequence[int]]=200,) -> List[np.ndarray]:
    """ finished, checked,

    prob --> qrs mask --> qrs intervals --> rpeaks

    Parameters
    ----------
    prob: ndarray,
        array of predicted probability
    fs: real number,
        sampling frequency of the ECG
    reduction: int,
        reduction (granularity) of `prob` w.r.t. the ECG
    bin_pred_thr: float, default 0.5,
        the threshold for making binary predictions from scalar predictions
    skip_dist: int, default 500,
        detected rpeaks with distance (units in ms) shorter than `skip_dist`
        to two ends of the ECG will be discarded
    duration_thr: int, default 4*16,
        minimum duration for a "true" qrs complex, units in ms
    dist_thr: int or sequence of int, default 200,
        if is sequence of int,
        (0-th element). minimum distance for two consecutive qrs complexes, units in ms;
        (1st element).(optional) maximum distance for checking missing qrs complexes, units in ms,
        e.g. [200, 1200]
        if is int, then is the case of (0-th element).
    """
    batch_size, prob_arr_len = prob.shape
    # print(batch_size, prob_arr_len)
    model_spacing = 1000 / fs  # units in ms
    input_len = reduction * prob_arr_len
    _skip_dist = skip_dist / model_spacing  # number of samples
    _duration_thr = duration_thr / model_spacing / reduction
    _dist_thr = [dist_thr] if isinstance(dist_thr, int) else dist_thr
    assert len(_dist_thr) <= 2

    # mask = (prob > bin_pred_thr).astype(int)
    rpeaks = []
    for b_idx in range(batch_size):
        b_prob = prob[b_idx,...]
        b_mask = (b_prob >= bin_pred_thr).astype(int)
        b_qrs_intervals = mask_to_intervals(b_mask, 1)
        # print(b_qrs_intervals)
        b_rpeaks = np.array([itv[0]+itv[1] for itv in b_qrs_intervals if itv[1]-itv[0] >= _duration_thr])
        b_rpeaks = (reduction * b_rpeaks / 2).astype(int)
        # print(f"before post-process, b_qrs_intervals = {b_qrs_intervals}")
        # print(f"before post-process, b_rpeaks = {b_rpeaks}")

        check = True
        dist_thr_inds = _dist_thr[0] / model_spacing
        while check:
            check = False
            b_rpeaks_diff = np.diff(b_rpeaks)
            for r in range(len(b_rpeaks_diff)):
                if b_rpeaks_diff[r] < dist_thr_inds:  # 200 ms
                    prev_r_ind = int(b_rpeaks[r]/reduction)  # ind in _prob
                    next_r_ind = int(b_rpeaks[r+1]/reduction)  # ind in _prob
                    if b_prob[prev_r_ind] > b_prob[next_r_ind]:
                        del_ind = r+1
                    else:
                        del_ind = r
                    b_rpeaks = np.delete(b_rpeaks, del_ind)
                    check = True
                    break
        if len(_dist_thr) == 1:
            b_rpeaks = b_rpeaks[np.where((b_rpeaks>=_skip_dist) & (b_rpeaks<input_len-_skip_dist))[0]]
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
                    prev_r_ind = int(b_rpeaks[r]/reduction)  # ind in _prob
                    next_r_ind = int(b_rpeaks[r+1]/reduction)  # ind in _prob
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
        b_rpeaks = b_rpeaks[np.where((b_rpeaks>=_skip_dist) & (b_rpeaks<input_len-_skip_dist))[0]]
        rpeaks.append(b_rpeaks)
    return rpeaks


def _main_task_post_process(prob:np.ndarray,
                            fs:Real,
                            reduction:int,
                            bin_pred_thr:float=0.5,
                            rpeaks:Sequence[Sequence[int]]=None,
                            siglens:Optional[Sequence[int]]=None,
                            episode_len_thr:int=5,) -> Tuple[List[List[List[int]]], np.ndarray]:
    """ finished, checked,

    post processing of the main task,
    converting mask into list of af episodes,
    and doing filtration, eliminating (both af and normal) episodes that are too short

    Parameters
    ----------
    prob: ndarray,
        predicted af mask, of shape (batch_size, seq_len)
    fs: real number,
        sampling frequency of the signal
    reduction: int,
        reduction ratio of the predicted af mask w.r.t. the signal
    bin_pred_thr: float, default 0.5,
        the threshold for making binary predictions from scalar predictions
    rpeaks: sequence of sequence of int, optional,
        sequences of r peak indices
    siglens: sequence of int, optional,
        original signal lengths,
        used to do padding for af intervals
    episode_len_thr: int, default 5,
        minimal length of (both af and normal) episodes,
        with units in number of beats (rpeaks)

    Returns
    -------
    af_episodes: list of list of intervals,
        af episodes, in the form of intervals of [start, end], right inclusive
    af_mask: ndarray,
        array (mask) of binary prediction of af, of shape (batch_size, seq_len)
    """
    batch_size, prob_arr_len = prob.shape
    model_spacing = 1000 / fs  # units in ms
    input_len = reduction * prob_arr_len
    default_rr = int(fs * 0.8)

    af_mask = (prob >= bin_pred_thr).astype(int)

    af_episodes = []
    for b_idx in range(batch_size):
        b_mask = af_mask[b_idx]
        intervals = mask_to_intervals(b_mask, [0,1])
        b_af_episodes = [
            [itv[0]*reduction, itv[1]*reduction] for itv in intervals[1]
        ]
        b_n_episodes = [
            [itv[0]*reduction, itv[1]*reduction] for itv in intervals[0]
        ]
        if siglens is not None and siglens[b_idx] % reduction > 0:
            b_n_episodes.append([siglens[b_idx] // reduction * reduction, siglens[b_idx]])
        if rpeaks is not None:
            b_rpeaks = rpeaks[b_idx]
            # merge non-af episodes shorter than `episode_len_thr`
            b_af_episodes.extend([
                itv for itv in b_n_episodes \
                    if len([r for r in b_rpeaks if itv[0] <= r < itv[1]]) < episode_len_thr
            ])
            b_af_episodes = intervals_union(b_af_episodes)
            # eliminate af episodes shorter than `episode_len_thr`
            # and make right inclusive
            b_af_episodes = [
                [itv[0], itv[1]-1] for itv in b_af_episodes \
                    if len([r for r in b_rpeaks if itv[0] <= r < itv[1]]) >= episode_len_thr
            ]
        else:
            # merge non-af episodes shorter than `episode_len_thr`
            b_af_episodes.extend([
                itv for itv in b_n_episodes \
                    if itv[1] - itv[0] < default_rr * episode_len_thr
            ])
            b_af_episodes = intervals_union(b_af_episodes)
            # eliminate af episodes shorter than `episode_len_thr`
            # and make right inclusive
            b_af_episodes = [
                [itv[0], itv[1]-1] for itv in b_af_episodes \
                    if itv[1] - itv[0] >= default_rr * episode_len_thr
            ]
        af_episodes.append(b_af_episodes)
    return af_episodes, af_mask
