"""
"""

import shutil
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, Optional, Any, Sequence, Union, Tuple, Dict, List

import pytest
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP

try:
    import torch_ecg
except:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))
    import torch_ecg

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.databases.datasets.cpsc2019 import CPSC2019Dataset, CPSC2019TrainCfg
from torch_ecg.databases.cpsc_databases.cpsc2019 import (
    compute_metrics as compute_cpsc2019_metrics,
)
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.model_configs import ECG_SEQ_LAB_NET_CONFIG
from torch_ecg.utils.utils_nn import (
    default_collate_fn as collate_fn,
    adjust_cnn_filter_lengths,
)
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import mask_to_intervals, add_docstring
from torch_ecg.utils.utils_signal import remove_spikes_naive
from torch_ecg.components.outputs import RPeaksDetectionOutput


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parent.parent / "tmp"
_DB_DIR = _CWD.parent.parent / "sample-data" / "cpsc2019"
###############################################################################

###############################################################################
# set up configs

ModelCfg = CFG()
ModelCfg.fs = 500  # Hz, CPSC2019 data fs
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.classes = [
    "N",
]
# NOTE(update): "background" now do not count as a class
# ModelCfg.classes = ["i", "N"]  # N for qrs, i for other parts
# ModelCfg.class_map = {c:i for i,c in enumerate(ModelCfg.classes)}
ModelCfg.n_leads = 1
ModelCfg.db_dir = None
ModelCfg.bias_thr = (
    0.075 * ModelCfg.fs
)  # keep the same with `THR` in `cpsc2019_score.py`
# detected rpeaks that are within `skip_dist` from two ends of the signal will be ignored,
# as in the official entry function
ModelCfg.skip_dist = 0.5 * ModelCfg.fs
ModelCfg.torch_dtype = DEFAULTS.torch_dtype

ModelCfg.seq_lab_crnn = deepcopy(ModelCfg)
ModelCfg.seq_lab_crnn.update(
    adjust_cnn_filter_lengths(
        deepcopy(ECG_SEQ_LAB_NET_CONFIG),
        ModelCfg.fs,
    )
)
ModelCfg.seq_lab_crnn.reduction = 1
ModelCfg.seq_lab_crnn.recover_length = True

# NOTE: one can adjust any of the cnn, rnn, attn, clf part of ModelCfg.seq_lab_crnn like ModelCfg.seq_lab_cnn

ModelCfg.seq_lab_cnn = deepcopy(ModelCfg.seq_lab_crnn)

ModelCfg.seq_lab_cnn.rnn = CFG()
ModelCfg.seq_lab_cnn.rnn.name = "none"  # "lstm"
###############################################################################


class ECG_SEQ_LAB_NET_CPSC2019(ECG_SEQ_LAB_NET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_SEQ_LAB_NET_CPSC2019"

    def __init__(
        self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any
    ) -> NoReturn:
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
        b_rpeaks = np.array(
            [
                (itv[0] + itv[1]) // 2
                for itv in b_qrs_intervals
                if itv[1] - itv[0] >= _duration_thr
            ]
        )
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
            b_rpeaks = b_rpeaks[
                np.where((b_rpeaks >= skip_dist) & (b_rpeaks < input_len - skip_dist))[
                    0
                ]
            ]
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
                    prev_qrs = [
                        itv for itv in b_qrs_intervals if itv[0] <= prev_r_ind <= itv[1]
                    ][0]
                    next_qrs = [
                        itv for itv in b_qrs_intervals if itv[0] <= next_r_ind <= itv[1]
                    ][0]
                    check_itv = [prev_qrs[1], next_qrs[0]]
                    l_new_itv = mask_to_intervals(
                        b_mask[check_itv[0] : check_itv[1]], 1
                    )
                    if len(l_new_itv) == 0:
                        continue
                    l_new_itv = [
                        [itv[0] + check_itv[0], itv[1] + check_itv[0]]
                        for itv in l_new_itv
                    ]
                    new_itv = max(l_new_itv, key=lambda itv: itv[1] - itv[0])
                    new_max_prob = (b_prob[new_itv[0] : new_itv[1]]).max()
                    for itv in l_new_itv:
                        itv_prob = (b_prob[itv[0] : itv[1]]).max()
                        if (
                            itv[1] - itv[0] == new_itv[1] - new_itv[0]
                            and itv_prob > new_max_prob
                        ):
                            new_itv = itv
                            new_max_prob = itv_prob
                    b_rpeaks = np.insert(b_rpeaks, r + 1, 4 * (new_itv[0] + new_itv[1]))
                    check = True
                    break
        b_rpeaks = b_rpeaks[
            np.where((b_rpeaks >= skip_dist) & (b_rpeaks < input_len - skip_dist))[0]
        ]
        rpeaks.append(b_rpeaks)
    return rpeaks


class CPSC2019Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2019Trainer"

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """

        Parameters
        ----------
        model: Module,
            the model to be trained
        model_config: dict,
            the configuration of the model,
            used to keep a record in the checkpoints
        train_config: dict,
            the configuration of the training,
            including configurations for the data loader, for the optimization, etc.
            will also be recorded in the checkpoints.
            `train_config` should at least contain the following keys:
                "monitor": str,
                "loss": str,
                "n_epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "lr_scheduler": str,
                    "lr_step_size": int, optional, depending on the scheduler
                    "lr_gamma": float, optional, depending on the scheduler
                    "max_lr": float, optional, depending on the scheduler
                "optimizer": str,
                    "decay": float, optional, depending on the optimizer
                    "momentum": float, optional, depending on the optimizer
        device: torch.device, optional,
            the device to be used for training
        lazy: bool, default True,
            whether to initialize the data loader lazily
        """
        super().__init__(
            model, CPSC2019Dataset, model_config, train_config, device, lazy
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> NoReturn:
        """

        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset
        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config, training=True, lazy=False
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config, training=False, lazy=False
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        num_workers = 4

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def run_one_step(
        self, *data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors

        Returns
        -------
        preds: Tensor,
            the predictions of the model for the given data
        labels: Tensor,
            the labels of the given data
        """
        signals, labels = data
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        preds = self.model(signals)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        if self.train_config.get("recover_length", False):
            reduction = 1
        else:
            reduction = self.train_config.reduction

        all_rpeak_preds = []
        all_rpeak_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            labels = [
                mask_to_intervals(item, 1) for item in labels
            ]  # intervals of qrs complexes
            labels = [  # to indices of rpeaks in the original signal sequence
                (reduction * np.array([itv[0] + itv[1] for itv in item]) / 2).astype(
                    int
                )
                for item in labels
            ]
            labels = [
                item[
                    np.where(
                        (item >= self.train_config.skip_dist)
                        & (
                            item
                            < self.train_config.input_len - self.train_config.skip_dist
                        )
                    )[0]
                ]
                for item in labels
            ]
            all_rpeak_labels += labels

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output = self._model.inference(
                signals,
                bin_pred_thr=0.5,
                duration_thr=4 * 16,
                dist_thr=200,
                correction=False,
            )
            all_rpeak_preds += model_output.rpeak_indices

        qrs_score = compute_cpsc2019_metrics(
            rpeaks_truths=all_rpeak_labels,
            rpeaks_preds=all_rpeak_preds,
            fs=self.train_config.fs,
            thr=self.train_config.bias_thr / self.train_config.fs,
        )
        eval_res = dict(
            qrs_score=qrs_score,
        )

        del all_rpeak_labels, all_rpeak_preds

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return []

    @property
    def save_prefix(self) -> str:
        return f"{self._model.__name__}_{self.model_config.cnn.name}_epoch"

    def extra_log_suffix(self) -> str:
        return super().extra_log_suffix() + f"_{self.model_config.cnn.name}"


def test_seq_lab_cpsc2019_pipeline() -> NoReturn:
    """ """

    train_config = deepcopy(CPSC2019TrainCfg)
    train_config.db_dir = _DB_DIR
    train_config.log_dir = _CWD / "logs"
    train_config.model_dir = _CWD / "saved_models"
    train_config.checkpoints = _CWD / "checkpoints"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.checkpoints.mkdir(parents=True, exist_ok=True)

    train_config.keep_checkpoint_max = 0
    # train_config.monitor = None
    train_config.n_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in ["seq_lab_cnn", "seq_lab_crnn"]:
        train_config.model_name = model_name

        model_config = deepcopy(ModelCfg[train_config.model_name])

        model = ECG_SEQ_LAB_NET_CPSC2019(model_config.n_leads, model_config)

        if torch.cuda.device_count() > 1:
            model = DP(model)
            # model = DDP(model)
        model.to(device=device)

        trainer = CPSC2019Trainer(
            model=model,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=False,
        )

        bmd = trainer.train()

        del bmd, trainer, model

    shutil.rmtree(_CWD)


if __name__ == "__main__":
    test_seq_lab_cpsc2019_pipeline()
