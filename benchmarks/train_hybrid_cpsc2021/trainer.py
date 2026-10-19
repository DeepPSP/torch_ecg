"""Trainer of the CPSC2021 benchmark (paroxysmal AF event detection)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases.cpsc_databases.cpsc2019 import compute_metrics as compute_cpsc2019_metrics
from torch_ecg.databases.datasets.cpsc2021 import CPSC2021Dataset
from torch_ecg.utils import worker_init_fn
from torch_ecg.utils.utils_data import mask_to_intervals
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from .aux_metrics import compute_main_task_metric, compute_rr_metric
from .model import (  # noqa: F401
    ECG_SEQ_LAB_NET_CPSC2021,
    ECG_SUBTRACT_UNET_CPSC2021,
    ECG_UNET_CPSC2021,
    RR_LSTM_CPSC2021,
)

__all__ = [
    "CPSC2021Trainer",
]


class CPSC2021Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2021Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
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
            model=model,
            dataset_cls=CPSC2021Dataset,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=lazy,
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
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
                config=self.train_config,
                task=self.train_config.task,
                training=True,
                lazy=False,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

        num_workers = self.train_config.get("num_workers", 4 if torch.cuda.is_available() else 0)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
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
                worker_init_fn=worker_init_fn if num_workers > 0 else None,
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
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
        )

    def run_one_step(self, *data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if self.train_config.task == "rr_lstm":
            signals, labels, weight_masks = data
            # (batch_size, seq_len, n_channel) -> (seq_len, batch_size, n_channel)
            signals = signals.permute(1, 0, 2)
            weight_masks = weight_masks.to(device=self.device, dtype=self.dtype)
        elif self.train_config.task == "qrs_detection":
            signals, labels = data
        else:  # main task
            signals, labels, weight_masks = data
            weight_masks = weight_masks.to(device=self.device, dtype=self.dtype)
        signals = signals.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=self.dtype)
        preds = self.model(signals)
        if self.train_config.task == "qrs_detection":
            return preds, labels
        else:
            return preds, labels, weight_masks

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        if self.train_config.task == "qrs_detection":
            all_rpeak_preds = []
            all_rpeak_labels = []
            for signals, labels in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy()
                labels = [mask_to_intervals(item, 1) for item in labels]  # intervals of qrs complexes
                labels = [  # to indices of rpeaks in the original signal sequence
                    (self.train_config.qrs_detection.reduction * np.array([itv[0] + itv[1] for itv in item]) / 2).astype(int)
                    for item in labels
                ]
                labels = [
                    item[
                        np.where(
                            (item >= self.train_config.rpeaks_dist2border)
                            & (item < self.train_config.qrs_detection.input_len - self.train_config.rpeaks_dist2border)
                        )[0]
                    ]
                    for item in labels
                ]
                all_rpeak_labels += labels

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_output = self._model.inference(signals)
                all_rpeak_preds += model_output.rpeak_indices
            # the R-peak score is identical to the CPSC2019 challenge metric
            eval_res = {
                "qrs_score": compute_cpsc2019_metrics(
                    rpeaks_truths=all_rpeak_labels,
                    rpeaks_preds=all_rpeak_preds,
                    fs=self.train_config.fs,
                    thr=self.train_config.qrs_mask_bias / self.train_config.fs,
                ),
            }
            # in case possible memeory leakage?
            del all_rpeak_preds, all_rpeak_labels
        elif self.train_config.task == "rr_lstm":
            all_preds = np.array([]).reshape((0, self.train_config[self.train_config.task].input_len))
            all_labels = np.array([]).reshape((0, self.train_config[self.train_config.task].input_len))
            all_weight_masks = np.array([]).reshape((0, self.train_config[self.train_config.task].input_len))
            for signals, labels, weight_masks in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy().squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                weight_masks = weight_masks.numpy().squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                all_labels = np.concatenate((all_labels, labels))
                all_weight_masks = np.concatenate((all_weight_masks, weight_masks))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_output = self._model.inference(signals)
                all_preds = np.concatenate((all_preds, model_output.af_mask))  # or model_output.prob ?
            eval_res = compute_rr_metric(all_labels, all_preds, all_weight_masks)
            # in case possible memeory leakage?
            del all_preds, all_labels, all_weight_masks
        elif self.train_config.task == "main":
            all_preds = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len // self.train_config.main.reduction,
                )
            )
            all_labels = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len // self.train_config.main.reduction,
                )
            )
            all_weight_masks = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len // self.train_config.main.reduction,
                )
            )
            for signals, labels, weight_masks in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy().squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                weight_masks = weight_masks.numpy().squeeze(-1)  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                all_labels = np.concatenate((all_labels, labels))
                all_weight_masks = np.concatenate((all_weight_masks, weight_masks))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_output = self._model.inference(signals)
                all_preds = np.concatenate((all_preds, model_output.af_mask))  # or model_output.prob ?
            eval_res = compute_main_task_metric(
                mask_truths=all_labels,
                mask_preds=all_preds,
                fs=self.train_config.fs,
                reduction=self.train_config.main.reduction,
                weight_masks=all_weight_masks,
            )
            # in case possible memeory leakage?
            del all_preds, all_labels, all_weight_masks

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        # return 1 if self.train_config.task in ["rr_lstm"] else 0
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        if self.train_config.task in ["rr_lstm"]:
            return f"task-{self.train_config.task}_{self._model.__name__}_epoch"
        else:
            return f"task-{self.train_config.task}_{self._model.__name__}_{self.model_config.cnn_name}_epoch"

    def extra_log_suffix(self) -> str:
        if self.train_config.task in ["rr_lstm"]:
            return f"task-{self.train_config.task}_{super().extra_log_suffix()}"
        else:
            return f"task-{self.train_config.task}_{super().extra_log_suffix()}_{self.model_config.cnn_name}"


_MODEL_MAP = {
    "seq_lab": ECG_SEQ_LAB_NET_CPSC2021,
    "unet": ECG_UNET_CPSC2021,
    "lstm_crf": RR_LSTM_CPSC2021,
    "lstm": RR_LSTM_CPSC2021,
}


def _set_task(task: str, config: dict) -> None:
    """Set the fields of `config` that vary per task (classes, monitor, loss, etc.)."""
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]
