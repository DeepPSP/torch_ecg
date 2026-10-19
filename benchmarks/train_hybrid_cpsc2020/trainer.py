"""Trainer of the CPSC2020 benchmark (premature beat detection)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases.cpsc_databases.cpsc2020 import compute_metrics as compute_cpsc2020_metrics
from torch_ecg.utils import worker_init_fn
from torch_ecg.utils.misc import list_sum
from torch_ecg.utils.utils_data import mask_to_intervals
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

from .dataset import CPSC2020
from .metrics import eval_score

__all__ = [
    "CPSC2020Trainer",
]


class CPSC2020Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2020Trainer"

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
            dataset_cls=CPSC2020,
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
                training=True,
            )

        if self.train_config.debug:
            val_train_dataset = self.dataset_cls(
                config=self.train_config,
                training=True,
            )
            val_train_dataset.disable_data_augmentation()
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                training=False,
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
        signals, labels = data
        signals = signals.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=self.dtype)
        preds = self.model(signals)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        if self.train_config.model_name == "crnn":
            all_scalar_preds = []
            all_bin_preds = []
            all_labels = []
            for signals, labels in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                all_labels.append(labels.numpy())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_output = self._model.inference(signals)
                all_scalar_preds.append(model_output.prob)
                all_bin_preds.append(model_output.pred)
            all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
            all_bin_preds = np.concatenate(all_bin_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            classes = data_loader.dataset.all_classes
            auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure = eval_score(
                classes=classes,
                truth=all_labels,
                scalar_pred=all_scalar_preds,
                binary_pred=all_bin_preds,
            )
            eval_res = {
                "auroc": auroc,
                "auprc": auprc,
                "accuracy": accuracy,
                "f_measure": f_measure,
                "f_beta_measure": f_beta_measure,
                "g_beta_measure": g_beta_measure,
            }
        else:  # sequence labelling
            all_spb_preds = []
            all_pvc_preds = []
            all_spb_labels = []
            all_pvc_labels = []
            for signals, labels in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy()  # (batch_size, seq_len, n_classes)
                classes = data_loader.dataset.all_classes
                spb_intervals = [mask_to_intervals(seq, 1) for seq in labels[..., classes.index("S")]]
                spb_labels = [
                    [self._model.reduction * (itv[0] + itv[1]) // 2 for itv in l_itv] if len(l_itv) > 0 else []
                    for l_itv in spb_intervals
                ]
                all_spb_labels.append(spb_labels)
                pvc_intervals = [mask_to_intervals(seq, 1) for seq in labels[..., classes.index("V")]]
                pvc_labels = [
                    [self._model.reduction * (itv[0] + itv[1]) // 2 for itv in l_itv] if len(l_itv) > 0 else []
                    for l_itv in pvc_intervals
                ]
                all_pvc_labels.append(pvc_labels)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_output = self._model.inference(signals)
                all_spb_preds.append(model_output.SPB_indices)
                all_pvc_preds.append(model_output.PVC_indices)

            all_spb_preds = [np.array(item) for item in list_sum(all_spb_preds)]
            all_pvc_preds = [np.array(item) for item in list_sum(all_pvc_preds)]
            all_spb_labels = [np.array(item) for item in list_sum(all_spb_labels)]
            all_pvc_labels = [np.array(item) for item in list_sum(all_pvc_labels)]

            # the official challenge score from the library;
            # (negative) penalty score per ectopic beat type, higher (closer to 0) is better
            spb_score, pvc_score = compute_cpsc2020_metrics(
                sbp_true=all_spb_labels,
                pvc_true=all_pvc_labels,
                sbp_pred=all_spb_preds,
                pvc_pred=all_pvc_preds,
                verbose=0,
            )
            eval_res = {
                "challenge_score": spb_score + pvc_score,
                "spb_challenge_score": spb_score,
                "pvc_challenge_score": pvc_score,
                "neg_challenge_loss": -(spb_score + pvc_score),
            }

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """ """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return []

    @property
    def save_prefix(self) -> str:
        prefix = f"{self._model.__name__}_{self.model_config.cnn_name}"
        if hasattr(self.model_config, "rnn_name"):
            prefix = f"{prefix}_{self.model_config.rnn_name}"
        return f"{prefix}_epoch"

    def extra_log_suffix(self) -> str:
        suffix = super().extra_log_suffix() + f"_{self.model_config.cnn_name}"
        if hasattr(self.model_config, "rnn_name"):
            suffix = f"{suffix}_{self.model_config.rnn_name}"
        return suffix
