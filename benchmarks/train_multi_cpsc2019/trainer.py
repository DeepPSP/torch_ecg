"""The CPSC2019 trainer."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases.cpsc_databases.cpsc2019 import compute_metrics as compute_cpsc2019_metrics
from torch_ecg.databases.datasets.cpsc2019 import CPSC2019Dataset
from torch_ecg.utils import worker_init_fn
from torch_ecg.utils.utils_data import mask_to_intervals
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

__all__ = [
    "CPSC2019Trainer",
]


class CPSC2019Trainer(BaseTrainer):
    """BaseTrainer wired to CPSC2019Dataset and the R-peak scoring metric."""

    __DEBUG__ = True
    __name__ = "CPSC2019Trainer"

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: CFG,
        train_config: CFG,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dataset_cls=CPSC2019Dataset,
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
        if train_dataset is None:
            train_dataset = self.dataset_cls(config=self.train_config, training=True, lazy=False)

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(config=self.train_config, training=False, lazy=False)

        # `worker_init_fn` reseeds DEFAULTS.RNG per worker, so that the
        # workers do not produce identical augmentation streams
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
        signals, labels = data
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        preds = self.model(signals)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate with the official CPSC2019 R-peak scoring metric."""
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
            labels = [mask_to_intervals(item, 1) for item in labels]  # intervals of qrs complexes
            labels = [  # to indices of rpeaks in the original signal sequence
                (reduction * np.array([itv[0] + itv[1] for itv in item]) / 2).astype(int) for item in labels
            ]
            labels = [
                item[
                    np.where(
                        (item >= self.train_config.skip_dist)
                        & (item < self.train_config.input_len - self.train_config.skip_dist)
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
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return []
