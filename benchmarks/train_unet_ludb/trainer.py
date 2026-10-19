"""The LUDB trainer."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases.datasets.ludb import LUDBDataset
from torch_ecg.databases.physionet_databases.ludb import compute_metrics as compute_ludb_metrics
from torch_ecg.utils import worker_init_fn
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

__all__ = [
    "LUDBTrainer",
]


class LUDBTrainer(BaseTrainer):
    """BaseTrainer wired to LUDBDataset and the delineation metrics."""

    __DEBUG__ = True
    __name__ = "LUDBTrainer"

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
            dataset_cls=LUDBDataset,
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
        """Evaluate with the wave-delineation metrics (onset/offset
        sensitivity, precision, f1, mean error and its std, per wave)."""
        self.model.eval()

        all_scalar_preds = []
        all_mask_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output = self._model.inference(signals)
            all_scalar_preds.append(model_output.prob)
            all_mask_preds.append(model_output.mask)

        # all_scalar_preds of shape (n_samples, seq_len, n_classes)
        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        # all_mask_preds of shape (n_samples, seq_len)
        all_mask_preds = np.concatenate(all_mask_preds, axis=0)
        # all_labels of shape (n_samples, seq_len) if loss is CrossEntropyLoss
        # otherwise of shape (n_samples, seq_len, n_classes)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.train_config.loss != "CrossEntropyLoss":
            all_labels = all_labels.argmax(axis=-1)  # (n_samples, seq_len, n_classes) -> (n_samples, seq_len)

        # eval_res_split are scorings of onsets and offsets of pwaves, qrs
        # complexes, twaves, each a dict of sensitivity, precision,
        # f1_score, mean_error, standard_deviation
        eval_res_split = compute_ludb_metrics(
            np.repeat(all_labels[:, np.newaxis, :], self.model_config.n_leads, axis=1),
            np.repeat(all_mask_preds[:, np.newaxis, :], self.model_config.n_leads, axis=1),
            self._cm,
            self.train_config.fs,
        )

        eval_res = {
            metric: np.nanmean([eval_res_split[f"{wf}_{pos}"][metric] for wf in self._cm for pos in ["onset", "offset"]])
            for metric in [
                "sensitivity",
                "precision",
                "f1_score",
                "mean_error",
                "standard_deviation",
            ]
        }

        self.model.train()

        return eval_res

    @property
    def _cm(self) -> Dict[str, int]:
        """class map from the library's wave names to the mask values"""
        return {
            "pwave": self.train_config.class_map["p"],
            "qrs": self.train_config.class_map["N"],
            "twave": self.train_config.class_map["t"],
        }

    @property
    def batch_dim(self) -> int:
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return []
