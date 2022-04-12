"""
References
----------
[1] https://github.com/milesial/Pytorch-UNet/blob/master/train.py
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # noqa: F401
except ModuleNotFoundError:
    from tqdm import tqdm  # noqa: F401

import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from cfg import ModelCfg, TrainCfg
from dataset import LUDB
from metrics import compute_metrics
from model import ECG_UNET_LUDB

from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

LUDB.__DEBUG__ = False
ECG_UNET_LUDB.__DEBUG__ = False

if TrainCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "LUDBTrainer",
]


class LUDBTrainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "LUDBTrainer"

    def __init__(
        self,
        model: nn.Module,
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
        super().__init__(model, LUDB, model_config, train_config, device, lazy)

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
        # labels of shape (batch_size, seq_len) if loss is CrossEntropyLoss
        # otherwise of shape (batch_size, seq_len, n_classes)
        labels = labels.to(self.device)
        preds = self.model(signals)  # of shape (batch_size, seq_len, n_classes)
        if self.train_config.loss == "CrossEntropyLoss":
            preds = preds.permute(0, 2, 1)  # of shape (batch_size, n_classes, seq_len)
            # or use the following
            # preds = pres.reshape(-1, preds.shape[-1])  # of shape (batch_size * seq_len, n_classes)
            # labels = labels.reshape(-1)  # of shape (batch_size * seq_len,)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
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
        # all_scalar_preds of shape (n_samples, seq_len)
        all_mask_preds = np.concatenate(all_mask_preds, axis=0)
        # all_labels of shape (n_samples, seq_len) if loss is CrossEntropyLoss
        # otherwise of shape (n_samples, seq_len, n_classes)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.train_config.loss != "CrossEntropyLoss":
            all_labels = all_labels.argmax(
                axis=-1
            )  # (n_samples, seq_len, n_classes) -> (n_samples, seq_len)

        # print(f"all_labels.shape: {all_labels.shape}, nan: {np.isnan(all_labels).any()}, inf: {np.isinf(all_labels).any()}")
        # print(f"all_scalar_preds.shape: {all_scalar_preds.shape}, nan: {np.isnan(all_scalar_preds).any()}, inf: {np.isinf(all_scalar_preds).any()}")
        # print(f"all_mask_preds.shape: {all_mask_preds.shape}, nan: {np.isnan(all_mask_preds).any()}, inf: {np.isinf(all_mask_preds).any()}")

        # eval_res are scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        # each scoring is a dict consisting of the following metrics:
        # sensitivity, precision, f1_score, mean_error, standard_deviation
        eval_res_split = compute_metrics(
            np.repeat(all_labels[:, np.newaxis, :], self.model_config.n_leads, axis=1),
            np.repeat(
                all_mask_preds[:, np.newaxis, :], self.model_config.n_leads, axis=1
            ),
            self._cm,
            self.train_config.fs,
        )

        # TODO: provide numerical values for the metrics from all of the dicts of eval_res
        eval_res = {
            metric: np.nanmean(
                [
                    eval_res_split[f"{wf}_{pos}"][metric]
                    for wf in self._cm
                    for pos in ["onset", "offset"]
                ]
            )
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
    def _cm(self) -> Dict[str, str]:
        """ """
        return {
            "pwave": self.train_config.class_map["p"],
            "qrs": self.train_config.class_map["N"],
            "twave": self.train_config.class_map["t"],
        }

    @property
    def batch_dim(self) -> int:
        """
        batch dimension,
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return []

    # @property
    # def save_prefix(self) -> str:
    #     return f"{self._model.__name__}_{self.model_config.cnn.name}_epoch"

    # def extra_log_suffix(self) -> str:
    #     return super().extra_log_suffix() + f"_{self.model_config.cnn.name}"


def get_args(**kwargs):
    """ """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on LUDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # parser.add_argument(
    #     "-l", "--learning-rate",
    #     metavar="LR", type=float, nargs="?", default=0.001,
    #     help="Learning rate",
    #     dest="learning_rate")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="the batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="unet",
        help="name of the model to train, `unet` or `subtract_unet`",
        dest="model_name",
    )
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=50,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_amsgrad",
        help="training optimizer",
        dest="train_optimizer",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="train with more debugging information",
        dest="debug",
    )

    args = vars(parser.parse_args())

    cfg.update(args)

    return CFG(cfg)


if __name__ == "__main__":
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = deepcopy(ModelCfg)

    model = ECG_UNET_LUDB(n_leads=model_config.n_leads, config=model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)

    model.to(device=device)

    trainer = LUDBTrainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=False,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
