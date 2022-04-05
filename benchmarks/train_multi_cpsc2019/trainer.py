"""
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, NoReturn, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

from cfg import ModelCfg, TrainCfg
from dataset import CPSC2019
from metrics import compute_metrics
from model import (
    ECG_SEQ_LAB_NET_CPSC2019,
    ECG_SUBTRACT_UNET_CPSC2019,
    ECG_UNET_CPSC2019,
)

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import mask_to_intervals, str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

ECG_SEQ_LAB_NET_CPSC2019.__DEBUG__ = False
ECG_UNET_CPSC2019.__DEBUG__ = False
ECG_SUBTRACT_UNET_CPSC2019.__DEBUG__ = False
CPSC2019.__DEBUG__ = False

if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2019Trainer",
]


class CPSC2019Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2019Trainer"

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
        super().__init__(model, CPSC2019, model_config, train_config, device, lazy)

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

        qrs_score = compute_metrics(
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


def get_args(**kwargs):
    """ """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2019",
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
        default=128,
        help="the batch size for training",
        dest="batch_size",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        default="crnn",
        help="name of the model to train, `cnn` or `crnn`",
        dest="model_name",
    )
    parser.add_argument(
        "-c",
        "--cnn-name",
        type=str,
        default="multi_scopic",
        help="choice of cnn feature extractor",
        dest="cnn_name",
    )
    parser.add_argument(
        "-r",
        "--rnn-name",
        type=str,
        default="lstm",
        help="choice of rnn structures",
        dest="rnn_name",
    )
    parser.add_argument(
        "-a",
        "--attn-name",
        type=str,
        default="se",
        help="choice of attention block",
        dest="attn_name",
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
        default="adam",
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


_MODEL_MAP = dict(
    seq_lab_crnn=ECG_SEQ_LAB_NET_CPSC2019,
    seq_lab_cnn=ECG_SEQ_LAB_NET_CPSC2019,
    unet=ECG_UNET_CPSC2019,
    subtract_unet=ECG_SUBTRACT_UNET_CPSC2019,
)


if __name__ == "__main__":
    train_config = get_args(**TrainCfg)
    model_name = f"seq_lab_{train_config.model_name.lower()}"
    model_config = deepcopy(ModelCfg[model_name])
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    model_cls = _MODEL_MAP[model_name]

    model = model_cls(
        n_leads=train_config.n_leads,
        input_len=train_config.input_len,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=DEFAULTS.device)

    trainer = CPSC2019Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEFAULTS.device,
        lazy=False,
    )

    try:
        best_model_state_dict = trainer.train()
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
