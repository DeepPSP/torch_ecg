"""
"""

import argparse
import os
import sys
import textwrap
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from cfg import ModelCfg, TrainCfg
from dataset import CinC2023Dataset
from models import CRNN_CINC2023  # TODO: implement and add more models
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from utils.scoring_metrics import compute_challenge_metrics

from torch_ecg.augmenters import AugmenterManager
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.models.loss import AsymmetricLoss, BCEWithLogitsWithClassWeightLoss, FocalLoss, MaskedBCEWithLogitsLoss
from torch_ecg.utils.misc import str2bool
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

__all__ = [
    "CINC2023Trainer",
]


class CINC2023Trainer(BaseTrainer):
    """Trainer for the CinC2023 challenge.

    Parameters
    ----------
    model : torch.nnModule
        the model to be trained
    model_config : dict
        the configuration of the model,
        used to keep a record in the checkpoints
    train_config : dict
        the configuration of the training,
        including configurations for the data loader, for the optimization, etc.
        will also be recorded in the checkpoints.
        `train_config` should at least contain the following keys:

            - "monitor": obj:`str`,
            - "loss": obj:`str`,
            - "n_epochs": obj:`int`,
            - "batch_size": obj:`int`,
            - "learning_rate": obj:`float`,
            - "lr_scheduler": obj:`str`,
            - "lr_step_size": obj:`int`, optional, depending on the scheduler
            - "lr_gamma": obj:`float`, optional, depending on the scheduler
            - "max_lr": obj:`float`, optional, depending on the scheduler
            - "optimizer": obj:`str`,
            - "decay": obj:`float`, optional, depending on the optimizer
            - "momentum": obj:`float`, optional, depending on the optimizer

    device : torch.device, optional
        the device to be used for training
    lazy : bool, default True
        whether to initialize the data loader lazily

    """

    __DEBUG__ = True
    __name__ = "CINC2023Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dataset_cls=CinC2023Dataset,
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

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if self.device == torch.device("cpu"):
            num_workers = 1
        else:
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

    def _setup_augmenter_manager(self) -> None:
        """ """
        self.augmenter_manager = AugmenterManager.from_config(config=self.train_config[self.train_config.task])

    def _setup_criterion(self) -> None:
        """ """
        loss_kw = self.train_config[self.train_config.task].get("loss_kw", {}).get(self._criterion_key, {})
        if self.train_config.loss[self._criterion_key] == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "BCEWithLogitsWithClassWeightLoss":
            self.criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "BCELoss":
            self.criterion = nn.BCELoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "FocalLoss":
            self.criterion = FocalLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "AsymmetricLoss":
            self.criterion = AsymmetricLoss(**loss_kw)
        elif self.train_config.loss[self._criterion_key] == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss(**loss_kw)
        else:
            raise NotImplementedError(
                f"loss `{self.train_config.loss}` not implemented! "
                "Please use one of the following: `BCEWithLogitsLoss`, `BCEWithLogitsWithClassWeightLoss`, "
                "`BCELoss`, `MaskedBCEWithLogitsLoss`, `MaskedBCEWithLogitsLoss`, `FocalLoss`, "
                "`AsymmetricLoss`, `CrossEntropyLoss`, or override this method to setup your own criterion."
            )
        self.criterion.to(device=self.device, dtype=self.dtype)

    def train_one_epoch(self, pbar: tqdm) -> None:
        """Train one epoch, and update the progress bar

        Parameters
        ----------
        pbar : tqdm
            the progress bar for training

        """
        if self.train_config.reload_data_every > 0 and self.epoch > 0 and self.epoch % self.train_config.reload_data_every == 0:
            self.log_manager.log_message(f"Reloading data at epoch {self.epoch}...")
            # reload data of the `Dataset` instances
            self.train_loader.dataset.empty_cache()
            # shuffle the list of records
            # DEFAULTS.RNG.shuffle(self.train_loader.dataset.records)
            self.train_loader.dataset.shuffle_records()
            self.train_loader.dataset._load_all_data()
            self.val_loader.dataset.empty_cache()
            self.val_loader.dataset._load_all_data()
            self.log_manager.log_message("Reloading data finished.")
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            n_samples = input_tensors["waveforms"].shape[self.batch_dim]
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "cpc" (optional): the cpc labels, for classification task or regression task
            # "outcome" (optional): the outcome labels, for classification task

            # move input_tensors to device
            input_tensors = {k: v.to(device=self.device, dtype=self.dtype) for k, v in input_tensors.items()}

            # out_tensors is a dict of tensors, with the following items (some are optional):
            # - "cpc": the cpc predictions, of shape (batch_size, n_classes) or (batch_size,)
            # - "outcome": the outcome predictions, of shape (batch_size, n_classes)
            out_tensors = self.run_one_step(input_tensors)

            # WARNING:
            # When `module` (self._model) returns a scalar (i.e., 0-dimensional tensor) in forward(),
            # `DataParallel` will return a vector of length equal to number of devices used in data parallelism,
            # containing the result from each device.
            # ref. https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
            loss = self.criterion(
                out_tensors[self._criterion_key],
                input_tensors[self._criterion_key],
            )

            if self.train_config.flooding_level > 0:
                flood = (loss - self.train_config.flooding_level).abs() + self.train_config.flooding_level
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                flood.backward()
            else:
                self.epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
            self.optimizer.step()
            self._update_lr()

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_last_lr()[0]})
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }
                    )
                else:
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                        }
                    )
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run one step (batch) of training

        Parameters
        ----------
        input_tensors : dict
            the tensors to be processed for training one step (batch), with the following items:
                - "waveforms" (required): the input waveforms
                - "cpc" (optional): the cpc labels, for classification task or regression task
                - "outcome" (optional): the outcome labels, for classification task

        Returns
        -------
        out_tensors : dict
            with the following items (some are optional):
                - "cpc": the cpc predictions, of shape (batch_size, n_classes) or (batch_size,)
                - "outcome": the outcome predictions, of shape (batch_size, n_classes)

        """
        waveforms = input_tensors.pop("waveforms").to(self.device)
        input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        waveforms, input_tensors[self._criterion_key] = self.augmenter_manager(waveforms, input_tensors[self._criterion_key])
        out_tensors = self.model(waveforms, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on the given data loader"""

        self.model.eval()

        all_outputs = []
        all_labels = []
        all_hospitals = []  # required for computing the metrics in the official phase

        for input_tensors in data_loader:
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "cpc" (optional): the cpc labels, for classification task or regression task
            # "outcome" (optional): the outcome labels, for classification task
            waveforms = input_tensors.pop("waveforms")
            waveforms = waveforms.to(device=self.device, dtype=self.dtype)
            hospitals = input_tensors.pop("hospitals").numpy().flatten().tolist()
            labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}

            all_labels.append(labels)
            all_hospitals.append(hospitals)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            all_outputs.append(self._model.inference(waveforms))

        if self.val_train_loader is not None and self._criterion_key == "cpc":
            log_head_num = 5
            head_scalar_preds = all_outputs[0].cpc_output.prob[:log_head_num]
            head_bin_preds = all_outputs[0].cpc_output.bin_pred[:log_head_num]
            head_preds_classes = [np.array(all_outputs[0].cpc_output.classes)[np.where(row)[0]] for row in head_bin_preds]
            head_labels = all_labels[0]["cpc"][:log_head_num]
            head_labels_classes = [
                (
                    np.array(all_outputs[0].cpc_output.classes)[np.where(row)]
                    if head_labels.ndim == 2
                    else np.array(all_outputs[0].cpc_output.classes)[row]
                )
                for row in head_labels
            ]
            log_head_num = min(log_head_num, len(head_scalar_preds))
            for n in range(log_head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                cpc scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                cpc binary prediction:    {head_bin_preds[n].tolist()}
                cpc labels:               {head_labels[n].astype(int).tolist()}
                cpc predicted classes:    {head_preds_classes[n].tolist()}
                cpc label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        eval_res = compute_challenge_metrics(
            labels=all_labels,
            outputs=all_outputs,
            hospitals=all_hospitals,
        )

        # in case possible memeory leakage?
        del all_labels
        del all_outputs

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        prefix = f"task-{self.train_config.task}_{self._model.__name__}"
        if hasattr(self.model_config, "cnn"):
            prefix = f"{prefix}_{self.model_config.cnn.name}_epoch"
        else:
            prefix = f"{prefix}_epoch"
        return prefix

    def extra_log_suffix(self) -> str:
        suffix = f"task-{self.train_config.task}_{super().extra_log_suffix()}"
        if hasattr(self.model_config, "cnn"):
            suffix = f"{suffix}_{self.model_config.cnn.name}"
        return suffix

    @property
    def _criterion_key(self) -> str:
        return {
            "classification": self.train_config.output_target,
        }[self.train_config.task]


def get_args(**kwargs: Any):
    """NOT checked,"""
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2023 database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="the batch size for training",
        dest="batch_size",
    )
    # parser.add_argument(
    #     "-c", "--cnn-name",
    #     type=str, default="multi_scopic_leadwise",
    #     help="choice of cnn feature extractor",
    #     dest="cnn_name")
    # parser.add_argument(
    #     "-r", "--rnn-name",
    #     type=str, default="none",
    #     help="choice of rnn structures",
    #     dest="rnn_name")
    # parser.add_argument(
    #     "-a", "--attn-name",
    #     type=str, default="se",
    #     help="choice of attention structures",
    #     dest="attn_name")
    parser.add_argument(
        "--keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )
    # parser.add_argument(
    #     "--optimizer", type=str, default="adam",
    #     help="training optimizer",
    #     dest="train_optimizer")
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


_MODEL_MAP = {
    "crnn": CRNN_CINC2023,
}


def _set_task(task: str, config: CFG) -> None:
    assert task in config.tasks
    config.task = task
    for item in [
        "classes",
        "monitor",
        "final_model_name",
        "loss",
    ]:
        config[item] = config[task][item]


if __name__ == "__main__":
    # WARNING: most training were done in notebook,
    # NOT in cli
    train_config = get_args(**TrainCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: adjust for CINC2023
    for task in train_config.tasks:
        _set_task(task, train_config)
        model_config = deepcopy(ModelCfg[task])
        model_config = deepcopy(ModelCfg[task])

        # adjust model choices if needed
        model_name = model_config.model_name = train_config[task].model_name
        if "cnn" in model_config[model_name]:
            model_config[model_name].cnn.name = train_config[task].cnn_name
        if "rnn" in model_config[model_name]:
            model_config[model_name].rnn.name = train_config[task].rnn_name
        if "attn" in model_config[model_name]:
            model_config[model_name].attn.name = train_config[task].attn_name

        model_cls = _MODEL_MAP[train_config[task].model_name]
        model_cls.__DEBUG__ = False
        model = model_cls(config=model_config)

        if torch.cuda.device_count() > 1:
            model = DP(model)
            # model = DDP(model)
        model.to(device=device)

        trainer = CINC2023Trainer(
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
