""" """

import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.models.loss import AsymmetricLoss, BCEWithLogitsWithClassWeightLoss, FocalLoss, MaskedBCEWithLogitsLoss
from torch_ecg.utils import worker_init_fn
from torch_ecg.utils.utils_data import mask_to_intervals  # noqa: F401
from torch_ecg.utils.utils_nn import default_collate_fn

from .cfg import BaseCfg, ModelCfg, TrainCfg  # noqa: F401
from .dataset import CinC2022Dataset
from .models import CRNN_CINC2022, SEQ_LAB_NET_CINC2022, UNET_CINC2022, HFWav2Vec2_CINC2022, Wav2Vec2_CINC2022  # noqa: F401
from .utils.augmentations import AugmenterManager
from .utils.scoring_metrics import compute_challenge_metrics  # noqa: F401

__all__ = [
    "CINC2022Trainer",
]


class CINC2022Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CINC2022Trainer"

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
            dataset_cls=CinC2022Dataset,
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
        """
        train one epoch, and update the progress bar

        Parameters
        ----------
        pbar: tqdm,
            the progress bar for training

        """
        if self.epoch >= self.train_config[self.train_config.task].freeze_backbone_at > 0:
            self._model.freeze_backbone(True)
        else:
            self._model.freeze_backbone(False)
        for epoch_step, input_tensors in enumerate(self.train_loader):
            self.global_step += 1
            n_samples = input_tensors["waveforms"].shape[self.batch_dim]
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            input_tensors["waveforms"] = self.augmenter_manager(input_tensors["waveforms"])

            # out_tensors is a dict of tensors, with the following items (some are optional):
            # - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            # - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            # - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            # - "outcome_loss": loss of the outcome predictions
            # - "segmentation_loss": loss of the segmentation predictions
            # - "total_extra_loss": total loss of the extra heads
            out_tensors = self.run_one_step(input_tensors)

            # WARNING:
            # When `module` (self._model) returns a scalar (i.e., 0-dimensional tensor) in forward(),
            # `DataParallel` will return a vector of length equal to number of devices used in data parallelism,
            # containing the result from each device.
            # ref. https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
            loss = self.criterion(
                out_tensors[self._criterion_key],
                input_tensors[self._criterion_key].to(dtype=self.dtype, device=self.device),
            ).to(dtype=self.dtype, device=self.device) + out_tensors.get(
                "total_extra_loss",
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
            ).mean().to(
                dtype=self.dtype, device=self.device
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
        """

        Parameters
        ----------
        input_tensors: dict of Tensors,
            the tensors to be processed for training one step (batch), with the following items:
                - "waveforms" (required): the input waveforms
                - "murmur" (optional): the murmur labels, for classification task and multi task
                - "outcome" (optional): the outcome labels, for classification task and multi task
                - "segmentation" (optional): the segmentation labels, for segmentation task and multi task

        Returns
        -------
        out_tensors: dict of Tensors, with the following items (some are optional):
            - "murmur": the murmur predictions, of shape (batch_size, n_classes)
            - "outcome": the outcome predictions, of shape (batch_size, n_outcomes)
            - "segmentation": the segmentation predictions, of shape (batch_size, seq_len, n_states)
            - "outcome_loss": loss of the outcome predictions
            - "segmentation_loss": loss of the segmentation predictions
            - "total_extra_loss": total loss of the extra heads

        """
        waveforms = input_tensors.pop("waveforms").to(self.device)
        input_tensors = {k: v.to(self.device) for k, v in input_tensors.items()}
        out_tensors = self.model(waveforms, input_tensors)
        return out_tensors

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """

        self.model.eval()

        all_outputs = []
        all_labels = []

        for input_tensors in data_loader:
            # input_tensors is assumed to be a dict of tensors, with the following items:
            # "waveforms" (required): the input waveforms
            # "murmur" (optional): the murmur labels, for classification task and multi task
            # "outcome" (optional): the outcome labels, for classification task and multi task
            # "segmentation" (optional): the segmentation labels, for segmentation task and multi task
            waveforms = input_tensors.pop("waveforms")
            waveforms = waveforms.to(device=self.device, dtype=self.dtype)
            labels = {k: v.numpy() for k, v in input_tensors.items() if v is not None}

            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            all_outputs.append(self._model.inference(waveforms))

        if self.val_train_loader is not None and self.train_config.task not in ["segmentation"]:
            log_head_num = 5
            head_scalar_preds = all_outputs[0].murmur_output.prob[:log_head_num]
            head_bin_preds = all_outputs[0].murmur_output.bin_pred[:log_head_num]
            head_preds_classes = [np.array(all_outputs[0].murmur_output.classes)[np.where(row)[0]] for row in head_bin_preds]
            head_labels = all_labels[0]["murmur"][:log_head_num]
            head_labels_classes = [
                (
                    np.array(all_outputs[0].murmur_output.classes)[np.where(row)]
                    if head_labels.ndim == 2
                    else np.array(all_outputs[0].murmur_output.classes)[row]
                )
                for row in head_labels
            ]
            log_head_num = min(log_head_num, len(head_scalar_preds))
            for n in range(log_head_num):
                msg = textwrap.dedent(f"""
                ----------------------------------------------
                murmur scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                murmur binary prediction:    {head_bin_preds[n].tolist()}
                murmur labels:               {head_labels[n].astype(int).tolist()}
                murmur predicted classes:    {head_preds_classes[n].tolist()}
                murmur label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """)
                self.log_manager.log_message(msg)
            if "outcome" in input_tensors:
                head_scalar_preds = all_outputs[0].outcome_output.prob[:log_head_num]
                head_bin_preds = all_outputs[0].outcome_output.bin_pred[:log_head_num]
                head_preds_classes = [
                    np.array(all_outputs[0].outcome_output.classes)[np.where(row)[0]] for row in head_bin_preds
                ]
                head_labels = all_labels[0]["outcome"][:log_head_num]
                head_labels_classes = [
                    (
                        np.array(all_outputs[0].outcome_output.classes)[np.where(row)[0]]
                        if head_labels.ndim == 2
                        else np.array(all_outputs[0].outcome_output.classes)[row]
                    )
                    for row in head_labels
                ]
                log_head_num = min(log_head_num, len(head_scalar_preds))
                for n in range(log_head_num):
                    msg = textwrap.dedent(f"""
                    ----------------------------------------------
                    outcome scalar prediction:    {[round(item, 3) for item in head_scalar_preds[n].tolist()]}
                    outcome binary prediction:    {head_bin_preds[n].tolist()}
                    outcome labels:               {head_labels[n].astype(int).tolist()}
                    outcome predicted classes:    {head_preds_classes[n].tolist()}
                    outcome label classes:        {head_labels_classes[n].tolist()}
                    ----------------------------------------------
                    """)
                    self.log_manager.log_message(msg)

        eval_res = compute_challenge_metrics(
            labels=all_labels,
            outputs=all_outputs,
            require_both=False,
        )
        # eval_res contains the following items:
        # murmur_auroc: float,
        #     the macro-averaged area under the receiver operating characteristic curve for the murmur predictions
        # murmur_auprc: float,
        #     the macro-averaged area under the precision-recall curve for the murmur predictions
        # murmur_f_measure: float,
        #     the macro-averaged F-measure for the murmur predictions
        # murmur_accuracy: float,
        #     the accuracy for the murmur predictions
        # murmur_weighted_accuracy: float,
        #     the weighted accuracy for the murmur predictions
        # murmur_cost: float,
        #     the challenge cost for the murmur predictions
        # outcome_auroc: float,
        #     the macro-averaged area under the receiver operating characteristic curve for the outcome predictions
        # outcome_auprc: float,
        #     the macro-averaged area under the precision-recall curve for the outcome predictions
        # outcome_f_measure: float,
        #     the macro-averaged F-measure for the outcome predictions
        # outcome_accuracy: float,
        #     the accuracy for the outcome predictions
        # outcome_weighted_accuracy: float,
        #     the weighted accuracy for the outcome predictions
        # outcome_cost: float,
        #     the challenge cost for the outcome predictions

        weighted_cost = 0
        if eval_res.get("murmur_cost", None) is not None:
            weighted_cost += eval_res["murmur_cost"] * self.train_config[self.train_config.task].head_weights.murmur
        if eval_res.get("outcome_cost", None) is not None:
            weighted_cost += eval_res["outcome_cost"] * self.train_config[self.train_config.task].head_weights.outcome
        eval_res["neg_weighted_cost"] = -weighted_cost

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
        """ """
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
            "multi_task": "murmur",
            "classification": "murmur",
            "segmentation": "segmentation",
        }[self.train_config.task]


def collate_fn(
    batch: Sequence[Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]],
) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
    """ """
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        collated = default_collate_fn([tuple(b[k] for k in keys) for b in batch])
        return {k: collated[i] for i, k in enumerate(keys)}
    else:
        return default_collate_fn(batch)
