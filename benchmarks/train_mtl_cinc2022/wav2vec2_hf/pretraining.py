"""
pretrain a Wav2Vec model from scratch
modified from:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py
"""

import re
from typing import Dict, List, Optional, Any, NoReturn

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import (  # noqa: F401
    DistributedDataParallel as DDP,
    DataParallel as DP,
)  # noqa: F401

# from accelerate import Accelerator  # noqa: F401
from transformers import (  # noqa: F401
    AdamW,
    Trainer,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from torch_ecg.components.trainer import BaseTrainer

try:
    from torch_ecg.utils import get_kwargs
except ImportError:
    from .utils import get_kwargs

from .pretraining_cfg import PreTrainCfg, PreTrainModelCfg
from .pretraining_data import get_pretraining_datacollator, Wav2Vec2PretrainingDataset
from .pretraining_models import Wav2Vec2ForPreTraining


__all__ = [
    "Wav2Vec2PreTrainingTrainer",
]


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class Wav2Vec2PreTrainingTrainer(BaseTrainer):
    """ """

    __name__ = "Wav2Vec2PreTrainingTrainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """ """
        super().__init__(
            model,
            Wav2Vec2PretrainingDataset,
            model_config,
            train_config,
            device,
            lazy,
            **kwargs,
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
        data_collator = get_pretraining_datacollator(self.model_config)

        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config,
                feature_extractor=self.model_config.get_Wav2Vec2FeatureExtractor(),
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
                feature_extractor=self.model_config.get_Wav2Vec2FeatureExtractor(),
                training=False,
                lazy=False,
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        num_workers = 4

        # https://github.com/pytorch/pytorch/issues/47445
        # bugs about pin_memory

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=data_collator,
        )

        if self.train_config.debug:
            self.val_train_loader = DataLoader(
                dataset=val_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=data_collator,
            )
        else:
            self.val_train_loader = None
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=data_collator,
        )

    def _setup_optimizer(self) -> NoReturn:
        """ """
        if re.sub("[\\-_]*", "", self.train_config.optimizer.lower()) == "hfadamw":
            # AdamW from huggingface
            optimizer_kwargs = get_kwargs(AdamW)
            optimizer_kwargs.update(
                {k: self.train_config.get(k, v) for k, v in optimizer_kwargs.items()}
            )
            optimizer_kwargs.update(dict(lr=self.lr))
            self.optimizer = AdamW(
                params=self.model.parameters(),
                **optimizer_kwargs,
            )
        else:
            super()._setup_optimizer()

    def _setup_scheduler(self) -> NoReturn:
        """ """
        if (
            self.train_config.lr_scheduler is not None
            and self.train_config.lr_scheduler.upper() in SchedulerType.__members__
        ):
            scheduler_type = SchedulerType[self.train_config.lr_scheduler.upper()]
            scheduler_kwargs = get_kwargs(get_scheduler)
            scheduler_kwargs["num_training_steps"] = (
                max(
                    1,
                    len(self.train_loader)
                    // self.train_config.gradient_accumulation_steps,
                )
                * self.train_config.n_epochs
            )
            scheduler_kwargs.update(
                {k: self.train_config.get(k, v) for k, v in scheduler_kwargs.items()}
            )
            self.scheduler = get_scheduler(
                scheduler_type,
                self.optimizer,
                **scheduler_kwargs,
            )
        else:
            super()._setup_scheduler()

    def _setup_criterion(self) -> NoReturn:
        """ """
        # the loss is computed in the model's forward function
        # and stored in the model's output's loss attribute
        pass

    def train_one_epoch(self, pbar: tqdm) -> NoReturn:
        """
        train one epoch, and update the progress bar

        Parameters
        ----------
        pbar: tqdm,
            the progress bar for training

        """
        for epoch_step, batch in enumerate(self.train_loader):
            self.model.train()
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask
                if sub_attention_mask is not None
                else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # forward
            outputs = self.run_one_step(**batch)

            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / self.train_config.gradient_accumulation_steps
            # loss.backward()

            multiply_grads(self.model.parameters(), 1 / num_losses)

            # update step
            if (
                (epoch_step + 1) % self.train_config.gradient_accumulation_steps == 0
                or epoch_step == len(self.train_loader) - 1
            ):

                # compute grad norm for monitoring
                scale = 1
                grad_norm = get_grad_norm(self.model.parameters(), scale)

                # update parameters
                if self.train_config.flooding_level > 0:
                    flood = (
                        loss - self.train_config.flooding_level
                    ).abs() + self.train_config.flooding_level
                    self.epoch_loss += loss.item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    flood.backward()
                else:
                    self.epoch_loss += loss.item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss.backward()

                if self.scheduler is not None:
                    self.scheduler.step()

                # update gumbel temperature
                gumbel_temperature = max(
                    self.train_config.max_gumbel_temperature
                    * self.train_config.gumbel_temperature_decay**self.global_step,
                    self.train_config.min_gumbel_temperature,
                )
                self._model.set_gumbel_temperature(gumbel_temperature)

                self.global_step += 1

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
            pbar.update(batch[self._model.main_input_name].shape[self.batch_dim])

    def run_one_step(self, **batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        batch: dict of Tensors,
            the data to be processed for training one step (batch)

        Returns
        -------
        outputs: dict of Tensors,
            the outputs of the model

        """
        # forward
        for k in batch:
            batch[k] = batch[k].to(self.device)
        outputs = self.model(**batch)
        return outputs

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()
        eval_res = {
            "loss": 0,
            "contrastive_loss": 0,
            "diversity_loss": 0,
            "num_losses": 0,
        }
        for step, batch in enumerate(data_loader):
            batch.pop("sub_attention_mask", None)
            outputs = self.model(**batch)

            eval_res["loss"] += outputs.loss
            eval_res["contrastive_loss"] += outputs.contrastive_loss
            eval_res["diversity_loss"] += outputs.diversity_loss
            eval_res["num_losses"] += batch["mask_time_indices"].sum()

        eval_res = {k: v / eval_res["num_losses"] for k, v in eval_res.items()}

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
        return []

    @property
    def required_train_config_fields(self) -> List[str]:
        """ """
        return [
            # "classes",
            # "monitor",  # can be None
            "n_epochs",
            "batch_size",
            "log_step",
            "optimizer",
            "lr_scheduler",
            "learning_rate",
        ] + self.extra_required_train_config_fields

    @property
    def save_prefix(self) -> str:
        return f"HF-Wav2Vec2-Pretrain-{self.model_config.model_name}"

    def extra_log_suffix(self) -> str:
        return (
            super().extra_log_suffix()
            + f"-HF-Wav2Vec2-Pretrain-{self.model_config.model_name}"
        )


def parse_args() -> dict:
    """ """
    raise NotImplementedError


if __name__ == "__main__":
    model_config = PreTrainModelCfg.get_Wav2Vec2Config()
    model = Wav2Vec2ForPreTraining(model_config)
    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)

    trainer = Wav2Vec2PreTrainingTrainer(
        model,
        # {k: v for k, v in PreTrainModelCfg.items() if not callable(v)},
        PreTrainModelCfg,
        PreTrainCfg,
    )
