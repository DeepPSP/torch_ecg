"""
Abstract base class for trainers,
in order to replace the functions for classes in the training pipelines
"""

import os
import textwrap
from copy import deepcopy
from abc import ABC, abstractmethod
from collections import deque, OrderedDict
from typing import NoReturn, Optional, Union, Tuple, Dict

import numpy as np
np.set_printoptions(precision=5, suppress=True)
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch_optimizer as extra_optim
from easydict import EasyDict as ED

from .utils_nn import default_collate_fn as collate_fn
from .misc import (
    dicts_equal, init_logger, get_date_str, dict_to_str, str2bool,
)
from .loggers import LoggerManager
from ..augmenters import AugmenterManager
from ..models.loss import (
    BCEWithLogitsWithClassWeightLoss,
    MaskedBCEWithLogitsLoss,
    FocalLoss, AsymmetricLoss,
)


__all__ = ["BaseTrainer",]


class BaseTrainer(ABC):
    """
    """
    __name__ = "BaseTrainer"
    __DEFATULT_CONFIGS__ = {
        "debug": True,
        "final_model_name": None,
        "log_step": 10,
        "flooding_level": 0,
        "early_stopping": {},
    }

    def __init__(self,
                 model:nn.Module,
                 dataset_cls:Dataset,
                 model_config:dict,
                 train_config:dict,
                 device:Optional[torch.device]=None,) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        model: Module,
            the model to be trained
        dataset_cls: Dataset,
            the class of dataset to be used for training,
            `dataset_cls` should be inherited from `torch.utils.data.Dataset`,
            and be initialized like `dataset_cls(config, training=True)`,
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
            the device to be used for training,
        """
        self.model = model
        if type(self.model).__name__ in ["DataParallel",]:
            # TODO: further consider "DistributedDataParallel"
            self._model = self.model.module
        else:
            self._model = self.model
        self.dataset_cls = dataset_cls
        self.model_config = ED(deepcopy(model_config))
        self._train_config = ED(deepcopy(train_config))
        self.device = device or next(self._model.parameters()).device
        self.dtype = next(self._model.parameters()).dtype
        self.model.to(self.device)

        self.log_manager = None
        self.augmenter_manager = None
        self._setup_from_config(self._train_config)

        # monitor for training: challenge metric
        self.best_state_dict = OrderedDict()
        self.best_metric = -np.inf
        self.best_eval_res = dict()
        self.best_epoch = -1
        self.pseudo_best_epoch = -1

        self.saved_models = deque()
        self.model.train()
        self.global_step = 0
        self.epoch = 0
        self.epoch_loss = 0

    def train(self) -> OrderedDict:
        """ finished, NOT checked,
        """
        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            # train one epoch
            self.model.train()
            self.epoch_loss = 0
            with tqdm(total=self.n_train, desc=f"Epoch {self.epoch}/{self.n_epochs}", unit="signals") as pbar:
                self.log_manager.epoch_start(self.epoch)
                # train one epoch
                self.train_one_epoch(pbar)

                # evaluate on train set, if debug is True
                if self.train_config.debug:
                    eval_train_res = self.evaluate(self.val_train_loader)
                    self.log_manager.log_metrics(
                        metrics=eval_train_res,
                        step=self.global_step,
                        epoch=self.epoch,
                        part="train",
                    )
                # evaluate on val set
                eval_res = self.evaluate(self.val_loader)
                self.log_manager.log_metrics(
                    metrics=eval_res,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="val",
                )

                # update best model and best metric
                if eval_res[self.train_config.monitor] > self.best_metric:
                    self.best_metric = eval_res[self.train_config.monitor]
                    self.best_state_dict = self._model.state_dict()
                    self.best_eval_res = deepcopy(eval_res)
                    self.best_epoch = self.epoch
                    self.pseudo_best_epoch = self.epoch
                elif self.train_config.early_stopping:
                    if eval_res[self.train_config.monitor] >= self.best_metric - self.train_config.early_stopping.min_delta:
                        self.pseudo_best_epoch = self.epoch
                    elif self.epoch - self.pseudo_best_epoch >= self.train_config.early_stopping.patience:
                        msg = f"early stopping is triggered at epoch {self.epoch}"
                        self.log_manager.log_message(msg)
                        break

                msg = textwrap.dedent(f"""
                    best metric = {self.best_metric},
                    obtained at epoch {self.best_epoch}
                """)
                self.log_manager.log_message(msg)

                # save checkpoint
                save_suffix = f"epochloss_{self.epoch_loss:.5f}_metric_{eval_res[self.train_config.monitor]:.2f}"
                save_filename = f"{self.save_prefix}{self.epoch}_{get_date_str()}_{save_suffix}.pth.tar"
                save_path = os.path.join(self.train_config.checkpoints, save_filename)
                self.save_checkpoint(save_path)

                # update learning rate using lr_scheduler
                self._update_lr(eval_res)

                self.log_manager.epoch_end(self.epoch)

            self.epoch += 1

        # save the best model
        if self.best_metric > -np.inf:
            if self.train_config.final_model_name:
                save_filename = self.train_config.final_model_name
            else:
                save_suffix = f"metric_{self.best_eval_res[self.train_config.monitor]:.2f}"
                save_filename = f"BestModel_{self.save_prefix}{self.best_epoch}_{get_date_str()}_{save_suffix}.pth.tar"
            save_path = os.path.join(self.train_config.model_dir, save_filename)
            self.save_checkpoint(path=save_path)
        else:
            raise ValueError("No best model found!")

        self.log_manager.close()

        return self.best_state_dict

    def train_one_epoch(self, pbar:tqdm) -> NoReturn:
        """ finished, NOT checked,

        train one epoch, and update the progress bar

        Parameters
        ----------
        pbar: tqdm,
            the progress bar for training
        """
        for epoch_step, data in enumerate(self.train_loader):
            self.global_step += 1
            # data is assumed to be a tuple of tensors, of the following order:
            # signals, labels, *extra_tensors
            data = self.augmenter_manager(*data)
            out_tensors = self.run_one_step(*data)

            loss = self.criterion(*out_tensors).to(self.dtype)
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

            if self.global_step % self.train_config.log_step == 0:
                train_step_metrics = {"loss": loss.item()}
                if self.scheduler:
                    train_step_metrics.update({"lr": self.scheduler.get_lr()[0]})
                    pbar.set_postfix(**{
                        "loss (batch)": loss.item(),
                        "lr": self.scheduler.get_lr()[0],
                    })
                else:
                    pbar.set_postfix(**{
                        "loss (batch)": loss.item(),
                    })
                if self.train_config.flooding_level > 0:
                    train_step_metrics.update({"flood": flood.item()})
                self.log_manager.log_metrics(
                    metrics=train_step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(data[0].shape[self.batch_dim])

    @property
    @abstractmethod
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        raise NotImplementedError

    @property
    def save_prefix(self) -> str:
        return f"{self._model.__name__}_epoch"

    @property
    def train_config(self) -> ED:
        """
        """
        return self._train_config

    @abstractmethod
    def run_one_step(self, *data:Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        run one step of training on one batch of data,

        Parameters
        ----------
        data: tuple of Tensors,
            the data to be processed for training one step (batch),
            should be of the following order:
            signals, labels, *extra_tensors

        Returns
        -------
        tuple of Tensors,
            the output of the model for one step (batch) data,
            along with labels and extra tensors,
            should be of the following order:
            preds, labels, *extra_tensors,
            preds usually are NOT the logits,
            but tensors before fed into `sigmoid` or `softmax` to get the logits
        """
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def evaluate(self, data_loader:DataLoader) -> Dict[str, float]:
        """
        do evaluation on the given data loader

        Parameters
        ----------
        data_loader: DataLoader,
            the data loader to evaluate on

        Returns
        -------
        dict,
            the evaluation results (metrics)
        """
        raise NotImplementedError

    def _update_lr(self, eval_res:dict) -> NoReturn:
        """ finished, NOT checked,

        update learning rate using lr_scheduler, perhaps based on the eval_res

        Parameters
        ----------
        eval_res: dict,
            the evaluation results (metrics)
        """
        if self.train_config.lr_scheduler is None:
            pass
        elif self.train_config.lr_scheduler.lower() == "plateau":
            metrics = eval_res[self.train_config.monitor]
            if isinstance(metrics, torch.Tensor):
                metrics = metrics.item()
            self.scheduler.step(metrics)
        elif self.train_config.lr_scheduler.lower() == "step":
            self.scheduler.step()
        elif self.train_config.lr_scheduler.lower() in ["one_cycle", "onecycle",]:
            self.scheduler.step()

    def _setup_from_config(self, train_config:dict) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        train_config: dict,
            training configuration
        """
        _default_config = ED(deepcopy(self.__DEFATULT_CONFIGS__))
        _default_config.update(train_config)
        self._train_config = ED(deepcopy(_default_config))
        if self.train_config.get("model_dir", None):
            self._train_config.model_dir = self.train_config.checkpoints

        self.n_epochs = self.train_config.n_epochs
        self.batch_size = self.train_config.batch_size
        self.lr = self.train_config.learning_rate

        self._setup_dataloaders()

        self.n_train = len(self.train_loader.dataset)
        self.n_val = len(self.val_loader.dataset)

        self._setup_log_manager()

        msg = f"training configurations are as follows:\n{dict_to_str(self.train_config)}"
        self.log_manager.log_message(msg)

        msg = textwrap.dedent(f"""
            Starting training:
            ------------------
            Epochs:          {self.n_epochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.lr}
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Device:          {self.device.type}
            Optimizer:       {self.train_config.optimizer}
            Dataset classes: {self.train_loader.dataset.all_classes}
            Class weights:   {self.train_loader.dataset.class_weights}
            -----------------------------------------
            """)
        self.log_manager.log_message(msg)

        self._setup_augmenter_manager()

        self._setup_optimizer()

        self._setup_scheduler()

        self._setup_criterion()

        os.makedirs(self.train_config.checkpoints, exist_ok=True)
        os.makedirs(self.train_config.model_dir, exist_ok=True)

    def extra_log_suffix(self) -> str:
        """
        """
        return f"{self._model.__name__}_{self.train_config.optimizer}_LR_{self.lr}_BS_{self.batch_size}"

    def _setup_log_manager(self) -> NoReturn:
        """ finished, NOT checked,
        """
        config = {"log_suffix": self.extra_log_suffix()}
        config.update(self.train_config)
        self.log_manager = LoggerManager.from_config(config=config)

    def _setup_augmenter_manager(self) -> NoReturn:
        """ finished, NOT checked,
        """
        self.augmenter_manager = AugmenterManager.from_config(config=self.train_config)

    def _setup_dataloaders(self) -> NoReturn:
        """ finished, NOT checked,
        """
        train_dataset = self.dataset_cls(config=self.train_config, training=True)

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        val_dataset = self.dataset_cls(config=self.train_config, training=False)

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

    def _setup_optimizer(self) -> NoReturn:
        """ finished, NOT checked,
        """
        if self.train_config.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=self.lr,
            )
        elif self.train_config.optimizer.lower() in ["adamw", "adamw_amsgrad"]:
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
                weight_decay=self.train_config.decay,
                eps=1e-08,  # default
                amsgrad=self.train_config.optimizer.lower().endswith("amsgrad"),
            )
        elif self.train_config.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=self.train_config.momentum,
                weight_decay=self.train_config.decay,
            )
        else:
            raise NotImplementedError(f"optimizer `{self.train_config.optimizer}` not implemented!")

    def _setup_scheduler(self) -> NoReturn:
        """ finished, NOT checked,
        """
        if self.train_config.lr_scheduler is None:
            self.scheduler = None
        elif self.train_config.lr_scheduler.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "max", patience=2)
        elif self.train_config.lr_scheduler.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, self.train_config.lr_step_size, self.train_config.lr_gamma
            )
        elif self.train_config.lr_scheduler.lower() in ["one_cycle", "onecycle",]:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.train_config.max_lr,
                epochs=self.n_epochs,
                steps_per_epoch=len(self.train_loader),
            )
        else:
            raise NotImplementedError(f"lr scheduler `{self.train_config.lr_scheduler.lower()}` not implemented for training")

    def _setup_criterion(self) -> NoReturn:
        """ finished, NOT checked,
        """
        loss_kw = self.train_config.get("loss_kw", {})
        for k, v in loss_kw.items():
            if isinstance(v, torch.Tensor):
                loss_kw[k] = v.to(device=self.device, dtype=self.dtype)
        if self.train_config.loss == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss == "BCEWithLogitsWithClassWeightLoss":
            self.criterion = BCEWithLogitsWithClassWeightLoss(**loss_kw)
        elif self.train_config.loss == "BCELoss":
            self.criterion = nn.BCELoss(**loss_kw)
        elif self.train_config.loss == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss(**loss_kw)
        elif self.train_config.loss == "FocalLoss":
            self.criterion = FocalLoss(**loss_kw)
        elif self.train_config.loss == "AsymmetricLoss":
            self.criterion = AsymmetricLoss(**loss_kw)
        else:
            raise NotImplementedError(f"loss `{self.train_config.loss}` not implemented!")

    def _check_model_config_compatability(self, model_config:dict) -> bool:
        """ finished, NOT checked,

        Parameters
        ----------
        model_config: dict,
            model configuration from elsewhere (e.g. from a checkpoint),
            which should be compatible with the current model configuration

        Returns
        -------
        bool, True if compatible, False otherwise
        """
        return dicts_equal(self.model_config, model_config)

    def resume_from_checkpoint(self, checkpoint:Union[str,dict]) -> NoReturn:
        """ finished, NOT checked,

        resume a training process from a checkpoint

        Parameters
        ----------
        checkpoint: str or dict,
            if is str, the path of the checkpoint, which is a `.pth.tar` file containing a dict,
            `checkpoint` should contain "model_state_dict", "optimizer_state_dict", "model_config", "train_config", "epoch"
            to resume a training process
        """
        if isinstance(checkpoint, str):
            ckpt = torch.load(checkpoint, map_location=self.device)
        else:
            ckpt = checkpoint
        insufficient_msg = "this checkpoint has no sufficient data to resume training"
        assert isinstance(ckpt, dict), insufficient_msg
        assert set(["model_state_dict", "optimizer_state_dict", "model_config", "train_config", "epoch",]).issubset(ckpt.keys()), \
            insufficient_msg
        if not self._check_model_config_compatability(self, ckpt["model_config"]):
            raise ValueError("model config of the checkpoint is not compatible with the config of the current model")
        self._model.load_state_dict(ckpt["model_state_dict"])
        self.epoch = ckpt["epoch"]
        self._setup_from_config(ckpt["train_config"])
        # TODO: resume optimizer, etc.

    def save_checkpoint(self, path:str) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        path: str,
            path to save the checkpoint
        """
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.train_config,
            "epoch": self.epoch,
        }, path)
