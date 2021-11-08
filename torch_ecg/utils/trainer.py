"""
Abstract base class for trainers,
in order to replace the functions for classes in the training pipelines
"""

import os
import textwrap
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import NoReturn

import torch
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data.dataset import Dataset
import torch_optimizer as extra_optim

from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.misc import (
    dicts_equal, init_logger, get_date_str, dict_to_str, str2bool,
)


__all__ = ["BaseTrainer",]


class BaseTrainer(ABC):
    """
    """
    __name__ = "BaseTrainer"
    __DEFATULT_CONFIGS__ = {
        "debug": True,
        "final_model_name": None,
    }

    def __init__(self,
                 model:nn.Module,
                 dataset_cls:Dataset,
                 model_config:dict,
                 train_config:dict,
                 device:Optional[torch.device]=None,
                 logger:Optional[logging.Logger]=None,) -> NoReturn:
        """ NOT finished, NOT checked,
        """
        self.model = model
        if type(self.model).__name__ in ["DataParallel",]:
            # TODO: further consider "DistributedDataParallel"
            self._model = self.model.module
        else:
            self._model = self.model
        self.dataset_cls = dataset_cls
        self.model_config = model_config
        self.train_config = train_config
        self.logger = logger
        self.device = device or next(self._model.parameters()).device
        self.dtype = next(self._model.parameters()).dtype
        self.model.to(self.device)

        self._setup_from_config(self.train_config)

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
        """ NOT finished, NOT checked,
        """
        start_epoch = self.epoch
        for _ in range(start_epoch, self.n_epochs):
            self.epoch += 1
            # train one epoch
            self.model.train()
            self.epoch_loss = 0
            self.train_one_epoch()

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

        self.writer.close()

        if self.logger:
            for h in self.logger.handlers:
                h.close()
                self.logger.removeHandler(h)
            del self.logger
        logging.shutdown()

        return sel.best_state_dict

    @abstractmethod
    def train_one_epoch(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_from_config(self, train_config:dict) -> NoReturn:
        """

        Parameters
        ----------
        train_config: dict,
            training configuration
        """
        msg = f"training configurations are as follows:\n{dict_to_str(config)}"
        # if self.logger:
        #     self.logger.info(msg)
        # else:
        #     print(msg)
        self.train_config = ED(deepcopy(self.__DEFATULT_CONFIGS__))
        self.train_config.update(train_config)

        self.n_train = len(train_dataset)
        self.n_val = len(val_dataset)

        self.n_epochs = self.train_config.n_epochs
        self.batch_size = self.train_config.batch_size
        self.lr = self.train_config.learning_rate

        self._setup_dataloaders()

        self.writer = SummaryWriter(
            log_dir=config.log_dir,
            filename_suffix=f"OPT_{self._model.__name__}_{self.train_config.train_optimizer}_LR_{self.lr}_BS_{self.batch_size}",
            comment=f"OPT_{self._model.__name__}_{self.train_config.train_optimizer}_LR_{self.lr}_BS_{self.batch_size}",
        )

        msg = textwrap.dedent(f"""
            Starting training:
            ------------------
            Epochs:          {n_epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Device:          {device.type}
            Optimizer:       {config.train_optimizer}
            Dataset classes: {train_dataset.all_classes}
            Class weights:   {train_dataset.class_weights}
            -----------------------------------------
            """)
        # if logger:
        #     logger.info(msg)
        # else:
        #     print(msg)

        self._setup_optimizer()

        self._setup_scheduler()

        self._setup_criterion()

        self.save_prefix = f"{self._model.__name__}_epoch"

        os.makedirs(self.train_config.checkpoints, exist_ok=True)
        os.makedirs(self.train_config.model_dir, exist_ok=True)

    def _setup_dataloaders(self) -> NoReturn:
        """ NOT finished, NOT checked,
        """
        train_dataset = self.dataset_cls(config=config, training=True)

        if self.train_config.debug:
            val_train_dataset = self.dataset_cls(config=config, training=True)
            val_train_dataset.disable_data_augmentation()
        val_dataset = self.dataset_cls(config=config, training=False)

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
        """ NOT finished, NOT checked,
        """
        if self.train_config.train_optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                params=self.model.parameters(),
                lr=self.lr,
                betas=self.train_config.betas,
                eps=1e-08,  # default
            )
        elif self.train_config.train_optimizer.lower() in ["adamw", "adamw_amsgrad"]:
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
                betas=self.train_config.betas,
                weight_decay=self.train_config.decay,
                eps=1e-08,  # default
                amsgrad=self.train_config.train_optimizer.lower().endswith("amsgrad"),
            )
        elif self.train_config.train_optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=self.train_config.momentum,
                weight_decay=self.train_config.decay,
            )
        else:
            raise NotImplementedError(f"optimizer `{self.train_config.train_optimizer}` not implemented!")

    def _setup_scheduler(self) -> NoReturn:
        """ NOT finished, NOT checked,
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
        # scheduler = ReduceLROnPlateau(optimizer, mode="max", verbose=True, patience=6, min_lr=1e-7)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    def _setup_criterion(self) -> NoReturn:
        """ NOT finished, NOT checked,
        """
        if self.train_config.loss == "BCEWithLogitsLoss":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.train_config.loss == "BCEWithLogitsWithClassWeightLoss":
            self.criterion = BCEWithLogitsWithClassWeightLoss(
                class_weight=train_dataset.class_weights.to(device=self.device, dtype=self.dtype)
            )
        elif config.loss == "BCELoss":
            self.criterion = nn.BCELoss()
        elif config.loss == "MaskedBCEWithLogitsLoss":
            self.criterion = MaskedBCEWithLogitsLoss()
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
        """ NOT finished, NOT checked,

        resume a training process from a checkpoint

        Parameters
        ----------
        checkpoint: str or dict,
            if is str, the path of the checkpoint, which is a `.pth.tar` file containing a dict,
            `checkpoint` should contain "model_state_dict", "optimizer_state_dict", "model_config", "train_config", "epoch"
            to resume a training process
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if isinstance(checkpoint, str):
            ckpt = torch.load(checkpoint, map_location=device)
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
        """
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.config,
            "epoch": self.epoch+1,
        }, path)
