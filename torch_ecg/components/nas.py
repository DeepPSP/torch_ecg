"""
neural architecture search

"""

from typing import NoReturn, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader  # noqa: F401
from torch.utils.data.dataset import Dataset

from ..cfg import CFG
from .trainer import BaseTrainer


__all__ = [
    "NAS",
]


class NAS:
    """ """

    __name__ = "NAS"

    def __init__(
        self,
        trainer_cls: BaseTrainer,
        model_cls: nn.Module,
        dataset_cls: Dataset,
        train_config: dict,
        model_configs: Sequence[dict],
        lazy: bool = False,
    ) -> NoReturn:
        """

        Parameters
        ----------
        trainer_cls: BaseTrainer,
            trainer class
        model_cls: nn.Module,
            model class
        dataset_cls: Dataset,
            dataset class
        train_config: dict,
            train configurations
        model_configs: sequence of dict,
            model configurations, each with a different network architecture
        lazy: bool, default False,
            whether to load the dataset in the trainer at initialization

        """
        self.trainer_cls = trainer_cls
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls
        self.train_config = CFG(train_config)
        self.model_configs = model_configs
        self.lazy = lazy
        if not lazy:
            self.ds_train = self.dataset_cls(
                self.train_config, training=True, lazy=False
            )
            self.ds_val = self.dataset_cls(
                self.train_config, training=False, lazy=False
            )
        else:
            self.ds_train = None
            self.ds_val = None

    def search(self) -> NoReturn:
        """ """
        if self.ds_train is None or self.ds_val is None:
            raise ValueError("training dataset or validation dataset is not set")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_config in self.model_configs:
            model = self.model_cls(
                classes=self.train_config.classes,
                n_leads=self.train_config.n_leads,
                config=model_config,
            )
            if torch.cuda.device_count() > 1:
                model = DP(model)
                # model = DDP(model)
            model.to(device=device)
            model.train()
            trainer = self.trainer_cls(
                model=model,
                dataset_cls=self.dataset_cls,
                train_config=self.train_config,
                model_config=model_config,
                device=device,
                lazy=True,
            )
            trainer._setup_dataloaders(self.ds_train, self.ds_val)
            trainer.train()

            del model
            del trainer
            torch.cuda.empty_cache()

    def _setup_dataset(self, ds_train: Dataset, ds_val: Dataset) -> NoReturn:
        """

        Parameters
        ----------
        ds_train: Dataset,
            training dataset
        ds_val: Dataset,
            validation dataset

        """
        self.ds_train = ds_train
        self.ds_val = ds_val
