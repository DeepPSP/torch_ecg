"""
neural architecture search
"""

from typing import NoReturn, Optional, Tuple, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from easydict import EasyDict as ED

from .trainer import BaseTrainer


__all__ = ["NAS",]


class NAS:
    """
    """
    __name__ = "NAS"

    def __init__(self,
                 trainer_cls:BaseTrainer,
                 model_cls:nn.Module,
                 dataset_cls:Dataset,
                 train_config:dict,
                 model_configs:Sequence[dict],) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        to write
        """
        self.trainer_cls = trainer_cls
        self.model_cls = model_cls
        self.dataset_cls = dataset_cls
        self.train_config = ED(train_config)
        self.model_configs = ED(model_configs)

    def search(self) -> NoReturn:
        """ finished, NOT checked,
        """
        model = self.model_cls(
            classes=self.train_config.classes,
            n_leads=self.train_config.n_leads,
            config=self.model_configs[0],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            model = DP(model)
            # model = DDP(model)
        model.to(device=device)
        model.train()

        trainer = self.trainer_cls(
            model=model,
            dataset_cls=self.dataset_cls,
            train_config=self.train_config,
            model_configs=self.model_configs[0],
            device=device,
            lazy=False,
        )
        ds_train = trainer.train_loader.dataset
        ds_val = trainer.val_loader.dataset

        trainer.train()

        del model
        del trainer
        torch.cuda.empty_cache()

        for model_config in self.model_configs[1:]:
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
                model_configs=model_config,
                device=device,
                lazy=True,
            )
            trainer._setup_train_loader(ds_train, ds_val)

            trainer.train()

            del model
            del trainer
            torch.cuda.empty_cache()
