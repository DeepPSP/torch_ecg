"""
Abstract base class for trainers,
in order to replace the functions for classes in the training pipelines
"""

from abc import ABC, abstractmethod
from typing import NoReturn

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.misc import dicts_equal


__all__ = ["Trainer",]


class Trainer(ABC):
    """
    """
    __name__ = "Trainer"

    @abstractmethod
    def train(self, model:nn.Module) -> NoReturn:
        """
        """
        raise NotImplementedError

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
    def _setup_from_config(self) -> NoReturn:
        """
        """
        raise NotImplementedError

    def _check_model_config_compatability(self, model_config:dict) -> bool:
        """

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
        """

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
        self.config = ckpt["train_config"]
        self._setup_from_config()

    def save_checkpoint(self, path:str) -> NoReturn:
        """
        """
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_config,
            "train_config": self.config,
            "epoch": self.epoch+1,
        }, path)
