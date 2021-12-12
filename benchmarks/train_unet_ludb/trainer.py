"""
References
----------
[1] https://github.com/milesial/Pytorch-UNet/blob/master/train.py
"""
import os
import sys
import time
import logging
import argparse
import textwrap
from copy import deepcopy
from collections import deque, OrderedDict
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

try:
    import torch_ecg
except ModuleNotFoundError:
    import sys
    from os.path import dirname, abspath
    sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.trainer import BaseTrainer

from cfg import TrainCfg
from dataset import LUDB
from metrics import compute_metrics
from model import ECG_UNET_LUDB

if TrainCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "train",
]


class LUDBTrainer(BaseTrainer):
    """
    """
    __DEBUG__ = True
    __name__ = "LUDBTrainer"

    def __init__(self,
                 model:nn.Module,
                 model_config:dict,
                 train_config:dict,
                 device:Optional[torch.device]=None,
                 lazy:bool=True,
                 **kwargs:Any,) -> NoReturn:
        """ finished, checked,

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
        super().__init__(model, CINC2021, model_config, train_config, device, lazy)

    def _setup_dataloaders(self, train_dataset:Optional[Dataset]=None, val_dataset:Optional[Dataset]=None) -> NoReturn:
        """ finished, checked,
        
        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset
        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(config=self.train_config, training=True, lazy=False)

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(config=self.train_config, training=False, lazy=False)

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

    def run_one_step(self, *data:Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def evaluate(self, data_loader:DataLoader) -> Dict[str, float]:
        """
        """
        self.model.eval()

        all_scalar_preds = []
        all_bin_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            preds, bin_preds = self._model.inference(signals)
            all_scalar_preds.append(preds)
            all_bin_preds.append(bin_preds)
        
        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        classes = data_loader.dataset.all_classes

        if self.val_train_loader is not None:
            msg = f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}"
            self.log_manager.log_message(msg, level=logging.DEBUG)
            head_num = 5
            head_scalar_preds = all_scalar_preds[:head_num,...]
            head_bin_preds = all_bin_preds[:head_num,...]
            head_preds_classes = [np.array(classes)[np.where(row)] for row in head_bin_preds]
            head_labels = all_labels[:head_num,...]
            head_labels_classes = [np.array(classes)[np.where(row)] for row in head_labels]
            for n in range(head_num):
                msg = textwrap.dedent(f"""
                ----------------------------------------------
                scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
                binary prediction:    {head_bin_preds[n].tolist()}
                labels:               {head_labels[n].astype(int).tolist()}
                predicted classes:    {head_preds_classes[n].tolist()}
                label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """)
                self.log_manager.log_message(msg)

        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric = \
            evaluate_scores(
                classes=classes,
                truth=all_labels,
                scalar_pred=all_scalar_preds,
                binary_pred=all_bin_preds,
            )
        eval_res = dict(
            auroc=auroc,
            auprc=auprc,
            accuracy=accuracy,
            f_measure=f_measure,
            f_beta_measure=f_beta_measure,
            g_beta_measure=g_beta_measure,
            challenge_metric=challenge_metric,
        )

        # in case possible memeory leakage?
        del all_scalar_preds, all_bin_preds, all_labels

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, for CinC2021, it is 0,
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """
        """
        return []

    @property
    def save_prefix(self) -> str:
        return f"{self._model.__name__}_{self.model_config.cnn.name}_epoch"

    def extra_log_suffix(self) -> str:
        return super().extra_log_suffix() + f"_{self.model_config.cnn.name}"


def get_args(**kwargs):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on LUDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     "-l", "--learning-rate",
    #     metavar="LR", type=float, nargs="?", default=0.001,
    #     help="Learning rate",
    #     dest="learning_rate")
    parser.add_argument(
        "-b", "--batch-size",
        type=int, default=128,
        help="the batch size for training",
        dest="batch_size")
    parser.add_argument(
        "-m", "--model-name",
        type=str, default="unet",
        help="name of the model to train, `unet` or `subtract_unet`",
        dest="model_name")
    parser.add_argument(
        "--keep-checkpoint-max", type=int, default=50,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max")
    parser.add_argument(
        "--optimizer", type=str, default="adam",
        help="training optimizer",
        dest="train_optimizer")
    parser.add_argument(
        "--debug", type=str2bool, default=False,
        help="train with more debugging information",
        dest="debug")
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)



if __name__ == "__main__":
    config = get_args(**TrainCfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = init_logger(log_dir=config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {device}")
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f"with configuration {config}")
    # print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    # print(f"Using device {device}")
    # print(f"Using torch of version {torch.__version__}")
    # print(f"with configuration {config}")

    model_config = deepcopy(ECG_UNET_VANILLA_CONFIG)

    model = ECG_UNET(classes=config.classes, config=model_config)

    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        model = torch.nn.parallel.DistributedDataParallel(model)

    model.to(device=device)
    model.__DEBUG__ = False

    try:
        train(
            model=model,
            config=config,
            device=device,
            logger=logger,
            debug=config.debug,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(config.checkpoints, "INTERRUPTED.pth"))
        logger.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
