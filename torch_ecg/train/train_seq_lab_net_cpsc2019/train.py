"""
"""
import os
import sys
import time
import logging
import argparse
from copy import deepcopy
from collections import deque
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
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED
import biosppy.signals.ecg as BSE

from ...models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from ...models.nets import (
    BCEWithLogitsWithClassWeightLoss,
    default_collate_fn as collate_fn,
)
from ...model_configs import ECG_SEQ_LAB_NET_CONFIG
from ...utils.misc import init_logger, get_date_str, dict_to_str, str2bool
from .cfg import ModelCfg, TrainCfg
from .dataset import CPSC2019
from .metrics import compute_metrics

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


__all__ = [
    "train",
]


def train(model:nn.Module, device:torch.device, config:dict, log_step:int=20, logger:Optional[logging.Logger]=None, debug:bool=False) -> NoReturn:
    """ finished, checked,

    Parameters:
    -----------
    model: Module,
        the model to train
    device: torch.device,
        device on which the model trains
    config: dict,
        configurations of training, ref. `ModelCfg`, `TrainCfg`, etc.
    log_step: int, default 20,
        number of training steps between loggings
    logger: Logger, optional,
    debug: bool, default False,
        if True, the training set itself would be evaluated 
        to check if the model really learns from the training set
    """
    print(f"training configurations are as follows:\n{dict_to_str(config)}")

    train_dataset = CPSC2019(config=config, training=True)

    if debug:
        val_train_dataset = CPSC2019(config=config, training=True)
        val_train_dataset.disable_data_augmentation()
    val_dataset = CPSC2019(config=config, training=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    n_epochs = config.n_epochs
    batch_size = config.batch_size
    lr = config.learning_rate

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if debug:
        val_train_loader = DataLoader(
            dataset=val_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    writer = SummaryWriter(
        log_dir=config.log_dir,
        filename_suffix=f"OPT_{model.__name__}_{config.cnn_name}_{config.rnn_name}_{config.attn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
        comment=f"OPT_{model.__name__}_{config.cnn_name}_{config.rnn_name}_{config.attn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
    )

    msg = f'''
        Starting training:
        ------------------
        Epochs:          {n_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Optimizer:       {config.train_optimizer}
        -----------------------------------------
    '''
    print(msg)  # in case no logger
    if logger:
        logger.info(msg)

    if config.train_optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),  # default
            eps=1e-08,  # default
        )
    elif config.train_optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    else:
        raise NotImplementedError(f"optimizer `{config.train_optimizer}` not implemented!")
    
    if config.lr_scheduler is None:
        scheduler = None
    elif config.lr_scheduler.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    elif config.lr_scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step_size, config.lr_gamma)
    else:
        raise NotImplementedError("lr scheduler `{config.lr_scheduler.lower()}` not implemented for training")
    raise NotImplementedError


@torch.no_grad()
def evaluate(model:nn.Module, data_loader:DataLoader, config:dict, device:torch.device, debug:bool=False) -> Tuple[float]:
    """ finished, checked,

    Parameters:
    -----------
    model: Module,
        the model to evaluate
    data_loader: DataLoader,
        the data loader for loading data for evaluation
    config: dict,
        evaluation configurations
    device: torch.device,
        device for evaluation
    debug: bool, default False

    Returns:
    --------
    eval_res: tuple of float,
        evaluation results, including
        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric
    """
    raise NotImplementedError


def get_args(**kwargs):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description='Train the Model on CINC2020',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-l', '--learning-rate',
    #     metavar='LR', type=float, nargs='?', default=0.001,
    #     help='Learning rate',
    #     dest='learning_rate')
    parser.add_argument(
        '-b', '--batch-size',
        type=int, default=128,
        help='the batch size for training',
        dest='batch_size')
    parser.add_argument(
        '-c', '--cnn-name',
        type=str, default='resnet',
        help='choice of cnn feature extractor',
        dest='cnn_name')
    parser.add_argument(
        '-r', '--rnn-name',
        type=str, default='lstm',
        help='choice of rnn structures',
        dest='rnn_name')
    parser.add_argument(
        '-a', '--attn-name',
        type=str, default='se',
        help='choice of attention block',
        dest='attn_name')
    parser.add_argument(
        '--keep-checkpoint-max', type=int, default=20,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    parser.add_argument(
        '--optimizer', type=str, default='adam',
        help='training optimizer',
        dest='train_optimizer')
    parser.add_argument(
        '--debug', type=str2bool, default=False,
        help='train with more debugging information',
        dest='debug')
    
    args = vars(parser.parse_args())

    cfg.update(args)
    
    return ED(cfg)



DAS = True  # JD DAS platform

if __name__ == "__main__":
    config = get_args(**TrainCfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    if not DAS:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda')
    logger = init_logger(log_dir=config.log_dir)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f'Using device {device}')
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f'with configuration\n{dict_to_str(config)}')
    print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    print(f'Using device {device}')
    print(f"Using torch of version {torch.__version__}")
    print(f'with configuration\n{dict_to_str(config)}')

    model_config = deepcopy(ECG_SEQ_LAB_NET_CONFIG)
    model_config.cnn.name = config.cnn_name
    model_config.rnn.name = config.rnn_name
    model_config.attn.name = config.attn_name

    model = ECG_SEQ_LAB_NET(
        classes=config.classes,
        n_leads=config.n_leads,
        input_len=config.input_len,
        config=model_config,
    )

    if not DAS and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device=device)

    try:
        train(
            model=model,
            config=config,
            device=device,
            logger=logger,
            debug=config.debug,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(config.checkpoints, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
