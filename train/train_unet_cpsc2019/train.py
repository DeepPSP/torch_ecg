"""
"""
import os
import sys
import time
import logging
import argparse
import textwrap
from copy import deepcopy
from collections import deque
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
# try:
#     from tqdm.auto import tqdm
# except ModuleNotFoundError:
#     from tqdm import tqdm
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

from torch_ecg.models.nets import BCEWithLogitsWithClassWeightLoss
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from .model import ECG_SUBTRACT_UNET_CPSC2019, ECG_UNET_CPSC2019
from .utils import (
    init_logger, get_date_str, dict_to_str, str2bool,
    mask_to_intervals,
)
from .cfg import ModelCfg, TrainCfg
from .dataset import CPSC2019
from .metrics import compute_metrics

if ModelCfg.torch_dtype.lower() == "double":
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
    msg = f"training configurations are as follows:\n{dict_to_str(config)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    train_dataset = CPSC2019(config=config, training=True)
    train_dataset.__DEBUG__ = False

    if debug:
        val_train_dataset = CPSC2019(config=config, training=True)
        val_train_dataset.disable_data_augmentation()
        val_train_dataset.__DEBUG__ = False
    val_dataset = CPSC2019(config=config, training=False)
    val_dataset.__DEBUG__ = False

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
        filename_suffix=f"OPT_{model.__name__}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
        comment=f"OPT_{model.__name__}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
    )

    msg = textwrap.dedent( f"""
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
        """)
    # print(msg)  # in case no logger
    if logger:
        logger.info(msg)
    else:
        print(msg)

    if config.train_optimizer.lower() == "adam":
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),  # default
            eps=1e-08,  # default
        )
    elif config.train_optimizer.lower() == "sgd":
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
    elif config.lr_scheduler.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    elif config.lr_scheduler.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step_size, config.lr_gamma)
    else:
        raise NotImplementedError(f"lr scheduler `{config.lr_scheduler.lower()}` not implemented for training")

    if config.loss == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss == "BCEWithLogitsWithClassWeightLoss":
        criterion = BCEWithLogitsWithClassWeightLoss(
            class_weight=train_dataset.class_weights.to(device=device, dtype=_DTYPE)
        )
    else:
        raise NotImplementedError(f"loss `{config.loss}` not implemented!")
    # scheduler = ReduceLROnPlateau(optimizer, mode="max", verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = f"{model.__name__}_epoch"

    saved_models = deque()
    model.train()
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{n_epochs}", ncols=100) as pbar:
            for epoch_step, (signals, labels) in enumerate(train_loader):
                global_step += 1
                signals = signals.to(device=device, dtype=_DTYPE)
                labels = labels.to(device=device, dtype=_DTYPE)

                preds = model(signals)
                loss = criterion(preds, labels)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % log_step == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    if scheduler:
                        writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        pbar.set_postfix(**{
                            "loss (batch)": loss.item(),
                            "lr": scheduler.get_lr()[0],
                        })
                        msg = f"Train step_{global_step}: loss : {loss.item()}, lr : {scheduler.get_lr()[0] * batch_size}"
                    else:
                        pbar.set_postfix(**{
                            "loss (batch)": loss.item(),
                        })
                        msg = f"Train step_{global_step}: loss : {loss.item()}"
                    # print(msg)  # in case no logger
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                pbar.update(signals.shape[0])
            
            writer.add_scalar("train/epoch_loss", epoch_loss, global_step)

            # eval for each epoch using corresponding `evaluate` function
            if debug:
                eval_train_res = evaluate(
                    model, val_train_loader, config, device, debug
                )
                writer.add_scalar("train/qrs_score", eval_train_res, global_step)

            eval_res = evaluate(
                model, val_loader, config, device, debug
            )
            model.train()
            writer.add_scalar("test/qrs_score", eval_res, global_step)

            if config.lr_scheduler is None:
                pass
            elif config.lr_scheduler.lower() == "plateau":
                scheduler.step(metrics=eval_res[6])
            elif config.lr_scheduler.lower() == "step":
                scheduler.step()
            
            if debug:
                eval_train_msg = f"""
                train/qrs_score:         {eval_train_res}
                """
            else:
                eval_train_msg = ""
            msg = textwrap.dedent(f"""
                Train epoch_{epoch + 1}:
                --------------------
                train/epoch_loss:        {epoch_loss}{eval_train_msg}
                test/qrs_score:          {eval_res}
                ---------------------------------
                """)

            # print(msg)  # in case no logger
            if logger:
                logger.info(msg)
            else:
                print(msg)

            try:
                os.makedirs(config.checkpoints, exist_ok=True)
                # if logger:
                #     logger.info("Created checkpoint directory")
            except OSError:
                pass
            save_suffix = f"epochloss_{epoch_loss:.5f}_challenge_loss(qrs_score)_{eval_res}"
            save_filename = f"{save_prefix}{epoch + 1}_{get_date_str()}_{save_suffix}.pth"
            save_path = os.path.join(config.checkpoints, save_filename)
            torch.save(model.state_dict(), save_path)
            if logger:
                logger.info(f"Checkpoint {epoch + 1} saved!")
            saved_models.append(save_path)
            # remove outdated models
            if len(saved_models) > config.keep_checkpoint_max > 0:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    logger.info(f"failed to remove {model_to_remove}")

    writer.close()



@torch.no_grad()
def evaluate(model:nn.Module, data_loader:DataLoader, config:dict, device:torch.device, debug:bool=True, logger:Optional[logging.Logger]=None) -> float:
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
    debug: bool, default True,
        more detailed evaluation output
    logger: Logger, optional,
        logger to record detailed evaluation output,
        if is None, detailed evaluation output will be printed

    Returns:
    --------
    qrs_score: float,
        evaluation results, a score defined in `compute_metrics`
    """
    model.eval()
    all_rpeak_preds = []
    all_rpeak_labels = []

    for signals, labels in data_loader:
        signals = signals.to(device=device, dtype=_DTYPE)
        labels = labels.numpy()
        labels = [mask_to_intervals(item, 1) for item in labels]  # intervals of qrs complexes
        labels = [ # to indices of rpeaks in the original signal sequence
            np.array([(itv[0]+itv[1])//2 for itv in item]) for item in labels
        ]
        labels = [
            item[np.where((item>=config.skip_dist) & (item<config.input_len-config.skip_dist))[0]] \
                for item in labels
        ]
        all_rpeak_labels += labels

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prob, rpeak_preds = model.inference(
            signals,
            bin_pred_thr=0.5,
            duration_thr=4*16,
            dist_thr=200,
            correction=False
        )
        all_rpeak_preds += rpeak_preds

    qrs_score = compute_metrics(
        rpeaks_truths=all_rpeak_labels,
        rpeaks_preds=all_rpeak_preds,
        fs=config.fs,
        thr=config.bias_thr/config.fs,
    )

    model.train()

    return qrs_score


def get_args(**kwargs):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train the Model on CINC2019",
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
    sys.path.append(os.path.dirname(__file__))
    
    config = get_args(**TrainCfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = init_logger(log_dir=config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {device}")
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f"with configuration\n{dict_to_str(config)}")
    # print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    # print(f"Using device {device}")
    # print(f"Using torch of version {torch.__version__}")
    # print(f"with configuration\n{dict_to_str(config)}")

    model_name = config.model_name.lower()
    model_config = deepcopy(ModelCfg[model_name])

    if model_name == "subtract_unet":
        model = ECG_SUBTRACT_UNET_CPSC2019(
            n_leads=config.n_leads,
            config=model_config,
        )
    elif model_name == "unet":
        model = ECG_UNET_CPSC2019(
            n_leads=config.n_leads,
            config=model_config,
        )

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
