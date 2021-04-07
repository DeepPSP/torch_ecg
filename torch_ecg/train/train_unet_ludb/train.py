"""
References:
-----------
[1] https://github.com/milesial/Pytorch-UNet/blob/master/train.py
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
from torch.nn import CrossEntropyLoss
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED

from torch_ecg.models import ECG_UNET
# from models.utils.torch_utils import BCEWithLogitsWithClassWeightLoss
from torch_ecg.models.nets import default_collate_fn as collate_fn
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG
from torch_ecg.utils.misc import init_logger, get_date_str, dict_to_str, str2bool
from .cfg import TrainCfg
from .dataset import LUDB
from .metrics import compute_metrics

if TrainCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "train",
]


def train(model:nn.Module, device:torch.device, config:dict, log_step:int=20, logger:Optional[logging.Logger]=None, debug:bool=False):
    """
    """
    msg = f"training configurations are as follows:\n{dict_to_str(config)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    train_dataset = LUDB(config=config, training=True)

    if debug:
        val_train_dataset = LUDB(config=config, training=True)
        val_train_dataset.disable_data_augmentation()
    val_dataset = LUDB(config=config, training=False)

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
        drop_last=True,  # setting False would result in error
        collate_fn=collate_fn,
    )

    if debug:
        val_train_loader = DataLoader(
            dataset=val_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,  # setting False would result in error
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,  # setting False would result in error
        collate_fn=collate_fn,
    )

    writer = SummaryWriter(
        log_dir=config.log_dir,
        filename_suffix=f"OPT_{model.__name__}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
        comment=f"OPT_{model.__name__}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
    )
    
    # max_itr = n_epochs * n_train

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
        scheduler = None
    elif config.train_optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step_size, config.lr_gamma)
    elif config.train_optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            params=model.parameters,
            lr=lr,
            alpha=0.99,  # default
            eps=1e-08,  # default
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min" if model.n_classes > 1 else "max", patience=2)
    else:
        raise NotImplementedError(f"optimizer `{config.train_optimizer}` not implemented!")
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    if config.loss == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"loss `{config.loss}` not implemented!")

    save_prefix = f"{model.__name__}_epoch"

    saved_models = deque()
    model.train()
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{n_epochs}", ncols=110) as pbar:
            for epoch_step, (signals, truth_masks) in enumerate(train_loader):
                global_step += 1
                signals = signals.to(device=device, dtype=torch.float64)
                truth_masks = truth_masks.to(device=device, dtype=torch.float64)
                
                optimizer.zero_grad()

                pred_masks = model(signals)
                loss = criterion(pred_masks, truth_masks)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                epoch_loss += loss.item()

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

            if debug:
                eval_train_res = evaluate(model, val_train_loader, config, device, debug)
                for wave in ["pwave", "qrs", "twave",]:
                    for term in ["onset", "offset"]:
                        for metric in ["sensitivity", "precision", "f1_score", "mean_error", "standard_deviation"]:
                            scalar_name = f"{wave}_{term}_{metric}"
                            scalar = eval(f"eval_train_res.{wave}_{term}.{metric}")
                            writer.add_scalar(f"train/{scalar_name}", scalar, global_step)
            
            eval_res = evaluate(model, val_loader, config, device, debug)
            model.train()
            for wave in ["pwave", "qrs", "twave",]:
                for term in ["onset", "offset"]:
                    for metric in ["sensitivity", "precision", "f1_score", "mean_error", "standard_deviation"]:
                        scalar_name = f"{wave}_{term}_{metric}"
                        scalar = eval(f"eval_res.{wave}_{term}.{metric}")
                        writer.add_scalar(f"test/{scalar_name}", scalar, global_step)

            try:
                os.makedirs(config.checkpoints, exist_ok=True)
                # if logger:
                #     logger.info("Created checkpoint directory")
            except OSError:
                pass
            save_suffix = f"epochloss_{epoch_loss:.5f}"
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
def evaluate(model:nn.Module, data_loader:DataLoader, config:dict, device:torch.device, debug:bool=True, logger:Optional[logging.Logger]=None) -> Tuple[float]:
    """ NOT finished, NOT checked,

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
    eval_res: tuple of float,
        evaluation results, including
        sensitivity, precision, f1_score, mean_error, standard_deviation, etc.
    """
    model.eval()
    data_loader.dataset.disable_data_augmentation()

    all_masks_pred = []
    all_masks_truth = []

    for signals, masks_truth in data_loader:
        signals = signals.to(device=device, dtype=torch.float64)
        masks_truth = masks_truth.numpy()
        all_masks_truth.append(masks_truth)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        masks_pred, _ = model.inference(signals)
        all_masks_pred.append(masks_pred.cpu().detach().numpy())
    
    # all_masks_pred = np.concatenate(all_masks_pred, axis=0)
    # all_masks_truth = np.concatenate(all_masks_truth, axis=0)

    eval_res = compute_metrics(
        truth_masks=all_masks_truth,
        pred_masks=all_masks_pred,
        class_map=config.class_map,
        fs=config.fs,
        mask_format="channel_first",
    )
    model.train()

    return eval_res


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
    if not DAS:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda")
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
