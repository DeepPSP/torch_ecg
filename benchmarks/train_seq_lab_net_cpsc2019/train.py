"""
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
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from tensorboardX import SummaryWriter
from easydict import EasyDict as ED
import biosppy.signals.ecg as BSE

from torch_ecg.models.loss import BCEWithLogitsWithClassWeightLoss
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from train.train_seq_lab_net_cpsc2019.model import ECG_SEQ_LAB_NET_CPSC2019
from train.train_seq_lab_net_cpsc2019.utils import (
    init_logger, get_date_str, dict_to_str, str2bool,
    mask_to_intervals,
)
from train.train_seq_lab_net_cpsc2019.cfg import ModelCfg, TrainCfg
from train.train_seq_lab_net_cpsc2019.dataset import CPSC2019
from train.train_seq_lab_net_cpsc2019.metrics import compute_metrics

if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


__all__ = [
    "train",
]


def train(model:nn.Module,
          model_config:dict,
          device:torch.device,
          config:dict,
          logger:Optional[logging.Logger]=None,
          debug:bool=False) -> OrderedDict:
    """ finished, checked,

    Parameters
    ----------
    model: Module,
        the model to train
    model_config: dict,
        config of the model, to store into the checkpoints
    device: torch.device,
        device on which the model trains
    config: dict,
        configurations of training, ref. `ModelCfg`, `TrainCfg`, etc.
    logger: Logger, optional,
        logger
    debug: bool, default False,
        if True, the training set itself would be evaluated 
        to check if the model really learns from the training set

    Returns
    -------
    best_state_dict: OrderedDict,
        state dict of the best model
    """
    msg = f"training configurations are as follows:\n{dict_to_str(config)}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    if type(model).__name__ in ["DataParallel",]:  # TODO: further consider "DistributedDataParallel"
        _model = model.module
    else:
        _model = model

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

    # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
    num_workers = 4 * (torch.cuda.device_count() or 1)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if debug:
        val_train_loader = DataLoader(
            dataset=val_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    writer = SummaryWriter(
        log_dir=config.log_dir,
        filename_suffix=f"OPT_{_model.__name__}_{config.cnn_name}_{config.rnn_name}_{config.attn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
        comment=f"OPT_{_model.__name__}_{config.cnn_name}_{config.rnn_name}_{config.attn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
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
            betas=config.betas,
            eps=1e-08,  # default
        )
    elif config.train_optimizer.lower() in ["adamw", "adamw_amsgrad"]:
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=lr,
            betas=config.betas,
            weight_decay=config.decay,
            eps=1e-08,  # default
            amsgrad=config.train_optimizer.lower().endswith("amsgrad"),
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
    elif config.lr_scheduler.lower() in ["one_cycle", "onecycle",]:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.max_lr,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
        )
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

    save_prefix = f"{_model.__name__}_{config.cnn_name}_{config.rnn_name}_epoch"

    os.makedirs(config.checkpoints, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    # monitor for training: challenge metric
    # TODO: add early_stopping using this monitor
    best_state_dict = OrderedDict()
    best_challenge_metric = -np.inf
    best_eval_res = tuple()
    best_epoch = -1
    pseudo_best_epoch = -1

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
                if config.flooding_level > 0:
                    flood = (loss - config.flooding_level).abs() + config.flooding_level
                    epoch_loss += loss.item()
                    optimizer.zero_grad()
                    flood.backward()
                else:
                    epoch_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()

                if global_step % config.log_step == 0:
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
                scheduler.step(metrics=eval_res)
            elif config.lr_scheduler.lower() == "step":
                scheduler.step()
            elif config.lr_scheduler.lower() in ["one_cycle", "onecycle",]:
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
            
            monitor = eval_res
            if monitor > best_challenge_metric:
                best_challenge_metric = monitor
                best_state_dict = _model.state_dict()
                best_eval_res = deepcopy(eval_res)
                best_epoch = epoch + 1
                pseudo_best_epoch = epoch + 1
            elif config.early_stopping:
                if monitor >= best_challenge_metric - config.early_stopping.min_delta:
                    pseudo_best_epoch = epoch + 1
                elif epoch - pseudo_best_epoch > config.early_stopping.patience:
                    msg = f"early stopping is triggered at epoch {epoch + 1}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                    break

            try:
                os.makedirs(config.checkpoints, exist_ok=True)
                # if logger:
                #     logger.info("Created checkpoint directory")
            except OSError:
                pass
            save_suffix = f"epochloss_{epoch_loss:.5f}_challenge_score(qrs_score)_{eval_res}"
            save_filename = f"{save_prefix}{epoch + 1}_{get_date_str()}_{save_suffix}.pth.tar"
            save_path = os.path.join(config.checkpoints, save_filename)
            torch.save({
                "model_state_dict": _model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "train_config": config,
                "epoch": epoch+1,
            }, save_path)
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

    # save the best model
    if best_challenge_metric > -np.inf:
        if config.final_model_name:
            save_filename = config.final_model_name
        else:
            save_suffix = f"BestModel_challenge_score(qrs_score)_{best_eval_res}"
            save_filename = f"{save_prefix}_{get_date_str()}_{save_suffix}.pth.tar"
        save_path = os.path.join(config.model_dir, save_filename)
        torch.save({
            "model_state_dict": best_state_dict,
            "model_config": model_config,
            "train_config": config,
            "epoch": best_epoch,
        }, save_path)
        if logger:
            logger.info(f"Best model saved to {save_path}!")

    writer.close()
    
    if logger:
        for h in logger.handlers:
            h.close()
            logger.removeHandler(h)
        del logger
    logging.shutdown()

    return best_state_dict



@torch.no_grad()
def evaluate(model:nn.Module,
             data_loader:DataLoader,
             config:dict,
             device:torch.device,
             debug:bool=True,
             logger:Optional[logging.Logger]=None) -> float:
    """ finished, checked,

    Parameters
    ----------
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

    Returns
    -------
    qrs_score: float,
        evaluation results, a score defined in `compute_metrics`
    """
    model.eval()
    all_rpeak_preds = []
    all_rpeak_labels = []

    if type(model).__name__ in ["DataParallel",]:  # TODO: further consider "DistributedDataParallel"
        _model = model.module
    else:
        _model = model

    for signals, labels in data_loader:
        signals = signals.to(device=device, dtype=_DTYPE)
        labels = labels.numpy()
        labels = [mask_to_intervals(item, 1) for item in labels]  # intervals of qrs complexes
        labels = [ # to indices of rpeaks in the original signal sequence
            (TrainCfg.seq_lab_reduction//2) * np.array([itv[0]+itv[1] for itv in item]) \
                for item in labels
        ]
        labels = [
            item[np.where((item>=config.skip_dist) & (item<config.input_len-config.skip_dist))[0]] \
                for item in labels
        ]
        all_rpeak_labels += labels

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prob, rpeak_preds = _model.inference(
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
        type=str, default="crnn",
        help="name of the model to train, `cnn` or `crnn`",
        dest="model_name")
    parser.add_argument(
        "-c", "--cnn-name",
        type=str, default="multi_scopic",
        help="choice of cnn feature extractor",
        dest="cnn_name")
    parser.add_argument(
        "-r", "--rnn-name",
        type=str, default="lstm",
        help="choice of rnn structures",
        dest="rnn_name")
    parser.add_argument(
        "-a", "--attn-name",
        type=str, default="se",
        help="choice of attention block",
        dest="attn_name")
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

    model_name = f"seq_lab_{config.model_name.lower()}"
    model_config = deepcopy(ModelCfg[model_name])
    model_config.cnn.name = config.cnn_name
    model_config.rnn.name = config.rnn_name
    model_config.attn.name = config.attn_name

    model = ECG_SEQ_LAB_NET_CPSC2019(
        n_leads=config.n_leads,
        input_len=config.input_len,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)

    model.to(device=device)
    model.__DEBUG__ = False

    try:
        train(
            model=model,
            model_config=model_config,
            config=train_config,
            device=device,
            logger=logger,
            debug=train_config.debug,
        )
    except KeyboardInterrupt:
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "train_config": config,
        }, os.path.join(config.checkpoints, "INTERRUPTED.pth.tar"))
        logger.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
