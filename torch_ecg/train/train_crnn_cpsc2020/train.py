"""
"""
import os
import sys
import time
import logging
import argparse
from copy import deepcopy
from collections import deque
from typing import Union, Optional, Tuple, Dict, Sequence, NoReturn
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

from torch_ecg.models.nets import (
    BCEWithLogitsWithClassWeightLoss,
    default_collate_fn as collate_fn,
)
# from torch_ecg.torch_ecg.models.ecg_crnn import ECG_CRNN
from .model import ECG_CRNN_CPSC2020, ECG_SEQ_LAB_NET_CPSC2020
from .utils import (
    get_date_str, dict_to_str, str2bool,
    mask_to_intervals, list_sum,
)
from .cfg import ModelCfg, TrainCfg
# from dataset import CPSC2020
from .dataset_simplified import CPSC2020 as CPSC2020_SIMPLIFIED
from .metrics import eval_score, CPSC2020_loss, CPSC2020_score

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

    ds = CPSC2020_SIMPLIFIED
    train_dataset = ds(config=config, training=True)
    train_dataset.__DEBUG__ = False

    if debug:
        val_train_dataset = ds(config=config, training=True)
        val_train_dataset.disable_data_augmentation()
        val_train_dataset.__DEBUG__ = False
    val_dataset = ds(config=config, training=False)
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
        filename_suffix=f"OPT_{config.model_name}_{config.cnn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
        comment=f"OPT_{config.model_name}_{config.cnn_name}_{config.train_optimizer}_LR_{lr}_BS_{batch_size}",
    )
    
    # max_itr = n_epochs * n_train

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
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    if config.lr_scheduler is None:
        scheduler = None
    elif config.lr_scheduler.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    elif config.lr_scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.lr_step_size, config.lr_gamma)
    else:
        raise NotImplementedError("lr scheduler `{config.lr_scheduler.lower()}` not implemented for training")

    if config.loss == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif config.loss == "BCEWithLogitsWithClassWeightLoss":
        criterion = BCEWithLogitsWithClassWeightLoss(
            class_weight=train_dataset.class_weights.to(device=device, dtype=_DTYPE)
        )
    else:
        raise NotImplementedError(f"loss `{config.loss}` not implemented!")
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = f"{model.__name__}_{config.cnn_name}_{config.rnn_name}_epoch"

    saved_models = deque()
    model.train()
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{n_epochs}', ncols=100) as pbar:
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
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    if scheduler:
                        writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        pbar.set_postfix(**{
                            'loss (batch)': loss.item(),
                            'lr': scheduler.get_lr()[0],
                        })
                        msg = f'Train step_{global_step}: loss : {loss.item()}, lr : {scheduler.get_lr()[0] * batch_size}'
                    else:
                        pbar.set_postfix(**{
                            'loss (batch)': loss.item(),
                        })
                        msg = f'Train step_{global_step}: loss : {loss.item()}'
                    print(msg)  # in case no logger
                    if logger:
                        logger.info(msg)
                pbar.update(signals.shape[0])

            writer.add_scalar('train/epoch_loss', epoch_loss, global_step)

            # eval for each epoch using corresponding `evaluate` function
            if debug:
                if config.model_name == "crnn":
                    eval_train_res = evaluate_crnn(
                        model, val_train_loader, config, device, debug
                    )
                    writer.add_scalar('train/auroc', eval_train_res[0], global_step)
                    writer.add_scalar('train/auprc', eval_train_res[1], global_step)
                    writer.add_scalar('train/accuracy', eval_train_res[2], global_step)
                    writer.add_scalar('train/f_measure', eval_train_res[3], global_step)
                    writer.add_scalar('train/f_beta_measure', eval_train_res[4], global_step)
                    writer.add_scalar('train/g_beta_measure', eval_train_res[5], global_step)
                elif config.model_name == "seq_lab":
                    eval_train_res = evaluate_seq_lab(
                        model, val_train_loader, config, device, debug
                    )
                    writer.add_scalar('train/total_loss', eval_train_res.total_loss, global_step)
                    writer.add_scalar('train/spb_loss', eval_train_res.spb_loss, global_step)
                    writer.add_scalar('train/pvc_loss', eval_train_res.pvc_loss, global_step)
                    writer.add_scalar('train/spb_tp', eval_train_res.spb_tp, global_step)
                    writer.add_scalar('train/pvc_tp', eval_train_res.pvc_tp, global_step)
                    writer.add_scalar('train/spb_fp', eval_train_res.spb_fp, global_step)
                    writer.add_scalar('train/pvc_fp', eval_train_res.pvc_fp, global_step)
                    writer.add_scalar('train/spb_fn', eval_train_res.spb_fn, global_step)
                    writer.add_scalar('train/pvc_fn', eval_train_res.pvc_fn, global_step)
            
            if config.model_name == "crnn":
                eval_res = evaluate_crnn(
                    model, val_loader, config, device, debug
                )
                model.train()
                writer.add_scalar('test/auroc', eval_res[0], global_step)
                writer.add_scalar('test/auprc', eval_res[1], global_step)
                writer.add_scalar('test/accuracy', eval_res[2], global_step)
                writer.add_scalar('test/f_measure', eval_res[3], global_step)
                writer.add_scalar('test/f_beta_measure', eval_res[4], global_step)
                writer.add_scalar('test/g_beta_measure', eval_res[5], global_step)

                if config.lr_scheduler is None:
                    pass
                elif config.lr_scheduler.lower() == 'plateau':
                    scheduler.step(metrics=eval_res[6])
                elif config.lr_scheduler.lower() == 'step':
                    scheduler.step()

                if debug:
                    eval_train_msg = f"""
                    train/auroc:             {eval_train_res[0]}
                    train/auprc:             {eval_train_res[1]}
                    train/accuracy:          {eval_train_res[2]}
                    train/f_measure:         {eval_train_res[3]}
                    train/f_beta_measure:    {eval_train_res[4]}
                    train/g_beta_measure:    {eval_train_res[5]}
                """
                else:
                    eval_train_msg = ""
                msg = f"""
                    Train epoch_{epoch + 1}:
                    --------------------
                    train/epoch_loss:        {epoch_loss}{eval_train_msg}
                    test/auroc:              {eval_res[0]}
                    test/auprc:              {eval_res[1]}
                    test/accuracy:           {eval_res[2]}
                    test/f_measure:          {eval_res[3]}
                    test/f_beta_measure:     {eval_res[4]}
                    test/g_beta_measure:     {eval_res[5]}
                    ---------------------------------
                """
            elif config.model_name == "seq_lab":
                eval_res = evaluate_seq_lab(
                    model, val_loader, config, device, debug
                )
                model.train()
                writer.add_scalar('test/total_loss', eval_res.total_loss, global_step)
                writer.add_scalar('test/spb_loss', eval_res.spb_loss, global_step)
                writer.add_scalar('test/pvc_loss', eval_res.pvc_loss, global_step)
                writer.add_scalar('test/spb_tp', eval_res.spb_tp, global_step)
                writer.add_scalar('test/pvc_tp', eval_res.pvc_tp, global_step)
                writer.add_scalar('test/spb_fp', eval_res.spb_fp, global_step)
                writer.add_scalar('test/pvc_fp', eval_res.pvc_fp, global_step)
                writer.add_scalar('test/spb_fn', eval_res.spb_fn, global_step)
                writer.add_scalar('test/pvc_fn', eval_res.pvc_fn, global_step)

                if config.lr_scheduler is None:
                    pass
                elif config.lr_scheduler.lower() == 'plateau':
                    scheduler.step(metrics=eval_res.total_loss)
                elif config.lr_scheduler.lower() == 'step':
                    scheduler.step()

                if debug:
                    eval_train_msg = f"""
                    train/total_loss:        {eval_train_res.total_loss}
                    train/spb_loss:          {eval_train_res.spb_loss}
                    train/pvc_loss:          {eval_train_res.pvc_loss}
                    train/spb_tp:            {eval_train_res.spb_tp}
                    train/pvc_tp:            {eval_train_res.pvc_tp}
                    train/spb_fp:            {eval_train_res.spb_fp}
                    train/pvc_fp:            {eval_train_res.pvc_fp}
                    train/spb_fn:            {eval_train_res.spb_fn}
                    train/pvc_fn:            {eval_train_res.pvc_fn}
                """
                else:
                    eval_train_msg = ""
                msg = f"""
                    Train epoch_{epoch + 1}:
                    --------------------
                    train/epoch_loss:        {epoch_loss}{eval_train_msg}
                    test/total_loss:         {eval_res.total_loss}
                    test/spb_loss:           {eval_res.spb_loss}
                    test/pvc_loss:           {eval_res.pvc_loss}
                    test/spb_tp:             {eval_res.spb_tp}
                    test/pvc_tp:             {eval_res.pvc_tp}
                    test/spb_fp:             {eval_res.spb_fp}
                    test/pvc_fp:             {eval_res.pvc_fp}
                    test/spb_fn:             {eval_res.spb_fn}
                    test/pvc_fn:             {eval_res.pvc_fn}
                    ---------------------------------
                """

            print(msg)  # in case no logger
            if logger:
                logger.info(msg)

            try:
                os.makedirs(config.checkpoints, exist_ok=True)
                if logger:
                    logger.info('Created checkpoint directory')
            except OSError:
                pass
            if config.model_name == "crnn":
                save_suffix = f'epochloss_{epoch_loss:.5f}_fb_{eval_res[4]:.2f}_gb_{eval_res[5]:.2f}'
            elif config.model_name == "seq_lab":
                save_suffix = f'epochloss_{epoch_loss:.5f}_challenge_loss_{eval_res.total_loss}'
            save_filename = f'{save_prefix}{epoch + 1}_{get_date_str()}_{save_suffix}.pth'
            save_path = os.path.join(config.checkpoints, save_filename)
            torch.save(model.state_dict(), save_path)
            if logger:
                logger.info(f'Checkpoint {epoch + 1} saved!')
            saved_models.append(save_path)
            # remove outdated models
            if len(saved_models) > config.keep_checkpoint_max > 0:
                model_to_remove = saved_models.popleft()
                try:
                    os.remove(model_to_remove)
                except:
                    logger.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate_crnn(model:nn.Module, data_loader:DataLoader, config:dict, device:torch.device, debug:bool=False) -> Tuple[float]:
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
        auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure
    """
    model.eval()
    # data_loader.dataset.disable_data_augmentation()

    all_scalar_preds = []
    all_bin_preds = []
    all_labels = []

    for signals, labels in data_loader:
        signals = signals.to(device=device, dtype=_DTYPE)
        labels = labels.numpy()
        all_labels.append(labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preds, bin_preds = model.inference(signals)
        all_scalar_preds.append(preds)
        all_bin_preds.append(bin_preds)
    
    all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
    all_bin_preds = np.concatenate(all_bin_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    classes = data_loader.dataset.all_classes

    if debug:
        print(f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}")
        head_num = 5
        head_scalar_preds = all_scalar_preds[:head_num,...]
        head_bin_preds = all_bin_preds[:head_num,...]
        head_preds_classes = [np.array(classes)[np.where(row)] for row in head_bin_preds]
        head_labels = all_labels[:head_num,...]
        head_labels_classes = [np.array(classes)[np.where(row)] for row in head_labels]
        for n in range(head_num):
            print(f"""
            ----------------------------------------------
            scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
            binary prediction:    {head_bin_preds[n].tolist()}
            labels:               {head_labels[n].astype(int).tolist()}
            predicted classes:    {head_preds_classes[n].tolist()}
            label classes:        {head_labels_classes[n].tolist()}
            ----------------------------------------------
            """)

    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure = \
        eval_score(
            classes=classes,
            truth=all_labels,
            scalar_pred=all_scalar_preds,
            binary_pred=all_bin_preds,
        )
    eval_res = auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure

    model.train()

    return eval_res


@torch.no_grad()
def evaluate_seq_lab(model:nn.Module, data_loader:DataLoader, config:dict, device:torch.device, debug:bool=False) -> Dict[str, int]:
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

    CAUTION:
    --------
    without rpeaks detection, consecutive SPBs or consecutive PVCs might be falsely missed,
    hence resulting higher than normal false negatives.
    for a more suitable eval pipeline, ref. CPSC2020_challenge.py
    """
    model.eval()
    # data_loader.dataset.disable_data_augmentation()

    all_scalar_preds = []
    all_spb_preds = []
    all_pvc_preds = []
    all_spb_labels = []
    all_pvc_labels = []

    for signals, labels in data_loader:
        signals = signals.to(device=device, dtype=_DTYPE)
        labels = labels.numpy()  # (batch_size, seq_len, 2 or 3)
        spb_intervals = [
            mask_to_intervals(seq, 1) for seq in labels[..., config.classes.index("S")]
        ]
        # print(spb_intervals)
        spb_labels = [
            [model.reduction * (itv[0]+itv[1])//2 for itv in l_itv]  if len(l_itv) > 0 else [] \
                for l_itv in spb_intervals
        ]
        # print(spb_labels)
        all_spb_labels.append(spb_labels)
        pvc_intervals = [
            mask_to_intervals(seq, 1) for seq in labels[..., config.classes.index("V")]
        ]
        pvc_labels = [
            [model.reduction * (itv[0]+itv[1])//2 for itv in l_itv]  if len(l_itv) > 0 else [] \
                for l_itv in pvc_intervals
        ]
        all_pvc_labels.append(pvc_labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preds, spb_preds, pvc_preds = model.inference(signals)
        all_scalar_preds.append(preds)
        all_spb_preds.append(spb_preds)
        all_pvc_preds.append(pvc_preds)

    all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
    # all_spb_preds = np.concatenate(all_spb_preds, axis=0)
    # all_pvc_preds = np.concatenate(all_pvc_preds, axis=0)
    # all_spb_labels = np.concatenate(all_spb_labels, axis=0)
    # all_pvc_labels = np.concatenate(all_pvc_labels, axis=0)
    all_spb_preds = [np.array(item) for item in list_sum(all_spb_preds)]
    all_pvc_preds = [np.array(item) for item in list_sum(all_pvc_preds)]
    all_spb_labels = [np.array(item) for item in list_sum(all_spb_labels)]
    all_pvc_labels = [np.array(item) for item in list_sum(all_pvc_labels)]

    eval_res_tmp = ED(CPSC2020_score(
        spb_true=all_spb_labels,
        pvc_true=all_pvc_labels,
        spb_pred=all_spb_preds,
        pvc_pred=all_pvc_preds,
        verbose=1
    ))

    eval_res = ED(
        total_loss=eval_res_tmp.total_loss,
        spb_loss=eval_res_tmp.class_loss.S,
        pvc_loss=eval_res_tmp.class_loss.V,
        spb_tp=eval_res_tmp.true_positive.S,
        pvc_tp=eval_res_tmp.true_positive.V,
        spb_fp=eval_res_tmp.false_positive.S,
        pvc_fp=eval_res_tmp.false_positive.V,
        spb_fn=eval_res_tmp.false_negative.S,
        pvc_fn=eval_res_tmp.false_negative.V,
    )

    model.train()

    return eval_res


def get_args(**kwargs):
    """
    """
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description='Train the Model on CPSC2020',
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
        '-m', '--model-name',
        type=str, default="crnn",
        help='name of the model to train',
        dest='model_name')
    parser.add_argument(
        '-c', '--cnn-name',
        type=str, default='multi_scopic',
        help='choice of cnn feature extractor',
        dest='cnn_name')
    parser.add_argument(
        '-r', '--rnn-name',
        type=str, default='linear',
        help='choice of rnn structures',
        dest='rnn_name')
    parser.add_argument(
        '--keep-checkpoint-max', type=int, default=50,
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
    from utils import init_logger
    train_config = get_args(**TrainCfg)
    # os.environ["CUDA_VISIBLE_DEVICES"] = train_config.gpu
    if not DAS:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda')

    # classes = train_config.classes
    model_name = train_config.model_name.lower()
    classes = deepcopy(ModelCfg[model_name].classes)
    class_map = deepcopy(ModelCfg[model_name].class_map)

    if model_name == "crnn":
        model_config = deepcopy(ModelCfg.crnn)
    elif model_name == "seq_lab":
        model_config = deepcopy(ModelCfg.seq_lab)
        train_config.classes = deepcopy(model_config.classes)
        train_config.class_map = deepcopy(model_config.class_map)
    model_config.model_name = model_name
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name

    if model_name == "crnn":
        # model = ECG_CRNN(
        model = ECG_CRNN_CPSC2020(
            classes=classes,
            n_leads=train_config.n_leads,
            input_len=train_config.input_len,
            config=model_config,
        )
    elif model_name == "seq_lab":
        model = ECG_SEQ_LAB_NET_CPSC2020(
            classes=classes,
            n_leads=train_config.n_leads,
            input_len=train_config.input_len,
            config=model_config,
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported yet!")

    if not DAS and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    logger = init_logger(log_dir=train_config.log_dir)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Model name = {train_config.model_name}")
    logger.info(f'Using device {device}')
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f'with configuration\n{dict_to_str(train_config)}')
    print(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    print(f'Using device {device}')
    print(f"Using torch of version {torch.__version__}")
    print(f'with configuration\n{dict_to_str(train_config)}')

    try:
        train(
            model=model,
            config=train_config,
            device=device,
            logger=logger,
            debug=train_config.debug,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(train_config.checkpoints, 'INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
