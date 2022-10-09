"""
"""

import logging
import shutil
import textwrap
from copy import deepcopy
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[2]))

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.components.outputs import MultiLabelClassificationOutput
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases.datasets.cinc2021 import CINC2021Dataset, CINC2021TrainCfg
from torch_ecg.databases.physionet_databases.cinc2021 import (
    compute_metrics as compute_cinc2021_metrics,
)
from torch_ecg.model_configs.ecg_crnn import ECG_CRNN_CONFIG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[1] / "tmp" / "test_crnn_cinc2021_pipeline"
_CWD.mkdir(parents=True, exist_ok=True)
_DB_DIR = _CWD.parents[2] / "sample-data" / "cinc2021"
###############################################################################

###############################################################################
# set up configs

_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)
_BASE_MODEL_CONFIG.cnn.multi_scopic_leadwise.block.batch_norm = "group_norm"  # False

# detailed configs for 12-lead, 6-lead, 4-lead, 3-lead, 2-lead models
# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelArchCfg = CFG()

ModelArchCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)

# TODO: add adjustifications for "leadwise" configs for 6,4,3,2 leads models
ModelArchCfg.six_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.six_leads.cnn.vgg16_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.six_leads.cnn.resnet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.resnet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.groups = 6
_base_num_filters = 6 * 6  # 12 * 4
ModelArchCfg.six_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.six_leads.cnn.densenet_leadwise.groups = 6
ModelArchCfg.six_leads.cnn.densenet_leadwise.init_num_filters = 6 * 8  # 12 * 8
ModelArchCfg.six_leads.cnn.xception_leadwise.groups = 6
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.six_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.six_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.four_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.four_leads.cnn.vgg16_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.four_leads.cnn.resnet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.resnet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.groups = 4
_base_num_filters = 6 * 4  # 12 * 4
ModelArchCfg.four_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.four_leads.cnn.densenet_leadwise.groups = 4
ModelArchCfg.four_leads.cnn.densenet_leadwise.init_num_filters = 6 * 6  # 12 * 8
ModelArchCfg.four_leads.cnn.xception_leadwise.groups = 4
_base_num_filters = 6 * 2  # 12 * 2
ModelArchCfg.four_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.four_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.three_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.three_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.three_leads.cnn.resnet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.resnet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.groups = 3
_base_num_filters = 3 * 8  # 12 * 4
ModelArchCfg.three_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.three_leads.cnn.densenet_leadwise.groups = 3
ModelArchCfg.three_leads.cnn.densenet_leadwise.init_num_filters = 3 * 12  # 12 * 8
ModelArchCfg.three_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 3 * 4  # 12 * 2
ModelArchCfg.three_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.three_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

ModelArchCfg.two_leads = deepcopy(_BASE_MODEL_CONFIG)
ModelArchCfg.two_leads.cnn.vgg16_leadwise.groups = 3
_base_num_filters = 2 * 12  # 12 * 4
ModelArchCfg.two_leads.cnn.vgg16_leadwise.num_filters = [
    _base_num_filters * 4,
    _base_num_filters * 8,
    _base_num_filters * 16,
    _base_num_filters * 32,
    _base_num_filters * 32,
]
ModelArchCfg.two_leads.cnn.resnet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.resnet_leadwise.init_num_filters = 2 * 16  # 12 * 8
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.groups = 2
_base_num_filters = 2 * 8  # 12 * 4
ModelArchCfg.two_leads.cnn.multi_scopic_leadwise.num_filters = [
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
    [
        _base_num_filters * 4,
        _base_num_filters * 8,
        _base_num_filters * 16,
    ],
]
ModelArchCfg.two_leads.cnn.densenet_leadwise.groups = 2
ModelArchCfg.two_leads.cnn.densenet_leadwise.init_num_filters = 2 * 12  # 12 * 8
ModelArchCfg.two_leads.cnn.xception_leadwise.groups = 3
_base_num_filters = 2 * 6  # 12 * 2
ModelArchCfg.two_leads.cnn.xception_vanilla.entry_flow = CFG(
    init_num_filters=[_base_num_filters * 4, _base_num_filters * 8],
    init_filter_lengths=3,
    init_subsample_lengths=[2, 1],
    num_filters=[
        _base_num_filters * 16,
        _base_num_filters * 32,
        _base_num_filters * 91,
    ],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.middle_flow = CFG(
    num_filters=list(repeat(_base_num_filters * 91, 8)),
    filter_lengths=3,
)
ModelArchCfg.two_leads.cnn.xception_vanilla.exit_flow = CFG(
    final_num_filters=[_base_num_filters * 182, _base_num_filters * 256],
    final_filter_lengths=3,
    num_filters=[[_base_num_filters * 91, _base_num_filters * 128]],
    filter_lengths=3,
    subsample_lengths=2,
    subsample_kernels=3,
)

# constants for model inference
_bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
_bin_pred_look_again_tol = 0.03
_bin_pred_nsr_thr = 0.1

ModelCfg = CFG()
ModelCfg.torch_dtype = DEFAULTS.DTYPE.TORCH
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = _bin_pred_thr
ModelCfg.bin_pred_look_again_tol = _bin_pred_look_again_tol
ModelCfg.bin_pred_nsr_thr = _bin_pred_nsr_thr

ModelCfg.special_classes = []
ModelCfg.dl_classes = deepcopy(CINC2021TrainCfg.classes)
ModelCfg.tranche_classes = deepcopy(CINC2021TrainCfg.tranche_classes)
ModelCfg.full_classes = ModelCfg.dl_classes + ModelCfg.special_classes

ModelCfg.dl_siglen = CINC2021TrainCfg.siglen

ModelCfg.cnn_name = CINC2021TrainCfg.cnn_name
ModelCfg.rnn_name = CINC2021TrainCfg.rnn_name
ModelCfg.attn_name = CINC2021TrainCfg.attn_name

# model architectures configs
ModelCfg.update(ModelArchCfg)
for lead_set in ["twelve_leads", "six_leads", "four_leads", "three_leads", "two_leads"]:
    adjust_cnn_filter_lengths(ModelCfg[lead_set], ModelCfg.fs)
    ModelCfg[lead_set].cnn.name = ModelCfg.cnn_name
    ModelCfg[lead_set].rnn.name = ModelCfg.rnn_name
    ModelCfg[lead_set].attn.name = ModelCfg.attn_name
    # ModelCfg[lead_set].clf = CFG()
    # ModelCfg[lead_set].clf.out_channels = [
    # # not including the last linear layer, whose out channels equals n_classes
    # ]
    # ModelCfg[lead_set].clf.bias = True
    # ModelCfg[lead_set].clf.dropouts = 0.0
    # ModelCfg[lead_set].clf.activation = "mish"  # for a single layer `SeqLin`, activation is ignored


class ECG_CRNN_CINC2021(ECG_CRNN):
    """ """

    __DEBUG__ = False
    __name__ = "ECG_CRNN_CINC2021"

    def __init__(
        self, classes: Sequence[str], n_leads: int, config: Optional[CFG] = None
    ) -> None:
        """

        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        model_config = CFG(deepcopy(ModelCfg))
        model_config.update(deepcopy(config) or {})
        super().__init__(classes, n_leads, model_config)

    @torch.no_grad()
    def inference(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> MultiLabelClassificationOutput:
        """

        auxiliary function to `forward`, for CINC2021,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions

        Returns
        -------
        MultiLabelClassificationOutput, with the following items:
            classes: list,
                list of the classes for classification
            thr: float,
                threshold for making binary predictions from scalar predictions
            prob: ndarray or DataFrame,
                scalar predictions, (and binary predictions if `class_names` is True)
            prob: ndarray,
                the array (with values 0, 1 for each class) of binary prediction

        NOTE that when `input` is ndarray, one should make sure that it is transformed,
        e.g. bandpass filtered, normalized, etc.

        """
        if "NSR" in self.classes:
            nsr_cid = self.classes.index("NSR")
        elif "426783006" in self.classes:
            nsr_cid = self.classes.index("426783006")
        else:
            nsr_cid = None
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        # batch_size, channels, seq_len = _input.shape
        prob = self.sigmoid(self.forward(_input))
        pred = (prob >= bin_pred_thr).int()
        prob = prob.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        for row_idx, row in enumerate(pred):
            row_max_prob = prob[row_idx, ...].max()
            if row_max_prob < ModelCfg.bin_pred_nsr_thr and nsr_cid is not None:
                pred[row_idx, nsr_cid] = 1
            elif row.sum() == 0:
                pred[row_idx, ...] = (
                    (
                        (prob[row_idx, ...] + ModelCfg.bin_pred_look_again_tol)
                        >= row_max_prob
                    )
                    & (prob[row_idx, ...] >= ModelCfg.bin_pred_nsr_thr)
                ).astype(int)
        if class_names:
            prob = pd.DataFrame(prob)
            prob.columns = self.classes
            prob["pred"] = ""
            for row_idx in range(len(prob)):
                prob.at[row_idx, "pred"] = np.array(self.classes)[
                    np.where(pred[row_idx] == 1)[0]
                ].tolist()
        return MultiLabelClassificationOutput(
            classes=self.classes,
            thr=bin_pred_thr,
            prob=prob,
            pred=pred,
        )

    @add_docstring(inference.__doc__)
    def inference_CINC2021(
        self,
        input: Union[np.ndarray, Tensor],
        class_names: bool = False,
        bin_pred_thr: float = 0.5,
    ) -> MultiLabelClassificationOutput:
        """
        alias for `self.inference`
        """
        return self.inference(input, class_names, bin_pred_thr)


class CINC2021Trainer(BaseTrainer):
    """ """

    __name__ = "CINC2021Trainer"

    def __init__(
        self,
        model: torch.nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        """

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
        super().__init__(
            model, CINC2021Dataset, model_config, train_config, device, lazy
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        """

        setup the dataloaders for training and validation

        Parameters
        ----------
        train_dataset: Dataset, optional,
            the training dataset
        val_dataset: Dataset, optional,
            the validation dataset

        """
        if train_dataset is None:
            train_dataset = self.dataset_cls(
                config=self.train_config, training=True, lazy=False
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config, training=False, lazy=False
            )

        # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
        if torch.cuda.is_available():
            num_workers = 4
        else:
            num_workers = 0

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

    def run_one_step(
        self, *data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
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
            model_output = self._model.inference(signals)
            all_scalar_preds.append(model_output.prob)
            all_bin_preds.append(model_output.pred)

        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        classes = data_loader.dataset.all_classes

        if self.val_train_loader is not None:
            msg = f"all_scalar_preds.shape = {all_scalar_preds.shape}, all_labels.shape = {all_labels.shape}"
            self.log_manager.log_message(msg, level=logging.DEBUG)
            head_num = 5
            head_scalar_preds = all_scalar_preds[:head_num, ...]
            head_bin_preds = all_bin_preds[:head_num, ...]
            head_preds_classes = [
                np.array(classes)[np.where(row)] for row in head_bin_preds
            ]
            head_labels = all_labels[:head_num, ...]
            head_labels_classes = [
                np.array(classes)[np.where(row)] for row in head_labels
            ]
            for n in range(head_num):
                msg = textwrap.dedent(
                    f"""
                ----------------------------------------------
                scalar prediction:    {[round(n, 3) for n in head_scalar_preds[n].tolist()]}
                binary prediction:    {head_bin_preds[n].tolist()}
                labels:               {head_labels[n].astype(int).tolist()}
                predicted classes:    {head_preds_classes[n].tolist()}
                label classes:        {head_labels_classes[n].tolist()}
                ----------------------------------------------
                """
                )
                self.log_manager.log_message(msg)

        (
            auroc,
            auprc,
            accuracy,
            f_measure,
            f_beta_measure,
            g_beta_measure,
            challenge_metric,
        ) = compute_cinc2021_metrics(
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
        """ """
        return []

    @property
    def save_prefix(self) -> str:
        return f"{self._model.__name__}_{self.model_config.cnn.name}_epoch"

    def extra_log_suffix(self) -> str:
        return super().extra_log_suffix() + f"_{self.model_config.cnn.name}"


# fmt: off
# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
four_leads = ("I", "II", "III", "V2")
three_leads = ("I", "II", "V2")
two_leads = ("I", "II")
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)
# fmt: on


def test_crnn_cinc2021_pipeline() -> None:
    """ """

    train_config = deepcopy(CINC2021TrainCfg)
    train_config.db_dir = _DB_DIR
    train_config.log_dir = _CWD / "logs"
    train_config.model_dir = _CWD / "saved_models"
    train_config.checkpoints = _CWD / "checkpoints"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.checkpoints.mkdir(parents=True, exist_ok=True)
    train_config.debug = True

    train_config.cnn_name = "resnet_nature_comm_bottle_neck_se"
    train_config.rnn_name = "none"  # "none", "lstm"
    train_config.attn_name = "none"  # "none", "se", "gc", "nl"
    train_config.n_epochs = 2
    train_config.keep_checkpoint_max = 0

    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lead_set_name in [
        "twelve_leads",
        "six_leads",
        "four_leads",
        "three_leads",
        "two_leads",
    ]:
        train_config.leads = eval(lead_set_name)
        train_config.n_leads = len(train_config.leads)
        model_config = eval(f"deepcopy(ModelCfg.{lead_set_name})")
        model_config.cnn.name = train_config.cnn_name
        model_config.rnn.name = train_config.rnn_name
        model_config.attn.name = train_config.attn_name

        model = ECG_CRNN_CINC2021(
            classes=train_classes,
            n_leads=train_config.n_leads,
            config=model_config,
        )

        if torch.cuda.device_count() > 1:
            model = DP(model)
        model.to(device=device)

        trainer = CINC2021Trainer(
            model=model,
            model_config=model_config,
            train_config=train_config,
            device=device,
            lazy=False,
        )
        bmd = trainer.train()

        del bmd, trainer, model

    shutil.rmtree(_CWD)


if __name__ == "__main__":
    test_crnn_cinc2021_pipeline()
