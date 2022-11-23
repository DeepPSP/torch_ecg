"""
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader, Dataset

from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.components.outputs import WaveDelineationOutput
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.databases import LUDB
from torch_ecg.databases.datasets.ludb import LUDBDataset, LUDBTrainCfg
from torch_ecg.databases.physionet_databases.ludb import (
    compute_metrics as compute_ludb_metrics,
)
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.utils import ecg_arrhythmia_knowledge as EAK
from torch_ecg.utils.misc import add_docstring
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[1] / "tmp" / "test_unet_ludb_pipeline"
_CWD.mkdir(parents=True, exist_ok=True)
_DB_DIR = _CWD / "ludb"
_DB_DIR.mkdir(parents=True, exist_ok=True)
###############################################################################

###############################################################################
# download data
dr = LUDB(_DB_DIR)
dr.download(compressed=True)
dr._ls_rec()
del dr
###############################################################################

###############################################################################
# set up configs
ModelCfg = CFG()
ModelCfg.torch_dtype = DEFAULTS.DTYPE.TORCH
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs

ModelCfg.classes = [
    "p",  # pwave
    "N",  # qrs complex
    "t",  # twave
    "i",  # isoelectric
]
ModelCfg.class_map = CFG(p=1, N=2, t=3, i=0)
ModelCfg.mask_classes = deepcopy(ModelCfg.classes)
ModelCfg.mask_class_map = deepcopy(ModelCfg.class_map)

ModelCfg.leads = deepcopy(EAK.Standard12Leads)
ModelCfg.n_leads = len(ModelCfg.leads)

ModelCfg.skip_dist = int(0.5 * ModelCfg.fs)

ModelCfg.model_name = "unet"
ModelCfg.unet = deepcopy(ECG_UNET_VANILLA_CONFIG)
adjust_cnn_filter_lengths(ModelCfg.unet, ModelCfg.fs)
###############################################################################


class ECG_UNET_LUDB(ECG_UNET):
    """ """

    __DEBUG__ = True
    __name__ = "ECG_UNET_LUDB"

    def __init__(
        self, n_leads: int, config: Optional[CFG] = None, **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file

        """
        model_config = deepcopy(ModelCfg.unet)
        if config:
            model_config.update(deepcopy(config[config.model_name]))
            ModelCfg.update(deepcopy(config))
        _inv_class_map = {v: k for k, v in ModelCfg.class_map.items()}
        self._mask_map = CFG(
            {k: _inv_class_map[v] for k, v in ModelCfg.mask_class_map.items()}
        )
        super().__init__(ModelCfg.mask_classes, n_leads, model_config)

    @torch.no_grad()
    def inference(
        self,
        input: Union[Sequence[float], np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
    ) -> WaveDelineationOutput:
        """
        Parameters
        ----------
        input: array-like,
            input ECG signal
        bin_pred_thr: float, default 0.5,
            threshold for binary prediction,
            used only when the `background` class "i" is not included in `mask_classes`

        Returns
        -------
        output: WaveDelineationOutput, with items:
            - classes: list of str,
                list of classes
            - prob: np.ndarray,
                predicted probability map, of shape (n_samples, seq_len, n_classes)
            - mask: np.ndarray,
                predicted mask, of shape (n_samples, seq_len)

        """
        self.eval()
        _input = torch.as_tensor(input, dtype=self.dtype, device=self.device)
        if _input.ndim == 2:
            _input = _input.unsqueeze(0)  # add a batch dimension
        batch_size, channels, seq_len = _input.shape
        prob = self.forward(_input)
        if "i" in self.classes:
            prob = self.softmax(prob)
        else:
            prob = torch.sigmoid(prob)
        prob = prob.cpu().detach().numpy()

        if "i" in self.classes:
            mask = np.argmax(prob, axis=-1)
        else:
            mask = np.vectorize(lambda n: self._mask_map[n])(np.argmax(prob, axis=-1))
            mask *= (prob > bin_pred_thr).any(axis=-1)  # class "i" mapped to 0

        # TODO: shoule one add more post-processing to filter out false positives of the waveforms?

        return WaveDelineationOutput(
            classes=self.classes,
            prob=prob,
            mask=mask,
        )

    @add_docstring(inference.__doc__)
    def inference_LUDB(
        self,
        input: Union[np.ndarray, Tensor],
        bin_pred_thr: float = 0.5,
    ) -> WaveDelineationOutput:
        """
        alias of `self.inference`
        """
        return self.inference(input, bin_pred_thr)


class LUDBTrainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "LUDBTrainer"

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
        super().__init__(model, LUDBDataset, model_config, train_config, device, lazy)

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
        # labels of shape (batch_size, seq_len) if loss is CrossEntropyLoss
        # otherwise of shape (batch_size, seq_len, n_classes)
        labels = labels.to(self.device)
        preds = self.model(signals)  # of shape (batch_size, seq_len, n_classes)
        if self.train_config.loss == "CrossEntropyLoss":
            preds = preds.permute(0, 2, 1)  # of shape (batch_size, n_classes, seq_len)
            # or use the following
            # preds = pres.reshape(-1, preds.shape[-1])  # of shape (batch_size * seq_len, n_classes)
            # labels = labels.reshape(-1)  # of shape (batch_size * seq_len,)
        return preds, labels

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        all_scalar_preds = []
        all_mask_preds = []
        all_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            all_labels.append(labels)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_output = self._model.inference(signals)
            all_scalar_preds.append(model_output.prob)
            all_mask_preds.append(model_output.mask)

        # all_scalar_preds of shape (n_samples, seq_len, n_classes)
        all_scalar_preds = np.concatenate(all_scalar_preds, axis=0)
        # all_scalar_preds of shape (n_samples, seq_len)
        all_mask_preds = np.concatenate(all_mask_preds, axis=0)
        # all_labels of shape (n_samples, seq_len) if loss is CrossEntropyLoss
        # otherwise of shape (n_samples, seq_len, n_classes)
        all_labels = np.concatenate(all_labels, axis=0)

        if self.train_config.loss != "CrossEntropyLoss":
            all_labels = all_labels.argmax(
                axis=-1
            )  # (n_samples, seq_len, n_classes) -> (n_samples, seq_len)

        # eval_res are scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        # each scoring is a dict consisting of the following metrics:
        # sensitivity, precision, f1_score, mean_error, standard_deviation
        eval_res_split = compute_ludb_metrics(
            np.repeat(all_labels[:, np.newaxis, :], self.model_config.n_leads, axis=1),
            np.repeat(
                all_mask_preds[:, np.newaxis, :], self.model_config.n_leads, axis=1
            ),
            self._cm,
            self.train_config.fs,
        )

        # TODO: provide numerical values for the metrics from all of the dicts of eval_res
        eval_res = {
            metric: np.mean([eval_res_split[f"{wf}_{pos}"][metric]])
            for metric in [
                "sensitivity",
                "precision",
                "f1_score",
                "mean_error",
                "standard_deviation",
            ]
            for wf in self._cm
            for pos in [
                "onset",
                "offset",
            ]
        }

        self.model.train()

        return eval_res

    @property
    def _cm(self) -> Dict[str, str]:
        """ """
        return {
            "pwave": self.train_config.class_map["p"],
            "qrs": self.train_config.class_map["N"],
            "twave": self.train_config.class_map["t"],
        }

    @property
    def batch_dim(self) -> int:
        """
        batch dimension,
        """
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return []


def test_unet_ludb_pipeline() -> None:
    """ """
    train_cfg_fl = deepcopy(LUDBTrainCfg)
    train_cfg_fl.use_single_lead = False
    train_cfg_fl.loss = "FocalLoss"

    train_cfg_fl.db_dir = _DB_DIR
    train_cfg_fl.log_dir = _CWD / "logs"
    train_cfg_fl.model_dir = _CWD / "saved_models"
    train_cfg_fl.checkpoints = _CWD / "checkpoints"
    train_cfg_fl.log_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_fl.model_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_fl.checkpoints.mkdir(parents=True, exist_ok=True)

    # train_cfg_ce = deepcopy(LUDBTrainCfg)
    # train_cfg_ce.use_single_lead = False
    # train_cfg_ce.loss = "CrossEntropyLoss"

    # ds_train_fl = LUDB(train_cfg_fl, training=True, lazy=False)
    # ds_val_fl = LUDB(train_cfg_fl, training=False, lazy=False)

    train_cfg_fl.keep_checkpoint_max = 0
    train_cfg_fl.monitor = None
    train_cfg_fl.n_epochs = 2

    model_config = deepcopy(ModelCfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECG_UNET_LUDB(model_config.n_leads, model_config)

    if torch.cuda.device_count() > 1:
        model = DP(model)
        # model = DDP(model)
    model.to(device=device)

    trainer = LUDBTrainer(
        model=model,
        model_config=model_config,
        train_config=train_cfg_fl,
        device=device,
        lazy=False,
    )

    bmd = trainer.train()

    del model, trainer, bmd

    # shutil.rmtree(_CWD)
