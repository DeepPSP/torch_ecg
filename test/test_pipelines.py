"""
"""

from typing import NoReturn

import torch

try:
    import torch_ecg
except:
    import sys
    from pathlib import Path

    sys.path.append(Path(__file__).absolute().parent.parent)
    import torch_ecg

from torch_ecg.databases.datasets.cinc2021 import CINC2021Dataset, CINC2021TrainCfg
from torch_ecg.databases.datasets.cpsc2019 import CPSC2019Dataset, CPSC2019TrainCfg
from torch_ecg.databases.datasets.cpsc2021 import CPSC2021Dataset, CPSC2021TrainCfg
from torch_ecg.databases.physionet_databases.cinc2021 import (
    compute_metrics as compute_cinc2021_metrics,
)
from torch_ecg.databases.physionet_databases.cpsc2019 import (
    compute_metrics as compute_cpsc2019_metrics,
)
from torch_ecg.databases.physionet_databases.cpsc2021 import (
    compute_metrics as compute_cpsc2021_metrics,
)
from torch_ecg.cfg import CFG
from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.ecg_seq_lab_net import ECG_SEQ_LAB_NET
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.models.rr_lstm import RR_LSTM
from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn
from torch_ecg.utils.trainer import BaseTrainer


def test_cinc2021_pipeline() -> NoReturn:
    """ """
    pass


def test_cpsc2019_pipeline() -> NoReturn:
    """ """
    pass


def test_cpsc2021_pipeline() -> NoReturn:
    """ """
    pass


class CINC2021Trainer(BaseTrainer):
    """ """

    __name__ = "CINC2021Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """finished, checked,

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
    ) -> NoReturn:
        """finished, checked,

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


class CPSC2019Trainer(BaseTrainer):
    """ """

    __name__ = "CPSC2019Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """finished, checked,

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
            model, CPSC2019Dataset, model_config, train_config, device, lazy
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> NoReturn:
        """finished, checked,

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

        if self.train_config.get("recover_length", False):
            reduction = 1
        else:
            reduction = self.train_config.reduction

        all_rpeak_preds = []
        all_rpeak_labels = []

        for signals, labels in data_loader:
            signals = signals.to(device=self.device, dtype=self.dtype)
            labels = labels.numpy()
            labels = [
                mask_to_intervals(item, 1) for item in labels
            ]  # intervals of qrs complexes
            labels = [  # to indices of rpeaks in the original signal sequence
                (reduction * np.array([itv[0] + itv[1] for itv in item]) / 2).astype(
                    int
                )
                for item in labels
            ]
            labels = [
                item[
                    np.where(
                        (item >= self.train_config.skip_dist)
                        & (
                            item
                            < self.train_config.input_len - self.train_config.skip_dist
                        )
                    )[0]
                ]
                for item in labels
            ]
            all_rpeak_labels += labels

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prob, rpeak_preds = self._model.inference(
                signals,
                bin_pred_thr=0.5,
                duration_thr=4 * 16,
                dist_thr=200,
                correction=False,
            )
            all_rpeak_preds += rpeak_preds

        qrs_score = compute_metrics(
            rpeaks_truths=all_rpeak_labels,
            rpeaks_preds=all_rpeak_preds,
            fs=self.train_config.fs,
            thr=self.train_config.bias_thr / self.train_config.fs,
        )
        eval_res = dict(
            qrs_score=qrs_score,
        )

        del all_rpeak_labels, all_rpeak_preds

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension
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


class CPSC2021Trainer(BaseTrainer):
    """ """

    __DEBUG__ = True
    __name__ = "CPSC2021Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> NoReturn:
        """finished, checked,

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
            model, CPSC2021Dataset, model_config, train_config, device, lazy
        )

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> NoReturn:
        """finished, checked,

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
                config=self.train_config,
                task=self.train_config.task,
                training=True,
                lazy=False,
            )

        if self.train_config.debug:
            val_train_dataset = train_dataset
        else:
            val_train_dataset = None
        if val_dataset is None:
            val_dataset = self.dataset_cls(
                config=self.train_config,
                task=self.train_config.task,
                training=False,
                lazy=False,
            )

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
        if self.train_config.task == "rr_lstm":
            signals, labels, weight_masks = data
            # (batch_size, seq_len, n_channel) -> (seq_len, batch_size, n_channel)
            signals = signals.permute(1, 0, 2)
            weight_masks = weight_masks.to(device=self.device, dtype=self.dtype)
        elif self.train_config.task == "qrs_detection":
            signals, labels = data
        else:  # main task
            signals, labels, weight_masks = data
            weight_masks = weight_masks.to(device=self.device, dtype=self.dtype)
        signals = signals.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device, dtype=self.dtype)
        # print(f"signals: {signals.shape}")
        # print(f"labels: {labels.shape}")
        preds = self.model(signals)
        if self.train_config.task == "qrs_detection":
            return preds, labels
        else:
            return preds, labels, weight_masks

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """ """
        self.model.eval()

        if self.train_config.task == "qrs_detection":
            all_rpeak_preds = []
            all_rpeak_labels = []
            for signals, labels in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy()
                labels = [
                    mask_to_intervals(item, 1) for item in labels
                ]  # intervals of qrs complexes
                labels = [  # to indices of rpeaks in the original signal sequence
                    (
                        self.train_config.qrs_detection.reduction
                        * np.array([itv[0] + itv[1] for itv in item])
                        / 2
                    ).astype(int)
                    for item in labels
                ]
                labels = [
                    item[
                        np.where(
                            (item >= self.train_config.rpeaks_dist2border)
                            & (
                                item
                                < self.train_config.qrs_detection.input_len
                                - self.train_config.rpeaks_dist2border
                            )
                        )[0]
                    ]
                    for item in labels
                ]
                all_rpeak_labels += labels

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prob, rpeak_preds = self._model.inference(signals)
                all_rpeak_preds += rpeak_preds
            eval_res = compute_rpeak_metric(
                rpeaks_truths=all_rpeak_labels,
                rpeaks_preds=all_rpeak_preds,
                fs=self.train_config.fs,
                thr=self.train_config.qrs_mask_bias / self.train_config.fs,
            )
            # in case possible memeory leakage?
            del all_rpeak_preds, all_rpeak_labels
        elif self.train_config.task == "rr_lstm":
            all_preds = np.array([]).reshape(
                (0, self.train_config[self.train_config.task].input_len)
            )
            all_labels = np.array([]).reshape(
                (0, self.train_config[self.train_config.task].input_len)
            )
            all_weight_masks = np.array([]).reshape(
                (0, self.train_config[self.train_config.task].input_len)
            )
            for signals, labels, weight_masks in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy().squeeze(
                    -1
                )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                weight_masks = weight_masks.numpy().squeeze(
                    -1
                )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                all_labels = np.concatenate((all_labels, labels))
                all_weight_masks = np.concatenate((all_weight_masks, weight_masks))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                preds, _ = self._model.inference(signals)
                all_preds = np.concatenate((all_preds, preds))
            eval_res = compute_rr_metric(all_labels, all_preds, all_weight_masks)
            # in case possible memeory leakage?
            del all_preds, all_labels, all_weight_masks
        elif self.train_config.task == "main":
            all_preds = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len
                    // self.train_config.main.reduction,
                )
            )
            all_labels = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len
                    // self.train_config.main.reduction,
                )
            )
            all_weight_masks = np.array([]).reshape(
                (
                    0,
                    self.train_config.main.input_len
                    // self.train_config.main.reduction,
                )
            )
            for signals, labels, weight_masks in data_loader:
                signals = signals.to(device=self.device, dtype=self.dtype)
                labels = labels.numpy().squeeze(
                    -1
                )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                weight_masks = weight_masks.numpy().squeeze(
                    -1
                )  # (batch_size, seq_len, 1) -> (batch_size, seq_len)
                all_labels = np.concatenate((all_labels, labels))
                all_weight_masks = np.concatenate((all_weight_masks, weight_masks))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                preds, _ = self._model.inference(signals)
                all_preds = np.concatenate((all_preds, preds))
            eval_res = compute_main_task_metric(
                mask_truths=all_labels,
                mask_preds=all_preds,
                fs=self.train_config.fs,
                reduction=self.train_config.main.reduction,
                weight_masks=all_weight_masks,
            )
            # in case possible memeory leakage?
            del all_preds, all_labels, all_weight_masks

        self.model.train()

        return eval_res

    @property
    def batch_dim(self) -> int:
        """
        batch dimension, usually 0,
        but can be 1 for some models, e.g. RR_LSTM
        """
        # return 1 if self.train_config.task in ["rr_lstm"] else 0
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        """ """
        return [
            "task",
        ]

    @property
    def save_prefix(self) -> str:
        if self.train_config.task in ["rr_lstm"]:
            return f"task-{self.train_config.task}_{self._model.__name__}_epoch"
        else:
            return f"task-{self.train_config.task}_{self._model.__name__}_{self.model_config.cnn_name}_epoch"

    def extra_log_suffix(self) -> str:
        if self.train_config.task in ["rr_lstm"]:
            return f"task-{self.train_config.task}_{super().extra_log_suffix()}"
        else:
            return f"task-{self.train_config.task}_{super().extra_log_suffix()}_{self.model_config.cnn_name}"
