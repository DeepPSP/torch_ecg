"""
"""

import shutil
from copy import deepcopy
from pathlib import Path
from typing import NoReturn, Optional, Any, Sequence, Union

import pytest
import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP

try:
    import torch_ecg
except:
    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))
    import torch_ecg

from torch_ecg.databases import LUDB
from torch_ecg.databases.datasets.ludb import LUDBDataset, LUDBTrainCfg
from torch_ecg.databases.physionet_databases.ludb import (
    compute_metrics as compute_ludb_metrics,
)
from torch_ecg.components.outputs import WaveDelineationOutput
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils import ecg_arrhythmia_knowledge as EAK
from torch_ecg.models.unets.ecg_unet import ECG_UNET
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG
from torch_ecg.utils.utils_nn import (
    default_collate_fn as collate_fn,
    adjust_cnn_filter_lengths,
)
from torch_ecg.components.trainer import BaseTrainer


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parent.parent / "tmp"
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
ModelCfg.torch_dtype = DEFAULTS.torch_dtype
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
    ) -> NoReturn:
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


def test_unet_ludb_pipeline() -> NoReturn:
    """ """

    train_cfg_fl = deepcopy(TrainCfg)
    train_cfg_fl.use_single_lead = False
    train_cfg_fl.loss = "FocalLoss"

    train_cfg_fl.db_dir = _DB_DIR
    train_cfg_fl.log_dir = _CWD / "logs"
    train_cfg_fl.model_dir = _CWD / "saved_models"
    train_cfg_fl.checkpoints = _CWD / "checkpoints"
    train_cfg_fl.log_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_fl.model_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_fl.checkpoints.mkdir(parents=True, exist_ok=True)

    # train_cfg_ce = deepcopy(TrainCfg)
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

    shutil.rmtree(_CWD)
