"""Configs for the CPSC2019 R-peak detection benchmark example.

Three architectures are provided for comparison ("multi"): a sequence
labelling CRNN, a UNet, and a subtract-UNet; the training config comes
from the library's `CPSC2019TrainCfg`.
"""

from copy import deepcopy

from torch_ecg.cfg import CFG
from torch_ecg.databases.datasets.cpsc2019 import CPSC2019TrainCfg
from torch_ecg.model_configs import ECG_SEQ_LAB_NET_CONFIG, ECG_SUBTRACT_UNET_CONFIG, ECG_UNET_VANILLA_CONFIG
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths

__all__ = [
    "TrainCfg",
    "ModelCfg",
]


TrainCfg = deepcopy(CPSC2019TrainCfg)


_COMMON_MODEL_CONFIGS = CFG()
_COMMON_MODEL_CONFIGS.skip_dist = TrainCfg.skip_dist
_COMMON_MODEL_CONFIGS.torch_dtype = TrainCfg.torch_dtype
_COMMON_MODEL_CONFIGS.fs = TrainCfg.fs
_COMMON_MODEL_CONFIGS.spacing = 1000 / TrainCfg.fs
# NOTE(update): "background" now do not count as a class
_COMMON_MODEL_CONFIGS.classes = deepcopy(TrainCfg.classes)
# _COMMON_MODEL_CONFIGS.classes = ["i", "N"]  # N for qrs, i for other parts
# _COMMON_MODEL_CONFIGS.class_map = {c:i for i,c in enumerate(TrainCfg.classes)}
_COMMON_MODEL_CONFIGS.n_leads = TrainCfg.n_leads
_COMMON_MODEL_CONFIGS.skip_dist = TrainCfg.skip_dist


ModelCfg = deepcopy(_COMMON_MODEL_CONFIGS)
ModelCfg.seq_lab_crnn = adjust_cnn_filter_lengths(
    deepcopy(ECG_SEQ_LAB_NET_CONFIG),
    TrainCfg.fs,
)
ModelCfg.seq_lab_crnn.reduction = 2**3
ModelCfg.seq_lab_crnn.recover_length = True
ModelCfg.seq_lab_crnn.update(deepcopy(_COMMON_MODEL_CONFIGS))

# NOTE: one can adjust any of the cnn, rnn, attn, clf part of ModelCfg.seq_lab_crnn like ModelCfg.seq_lab_cnn


ModelCfg.seq_lab_cnn = deepcopy(ModelCfg.seq_lab_crnn)

ModelCfg.seq_lab_cnn.rnn = CFG()
ModelCfg.seq_lab_cnn.rnn.name = "none"  # "lstm"


ModelCfg.unet = adjust_cnn_filter_lengths(
    deepcopy(ECG_UNET_VANILLA_CONFIG),
    TrainCfg.fs,
)
ModelCfg.unet.reduction = 1
ModelCfg.unet.recover_length = True
ModelCfg.unet.update(deepcopy(_COMMON_MODEL_CONFIGS))
ModelCfg.unet.cnn = CFG(name="none")


ModelCfg.subtract_unet = adjust_cnn_filter_lengths(
    deepcopy(ECG_SUBTRACT_UNET_CONFIG),
    TrainCfg.fs,
)
ModelCfg.subtract_unet.reduction = 1
ModelCfg.subtract_unet.recover_length = True
ModelCfg.subtract_unet.update(deepcopy(_COMMON_MODEL_CONFIGS))
ModelCfg.subtract_unet.cnn = CFG(name="none")
