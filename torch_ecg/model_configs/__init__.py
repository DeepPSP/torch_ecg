"""
torch_ecg.model_configs
=======================

Ready-made configurations for the model architectures and their building
blocks.  A config is a plain :class:`~torch_ecg.cfg.CFG`; pick one, adapt
it, and pass it to the model constructor together with ``n_leads`` /
``n_classes`` (see the :ref:`tutorial <tutorial>`):

>>> from torch_ecg.model_configs import ECG_CRNN_CONFIG
>>> from torch_ecg.models import ECG_CRNN
>>> config = ECG_CRNN_CONFIG.copy()
>>> config.cnn.name = "resnet_block_leadwise"
>>> model = ECG_CRNN(config, n_leads=12, n_classes=26)

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Full model configs
------------------
Each top-level ``*_CONFIG`` collects the complete specification of one
architecture family (``fs``, ``cnn``, ``rnn``, ``attn``, ``clf``, ...);
its ``cnn`` / ``rnn`` / ``attn`` entries accept any of the building
blocks catalogued below.

===================  =========================================================
Config               Architecture
===================  =========================================================
ECG_CRNN_CONFIG      :class:`~torch_ecg.models.ECG_CRNN` (CNN + RNN + attention classifier)
ECG_SEQ_LAB_NET_CONFIG  :class:`~torch_ecg.models.ECG_SEQ_LAB_NET` (per-sample sequence labeling)
ECG_UNET_VANILLA_CONFIG  :class:`~torch_ecg.models.ECG_UNET` (U-Net segmentation)
ECG_SUBTRACT_UNET_CONFIG  subtractive U-Net (segmentation, e.g. QRS detection)
RR_LSTM_CONFIG       :class:`~torch_ecg.models.RR_LSTM` (RR-interval LSTM)
RR_AF_VANILLA_CONFIG / RR_AF_CRF_CONFIG  RR-LSTM heads: vanilla / CRF
ATI_CNN_CONFIG       Atari-CNN style small classifier
ECG_YOLO_CONFIG      YOLO-style detection
===================  =========================================================

CNN building blocks (``cnn`` submodule)
---------------------------------------
Grouped by family; ``*_leadwise`` variants use lead-wise (per-lead)
convolutions adapted to 12-lead ECG, ``*_gc`` / ``*_nl`` / ``*_se``
suffixes add global-context / non-local / squeeze-and-excitation
attention to the block.

- **VGG family**: ``vgg16``, ``vgg16_leadwise``, ``vgg_block_basic``, ``vgg_block_mish``, ``vgg_block_swish``
- **vanilla ResNet**: ``resnet_vanilla_18/34/50/101/152``, wide variants ``resnet_vanilla_wide_50_2``, ``resnet_vanilla_wide_101_2``, blocks ``resnet_block_basic``, ``resnet_bottle_neck``
- **CPSC2018-style ResNet** (small, contest-proven): ``resnet_cpsc2018``, ``resnet_cpsc2018_leadwise``, blocks ``cpsc_block_basic/mish/swish``, ``cpsc_2018``, ``cpsc_2018_leadwise``
- **scaled ResNet** (He et al.): ``resnetN``, ``resnetNB``, ``resnetNS``, ``resnetNBS``, bottle-neck variants ``resnet_bottle_neck_B/D``
- **stanford ResNet**: ``resnet_stanford``, ``resnet_block_stanford``
- **ResNet Nature Communications** (Moskalenko et al.): ``resnet_nature_comm`` + ``_bottle_neck`` and ``_gc``/``_nl``/``_se`` variants
- **TResNet** (together with the block configs above): ``tresnetN/S/M/L/P/F/XL``, ``tresnetM_V2``
- **DenseNet**: ``densenet_vanilla``, ``densenet_leadwise``
- **Xception**: ``xception_vanilla``, ``xception_leadwise``
- **MobileNet**: ``mobilenet_1_vanilla``, ``mobilenet_2_vanilla``, ``mobilenet_3_small``
- **MultiScopicCNN** (CPSC2019 winner): ``multi_scopic``, ``multi_scopic_block``, ``multi_scopic_leadwise``
- **ResNeXt**: ``resnext_vanilla_50_32x4d``, ``resnext_vanilla_101_32x8d``

Attention modules (``attn`` submodule)
--------------------------------------
Usable as the ``attn`` entry of the CRNN-family configs:

- ``squeeze_excitation`` (SE), ``global_context`` (GC), ``non_local`` (NL)
- ``transformer`` -- multi-head self-attention (``transformer_encoder`` style heads)

RNN configs (``rnn`` submodule)
-------------------------------
- ``lstm`` -- bidirectional LSTM used after the CNN feature extractor
- ``linear`` -- non-recurrent (linear) alternative

MLP / heads (``mlp`` submodule)
-------------------------------
- ``linear`` -- plain linear head
- ``mlp`` -- multi-layer perceptron classification head (with optional ``dropout``)

Notes
-----
``cnn.name`` selects a backbone (family name or full config name), e.g.
to switch the CRNN backbone to the SE-enhanced Nature-Communications
ResNet (the default of the cinc2021 benchmark)::

    config.cnn.name = "resnet_nature_comm_bottle_neck_se"

Concrete end-to-end usages are linked from the :ref:`examples <examples>`
(the benchmarks select and adapt these configs).
"""

from .ati_cnn import ATI_CNN_CONFIG
from .attn import global_context, non_local, squeeze_excitation, transformer
from .cnn import (  # vgg; ResNet; vanilla resnet; cpsc2018 resnet; smaller resnets; stanford resnet; ResNet Nature Communications; TResNet; cpsc2018 SOTA; multi_scopic; vanilla densenet; custom densenet; vanilla xception; custom xception; vanilla mobilenets
    cpsc_2018,
    cpsc_2018_leadwise,
    cpsc_block_basic,
    cpsc_block_mish,
    cpsc_block_swish,
    densenet_leadwise,
    densenet_vanilla,
    mobilenet_v1_vanilla,
    mobilenet_v2_vanilla,
    mobilenet_v3_small,
    multi_scopic,
    multi_scopic_block,
    multi_scopic_leadwise,
    resnet_block_basic,
    resnet_block_basic_gc,
    resnet_block_basic_nl,
    resnet_block_basic_se,
    resnet_block_stanford,
    resnet_bottle_neck,
    resnet_bottle_neck_B,
    resnet_bottle_neck_D,
    resnet_bottle_neck_gc,
    resnet_bottle_neck_nl,
    resnet_bottle_neck_se,
    resnet_cpsc2018,
    resnet_cpsc2018_leadwise,
    resnet_nature_comm,
    resnet_nature_comm_bottle_neck,
    resnet_nature_comm_bottle_neck_gc,
    resnet_nature_comm_bottle_neck_nl,
    resnet_nature_comm_bottle_neck_se,
    resnet_nature_comm_gc,
    resnet_nature_comm_nl,
    resnet_nature_comm_se,
    resnet_stanford,
    resnet_vanilla_18,
    resnet_vanilla_34,
    resnet_vanilla_50,
    resnet_vanilla_101,
    resnet_vanilla_152,
    resnet_vanilla_wide_50_2,
    resnet_vanilla_wide_101_2,
    resnetN,
    resnetNB,
    resnetNBS,
    resnetNS,
    resnext_vanilla_50_32x4d,
    resnext_vanilla_101_32x8d,
    tresnetF,
    tresnetL,
    tresnetM,
    tresnetM_V2,
    tresnetN,
    tresnetP,
    tresnetS,
    tresnetXL,
    vgg16,
    vgg16_leadwise,
    vgg_block_basic,
    vgg_block_mish,
    vgg_block_swish,
    xception_leadwise,
    xception_vanilla,
)
from .ecg_crnn import ECG_CRNN_CONFIG
from .ecg_seq_lab_net import ECG_SEQ_LAB_NET_CONFIG
from .ecg_subtract_unet import ECG_SUBTRACT_UNET_CONFIG
from .ecg_unet import ECG_UNET_VANILLA_CONFIG
from .ecg_yolo import ECG_YOLO_CONFIG
from .mlp import linear
from .rnn import attention, lstm
from .rr_lstm import RR_AF_CRF_CONFIG, RR_AF_VANILLA_CONFIG, RR_LSTM_CONFIG

__all__ = [
    # building blocks
    # CNN
    # vgg
    "vgg_block_basic",
    "vgg_block_mish",
    "vgg_block_swish",
    "vgg16",
    "vgg16_leadwise",
    # ResNet
    # resnet building blocks
    "resnet_block_basic",
    "resnet_bottle_neck",
    "resnet_bottle_neck_B",
    "resnet_bottle_neck_D",
    "resnet_block_basic_se",
    "resnet_bottle_neck_se",
    "resnet_block_basic_nl",
    "resnet_bottle_neck_nl",
    "resnet_block_basic_gc",
    "resnet_bottle_neck_gc",
    # smaller resnets
    "resnetN",
    "resnetNB",
    "resnetNS",
    "resnetNBS",
    # vanilla resnet
    "resnet_vanilla_18",
    "resnet_vanilla_34",
    "resnet_vanilla_50",
    "resnet_vanilla_101",
    "resnet_vanilla_152",
    "resnext_vanilla_50_32x4d",
    "resnext_vanilla_101_32x8d",
    "resnet_vanilla_wide_50_2",
    "resnet_vanilla_wide_101_2",
    # cpsc2018 resnet
    "resnet_cpsc2018",
    "resnet_cpsc2018_leadwise",
    # stanford resnet
    "resnet_block_stanford",
    "resnet_stanford",
    # ResNet Nature Communications
    "resnet_nature_comm",
    "resnet_nature_comm_se",
    "resnet_nature_comm_nl",
    "resnet_nature_comm_gc",
    "resnet_nature_comm_bottle_neck",
    "resnet_nature_comm_bottle_neck_se",
    "resnet_nature_comm_bottle_neck_gc",
    "resnet_nature_comm_bottle_neck_nl",
    # TresNet
    "tresnetF",
    "tresnetP",
    "tresnetN",
    "tresnetS",
    "tresnetM",
    "tresnetL",
    "tresnetXL",
    "tresnetM_V2",
    # cpsc2018 SOTA
    "cpsc_block_basic",
    "cpsc_block_mish",
    "cpsc_block_swish",
    "cpsc_2018",
    "cpsc_2018_leadwise",
    # multi_scopic
    "multi_scopic_block",
    "multi_scopic",
    "multi_scopic_leadwise",
    # vanilla densenet
    "densenet_vanilla",
    # custom densenet
    "densenet_leadwise",
    # vanilla xception
    "xception_vanilla",
    # custom xception
    "xception_leadwise",
    # vanilla mobilenets
    "mobilenet_v1_vanilla",
    "mobilenet_v2_vanilla",
    "mobilenet_v3_small",
    # RNN
    "lstm",
    "attention",
    # MLP
    "linear",
    # ATTN
    "non_local",
    "squeeze_excitation",
    "global_context",
    "transformer",
    # downstream tasks
    "ATI_CNN_CONFIG",
    "ECG_CRNN_CONFIG",
    "ECG_SEQ_LAB_NET_CONFIG",
    "ECG_SUBTRACT_UNET_CONFIG",
    "ECG_UNET_VANILLA_CONFIG",
    "ECG_YOLO_CONFIG",
    "RR_AF_CRF_CONFIG",
    "RR_AF_VANILLA_CONFIG",
    "RR_LSTM_CONFIG",
]
