"""
"""

from copy import deepcopy
from pathlib import Path
from typing import List, NoReturn

from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from torch_ecg.cfg import CFG

from cfg import BaseCfg, TrainCfg


__all__ = [
    "PreTrainCfg",
    "PreTrainModelCfg",
]


_cwd = Path(__file__).resolve().parent


PreTrainCfg = deepcopy(TrainCfg)
for t in PreTrainCfg.tasks + [
    "tasks",
    "db_dir",
    "loss",
    "loss_kw",
    "outcomes",
    "classes",
    "states",
    "ignore_unannotated",
    "merge_rule",
    "early_stopping",
]:
    PreTrainCfg.pop(t)

PreTrainCfg.update(
    CFG(
        # new attributes
        classes=[],
        max_gumbel_temperature=2.0,
        min_gumbel_temperature=0.5,
        gumbel_temperature_decay=0.999995,
        max_duration_in_seconds=30.0,
        min_duration_in_seconds=5.0,
        pad_to_multiple_of=None,
        # old attributes
        optimizer="hf-adamw",
        lr_scheduler="linear",
        num_warmup_steps=600,
        log_dir=_cwd / "log",
        checkpoints=_cwd / "checkpoints",
        model_dir=_cwd / "saved_models",
        gradient_accumulation_steps=1,
        # input_len=int(30 * PreTrainCfg.fs),  # 30seconds, deprecated, use `max_duration_in_seconds`
    )
)
PreTrainCfg.log_dir.mkdir(parents=True, exist_ok=True)
PreTrainCfg.checkpoints.mkdir(parents=True, exist_ok=True)
PreTrainCfg.model_dir.mkdir(parents=True, exist_ok=True)

# PreTrainCfg.resample = CFG(fs=PreTrainCfg.fs)


# data directories
PreTrainCfg.cinc2022_dir = None
PreTrainCfg.cinc2016_dir = None
PreTrainCfg.ephnogram_dir = None


_DefaultFeatureExtractorCfg = CFG(
    feature_size=1,
    sampling_rate=BaseCfg.fs,
    padding_value=0.0,
    return_attention_mask=False,
    do_normalize=True,
)


_PreTrainModelCfg = CFG()

_PreTrainModelCfg.model_name = None


def register_model(model_name: str, model_cfg: dict) -> NoReturn:
    """register a new model configuration"""
    if model_name in [k for k in _PreTrainModelCfg if k != "model_name"]:
        raise ValueError(f"Model {model_name} already exists, choose another name.")
    assert hasattr(
        model_cfg, "model_config_args"
    ), "`model_config_args` must be defined in `model_cfg`"

    if "feature_extractor" not in model_cfg:
        model_cfg["feature_extractor"] = deepcopy(_DefaultFeatureExtractorCfg)

    assert (
        model_cfg["feature_extractor"]["sampling_rate"] == BaseCfg.fs
    ), "sampling_rate must be equal to BaseCfg.fs"
    assert (
        model_cfg["feature_extractor"]["do_normalize"] is True
    ), "do_normalize must be True"
    # adjust "return_attention_mask" according to "feat_extract_norm" of "model_config_args"
    if model_cfg["model_config_args"]["feat_extract_norm"] == "layer":
        model_cfg["feature_extractor"]["return_attention_mask"] = True
    elif model_cfg["model_config_args"]["feat_extract_norm"] == "group":
        model_cfg["feature_extractor"]["return_attention_mask"] = False

    _PreTrainModelCfg[model_name] = model_cfg


def list_models() -> List[str]:
    """ """
    return [
        k
        for k in _PreTrainModelCfg
        if not k.startswith("_") and k not in ["model_name"]
    ]


_wav2vec_base = CFG(
    model_name="base",
    model_config_args=CFG(
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=3,
        output_hidden_size=None,
    ),
    feature_extractor=deepcopy(_DefaultFeatureExtractorCfg),
)

_wav2vec_small = deepcopy(_wav2vec_base)
_wav2vec_small.model_name = "small"
_wav2vec_small.model_config_args.hidden_size = 3 * 2**7  # 384
_wav2vec_small.model_config_args.intermediate_size = 3 * 2**9  # 1536
_wav2vec_small.model_config_args.num_hidden_layers = 8
_wav2vec_small.model_config_args.num_attention_heads = 8

register_model("base", _wav2vec_base)
register_model("small", _wav2vec_small)

_PreTrainModelCfg.model_name = "base"


# PreTrainModelCfg = Wav2Vec2Config(**_PreTrainModelCfg[_PreTrainModelCfg.model_name])
PreTrainModelCfg = deepcopy(_PreTrainModelCfg[_PreTrainModelCfg.model_name])


def change_model(model_name: str) -> NoReturn:
    """change model configuration to the one specified by `model_name`"""
    assert model_name in [k for k in _PreTrainModelCfg if k != "model_name"]
    PreTrainModelCfg.update(**deepcopy(_PreTrainModelCfg[model_name]))


def get_Wav2Vec2Config() -> Wav2Vec2Config:
    return Wav2Vec2Config(**(PreTrainModelCfg.model_config_args))


def get_Wav2Vec2FeatureExtractor() -> Wav2Vec2FeatureExtractor:
    """ """
    return Wav2Vec2FeatureExtractor(**(PreTrainModelCfg.feature_extractor))


PreTrainModelCfg.change_model = change_model
PreTrainModelCfg.register_model = register_model
PreTrainModelCfg.list_models = list_models
PreTrainModelCfg.get_Wav2Vec2Config = get_Wav2Vec2Config
PreTrainModelCfg.get_Wav2Vec2FeatureExtractor = get_Wav2Vec2FeatureExtractor
