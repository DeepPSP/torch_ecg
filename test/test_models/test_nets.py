"""
test of the classes from models._nets.py
"""

import torch
import pytest

from torch_ecg.models._nets import (  # noqa: F401
    Mish,
    Swish,
    Hardswish,
    Initializers,
    Activations,
    Bn_Activation,
    Conv_Bn_Activation,
    CBA,
    MultiConv,
    BranchedConv,
    SeparableConv,
    DeformConv,
    AntiAliasConv,
    DownSample,
    BlurPool,
    BidirectionalLSTM,
    StackedLSTM,
    AttentionWithContext,
    MultiHeadAttention,
    SelfAttention,
    AttentivePooling,
    ZeroPadding,
    ZeroPad1d,
    SeqLin,
    MLP,
    NonLocalBlock,
    SEBlock,
    GlobalContextBlock,
    CBAMBlock,
    CRF,
    ExtendedCRF,
    SpaceToDepth,
    MLDecoder,
    DropPath,
    make_attention_layer,
    get_activation,
    get_normalization,
)  # noqa: F401


BATCH_SIZE = 32
IN_CHANNELS = 12
SEQ_LEN = 5000
SAMPLE_INPUT = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)


def test_activations():
    # Mish, Swish, Hardswish
    # get_activation

    mish = Mish(inplace=False)
    swish = Swish()
    hard_swish = Hardswish()

    out = mish(SAMPLE_INPUT)
    assert out.shape == SAMPLE_INPUT.shape
    out = swish(SAMPLE_INPUT)
    assert out.shape == SAMPLE_INPUT.shape
    out = hard_swish(SAMPLE_INPUT)
    assert out.shape == SAMPLE_INPUT.shape

    assert get_activation(None) is None

    assert get_activation("mish") == Mish
    assert isinstance(get_activation(Mish, dict(inplace=False)), Mish)
    assert get_activation(Mish, dict(inplace=False)) != mish
    assert get_activation(mish, dict(inplace=True)) == mish

    assert isinstance(get_activation("mish", {}), Mish)
    assert isinstance(get_activation(Mish, {}), Mish)
    assert isinstance(get_activation(mish, {}), Mish)

    for name in Activations:
        act = get_activation(name)
        assert act == Activations[name]
        act = get_activation(name, {})
        if name not in ["glu"]:
            assert act(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    act = get_activation("leaky", kw_act=dict(negative_slope=0.1))
    assert act.negative_slope == 0.1

    class SomeClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    with pytest.raises(ValueError, match="activation `.+` not supported"):
        get_activation("not_supported")
    with pytest.raises(ValueError, match="activation `.+` not supported"):
        get_activation(123)
    with pytest.raises(ValueError, match="activation `.+` not supported"):
        get_activation(SomeClass(1, 2))
    with pytest.raises(ValueError, match="activation `.+` not supported"):
        get_activation(SomeClass)


def test_cba():
    cba = Conv_Bn_Activation(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=11,
        stride=3,
        activation="relu6",
        groups=1,
    )
    assert [item.__class__.__name__ for item in cba.children()] == [
        "Conv1d",
        "BatchNorm1d",
        "ReLU6",
    ]
    assert cba(SAMPLE_INPUT).shape == cba.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    cba_alpha = Conv_Bn_Activation(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=5,
        stride=1,
        activation="hardswish",
        groups=12,
        width_multiplier=1.5,
        depth_multiplier=2,
        conv_type="separable",
    )
    assert cba_alpha(SAMPLE_INPUT).shape == cba_alpha.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    cab = Conv_Bn_Activation(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=5,
        stride=1,
        activation="relu6",
        groups=12,
        ordering="cab",
    )
    assert [item.__class__.__name__ for item in cab.children()] == [
        "Conv1d",
        "ReLU6",
        "BatchNorm1d",
    ]
    assert cab(SAMPLE_INPUT).shape == cab.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    bac = Conv_Bn_Activation(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=5,
        stride=1,
        activation="mish",
        groups=12,
        ordering="bac",
    )
    assert [item.__class__.__name__ for item in bac.children()] == [
        "BatchNorm1d",
        "Mish",
        "Conv1d",
    ]
    assert bac(SAMPLE_INPUT).shape == bac.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )


def test_normalization():
    pass


def test_initializer():
    pass


def test_multi_conv():
    pass


def test_branched_conv():
    pass


def test_separable_conv():
    pass


def test_deform_conv():
    pass


def test_anti_alias_conv():
    pass


def test_down_sample():
    pass


def test_blur_pool():
    pass


def test_bidirectional_lstm():
    pass


def test_stacked_lstm():
    pass


def test_attention_with_context():
    pass


def test_multi_head_attention():
    pass


def test_self_attention():
    pass


def test_attentive_pooling():
    pass


def test_zero_padding():
    pass


def test_zero_pad_1d():
    pass


def test_mlp():
    pass


def test_attention_blocks():
    # NonLocalBlock, SEBlock, GlobalContextBlock, CBAMBlock,
    # make_attention_layer
    pass


def test_crf():
    # CRF, ExtendedCRF,
    pass


def test_s2d():
    # SpaceToDepth
    pass


def test_mldecoder():
    # MLDecoder
    pass


def test_droppath():
    # DropPath
    pass
