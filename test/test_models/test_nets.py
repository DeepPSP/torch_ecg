"""
test of the classes from models._nets.py
"""

import inspect
import itertools

import torch
import pytest

from torch_ecg.models._nets import (  # noqa: F401
    Mish,
    Swish,
    Hardswish,
    Initializers,
    Activations,
    Normalizations,
    Bn_Activation,
    Conv_Bn_Activation,
    CBA,
    MultiConv,
    BranchedConv,
    SeparableConv,
    # DeformConv,
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
    # "BAMBlock",
    # "CoordAttention",
    # "GEBlock",
    # "SKBlock",
    CRF,
    ExtendedCRF,
    SpaceToDepth,
    MLDecoder,
    DropPath,
    make_attention_layer,
    get_activation,
    get_normalization,
    # internal
    _DEFAULT_CONV_CONFIGS,
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


def test_normalization():
    assert get_normalization(None) is None

    bn = torch.nn.BatchNorm1d(IN_CHANNELS)

    assert get_normalization("batch_norm") == torch.nn.BatchNorm1d
    assert isinstance(
        get_normalization(
            "batch_normalization", kw_norm=dict(num_features=IN_CHANNELS)
        ),
        torch.nn.BatchNorm1d,
    )
    assert (
        get_normalization("batch_normalization", kw_norm=dict(num_features=IN_CHANNELS))
        != bn
    )
    assert get_normalization(bn, kw_norm=dict(num_features=IN_CHANNELS)) == bn

    norm = get_normalization(
        "batch_norm", kw_norm=dict(num_features=IN_CHANNELS, momentum=0.01)
    )
    assert norm.momentum == 0.01
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    norm = get_normalization(
        "group_norm", kw_norm=dict(num_channels=IN_CHANNELS, num_groups=IN_CHANNELS)
    )
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    norm = get_normalization(
        "group_norm", kw_norm=dict(num_features=IN_CHANNELS, num_groups=IN_CHANNELS)
    )
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    norm = get_normalization("layer_norm", kw_norm=dict(normalized_shape=SEQ_LEN))
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    norm = get_normalization("instance_norm", kw_norm=dict(num_features=IN_CHANNELS))
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    norm = get_normalization("local_response_norm", kw_norm=dict(size=IN_CHANNELS // 4))
    assert norm(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    class SomeClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        get_normalization("not_supported")
    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        get_normalization(123)
    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        get_normalization(SomeClass(1, 2))
    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        get_normalization(SomeClass)


def test_initializer():
    for name in Initializers:
        assert inspect.isfunction(Initializers[name])


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


def test_multi_conv():
    mc = MultiConv(
        in_channels=IN_CHANNELS,
        out_channels=[IN_CHANNELS * 4, IN_CHANNELS * 8, IN_CHANNELS * 16],
        filter_lengths=5,
        subsample_lengths=[1, 2, 1],
        dilations=2,
        groups=IN_CHANNELS // 2,
        dropouts=[0.1, 0.2, 0.0],
        activation="mish",
    )
    assert mc(SAMPLE_INPUT).shape == mc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )


def test_branched_conv():
    bc = BranchedConv(
        in_channels=IN_CHANNELS,
        out_channels=[
            [IN_CHANNELS * 4, IN_CHANNELS * 8, IN_CHANNELS * 16, IN_CHANNELS * 32],
            [IN_CHANNELS * 8, IN_CHANNELS * 32],
        ],
        filter_lengths=[5, [3, 7]],
        subsample_lengths=2,
        dilations=[[1, 2, 1, 2], [4, 8]],
        groups=IN_CHANNELS // 2,
        dropouts=0.1,
    )
    out_tensors = bc(SAMPLE_INPUT)
    assert isinstance(out_tensors, list) and len(out_tensors) == 2
    assert all(isinstance(t, torch.Tensor) for t in out_tensors)
    assert [t.shape for t in out_tensors] == bc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )


def test_separable_conv():
    sc = SeparableConv(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=7,
        stride=2,
        padding=3,
        dilation=2,
        groups=IN_CHANNELS // 3,
    )
    assert sc(SAMPLE_INPUT).shape == sc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )


def test_deform_conv():
    pass  # NOT IMPLEMENTED


def test_blur_pool():
    grid = itertools.product(
        [1, 2, 5],  # down_scale
        range(1, 8),  # filt_size
        ["reflect", "replicate", "zero"],  # pad_type
        [0, 1, 2],  # pad_off
    )
    for down_scale, filt_size, pad_type, pad_off in grid:
        bp = BlurPool(
            in_channels=IN_CHANNELS,
            down_scale=down_scale,
            filt_size=filt_size,
            pad_type=pad_type,
            pad_off=pad_off,
        )
        assert bp(SAMPLE_INPUT).shape == bp.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )


def test_anti_alias_conv():
    aac = AntiAliasConv(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=11,
        stride=1,
        padding=5,
        dilation=2,
        groups=IN_CHANNELS // 4,
    )
    assert aac(SAMPLE_INPUT).shape == aac.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    aac = AntiAliasConv(
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS * 4,
        kernel_size=11,
        stride=3,
        padding=None,
        dilation=2,
        groups=IN_CHANNELS // 2,
    )
    assert aac(SAMPLE_INPUT).shape == aac.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )


def test_down_sample():
    grid = itertools.product(
        [1, 2, 5],  # down_scale
        ["max", "avg", "lp", "conv", "blur"],  # mode
        [True, False],  # batch_norm
        [None, 5, 11],  # kernel_size
        [None, IN_CHANNELS * 2],  # out_channels
        # [0, 1, 2],  # padding
        [0],  # padding, TODO: test padding other than 0
    )
    for down_scale, mode, batch_norm, kernel_size, out_channels, padding in grid:
        # print(down_scale, mode, batch_norm, kernel_size, out_channels)
        ds = DownSample(
            in_channels=IN_CHANNELS,
            down_scale=down_scale,
            mode=mode,
            batch_norm=batch_norm,
            kernel_size=kernel_size,
            out_channels=out_channels,
            padding=padding,
        )
        assert ds(SAMPLE_INPUT).shape == ds.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )


def test_bidirectional_lstm():
    grid = itertools.product(
        [1, 2, 3],  # num_layers
        [True, False],  # bias
        [0.0, 0.1, 0.5],  # dropout
        [True, False],  # return_sequences
    )
    sample_input = torch.randn(SEQ_LEN // 50, BATCH_SIZE, IN_CHANNELS)
    for num_layers, bias, dropout, return_sequences in grid:
        if num_layers == 1:
            dropout = 0.0
        bi_lstm = BidirectionalLSTM(
            input_size=IN_CHANNELS,
            hidden_size=IN_CHANNELS * 2,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            return_sequences=return_sequences,
        )
        assert bi_lstm(sample_input).shape == bi_lstm.compute_output_shape(
            seq_len=SEQ_LEN // 50, batch_size=BATCH_SIZE
        )


def test_stacked_lstm():
    grid = itertools.product(
        [
            [IN_CHANNELS * 4],
            [IN_CHANNELS * 2, IN_CHANNELS * 4],
            [IN_CHANNELS * 2, IN_CHANNELS * 4, IN_CHANNELS * 8],
        ],  # hidden_sizes
        [True, False],  # bias
        [0.0, 0.1, 0.5],  # dropouts
        [True, False],  # bidirectional
        [True, False],  # return_sequences
    )
    sample_input = torch.randn(SEQ_LEN // 50, BATCH_SIZE, IN_CHANNELS)
    for hidden_sizes, bias, dropouts, bidirectional, return_sequences in grid:
        slstm = StackedLSTM(
            input_size=IN_CHANNELS,
            hidden_sizes=hidden_sizes,
            bias=bias,
            dropout=dropouts,
            bidirectional=bidirectional,
            return_sequences=return_sequences,
        )
        assert slstm(sample_input).shape == slstm.compute_output_shape(
            seq_len=SEQ_LEN // 50, batch_size=BATCH_SIZE
        )

    slstm = StackedLSTM(
        input_size=IN_CHANNELS,
        hidden_sizes=[IN_CHANNELS * 2, IN_CHANNELS * 4, IN_CHANNELS * 8],
        bias=[True, False, True],
        dropout=[0.0, 0.2, 0.1],
    )
    assert slstm(sample_input).shape == slstm.compute_output_shape(
        seq_len=SEQ_LEN // 50, batch_size=BATCH_SIZE
    )


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
