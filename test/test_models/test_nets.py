"""
test of the classes from models._nets.py

TODO: add more test for error raising
"""

import functools
import inspect
import itertools
import operator

import torch
import pytest
from tqdm.auto import tqdm

from torch_ecg.models._nets import (
    Initializers,
    Activations,
    # Normalizations,
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
    _ScaledDotProductAttention,
)


BATCH_SIZE = 4
IN_CHANNELS = 12
SEQ_LEN = 2000
SAMPLE_INPUT = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)


@torch.no_grad()
def test_activations():

    for name in Activations:
        act = get_activation(name)
        assert act == Activations[name]
        if name == "softmax":
            kw = dict(dim=-1)
        else:
            kw = {}
        act = get_activation(name, kw)
        if name not in ["glu"]:
            assert act(SAMPLE_INPUT).shape == SAMPLE_INPUT.shape

    act = get_activation("leaky", kw_act=dict(negative_slope=0.1))
    assert act.negative_slope == 0.1

    mish = torch.nn.Mish()
    act = get_activation(mish)
    assert act is mish

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


@torch.no_grad()
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


@torch.no_grad()
def test_ba():
    grid = itertools.product(
        ["batch_norm", "group_norm"],  # norm
        ["mish", "leaky"],  # activation
        [0.0, 0.1, {"p": 0.3, "type": None}, {"p": 0.3, "type": "1d"}],  # dropout
    )
    for norm, activation, dropout in grid:
        if norm == "group_norm":
            kw_norm = dict(num_groups=IN_CHANNELS)
        else:
            kw_norm = None
        ba = Bn_Activation(
            num_features=IN_CHANNELS,
            norm=norm,
            activation=activation,
            kw_norm=kw_norm,
            dropout=dropout,
        )
        ba.eval()
        assert ba(SAMPLE_INPUT).shape == ba.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )

    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        Bn_Activation(num_features=IN_CHANNELS, norm="not_supported")

    with pytest.raises(ValueError, match="unknown type of normalization: `.+`"):
        Bn_Activation(num_features=IN_CHANNELS, norm=1)


@torch.no_grad()
def test_cba():
    grid_dict = dict(
        kernel_size=[1, 2, 11, 16],  # kernel_size
        stride=[1, 2, 5],  # stride
        padding=[None, 0, 2],  # padding
        dilation=[1, 2, 7],  # dilation
        groups=[1, 2, IN_CHANNELS],  # groups
        norm=[None, True, "group_norm"],  # norm
        activation=[None, "leaky"],  # activation
        bias=[True, False],  # bias
        ordering=[
            "cab",
            "bac",
            "bca",
            "acb",
            "bc",
            "cb",
            "ac",
            "ca",
        ],  # ordering
        conv_type=[None, "separable", "aa"],  # conv_type
        alpha=[None, 2],  # alpha (width_multiplier)
    )
    grid = itertools.product(*grid_dict.values())
    grid_len = functools.reduce(operator.mul, map(len, grid_dict.values()))
    for (
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        norm,
        activation,
        bias,
        ordering,
        conv_type,
        alpha,
    ) in tqdm(
        grid, mininterval=3.0, total=grid_len, desc="Testing CBA", dynamic_ncols=True
    ):
        if not norm and "b" in ordering:
            continue
        if norm and "b" not in ordering:
            continue
        if not activation and "a" in ordering:
            continue
        if activation and "a" not in ordering:
            continue
        config = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            norm=norm,
            activation=activation,
            bias=bias,
            ordering=ordering,
            conv_type=conv_type,
            alpha=alpha,
        )
        cba = Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 8,
            **config,
        )
        cba.eval()
        assert cba(SAMPLE_INPUT).shape == cba.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )

    cba = CBA(
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
    cab = CBA(
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
    bac = CBA(
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

    with pytest.raises(AssertionError, match="convolution must be included"):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 4,
            kernel_size=5,
            stride=1,
            ordering="ab",
            activation="gelu",
        )

    with pytest.raises(
        AssertionError,
        match="`width_multiplier` .+ makes `out_channels` .+ not divisible by `groups`",
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            groups=12,
            width_multiplier=1.5,
            activation="gelu",
        )

    with pytest.raises(
        NotImplementedError, match="convolution of type `.+` not implemented yet"
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            conv_type="deformable",
        )

    with pytest.raises(ValueError, match="initializer `.+` not supported"):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            kernel_initializer="not_supported",
        )

    with pytest.raises(ValueError, match="normalization `.+` not supported"):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            norm="not_supported",
        )

    with pytest.raises(ValueError, match="unknown type of normalization: `.+`"):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            norm=1,
        )

    with pytest.raises(ValueError, match="`ordering` \\(.+\\) not supported"):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            activation="gelu",
            ordering="abc",
        )

    with pytest.warns(
        RuntimeWarning,
        match="normalization is specified by `norm` but not included in `ordering`",
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            groups=IN_CHANNELS // 2,
            norm="group_norm",
            ordering="ca",
        )
    with pytest.warns(
        RuntimeWarning,
        match="normalization is specified in `ordering` but not by `norm`",
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            groups=IN_CHANNELS // 2,
            norm=None,
            ordering="cab",
        )
    with pytest.warns(
        RuntimeWarning,
        match="activation is specified by `activation` but not included in `ordering`",
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            groups=IN_CHANNELS // 2,
            activation="relu",
            ordering="cb",
        )
    with pytest.warns(
        RuntimeWarning,
        match="activation is specified in `ordering` but not by `activation`",
    ):
        Conv_Bn_Activation(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 3,
            kernel_size=5,
            stride=1,
            groups=IN_CHANNELS // 2,
            activation=None,
            ordering="cab",
        )


@torch.no_grad()
def test_multi_conv():
    mc = MultiConv(
        in_channels=IN_CHANNELS,
        out_channels=[IN_CHANNELS * 4, IN_CHANNELS * 8, IN_CHANNELS * 16],
        filter_lengths=5,
        subsample_lengths=[1, 2, 1],
        dilations=2,
        groups=IN_CHANNELS // 2,
        dropouts=[0.1, {"p": 0.3, "type": None}, {"p": 0.3, "type": "1d"}],
        activation="mish",
    )
    mc.eval()
    assert mc(SAMPLE_INPUT).shape == mc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    assert mc.in_channels == IN_CHANNELS


@torch.no_grad()
def test_branched_conv():
    bc_config = dict(
        in_channels=IN_CHANNELS,
        out_channels=[
            [IN_CHANNELS * 4, IN_CHANNELS * 8, IN_CHANNELS * 16, IN_CHANNELS * 32],
            [IN_CHANNELS * 8, IN_CHANNELS * 32],
        ],
        filter_lengths=[5, [3, 7]],
        subsample_lengths=2,
        dilations=[[1, 2, 1, 2], [4, 8]],
        groups=IN_CHANNELS // 2,
        dropouts=[[0.0, 0.2, 0.2, {"p": 0.1, "type": "1d"}], {"p": 0.3, "type": "1d"}],
    )
    bc = BranchedConv(**bc_config)
    bc.eval()
    out_tensors = bc(SAMPLE_INPUT)
    assert isinstance(out_tensors, list) and len(out_tensors) == 2
    assert all(isinstance(t, torch.Tensor) for t in out_tensors)
    assert [t.shape for t in out_tensors] == bc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    assert bc.in_channels == IN_CHANNELS
    bc_config["dropouts"] = {"p": 0.2, "type": "1d"}
    bc = BranchedConv(**bc_config)
    bc.eval()
    out_tensors = bc(SAMPLE_INPUT)
    assert isinstance(out_tensors, list) and len(out_tensors) == 2
    assert all(isinstance(t, torch.Tensor) for t in out_tensors)
    assert [t.shape for t in out_tensors] == bc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    assert bc.in_channels == IN_CHANNELS


@torch.no_grad()
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
    sc.eval()
    assert sc(SAMPLE_INPUT).shape == sc.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    assert sc.in_channels == IN_CHANNELS


@torch.no_grad()
def test_deform_conv():
    pass  # NOT IMPLEMENTED


@torch.no_grad()
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
        bp.eval()
        assert bp(SAMPLE_INPUT).shape == bp.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert bp.in_channels == IN_CHANNELS
    repr(bp)

    with pytest.raises(
        NotImplementedError, match="Filter size of `\\d+` is not implemented"
    ):
        BlurPool(in_channels=IN_CHANNELS, down_scale=3, filt_size=10)
    with pytest.raises(
        NotImplementedError, match="Padding type of `.+` is not implemented"
    ):
        BlurPool(in_channels=IN_CHANNELS, down_scale=3, pad_type="xxx")


@torch.no_grad()
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
    aac.eval()
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
    aac.eval()
    assert aac(SAMPLE_INPUT).shape == aac.compute_output_shape(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    assert aac.in_channels == IN_CHANNELS


@torch.no_grad()
def test_down_sample():
    grid = itertools.product(
        [1, 2, 5],  # down_scale
        ["max", "avg", "lp", "conv", "blur"],  # mode
        [True, False],  # norm
        [None, 5, 11],  # kernel_size
        [None, IN_CHANNELS * 2],  # out_channels
        # [0, 1, 2],  # padding
        [0],  # padding, TODO: test padding other than 0
    )
    for down_scale, mode, norm, kernel_size, out_channels, padding in grid:
        # print(down_scale, mode, norm, kernel_size, out_channels)
        ds = DownSample(
            in_channels=IN_CHANNELS,
            down_scale=down_scale,
            mode=mode,
            norm=norm,
            kernel_size=kernel_size,
            out_channels=out_channels,
            padding=padding,
        )
        ds.eval()
        assert ds(SAMPLE_INPUT).shape == ds.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert ds.in_channels == IN_CHANNELS

    for mode in ["nearest", "area", "linear", "lse"]:
        with pytest.raises(NotImplementedError):
            ds = DownSample(in_channels=IN_CHANNELS, down_scale=2, mode=mode)


@torch.no_grad()
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
        bi_lstm.eval()
        assert bi_lstm(sample_input).shape == bi_lstm.compute_output_shape(
            seq_len=SEQ_LEN // 50, batch_size=BATCH_SIZE
        )


@torch.no_grad()
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
        slstm.eval()
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


@torch.no_grad()
def test_attention_with_context():
    grid = itertools.product(
        [True, False],  # bias
        [
            "glorot_uniform",
            "glorot_normal",
            "he_uniform",
            "he_normal",
            "xavier_normal",
            "xavier_uniform",
            "normal",
            "uniform",
            "orthogonal",
        ],  # initializer
    )
    for bias, initializer in grid:
        awc = AttentionWithContext(
            in_channels=IN_CHANNELS,
            bias=bias,
            initializer=initializer,
        )
        awc.eval()
        assert awc(SAMPLE_INPUT).shape == awc.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )

    with pytest.raises(
        AssertionError, match="at least one of `seq_len` and `batch_size` must be given"
    ):
        awc.compute_output_shape(seq_len=None, batch_size=None)


@torch.no_grad()
def test_multi_head_attention():
    grid = itertools.product(
        [True, False],  # bias
        [2, 6, 12],  # num_heads
    )
    q, k, v = (torch.randn(SEQ_LEN // 10, BATCH_SIZE, IN_CHANNELS) for _ in range(3))
    for bias, num_heads in grid:
        mha = MultiHeadAttention(
            embed_dim=IN_CHANNELS,
            num_heads=num_heads,
            bias=bias,
        )
        mha.eval()
        assert mha(q, k, v)[0].shape == mha.compute_output_shape(
            seq_len=SEQ_LEN // 10, batch_size=BATCH_SIZE
        )
    repr(mha)


@torch.no_grad()
def test_self_attention():
    grid = itertools.product(
        [True, False],  # bias
        [2, 6, 12],  # num_heads
        [0.0, 0.1],  # dropout
    )
    sample_input = torch.randn(SEQ_LEN // 10, BATCH_SIZE, IN_CHANNELS)
    for bias, num_heads, dropout in grid:
        sa = SelfAttention(
            embed_dim=IN_CHANNELS,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
        )
        sa.eval()
        assert sa(sample_input).shape == sa.compute_output_shape(
            seq_len=SEQ_LEN // 10, batch_size=BATCH_SIZE
        )
    with pytest.warns(
        RuntimeWarning,
        match="`embed_dim`\\(.+\\) is not divisible by `num_heads`\\(.+\\)",
    ):
        SelfAttention(embed_dim=IN_CHANNELS, num_heads=5)


@torch.no_grad()
def test_attentive_pooling():
    grid = itertools.product(
        [None, IN_CHANNELS // 2, IN_CHANNELS * 2],  # mid_channels
        ["tanh", "sigmoid", "softmax"],  # activation
        [0, 0.1, 0.5],  # dropout
    )
    for mid_channels, activation, dropout in grid:
        if activation == "softmax":
            kw_activation = dict(dim=-1)
        else:
            kw_activation = {}
        ap = AttentivePooling(
            in_channels=IN_CHANNELS,
            mid_channels=mid_channels,
            activation=activation,
            dropout=dropout,
            kw_activation=kw_activation,
        )
        ap.eval()
        assert ap(SAMPLE_INPUT).shape == ap.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert ap.in_channels == IN_CHANNELS


@torch.no_grad()
def test_zero_padding():
    grid = itertools.product(
        [IN_CHANNELS, IN_CHANNELS * 2],  # out_channels
        ZeroPadding.__LOC__,  # loc
    )
    for out_channels, loc in grid:
        zp = ZeroPadding(
            in_channels=IN_CHANNELS,
            out_channels=out_channels,
            loc=loc,
        )
        zp.eval()
        assert zp(SAMPLE_INPUT).shape == zp.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )

    with pytest.raises(AssertionError, match="`loc` must be in"):
        ZeroPadding(
            in_channels=IN_CHANNELS, out_channels=IN_CHANNELS * 2, loc="invalid"
        )

    with pytest.raises(AssertionError, match="`out_channels` must be >= `in_channels`"):
        ZeroPadding(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS // 2)


@torch.no_grad()
def test_zero_pad_1d():
    for padding in [2, [1, 1], [0, 3]]:
        zp = ZeroPad1d(padding=padding)
        zp.eval()
        assert zp(SAMPLE_INPUT).shape == zp.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE, in_channels=IN_CHANNELS
        )

    with pytest.raises(
        AssertionError,
        match="`padding` must be non-negative int or a 2-sequence of non-negative int",
    ):
        ZeroPad1d(padding=[1, 2, 3])
    with pytest.raises(
        AssertionError,
        match="`padding` must be non-negative int or a 2-sequence of non-negative int",
    ):
        ZeroPad1d(padding=-1)
    with pytest.raises(
        AssertionError,
        match="`padding` must be non-negative int or a 2-sequence of non-negative int",
    ):
        ZeroPad1d(padding=[1, 2.3])


@torch.no_grad()
def test_mlp():
    out_channels = [IN_CHANNELS * 2, IN_CHANNELS * 4, 26]  # out_channels
    grid = itertools.product(
        ["mish", "relu", "leaky", "gelu", "tanh"],  # activation
        [0, 0.1, [0.1, 0.2, 0.0]],  # dropout
        [True, False],  # bias
        [
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        ],  # kernel_initializer
    )
    for activation, dropout, bias, kernel_initializer in grid:
        mlp = MLP(
            in_channels=IN_CHANNELS,
            out_channels=out_channels,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias=bias,
            dropout=dropout,
        )
        mlp.eval()
        assert mlp(SAMPLE_INPUT.permute(0, 2, 1)).shape == mlp.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert mlp.in_channels == IN_CHANNELS

    with pytest.raises(
        AssertionError,
        match="`out_channels` indicates `\\d+` linear layers, while `dropouts` indicates `\\d+`",
    ):
        SeqLin(
            in_channels=IN_CHANNELS,
            out_channels=[IN_CHANNELS * 2, IN_CHANNELS * 4, 26],
            dropouts=[0.1, 0.2],
        )


@torch.no_grad()
def test_attention_blocks():
    # NonLocalBlock, SEBlock, GlobalContextBlock, CBAMBlock,
    # make_attention_layer

    grid_nl = itertools.product(
        [IN_CHANNELS, IN_CHANNELS * 2],  # mid_channels
        [2, {"g": 1, "phi": 3, "theta": 2, "W": 3}],  # filter_lengths
        [1, 2],  # subsample_length
    )
    for mid_channels, filter_lengths, subsample_length in grid_nl:
        config = dict(
            mid_channels=mid_channels,
            filter_lengths=filter_lengths,
            subsample_length=subsample_length,
        )
        nl = NonLocalBlock(in_channels=IN_CHANNELS, **config)
        nl.eval()
        assert nl(SAMPLE_INPUT).shape == nl.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
        nl = make_attention_layer(IN_CHANNELS, name="non_local", **config)
        nl.eval()
        assert nl(SAMPLE_INPUT).shape == nl.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert nl.in_channels == IN_CHANNELS

    with pytest.raises(
        AssertionError, match="`filter_lengths` must be an int or a dict, but got `.+`"
    ):
        NonLocalBlock(
            in_channels=IN_CHANNELS,
            mid_channels=IN_CHANNELS * 2,
            filter_lengths=[1, 2],
            subsample_length=1,
        )
    with pytest.raises(
        AssertionError,
        match="`filter_lengths` keys must be a subset of `.+`, but got `.+`",
    ):
        NonLocalBlock(
            in_channels=IN_CHANNELS,
            mid_channels=IN_CHANNELS * 2,
            filter_lengths={"g": 1, "gamma": 3, "theta": 2, "W": 3},
            subsample_length=1,
        )

    for reduction in [2, 4, 8]:
        se = SEBlock(in_channels=IN_CHANNELS, reduction=reduction)
        se.eval()
        assert se(SAMPLE_INPUT).shape == se.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
        se = make_attention_layer(IN_CHANNELS, name="se", reduction=reduction)
        se.eval()
        assert se(SAMPLE_INPUT).shape == se.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert se.in_channels == IN_CHANNELS

    grid_gc = itertools.product(
        [4, 16],  # ratio
        [True, False],  # reduction
        GlobalContextBlock.__POOLING_TYPES__,  # pooling_type
        [["add", "mul"], ["add"], ["mul"]],  # fusion_types
    )
    sample_input = torch.randn(BATCH_SIZE, IN_CHANNELS * 16, SEQ_LEN // 8)
    for ratio, reduction, pooling_type, fusion_types in grid_gc:
        config = dict(
            ratio=ratio,
            reduction=reduction,
            pooling_type=pooling_type,
            fusion_types=fusion_types,
        )
        gc = GlobalContextBlock(in_channels=IN_CHANNELS * 16, **config)
        gc.eval()
        assert gc(sample_input).shape == gc.compute_output_shape(
            seq_len=SEQ_LEN // 8, batch_size=BATCH_SIZE
        )
        gc = make_attention_layer(IN_CHANNELS * 16, name="gc", **config)
        gc.eval()
        assert gc(sample_input).shape == gc.compute_output_shape(
            seq_len=SEQ_LEN // 8, batch_size=BATCH_SIZE
        )
    assert gc.in_channels == IN_CHANNELS * 16

    with pytest.raises(
        AssertionError,
        match="`pooling_type` should be one of `.+`, but got `.+`",
    ):
        GlobalContextBlock(
            in_channels=IN_CHANNELS * 16,
            ratio=4,
            pooling_type="max",
        )
    with pytest.raises(
        AssertionError,
        match="`fusion_types` should be a subset of `.+`, but got `.+`",
    ):
        GlobalContextBlock(
            in_channels=IN_CHANNELS * 16,
            ratio=4,
            fusion_types=["add", "mul", "div"],
        )

    grid_cbam = itertools.product(
        [4, 16],  # reduction
        ["sigmoid", "tanh"],  # gate
        [["avg", "max"], ["avg"], ["max"], ["lp", "lse"]],  # pool_types
        [True, False],  # no_spatial
    )
    sample_input = torch.randn(BATCH_SIZE, IN_CHANNELS * 16, SEQ_LEN // 8)
    for reduction, gate, pool_types, no_spatial in grid_cbam:
        config = dict(
            reduction=reduction,
            gate=gate,
            pool_types=pool_types,
            no_spatial=no_spatial,
        )
        cbam = CBAMBlock(gate_channels=IN_CHANNELS * 16, **config)
        cbam.eval()
        assert cbam(sample_input).shape == cbam.compute_output_shape(
            seq_len=SEQ_LEN // 8, batch_size=BATCH_SIZE
        )
        cbam = make_attention_layer(IN_CHANNELS * 16, name="cbam", **config)
        cbam.eval()
        assert cbam(sample_input).shape == cbam.compute_output_shape(
            seq_len=SEQ_LEN // 8, batch_size=BATCH_SIZE
        )
    assert cbam.gate_channels == IN_CHANNELS * 16
    assert cbam.in_channels == IN_CHANNELS * 16

    for attn in ["ca", "sk", "ge", "bam"]:
        with pytest.raises(NotImplementedError):
            make_attention_layer(IN_CHANNELS, name=attn)

    with pytest.raises(ValueError, match="Unknown attention type: `.+`"):
        make_attention_layer(IN_CHANNELS, name="xxx")


@torch.no_grad()
def test_crf():
    # CRF
    num_tags = 26
    sample_input = torch.randn(SEQ_LEN // 20, BATCH_SIZE, num_tags)
    labels = torch.randint(0, num_tags, (SEQ_LEN // 20, BATCH_SIZE))
    mask = torch.randint(0, 2, (SEQ_LEN // 20, BATCH_SIZE))
    mask[0, :] = 1
    mask = mask.bool()

    crf = CRF(num_tags=num_tags, batch_first=False)
    crf.eval()
    assert crf(sample_input).shape == crf.compute_output_shape(
        seq_len=SEQ_LEN // 20, batch_size=BATCH_SIZE
    )
    repr(crf)
    nll = crf.neg_log_likelihood(sample_input, labels)
    assert nll.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask)
    assert nll_1.shape == torch.Size([])
    assert nll_1 < nll
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="mean")
    assert nll_1.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="sum")
    assert nll_1.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="none")
    assert nll_1.shape == torch.Size([BATCH_SIZE])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="token_mean")
    assert nll_1.shape == torch.Size([])
    with pytest.raises(
        ValueError, match="`reduction` should be one of `.+`, but got `.+`"
    ):
        crf.neg_log_likelihood(sample_input, labels, mask, reduction="max")

    sample_input = sample_input.permute(1, 0, 2)
    labels = labels.permute(1, 0)
    mask = mask.permute(1, 0)
    crf = CRF(num_tags=num_tags, batch_first=True)
    crf.eval()
    assert crf(sample_input).shape == crf.compute_output_shape(
        seq_len=SEQ_LEN // 20, batch_size=BATCH_SIZE
    )
    nll = crf.neg_log_likelihood(sample_input, labels)
    assert nll.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask)
    assert nll_1.shape == torch.Size([])
    assert nll_1 < nll
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="mean")
    assert nll_1.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="sum")
    assert nll_1.shape == torch.Size([])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="none")
    assert nll_1.shape == torch.Size([BATCH_SIZE])
    nll_1 = crf.neg_log_likelihood(sample_input, labels, mask, reduction="token_mean")
    assert nll_1.shape == torch.Size([])

    with pytest.raises(
        AssertionError,
        match="`num_tags` must be be positive, but got `.+`",
    ):
        CRF(num_tags=-1)

    # ExtendedCRF
    sample_input = torch.randn(BATCH_SIZE, SEQ_LEN // 20, IN_CHANNELS)
    for bias in [True, False]:
        crf = ExtendedCRF(in_channels=IN_CHANNELS, num_tags=num_tags, bias=bias)
        assert crf(sample_input).shape == crf.compute_output_shape(
            seq_len=SEQ_LEN // 20, batch_size=BATCH_SIZE
        )
    assert crf.in_channels == IN_CHANNELS


@torch.no_grad()
def test_s2d():
    # SpaceToDepth
    for block_size in [2, 4]:
        s2d = SpaceToDepth(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 8,
            block_size=block_size,
        )
        s2d.eval()
        assert s2d(SAMPLE_INPUT).shape == s2d.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert s2d.in_channels == IN_CHANNELS


@torch.no_grad()
def test_mldecoder():
    # MLDecoder
    grid = itertools.product(
        [-1, 50, 200],  # num_groups
        [False],  # zsl
    )
    for num_groups, zsl in grid:
        mldecoder = MLDecoder(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS * 8,
            num_groups=num_groups,
            zsl=zsl,
        )
        mldecoder.eval()
        assert mldecoder(SAMPLE_INPUT).shape == mldecoder.compute_output_shape(
            seq_len=SEQ_LEN, batch_size=BATCH_SIZE
        )
    assert mldecoder.in_channels == IN_CHANNELS

    with pytest.raises(NotImplementedError, match="Not implemented for `zsl` is `.+`"):
        MLDecoder(in_channels=IN_CHANNELS, out_channels=IN_CHANNELS * 8, zsl=True)


@torch.no_grad()
def test_droppath():
    # DropPath
    dp = DropPath()
    dp.train()
    assert dp(SAMPLE_INPUT).shape == dp.compute_output_shape(
        input_shape=SAMPLE_INPUT.shape
    )
    repr(dp)
    dp.eval()
    out = dp(SAMPLE_INPUT)
    assert out is SAMPLE_INPUT


@torch.no_grad()
def test_ScaledDotProductAttention():
    model = _ScaledDotProductAttention()
    model.eval()
    query, key, value = [torch.randn(2, 12, 2000) for _ in range(3)]
    assert model(query, key, value).shape == torch.Size([2, 12, 2000])
