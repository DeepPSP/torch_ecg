"""
"""

import itertools
from pathlib import Path

import numpy as np
import torch
import pytest
from tqdm.auto import tqdm

from torch_ecg.utils.utils_nn import (
    extend_predictions,
    compute_output_shape,
    compute_deconv_output_shape,
    compute_sequential_output_shape,
    compute_module_size,
    default_collate_fn,
    compute_receptive_field,
    adjust_cnn_filter_lengths,
    SizeMixin,
    CkptMixin,
)
from torch_ecg.models._nets import Conv_Bn_Activation
from torch_ecg.cfg import CFG, DEFAULTS


class Model1D(torch.nn.Sequential, SizeMixin, CkptMixin):
    def __init__(self, n_leads: int, config: CFG) -> None:
        self.config = config
        out_channels = self.config["out_channels"]
        super().__init__(
            Conv_Bn_Activation(n_leads, 32, 7, stride=2, padding=1, activation="mish"),
            Conv_Bn_Activation(
                32, out_channels, 5, stride=1, dilation=4, activation="leaky"
            ),
        )


class ModelDummy(torch.nn.Module, SizeMixin, CkptMixin):
    def __init__(self, n_leads: int, config: CFG) -> None:
        self.config = config
        super().__init__()

    def forward(self, x):
        return x


def test_extend_predictions():
    n_records, n_classes = 10, 3
    classes = ["NSR", "AF", "PVC"]
    extended_classes = ["AF", "RBBB", "PVC", "NSR"]
    perm = {idx: extended_classes.index(c) for idx, c in enumerate(classes)}

    scalar_pred = torch.rand(n_records, n_classes)
    extended_pred = extend_predictions(scalar_pred, classes, extended_classes)
    assert np.allclose(extended_pred[:, [3, 0, 2]], scalar_pred.numpy())

    bin_pred = torch.randint(0, 2, (n_records, n_classes))
    extended_pred = extend_predictions(bin_pred, classes, extended_classes)
    assert np.allclose(extended_pred[:, [3, 0, 2]], bin_pred.numpy())

    cate_pred = torch.randint(0, n_classes, (n_records,))
    extended_pred = extend_predictions(cate_pred, classes, extended_classes)
    cate_pred = cate_pred.numpy()
    for k, v in perm.items():
        cate_pred[cate_pred == k] = v
    assert np.allclose(extended_pred, cate_pred)

    classes = ["NSR", "AF", "PVC"]
    with pytest.raises(
        AssertionError, match="`extended_classes` is not a superset of `classes`,"
    ):
        extended_classes = ["LAnFB", "AF", "RBBB", "RAD", "NSR"]
        scalar_pred = torch.rand(n_records, n_classes)
        extend_predictions(scalar_pred, classes, extended_classes)
    with pytest.raises(
        AssertionError, match="`pred` indicates 4 classes, while `classes` has 3"
    ):
        extended_classes = ["AF", "RBBB", "PVC", "NSR"]
        scalar_pred = torch.rand(n_records, len(extended_classes))
        extend_predictions(scalar_pred, classes, extended_classes)


def test_compute_output_shape():
    in_channels = 12
    tensor_first = torch.rand(32, in_channels, 2000)
    shape_first = tensor_first.shape
    tensor_last = torch.rand(32, 2000, in_channels)
    shape_last = tensor_last.shape

    kw_grid = itertools.product(
        [1, 3, [9], 12],  # kernel_size
        [1, [2], 3, 8],  # stride
        [[0], 1, 2, 3],  # padding
        [1, 2, [5], 8],  # dilation
    )
    kw_grid = list(kw_grid)

    # conv
    num_filters = 32
    for kernel_size, stride, padding, dilation in tqdm(
        kw_grid, mininterval=1, desc="conv"
    ):
        conv_kw = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        for tensor, channel_last in zip([tensor_first, tensor_last], [False, True]):
            conv_output_shape = compute_output_shape(
                "conv",
                input_shape=tensor.shape,
                num_filters=num_filters,
                output_padding=0,
                channel_last=channel_last,
                **conv_kw
            )
            conv_output_tensor = torch.nn.Conv1d(in_channels, num_filters, **conv_kw)(
                tensor_first
            )
            if channel_last:
                conv_output_tensor = conv_output_tensor.permute(0, 2, 1)

            assert conv_output_shape == conv_output_tensor.shape

    # deconv
    for kernel_size, stride, padding, dilation in tqdm(
        kw_grid, mininterval=1, desc="deconv"
    ):
        deconv_kw = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        for tensor, channel_last in zip([tensor_first, tensor_last], [False, True]):
            deconv_output_shape = compute_output_shape(
                "deconv",
                input_shape=tensor.shape,
                num_filters=num_filters,
                output_padding=0,
                channel_last=channel_last,
                **deconv_kw
            )
            deconv_output_tensor = torch.nn.ConvTranspose1d(
                in_channels, num_filters, **deconv_kw
            )(tensor_first)
            if channel_last:
                deconv_output_tensor = deconv_output_tensor.permute(0, 2, 1)

            assert deconv_output_shape == deconv_output_tensor.shape
    compute_deconv_output_shape(
        input_shape=tensor.shape,
        num_filters=num_filters,
        output_padding=0,
        channel_last=channel_last,
        **deconv_kw
    )

    # maxpool
    for kernel_size, stride, padding, dilation in tqdm(
        kw_grid, mininterval=1, desc="maxpool"
    ):
        _padding = padding if isinstance(padding, int) else padding[0]
        _kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if _padding > _kernel_size / 2:
            continue
        maxpool_kw = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        for tensor, channel_last in zip([tensor_first, tensor_last], [False, True]):
            maxpool_output_shape = compute_output_shape(
                "maxpool",
                input_shape=tensor.shape,
                num_filters=1,
                output_padding=0,
                channel_last=channel_last,
                **maxpool_kw
            )
            maxpool_output_tensor = torch.nn.MaxPool1d(**maxpool_kw)(tensor_first)
            if channel_last:
                maxpool_output_tensor = maxpool_output_tensor.permute(0, 2, 1)

            assert maxpool_output_shape == maxpool_output_tensor.shape

    # avgpool
    for kernel_size, stride, padding, _ in tqdm(kw_grid, mininterval=1, desc="avgpool"):
        _padding = padding if isinstance(padding, int) else padding[0]
        _kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if _padding > _kernel_size / 2:
            continue
        avgpool_kw = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        for tensor, channel_last in zip([tensor_first, tensor_last], [False, True]):
            avgpool_output_shape = compute_output_shape(
                "avgpool",
                input_shape=tensor.shape,
                num_filters=1,
                output_padding=0,
                channel_last=channel_last,
                **avgpool_kw
            )
            avgpool_output_tensor = torch.nn.AvgPool1d(**avgpool_kw)(tensor_first)
            if channel_last:
                avgpool_output_tensor = avgpool_output_tensor.permute(0, 2, 1)

            assert avgpool_output_shape == avgpool_output_tensor.shape

    shape_1 = compute_output_shape("conv", [None, None, 224, 224], padding=[4, 8])
    shape_2 = compute_output_shape(
        "conv", [None, None, 224, 224], padding=[4, 8], asymmetric_padding=[1, 3]
    )
    assert shape_2[2:] == (shape_1[2] + 1 + 3, shape_1[3] + 1 + 3)
    shape_1 = compute_output_shape("conv", [None, None, 224, 224], padding=[4, 8])
    shape_2 = compute_output_shape(
        "conv",
        [None, None, 224, 224],
        padding=[4, 8],
        asymmetric_padding=[[1, 3], [0, 2]],
    )
    assert shape_2[2:] == (shape_1[2] + 1 + 3, shape_1[3] + 0 + 2)

    shape_1 = compute_output_shape(
        "conv",
        [None, None, None, None],
        padding=[4, 8],
        num_filters=32,
        channel_last=True,
    )
    assert shape_1 == (None, None, None, 32)

    with pytest.raises(
        AssertionError, match="Unknown layer type `pool`, should be one of"
    ):
        compute_output_shape("pool", tensor_first.shape)
    with pytest.raises(
        AssertionError, match="`input_shape` should be a sequence of length at least 3"
    ):
        compute_output_shape("conv", tensor_first.shape[:2])
    with pytest.raises(
        AssertionError,
        match="`input_shape` should be a sequence containing only `None` and positive integers",
    ):
        compute_output_shape("conv", [None, None, -1])
    with pytest.raises(
        AssertionError, match="`num_filters` should be `None` or positive integer"
    ):
        compute_output_shape("conv", tensor_first.shape, num_filters=-12)
    with pytest.raises(
        AssertionError, match="`kernel_size` should contain only positive integers"
    ):
        compute_output_shape("conv", tensor_first.shape, kernel_size=2.5)
    with pytest.raises(
        AssertionError, match="`kernel_size` should contain only positive integers"
    ):
        compute_output_shape("conv", tensor_first.shape, kernel_size=0)
    with pytest.raises(
        AssertionError, match="`stride` should contain only positive integers"
    ):
        compute_output_shape("conv", tensor_first.shape, stride=0)
    with pytest.raises(
        AssertionError, match="`padding` should contain only non-negative integers"
    ):
        compute_output_shape("conv", tensor_first.shape, padding=-3)
    with pytest.raises(
        AssertionError,
        match="`output_padding` should contain only non-negative integers",
    ):
        compute_output_shape("deconv", tensor_first.shape, output_padding=-3)
    with pytest.raises(
        ValueError, match="out channel dimension and spatial dimensions are all `None`"
    ):
        compute_output_shape("conv", [None, 12, None], num_filters=None)
    with pytest.raises(
        ValueError, match="out channel dimension and spatial dimensions are all `None`"
    ):
        compute_output_shape(
            "conv", [None, None, 12], num_filters=None, channel_last=True
        )
    with pytest.raises(
        ValueError, match="spatial dimensions should be all `None`, or all not `None`"
    ):
        compute_output_shape("conv", [None, 12, None, 224])  # spatial 2D (total 4D)
    with pytest.raises(
        ValueError, match="spatial dimensions should be all `None`, or all not `None`"
    ):
        compute_output_shape(
            "conv", [None, None, 224, 12], channel_last=True
        )  # spatial 2D (total 4D)
    with pytest.raises(
        ValueError, match="input has 1 dimensions, while kernel has 2 dimensions,"
    ):
        compute_output_shape("conv", tensor_first.shape, kernel_size=[3, 5])
    with pytest.raises(
        ValueError,
        match="input has 1 dimensions, while `output_padding` has 2 dimensions,",
    ):
        compute_output_shape("deconv", tensor_first.shape, output_padding=[1, 2])
    with pytest.raises(
        ValueError, match="input has 1 dimensions, while `dilation` has 2 dimensions,"
    ):
        compute_output_shape("conv", tensor_first.shape, dilation=[4, 8])
    with pytest.raises(
        ValueError, match="input has 1 dimensions, while `padding` has 2 dimensions,"
    ):
        compute_output_shape("conv", tensor_first.shape, padding=[1, 1])
    with pytest.raises(AssertionError, match="Invalid `asymmetric_padding`"):
        compute_output_shape("conv", tensor_first.shape, asymmetric_padding=2)
    with pytest.raises(AssertionError, match="Invalid `asymmetric_padding`"):
        compute_output_shape(
            "conv",
            tensor_first.shape,
            asymmetric_padding=[1, 2, 3],
        )
    with pytest.raises(AssertionError, match="Invalid `asymmetric_padding`"):
        compute_output_shape(
            "conv", [None, None, 224, 224], asymmetric_padding=[[1, 2]]
        )
    with pytest.raises(
        AssertionError,
        match="output shape `\\(2, None, 0\\)` is illegal, please check input arguments",
    ):
        compute_output_shape("conv", (2, 12, 400), kernel_size=5, dilation=100)


def test_compute_sequential_output_shape():
    seq_len, batch_size = 2000, 2
    tensor_1d = torch.rand(batch_size, 12, seq_len)
    out_channels = 128
    model_1d = Model1D(12, CFG(out_channels=out_channels))
    assert (
        compute_sequential_output_shape(model_1d, seq_len, batch_size)
        == model_1d(tensor_1d).shape
    )


def test_compute_module_size():
    in_features, out_features = 10, 20

    class Model(torch.nn.Sequential):
        def __init__(self):
            super().__init__()
            # linear with bias
            self.add_module(
                "linear",
                torch.nn.Linear(in_features, out_features, dtype=torch.float16),
            )
            self.register_buffer("hehe", torch.ones(20, 2, dtype=torch.float64))

    model = Model()
    model.linear.weight.requires_grad_(False)

    assert compute_module_size(model) == out_features
    assert (
        compute_module_size(model, requires_grad=False)
        == in_features * out_features + out_features
    )
    assert (
        compute_module_size(model, requires_grad=False, include_buffers=True)
        == in_features * out_features + out_features + 20 * 2
    )
    assert (
        compute_module_size(
            model, requires_grad=False, include_buffers=True, human=True
        )
        == "0.7K"
    )
    assert (
        compute_module_size(
            model, requires_grad=False, include_buffers=False, human=True
        )
        == "0.4K"
    )
    assert compute_module_size(model, human=True) == "40.0B"

    with pytest.warns(
        RuntimeWarning,
        match="`include_buffers` is ignored when `requires_grad` is True",
    ):
        compute_module_size(model, requires_grad=True, include_buffers=True)


def test_default_collate_fn():
    batch_size = 32
    shape_1 = (12, 2000)
    shape_2 = (26,)
    shape_3 = (2000, 4)

    batch_data = [
        (
            DEFAULTS.RNG.uniform(size=shape_1),
            DEFAULTS.RNG.uniform(size=shape_2),
            DEFAULTS.RNG.uniform(size=shape_3),
        )
        for _ in range(batch_size)
    ]
    tensor_1, tensor_2, tensor_3 = default_collate_fn(batch_data)
    assert tensor_1.shape == (batch_size, *shape_1)
    assert tensor_2.shape == (batch_size, *shape_2)
    assert tensor_3.shape == (batch_size, *shape_3)

    batch_data = [
        dict(
            tensor_1=DEFAULTS.RNG.uniform(size=shape_1),
            tensor_2=DEFAULTS.RNG.uniform(size=shape_2),
            tensor_3=DEFAULTS.RNG.uniform(size=shape_3),
        )
        for _ in range(batch_size)
    ]
    tensors = default_collate_fn(batch_data)
    assert tensors["tensor_1"].shape == (batch_size, *shape_1)
    assert tensors["tensor_2"].shape == (batch_size, *shape_2)
    assert tensors["tensor_3"].shape == (batch_size, *shape_3)

    with pytest.raises(ValueError, match="Invalid batch"):
        default_collate_fn([1])

    with pytest.raises(ValueError, match="No data"):
        default_collate_fn([tuple()])


def test_compute_receptive_field():
    assert (
        compute_receptive_field(
            kernel_sizes=[11, 2, 7, 7, 2, 5, 5, 5, 2],
            strides=[1, 2, 1, 1, 2, 1, 1, 1, 2],
        )
        == 90
    )
    assert (
        compute_receptive_field(
            kernel_sizes=[11, 2, 7, 7, 2, 5, 5, 5, 2],
            strides=[1, 2, 1, 1, 2, 1, 1, 1, 2],
            dilations=[2, 1, 2, 4, 1, 8, 8, 8, 1],
        )
        == 484
    )
    assert (
        compute_receptive_field(
            kernel_sizes=[11, 2, 7, 7, 2, 5, 5, 5, 2],
            strides=[1, 2, 1, 1, 2, 1, 1, 1, 2],
            dilations=[4, 1, 4, 8, 1, 16, 32, 64, 1],
        )
        == 1984
    )
    assert (
        compute_receptive_field(
            kernel_sizes=[11, 2, 7, 7, 2, 5, 5, 5, 2],
            strides=[1, 2, 1, 1, 2, 1, 1, 1, 2],
            dilations=[4, 1, 4, 8, 1, 16, 32, 64, 1],
            input_len=1000,
        )
        == 1000
    )
    assert (
        compute_receptive_field(
            kernel_sizes=[11, 2, 7, 7, 2, 5, 5, 5, 2],
            strides=[1, 2, 1, 1, 2, 1, 1, 1, 2],
            dilations=[4, 1, 4, 8, 1, 16, 32, 64, 1],
            input_len=2500,
        )
        == 1984
    )


def test_adjust_cnn_filter_lengths():
    fs = 200
    config = dict(
        fs=fs,
        block_1=dict(
            filter_length=15,
            fs=fs,
        ),
        block_2=dict(
            filter_lengths=[9, 9, 6],
            fs=fs,
        ),
        block_3=dict(
            filter_size=11,
            fs=fs,
        ),
        block_4=dict(
            filter_sizes=[9, 5, 5],
            fs=fs,
        ),
    )
    new_fs = 400
    new_config = adjust_cnn_filter_lengths(config, fs=new_fs)
    assert new_config == dict(
        fs=new_fs,
        block_1=dict(
            filter_length=31,
            fs=new_fs,
        ),
        block_2=dict(
            filter_lengths=[19, 19, 13],
            fs=new_fs,
        ),
        block_3=dict(
            filter_size=23,
            fs=new_fs,
        ),
        block_4=dict(
            filter_sizes=[19, 11, 11],
            fs=new_fs,
        ),
    )
    new_config = adjust_cnn_filter_lengths(config, fs=new_fs, ensure_odd=False)
    assert new_config == dict(
        fs=new_fs,
        block_1=dict(
            filter_length=30,
            fs=new_fs,
        ),
        block_2=dict(
            filter_lengths=[18, 18, 12],
            fs=new_fs,
        ),
        block_3=dict(
            filter_size=22,
            fs=new_fs,
        ),
        block_4=dict(
            filter_sizes=[18, 10, 10],
            fs=new_fs,
        ),
    )

    new_fs = 100
    new_config = adjust_cnn_filter_lengths(config, fs=new_fs)
    assert new_config == dict(
        fs=new_fs,
        block_1=dict(
            filter_length=9,
            fs=new_fs,
        ),
        block_2=dict(
            filter_lengths=[5, 5, 3],
            fs=new_fs,
        ),
        block_3=dict(
            filter_size=7,
            fs=new_fs,
        ),
        block_4=dict(
            filter_sizes=[5, 3, 3],
            fs=new_fs,
        ),
    )
    new_config = adjust_cnn_filter_lengths(config, fs=new_fs, ensure_odd=False)
    assert new_config == dict(
        fs=new_fs,
        block_1=dict(
            filter_length=8,
            fs=new_fs,
        ),
        block_2=dict(
            filter_lengths=[4, 4, 3],
            fs=new_fs,
        ),
        block_3=dict(
            filter_size=6,
            fs=new_fs,
        ),
        block_4=dict(
            filter_sizes=[4, 2, 2],
            fs=new_fs,
        ),
    )


def test_mixin_classes():
    model_1d = Model1D(12, CFG(out_channels=128))
    assert isinstance(model_1d.module_size, int)
    assert model_1d.module_size > 0
    assert isinstance(model_1d.sizeof, int)
    assert model_1d.sizeof > model_1d.module_size

    assert isinstance(model_1d.module_size_, str)
    assert isinstance(model_1d.sizeof_, str)

    assert isinstance(model_1d.dtype_, str)
    assert isinstance(model_1d.device_, str)

    save_path = Path(__file__).resolve().parents[1] / "tmp" / "test_mixin.pth"

    model_1d.save(save_path, dict(n_leads=12))

    assert save_path.is_file()

    loaded_model, _ = Model1D.from_checkpoint(save_path)

    assert repr(model_1d) == repr(loaded_model)

    save_path.unlink()

    model_dummy = ModelDummy(12, CFG(out_channels=128))
    assert model_dummy.module_size == model_dummy.sizeof == 0
    assert model_dummy.module_size_ == model_dummy.sizeof_ == "0.0B"
    assert model_dummy.dtype == torch.float32
    assert model_dummy.device == torch.device("cpu")
    inp = torch.randn(1, 12, 1000)
    out = model_dummy(inp)
    assert inp is out
