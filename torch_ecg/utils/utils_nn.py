"""
utilities for nn models

"""

import os
import re
import warnings
from copy import deepcopy
from itertools import chain, repeat
from math import floor
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..cfg import CFG, DEFAULTS
from .download import http_get
from .misc import add_docstring, make_serializable
from .utils_data import cls_to_bin

__all__ = [
    "extend_predictions",
    "compute_output_shape",
    "compute_conv_output_shape",
    "compute_deconv_output_shape",
    "compute_maxpool_output_shape",
    "compute_avgpool_output_shape",
    "compute_sequential_output_shape",
    "compute_sequential_output_shape_docstring",
    "compute_module_size",
    "default_collate_fn",
    "compute_receptive_field",
    "adjust_cnn_filter_lengths",
    "SizeMixin",
    "CkptMixin",
]


def extend_predictions(preds: Sequence, classes: List[str], extended_classes: List[str]) -> np.ndarray:
    """Extend the prediction arrays to prediction arrays in larger range of classes

    Parameters
    ----------
    preds : array_like
        Array of predictions (scalar or binary) of shape
        ``(n_records, n_classes)``,
        or categorical predictions of shape ``(n_classes,)``,
        where ``n_classes = len(classes)``.
    classes : List[str]
        Classes of the predictions of `preds`.
    extended_classes : List[str]
        A superset of `classes`.
        The predictions will be extended to this range of classes.

    Returns
    -------
    extended_preds : numpy.ndarray
        The extended array of predictions, with indices in `extended_classes`,
        of shape ``(n_records, n_classes)``, or ``(n_classes,)``.

    Examples
    --------
    .. code-block:: python

        n_records, n_classes = 10, 3
        classes = ["NSR", "AF", "PVC"]
        extended_classes = ["AF", "RBBB", "PVC", "NSR"]
        scalar_pred = torch.rand(n_records, n_classes)
        extended_pred = extend_predictions(scalar_pred, classes, extended_classes)
        bin_pred = torch.randint(0, 2, (n_records, n_classes))
        extended_pred = extend_predictions(bin_pred, classes, extended_classes)
        cate_pred = torch.randint(0, n_classes, (n_records,))
        extended_pred = extend_predictions(cate_pred, classes, extended_classes)

    """
    assert len(set(classes) - set(extended_classes)) == 0, (
        "`extended_classes` is not a superset of `classes`, "
        f"with {set(classes)-set(extended_classes)} in `classes` but not in `extended_classes`"
    )

    if isinstance(preds, Tensor):
        _preds = preds.numpy()
    else:
        _preds = np.array(preds)

    if np.ndim(_preds) == 1:  # categorical predictions
        extended_preds = cls_to_bin(_preds, len(classes))
        extended_preds = extend_predictions(extended_preds, classes, extended_classes)
        extended_preds = np.where(extended_preds == 1)[1]
        return extended_preds

    assert _preds.shape[1] == len(classes), f"`pred` indicates {_preds.shape[1]} classes, while `classes` has {len(classes)}"

    extended_preds = np.zeros((_preds.shape[0], len(extended_classes)))

    for idx, c in enumerate(classes):
        new_idx = extended_classes.index(c)
        extended_preds[..., new_idx] = _preds[..., idx]

    if np.array(preds).ndim == 1:
        extended_preds = extended_preds[0]

    return extended_preds


# utils for computing output shape
def compute_output_shape(
    layer_type: str,
    input_shape: Sequence[Union[int, None]],
    num_filters: Optional[int] = None,
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    output_padding: Union[Sequence[int], int] = 0,
    dilation: Union[Sequence[int], int] = 1,
    channel_last: bool = False,
    asymmetric_padding: Union[Sequence[int], Sequence[Sequence[int]]] = None,
) -> Tuple[Union[int, None]]:
    """Compute the output shape of a (transpose) convolution/maxpool/avgpool layer.

    This function is based on the discussion [#disc]_.

    Parameters
    ----------
    layer_type : str
        Type (conv, maxpool, avgpool, etc.) of the layer.
    input_shape : Sequence[Union[int, None]]
        Shape of an input :class:`~torch.Tensor`.
        The first dimension is the batch dimension,
        which is allowed to be `None`.
    num_filters : int, optional
        Number of filters, also the channel dimension.
    kernel_size : int or Sequence[int], default 1
        Kernel size (filter size) of the layer,
        should be compatible with `input_shape`.
    stride : int or Sequence[int], default 1
        Stride (down-sampling length) of the layer,
        should be compatible with `input_shape`.
    padding : int or Sequence[int], default 0
        Padding length(s) of the layer,
        should be compatible with `input_shape`.
    output_padding : int or Sequence[int], default 0
        Additional size added to one side of the output shape,
        used only for transpose convolution.
    dilation : int or Sequence[int], default 1
        Dilation of the layer, should be compatible with `input_shape`.
    channel_last : bool, default False
        Whether the channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention).
    asymmetric_padding : Sequence[int] or Sequence[Sequence[int]], optional
        (2-)sequence of int or sequence of (2-)sequence of int
        asymmetric paddings for all dimensions or for each dimension.

    Returns
    -------
    output_shape : tuple
        Shape of the output :class:`~torch.Tensor`.

    References
    ----------
    .. [#disc] https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5

    """
    # check validity of arguments
    __TYPES__ = [
        "conv",
        "convolution",
        "deconv",
        "deconvolution",
        "transposeconv",
        "transposeconvolution",
        "maxpool",
        "maxpooling",
        "avgpool",
        "avgpooling",
        "averagepool",
        "averagepooling",
    ]
    lt = "".join(layer_type.lower().split("_"))
    assert lt in __TYPES__, f"Unknown layer type `{layer_type}`, should be one of: {__TYPES__}"

    def assert_positive_integer(num):
        return isinstance(num, int) and num > 0

    def assert_non_negative_integer(num):
        return isinstance(num, int) and num >= 0

    dim = len(input_shape) - 2
    assert dim > 0, (
        "`input_shape` should be a sequence of length at least 3, "
        "to be a valid (with batch and channel) shape of a non-degenerate Tensor"
    )
    assert all(
        [s is None or assert_positive_integer(s) for s in input_shape]
    ), "`input_shape` should be a sequence containing only `None` and positive integers"

    if num_filters is not None:
        assert assert_positive_integer(num_filters), "`num_filters` should be `None` or positive integer"
    assert all(
        [assert_positive_integer(num) for num in np.asarray(kernel_size).flatten().tolist()]
    ), "`kernel_size` should contain only positive integers"
    assert all(
        [assert_positive_integer(num) for num in np.asarray(stride).flatten().tolist()]
    ), "`stride` should contain only positive integers"
    assert all(
        [assert_non_negative_integer(num) for num in np.asarray(padding).flatten().tolist()]
    ), "`padding` should contain only non-negative integers"
    assert all(
        [assert_non_negative_integer(num) for num in np.asarray(output_padding).flatten().tolist()]
    ), "`output_padding` should contain only non-negative integers"
    assert all(
        [assert_positive_integer(num) for num in np.asarray(dilation).flatten().tolist()]
    ), "`dilation` should contain only positive integers"

    if lt in [
        "conv",
        "convolution",
    ]:
        # as function of dilation, kernel_size
        def minus_term(d, k):
            return d * (k - 1) + 1

        out_channels = num_filters
    elif lt in [
        "maxpool",
        "maxpooling",
    ]:

        def minus_term(d, k):
            return d * (k - 1) + 1

        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in [
        "avgpool",
        "avgpooling",
        "averagepool",
        "averagepooling",
    ]:

        def minus_term(d, k):
            return k

        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in [
        "deconv",
        "deconvolution",
        "transposeconv",
        "transposeconvolution",
    ]:
        out_channels = num_filters

    def check_output_validity(shape):
        assert all(p is None or p > 0 for p in shape), f"output shape `{shape}` is illegal, please check input arguments"
        return shape

    # none_dim_msg = "only batch and channel dimension can be `None`"
    # if channel_last:
    #     assert all([n is not None for n in input_shape[1:-1]]), none_dim_msg
    # else:
    #     assert all([n is not None for n in input_shape[2:]]), none_dim_msg
    none_dim_msg = "spatial dimensions should be all `None`, or all not `None`"
    if channel_last:
        if all([n is None for n in input_shape[1:-1]]):
            if out_channels is None:
                raise ValueError("out channel dimension and spatial dimensions are all `None`")
            output_shape = tuple(list(input_shape[:-1]) + [out_channels])
            return check_output_validity(output_shape)
        elif any([n is None for n in input_shape[1:-1]]):
            raise ValueError(none_dim_msg)
    else:
        if all([n is None for n in input_shape[2:]]):
            if out_channels is None:
                raise ValueError("out channel dimension and spatial dimensions are all `None`")
            output_shape = tuple([input_shape[0], out_channels] + list(input_shape[2:]))
            return check_output_validity(output_shape)
        elif any([n is None for n in input_shape[2:]]):
            raise ValueError(none_dim_msg)

    if isinstance(kernel_size, int):
        _kernel_size = list(repeat(kernel_size, dim))
    elif len(kernel_size) == dim:
        _kernel_size = kernel_size
    else:
        raise ValueError(
            f"input has {dim} dimensions, while kernel has {len(kernel_size)} dimensions, "
            "both not including the channel dimension"
        )

    if isinstance(stride, int):
        _stride = list(repeat(stride, dim))
    elif len(stride) == dim:
        _stride = stride
    else:
        raise ValueError(
            f"input has {dim} dimensions, while `kernel` has {len(stride)} dimensions, "
            "both not including the channel dimension"
        )

    # NOTE: asymmetric padding along one spatial dimension
    # seems not supported yet by PyTorch's builtin Module classes
    if isinstance(padding, int):
        _padding = list(repeat(list(repeat(padding, 2)), dim))
    # elif len(padding) == 2 and isinstance(padding[0], int):
    #     _padding = list(repeat(padding, dim))
    # elif (
    #     len(padding) == dim
    #     and all([isinstance(p, Sequence) for p in padding])
    #     and all([len(p) == 2 for p in padding])
    # ):
    elif len(padding) == dim:
        _padding = [list(repeat(p, 2)) for p in padding]
    else:
        raise ValueError(
            f"input has {dim} dimensions, while `padding` has {len(padding)} dimensions, "
            "both not including the channel dimension"
        )
        # raise ValueError("Invalid `padding`")

    if asymmetric_padding is not None:
        assert hasattr(asymmetric_padding, "__len__"), "Invalid `asymmetric_padding`"
        if isinstance(asymmetric_padding[0], int):
            assert len(asymmetric_padding) == 2 and isinstance(asymmetric_padding[1], int), "Invalid `asymmetric_padding`"
            _asymmetric_padding = list(repeat(asymmetric_padding, dim))
        else:
            assert len(asymmetric_padding) == dim and all(
                len(ap) == 2 and all(isinstance(p, int) for p in ap) for ap in asymmetric_padding
            ), "Invalid `asymmetric_padding`"
            _asymmetric_padding = asymmetric_padding
        for idx in range(dim):
            _padding[idx][0] += _asymmetric_padding[idx][0]
            _padding[idx][1] += _asymmetric_padding[idx][1]

    if isinstance(output_padding, int):
        _output_padding = list(repeat(output_padding, dim))
    elif len(output_padding) == dim:
        _output_padding = output_padding
    else:
        raise ValueError(
            f"input has {dim} dimensions, while `output_padding` has {len(output_padding)} dimensions, "
            "both not including the channel dimension"
        )

    if isinstance(dilation, int):
        _dilation = list(repeat(dilation, dim))
    elif len(dilation) == dim:
        _dilation = dilation
    else:
        raise ValueError(
            f"input has {dim} dimensions, while `dilation` has {len(dilation)} dimensions, "
            "both not including the channel dimension"
        )

    if channel_last:
        _input_shape = list(input_shape[1:-1])
    else:
        _input_shape = list(input_shape[2:])

    if lt in [
        "deconv",
        "deconvolution",
        "transposeconv",
        "transposeconvolution",
    ]:
        output_shape = [
            (i - 1) * s - sum(p) + d * (k - 1) + o + 1
            for i, p, o, d, k, s in zip(
                _input_shape,
                _padding,
                _output_padding,
                _dilation,
                _kernel_size,
                _stride,
            )
        ]
    else:
        output_shape = [
            floor(((i + sum(p) - minus_term(d, k)) / s) + 1)
            for i, p, d, k, s in zip(_input_shape, _padding, _dilation, _kernel_size, _stride)
        ]
    if channel_last:
        output_shape = tuple([input_shape[0]] + output_shape + [out_channels])
    else:
        output_shape = tuple([input_shape[0], out_channels] + output_shape)

    return check_output_validity(output_shape)


def compute_conv_output_shape(
    input_shape: Sequence[Union[int, None]],
    num_filters: Optional[int] = None,
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    dilation: Union[Sequence[int], int] = 1,
    channel_last: bool = False,
    asymmetric_padding: Union[Sequence[int], Sequence[Sequence[int]]] = None,
) -> Tuple[Union[int, None]]:
    """Compute the output shape of a convolution layer.

    Parameters
    ----------
    input_shape : Sequence[Union[int, None]]
        Shape of an input :class:`~torch.Tensor`.
        The first dimension is the batch dimension,
        which is allowed to be `None`.
    num_filters : int, optional
        Number of filters, also the channel dimension.
    kernel_size : int or Sequence[int], default 1
        Kernel size (filter size) of the layer,
        should be compatible with `input_shape`.
    stride : int or Sequence[int], default 1
        Stride (down-sampling length) of the layer,
        should be compatible with `input_shape`.
    padding : int or Sequence[int], default 0
        Padding length(s) of the layer,
        should be compatible with `input_shape`.
    dilation : int or Sequence[int], default 1
        Dilation of the layer, should be compatible with `input_shape`.
    channel_last : bool, default False
        Whether the channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention).
    asymmetric_padding : Sequence[int] or Sequence[Sequence[int]], optional
        (2-)sequence of int or sequence of (2-)sequence of int
        asymmetric paddings for all dimensions or for each dimension.

    Returns
    -------
    output_shape : tuple
        Shape of the output :class:`~torch.Tensor`.

    """
    output_shape = compute_output_shape(
        "conv",
        input_shape,
        num_filters,
        kernel_size,
        stride,
        padding,
        0,
        dilation,
        channel_last,
        asymmetric_padding,
    )
    return output_shape


def compute_maxpool_output_shape(
    input_shape: Sequence[Union[int, None]],
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    dilation: Union[Sequence[int], int] = 1,
    channel_last: bool = False,
) -> Tuple[Union[int, None]]:
    """Compute the output shape of a maxpool layer.

    Parameters
    ----------
    input_shape : Sequence[Union[int, None]]
        Shape of an input :class:`~torch.Tensor`.
        The first dimension is the batch dimension,
        which is allowed to be `None`.
    kernel_size : int or Sequence[int], default 1
        Kernel size (filter size) of the layer,
        should be compatible with `input_shape`.
    stride : int or Sequence[int], default 1
        Stride (down-sampling length) of the layer,
        should be compatible with `input_shape`.
    padding : int or Sequence[int], default 0
        Padding length(s) of the layer,
        should be compatible with `input_shape`.
    dilation : int or Sequence[int], default 1
        Dilation of the layer, should be compatible with `input_shape`.
    channel_last : bool, default False
        Whether the channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention).

    Returns
    -------
    output_shape : tuple
        Shape of the output :class:`~torch.Tensor`.

    """
    output_shape = compute_output_shape(
        "maxpool",
        input_shape,
        1,
        kernel_size,
        stride,
        padding,
        0,
        dilation,
        channel_last,
    )
    return output_shape


def compute_avgpool_output_shape(
    input_shape: Sequence[Union[int, None]],
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    channel_last: bool = False,
) -> Tuple[Union[int, None]]:
    """Compute the output shape of a avgpool layer.

    Parameters
    ----------
    input_shape : Sequence[Union[int, None]]
        Shape of an input :class:`~torch.Tensor`.
        The first dimension is the batch dimension,
        which is allowed to be `None`.
    kernel_size : int or Sequence[int], default 1
        Kernel size (filter size) of the layer,
        should be compatible with `input_shape`.
    stride : int or Sequence[int], default 1
        Stride (down-sampling length) of the layer,
        should be compatible with `input_shape`.
    padding : int or Sequence[int], default 0
        Padding length(s) of the layer,
        should be compatible with `input_shape`.
    channel_last : bool, default False
        Whether the channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention).

    Returns
    -------
    output_shape : tuple
        Shape of the output :class:`~torch.Tensor`.

    """
    output_shape = compute_output_shape(
        "avgpool",
        input_shape,
        1,
        kernel_size,
        stride,
        padding,
        0,
        1,
        channel_last,
    )
    return output_shape


def compute_deconv_output_shape(
    input_shape: Sequence[Union[int, None]],
    num_filters: Optional[int] = None,
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    output_padding: Union[Sequence[int], int] = 0,
    dilation: Union[Sequence[int], int] = 1,
    channel_last: bool = False,
    asymmetric_padding: Union[Sequence[int], Sequence[Sequence[int]]] = None,
) -> Tuple[Union[int, None]]:
    """Compute the output shape of a transpose convolution layer

    Parameters
    ----------
    input_shape : Sequence[Union[int, None]]
        Shape of an input :class:`~torch.Tensor`.
        The first dimension is the batch dimension,
        which is allowed to be `None`.
    num_filters : int, optional
        Number of filters, also the channel dimension.
    kernel_size : int or Sequence[int], default 1
        Kernel size (filter size) of the layer,
        should be compatible with `input_shape`.
    stride : int or Sequence[int], default 1
        Stride (down-sampling length) of the layer,
        should be compatible with `input_shape`.
    padding : int or Sequence[int], default 0
        Padding length(s) of the layer,
        should be compatible with `input_shape`.
    output_padding : int or Sequence[int], default 0
        Additional size added to one side of the output shape.
    dilation : int or Sequence[int], default 1
        Dilation of the layer, should be compatible with `input_shape`.
    channel_last : bool, default False
        Whether the channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention).
    asymmetric_padding : Sequence[int] or Sequence[Sequence[int]], optional
        (2-)sequence of int or sequence of (2-)sequence of int
        asymmetric paddings for all dimensions or for each dimension.

    Returns
    -------
    output_shape : tuple
        Shape of the output :class:`~torch.Tensor`.

    """
    output_shape = compute_output_shape(
        "deconv",
        input_shape,
        num_filters,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
        channel_last,
        asymmetric_padding,
    )
    return output_shape


compute_sequential_output_shape_docstring = """

    Parameters
    ----------
    seq_len : int, optional
        Length of the input tensors.
    batch_size : int, optional
        Batch size of the input tensors.

    Returns
    -------
    output_shape : sequence
        The output shape of the module.

    """


@add_docstring(compute_sequential_output_shape_docstring, mode="append")
def compute_sequential_output_shape(
    model: nn.Sequential,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Sequence[Union[int, None]]:
    """Compute the output shape of a sequential model."""
    assert issubclass(type(model), nn.Sequential), f"model should be nn.Sequential, but got {type(model)}"
    _seq_len = seq_len
    for module in model:
        output_shape = module.compute_output_shape(_seq_len, batch_size)
        _, _, _seq_len = output_shape
    return output_shape


def compute_module_size(
    module: nn.Module,
    requires_grad: bool = True,
    include_buffers: bool = False,
    human: bool = False,
) -> Union[int, str]:
    """compute the size (number of parameters)
    of a :class:`~torch.nn.Module`.

    Parameters
    ----------
    module : torch.nn.Module
        The :class:`~torch.nn.Module` to compute the size.
    requires_grad : bool, default True
        Whether to only count the parameters that require gradients.
    include_buffers : bool, default False
        Whether to include the buffers.
        If `requires_grad` is True, then `include_buffers` is ignored.
    human : bool, default False
        Size is returned in a way that is easy to read by a human,
        by appending a suffix corresponding to the unit (B, K, M, G, T, P).

    Returns
    -------
    n_params : int or str
        Size (number of parameters) of this :class:`~torch.nn.Module`,
        or a string representing the memory size.

    Examples
    --------
    >>> import torch
    >>> class Model(torch.nn.Sequential):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.add_module("linear", torch.nn.Linear(10, 20, dtype=torch.float16))
    ...         self.register_buffer("hehe", torch.ones(20, 2, dtype=torch.float64))
    >>> model = Model()
    >>> model.linear.weight.requires_grad_(False)
    >>> compute_module_size(model)
    20
    >>> compute_module_size(model, requires_grad=False)
    220
    >>> compute_module_size(model, requires_grad=False, include_buffers=True)
    260
    >>> compute_module_size(model, requires_grad=False, include_buffers=True, human=True)
    '0.7K'
    >>> compute_module_size(model, requires_grad=False, include_buffers=False, human=True)
    '0.4K'
    >>> compute_module_size(model, human=True)
    '40.0B'

    """
    if requires_grad:
        tensor_containers = filter(lambda p: p.requires_grad, module.parameters())
        if include_buffers:
            warnings.warn(
                "`include_buffers` is ignored when `requires_grad` is True",
                RuntimeWarning,
            )
    elif include_buffers:
        tensor_containers = chain(module.parameters(), module.buffers())
    else:
        tensor_containers = module.parameters()
    if human:
        size_dict = {
            "torch.float16": 2,
            "torch.float32": 4,
            "torch.float64": 8,
            "torch.int8": 1,
            "torch.int16": 2,
            "torch.int32": 4,
            "torch.int64": 8,
            "torch.uint8": 1,
        }
        n_params = sum([np.prod(item.size()) * size_dict[str(item.dtype)] for item in tensor_containers])
        div_count = 0
        while n_params >= 1024 * 0.1:
            n_params /= 1024
            div_count += 1
        cvt_dict = {c: u for c, u in enumerate(list("BKMGTP"))}
        n_params = f"""{n_params:.1f}{cvt_dict[div_count]}"""
    else:
        n_params = int(sum([np.prod(item.size()) for item in tensor_containers]))
    return n_params


def compute_receptive_field(
    kernel_sizes: Union[Sequence[int], int] = 1,
    strides: Union[Sequence[int], int] = 1,
    dilations: Union[Sequence[int], int] = 1,
    input_len: Optional[int] = None,
    fs: Optional[Real] = None,
) -> Union[int, float]:
    """Compute the receptive field of several types of
    :class:`~torch.nn.Module`.

    Computes the (generic) receptive field of feature map of certain channel,
    from certain flow (if not merged, different flows, e.g.
    shortcut, must be computed separately),
    for convolutions, (non-global) poolings.
    "generic" refers to a general position, rather than specific positions,
    like on the edges, whose receptive field is definitely different.

    In CNNs, for any element of some layer, its receptive field refers to
    all the elements (from all the previous layers) that may affect the calculation of
    during the forward propagation [:footcite:ct:`zhang2023dive`].
    (See `this url <https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#feature-map-and-receptive-field>`_)

    The receptive field is computed as follows.
    Let the layers has kernel size, stride, dilation :math:`(k_n, s_n, d_n)` respectively.
    Let each feature map has receptive field length :math:`r_n`,
    and the difference of receptive fields of adjacent positions (layers) be :math:`f_n`.
    By convention, :math:`(r_0, f_0) = (1, 1)`. Then one has

    .. math::

        \\begin{eqnarray}
        r_{n+1} & = & r_n + d_n(k_n-1)f_n, \\\\
        f_{n+1} & = & s_n f_n.
        \\end{eqnarray}

    Hence

    .. math::

        \\begin{eqnarray}
        f_{n} & = & \\prod\\limits_{i=0}^{n-1} s_i, \\\\
        r_{n} & = & 1 + \\sum\\limits_{i=0}^{n-1} d_i(k_i-1) \\prod\\limits_{i=0}^{j-1} s_j,
        \\end{eqnarray}

    with empty products equaling 1 by convention.

    Parameters
    ----------
    kernel_sizes : int or Sequence[int], default 1
        The sequence of kernel size for all the layers in the flow.
    strides : int or Sequence[int], default 1
        The sequence of strides for all the layers in the flow
    dilations : int or Sequence[int], default 1
        The sequence of strides for all the layers in the flow
    input_len : int, optional
        Length of the first feature map in the flow.
    fs : numbers.Real, optional
        Sampling frequency of the input signal.
        If is not ``None``, then the receptive field is returned in seconds.

    Returns
    -------
    receptive_field : int or float
        (Length of) the receptive field, in samples if `fs` is ``None``,
        otherwise in seconds.

    Examples
    --------
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2])
    90
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2],fs=500)
    0.18
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2],[2,1,2,4,1,8,8,8,1])
    484
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2],[4,1,4,8,1,16,32,64,1])
    1984

    The above example exhibits the receptive fields of the output feature maps
    of the 3 branches of the multi-scopic net, using its original hyper-parameters,
    (note the 3 max pooling layers).


    .. footbibliography::

    """
    _kernel_sizes = [kernel_sizes] if isinstance(kernel_sizes, int) else list(kernel_sizes)
    num_layers = len(_kernel_sizes)
    _strides = list(repeat(strides, num_layers)) if isinstance(strides, int) else list(strides)
    _dilations = list(repeat(dilations, num_layers)) if isinstance(dilations, int) else list(dilations)
    assert num_layers == len(_strides) == len(_dilations)
    receptive_field = 1
    for idx, (k, d) in enumerate(zip(_kernel_sizes, _dilations)):
        s = np.prod(_strides[:idx]) if idx > 0 else 1
        receptive_field += d * (k - 1) * s
    if input_len is not None:
        receptive_field = min(receptive_field, input_len)
    if fs is not None:
        receptive_field /= fs
    return make_serializable(receptive_field)


def default_collate_fn(
    batch: Sequence[Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]]
) -> Union[Tuple[Tensor, ...], Dict[str, Tensor]]:
    """Default collate functions for model training.

    The data generator (:class:`~torch.utils.data.Dataset`) should
    generate (`__getitem__`) n-tuples ``signals, labels, ...``,
    or dictionaries of tensors.

    Parameters
    ----------
    batch : sequence
        Sequence of n-tuples,
        in which the first element is the signal, the second is the label, ...;
        or sequence of dictionaries of tensors.

    Returns
    -------
    tuple or dict
        Tuple or dict of :class:`~torch.Tensor`, which are
        the concatenated values to feed into neural networks.

    """
    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        collated = _default_collate_fn([tuple(b[k] for k in keys) for b in batch])
        return {k: collated[i] for i, k in enumerate(keys)}
    else:
        return _default_collate_fn(batch)


def _default_collate_fn(batch: Sequence[Tuple[np.ndarray, ...]]) -> Tuple[Tensor, ...]:
    """Collate functions for tuples of tensors.

    The data generator (:class:`~torch.utils.data.Dataset`) should
    generate (`__getitem__`) n-tuples ``signals, labels, ...``,
    or dictionaries of tensors.

    Parameters
    ----------
    batch : sequence
        Sequence of n-tuples,
        in which the first element is the signal, the second is the label, ...

    Returns
    -------
    Tuple[torch.Tensor]
        The concatenated values to feed into neural networks.

    """
    try:
        n_fields = len(batch[0])
    except Exception:
        raise ValueError("Invalid batch")
    if n_fields == 0:
        raise ValueError("No data")
    ret = []
    for i in range(n_fields):
        values = [[item[i]] for item in batch]
        values = np.concatenate(values, axis=0)
        values = torch.from_numpy(values)
        ret.append(values)
    return tuple(ret)


def _adjust_cnn_filter_lengths(
    config: dict,
    fs: int,
    ensure_odd: bool = True,
    pattern: str = "filter_length|filter_size",
) -> dict:
    """
    Adjust the filter lengths (kernel sizes) in the config for
    convolutional neural networks, according to the new sampling frequency.

    Parameters
    ----------
    config : dict
        The config dictionary. This `dict` is **NOT** modified.
    fs : int
        The new sampling frequency.
    ensure_odd : bool, default True
        If True, the new filter lengths are ensured to be odd.
    pattern : str, default "filter_length|filter_size"
        The pattern to search for in the config items related to filter lengths.

    Returns
    -------
    config : dict
        The adjusted config dictionary.

    """
    assert "fs" in config
    config = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, dict):
            tmp_config = deepcopy(v)
            original_fs = tmp_config.get("fs", None)
            tmp_config.update({"fs": config["fs"]})
            config[k] = _adjust_cnn_filter_lengths(tmp_config, fs, ensure_odd, pattern)
            if original_fs is None:
                config[k].pop("fs", None)
            else:
                config[k]["fs"] = fs
        elif re.findall(pattern, k):
            if isinstance(v, (Sequence, np.ndarray)):  # DO NOT use `Iterable`
                config[k] = [
                    _adjust_cnn_filter_lengths({"filter_length": fl, "fs": config["fs"]}, fs, ensure_odd)["filter_length"]
                    for fl in v
                ]
            elif isinstance(v, Real):
                # DO NOT use `int`, which might not work for numpy array elements
                if v > 1:
                    config[k] = int(round(v * fs / config["fs"]))
                    if ensure_odd:
                        config[k] = config[k] - config[k] % 2 + 1
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            tmp_configs = [_adjust_cnn_filter_lengths({k: item, "fs": config["fs"]}, fs, ensure_odd, pattern) for item in v]
            config[k] = [item[k] for item in tmp_configs]
    return config


def adjust_cnn_filter_lengths(
    config: dict,
    fs: int,
    ensure_odd: bool = True,
    pattern: str = "filter_length|filter_size",
) -> dict:
    """
    Adjust the filter lengths in the config for
    convolutional neural networks, according to the new sampling frequency.

    Parameters
    ----------
    config : dict
        The config dictionary. This `dict` is **NOT** modified.
    fs : int
        The new sampling frequency.
    ensure_odd : bool, default True
        If True, the new filter lengths are ensured to be odd.
    pattern : str, default "filter_length|filter_size"
        The pattern to search for in the config items related to filter lengths.

    Returns
    -------
    config : dict
        The adjusted config dictionary.

    """
    config = _adjust_cnn_filter_lengths(config, fs, ensure_odd, pattern)
    config["fs"] = fs
    return config


class SizeMixin(object):
    """Mixin class for size related methods"""

    @property
    def module_size(self) -> int:
        """Size of trainable parameters in the model in terms of number of parameters."""
        return compute_module_size(self)

    @property
    def module_size_(self) -> str:
        """Size of trainable parameters in the model in terms of memory capacity."""
        return compute_module_size(self, human=True)

    @property
    def sizeof(self) -> int:
        """Size of the model in terms of number of parameters, including non-trainable parameters and buffers."""
        return compute_module_size(self, requires_grad=False, include_buffers=True, human=False)

    @property
    def sizeof_(self) -> str:
        """Size of the model in terms of memory capacity, including non-trainable parameters and buffers."""
        return compute_module_size(self, requires_grad=False, include_buffers=True, human=True)

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
        except Exception as err:
            raise err  # unknown error

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
        except Exception as err:
            raise err  # unknown error

    @property
    def dtype_(self) -> str:
        return str(self.dtype).replace("torch.", "")

    @property
    def device_(self) -> str:
        return str(self.device)


class CkptMixin(object):
    """Mixin class for loading from checkpoint class methods"""

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, bytes, os.PathLike], device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, dict]:
        """Load a model from a checkpoint.

        Parameters
        ----------
        path : `path-like`
            Path to the checkpoint.
            If it is a directory, then this directory should contain only one checkpoint file
            (with the extension `.pth` or `.pt`).
        device : torch.device, optional
            Map location of the model parameters,
            defaults to "cuda" if available, otherwise "cpu".

        Returns
        -------
        model : torch.nn.Module
            The model loaded from a checkpoint.
        aux_config : dict
            Auxiliary configs that are needed for data preprocessing, etc.

        """
        if Path(path).is_dir():
            candidates = list(Path(path).glob("*.pth")) + list(Path(path).glob("*.pt"))
            assert len(candidates) == 1, "The directory should contain only one checkpoint file"
            path = candidates[0]
        _device = device or DEFAULTS.device
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        kwargs = dict(
            config=ckpt["model_config"],
        )
        if "classes" in aux_config:
            kwargs["classes"] = aux_config["classes"]
        if "n_leads" in aux_config:
            kwargs["n_leads"] = aux_config["n_leads"]
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, aux_config

    @classmethod
    def from_remote(
        cls,
        url: str,
        model_dir: Union[str, bytes, os.PathLike],
        filename: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, dict]:
        """Load the model from the remote model.

        Parameters
        ----------
        url : str
            URL of the remote model.
        model_dir : `path-like`
            Path for downloading the model.
        filename : str, optional
            Filename of the model to save, defaults to the basename of the URL.
        device : torch.device, optional
            Map location of the model parameters,
            defaults to "cuda" if available, otherwise "cpu".

        Returns
        -------
        model : torch.nn.Module
            The model loaded from a checkpoint.
        aux_config : dict
            Auxiliary configs that are needed for data preprocessing, etc.

        """
        http_get(url, model_dir, extract=True, filename=filename)
        return cls.from_checkpoint(model_dir, device=device)

    def save(self, path: Union[str, bytes, os.PathLike], train_config: CFG) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : `path-like`
            Path to save the model.
        train_config : CFG
            Config for training the model,
            used when one restores the model.

        Returns
        -------
        None

        """
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_config": self.config,
                "train_config": train_config,
            },
            path,
        )
