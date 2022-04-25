"""
utilities for nn models

"""

import re
from copy import deepcopy
from itertools import repeat
from math import floor
from numbers import Real
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from ..cfg import DEFAULTS

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


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


def extend_predictions(
    preds: Sequence, classes: List[str], extended_classes: List[str]
) -> np.ndarray:
    """

    extend the prediction arrays to prediction arrays in larger range of classes

    Parameters
    ----------
    preds: sequence,
        sequence of predictions (scalar or binary),
        of shape (n_records, n_classes), or (n_classes,),
        where n_classes = `len(classes)`
    classes: list of str,
        classes of the predictions of `preds`
    extended_classes: list of str,
        a superset of `classes`

    Returns
    -------
    extended_preds: ndarray,
        the extended array of predictions, with indices in `extended_classes`,
        of shape (n_records, n_classes), or (n_classes,)

    """
    _preds = np.atleast_2d(preds)
    assert _preds.shape[1] == len(
        classes
    ), f"`pred` indicates {_preds.shape[1]} classes, while `classes` has {len(classes)}"
    assert (
        len(set(classes) - set(extended_classes)) == 0
    ), f"`extended_classes` is not a superset of `classes`, with {set(classes)-set(extended_classes)} in `classes` but not in `extended_classes`"

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
) -> Tuple[Union[int, None]]:
    """

    compute the output shape of a (transpose) convolution/maxpool/avgpool layer

    Parameters
    ----------
    layer_type: str,
        type (conv, maxpool, avgpool, etc.) of the layer
    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    out_padding: int, or sequence of int, default 0,
        additional size added to one side of the output shape,
        used only for transpose convolution
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns
    -------
    output_shape: tuple,
        shape of the output Tensor

    References
    ----------
    [1] https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5

    """
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
    assert lt in __TYPES__
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
    dim = len(input_shape) - 2
    assert (
        dim > 0
    ), "input_shape should be a sequence of length at least 3, to be a valid (with batch and channel) shape of a non-degenerate Tensor"

    # none_dim_msg = "only batch and channel dimension can be `None`"
    # if channel_last:
    #     assert all([n is not None for n in input_shape[1:-1]]), none_dim_msg
    # else:
    #     assert all([n is not None for n in input_shape[2:]]), none_dim_msg
    none_dim_msg = "spatial dimensions should be all `None`, or all not `None`"
    if channel_last:
        if all([n is None for n in input_shape[1:-1]]):
            if out_channels is None:
                raise ValueError(
                    "out channel dimension and spatial dimensions are all `None`"
                )
            output_shape = tuple(list(input_shape[:-1]) + [out_channels])
            return output_shape
        elif any([n is None for n in input_shape[1:-1]]):
            raise ValueError(none_dim_msg)
    else:
        if all([n is None for n in input_shape[2:]]):
            if out_channels is None:
                raise ValueError(
                    "out channel dimension and spatial dimensions are all `None`"
                )
            output_shape = tuple([input_shape[0], out_channels] + list(input_shape[2:]))
            return output_shape
        elif any([n is None for n in input_shape[2:]]):
            raise ValueError(none_dim_msg)

    if isinstance(kernel_size, int):
        _kernel_size = list(repeat(kernel_size, dim))
    elif len(kernel_size) == dim:
        _kernel_size = kernel_size
    else:
        raise ValueError(
            f"input has {dim} dimensions, while kernel has {len(kernel_size)} dimensions, both not including the channel dimension"
        )

    if isinstance(stride, int):
        _stride = list(repeat(stride, dim))
    elif len(stride) == dim:
        _stride = stride
    else:
        raise ValueError(
            f"input has {dim} dimensions, while kernel has {len(stride)} dimensions, both not including the channel dimension"
        )

    if isinstance(padding, int):
        _padding = list(repeat(list(repeat(padding, 2)), dim))
    elif len(padding) == 2 and isinstance(padding[0], int):
        _padding = list(repeat(padding, dim))
    elif (
        len(padding) == dim
        and all([isinstance(p, Sequence) for p in padding])
        and all([len(p) == 2 for p in padding])
    ):
        _padding = padding
    else:
        # raise ValueError(
        #     f"input has {dim} dimensions, while kernel has {len(padding)} dimensions, both not including the channel dimension"
        # )
        raise ValueError("Invalid padding")

    if isinstance(output_padding, int):
        _output_padding = list(repeat(output_padding, dim))
    elif len(output_padding) == dim:
        _output_padding = output_padding
    else:
        raise ValueError(
            f"input has {dim} dimensions, while kernel has {len(output_padding)} dimensions, both not including the channel dimension"
        )

    if isinstance(dilation, int):
        _dilation = list(repeat(dilation, dim))
    elif len(dilation) == dim:
        _dilation = dilation
    else:
        raise ValueError(
            f"input has {dim} dimensions, while kernel has {len(dilation)} dimensions, both not including the channel dimension"
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
            for i, p, d, k, s in zip(
                _input_shape, _padding, _dilation, _kernel_size, _stride
            )
        ]
    if channel_last:
        output_shape = tuple([input_shape[0]] + output_shape + [out_channels])
    else:
        output_shape = tuple([input_shape[0], out_channels] + output_shape)

    return output_shape


def compute_conv_output_shape(
    input_shape: Sequence[Union[int, None]],
    num_filters: Optional[int] = None,
    kernel_size: Union[Sequence[int], int] = 1,
    stride: Union[Sequence[int], int] = 1,
    padding: Union[Sequence[int], int] = 0,
    dilation: Union[Sequence[int], int] = 1,
    channel_last: bool = False,
) -> Tuple[Union[int, None]]:
    """finished, cheched,

    compute the output shape of a convolution/maxpool/avgpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns
    -------
    output_shape: tuple,
        shape of the output Tensor

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
    """finished, cheched,

    compute the output shape of a maxpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns
    -------
    output_shape: tuple,
        shape of the output Tensor

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
    """finished, cheched,

    compute the output shape of a avgpool layer

    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns
    -------
    output_shape: tuple,
        shape of the output Tensor

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
) -> Tuple[Union[int, None]]:
    """

    compute the output shape of a transpose convolution layer

    Parameters
    ----------
    input_shape: sequence of int or None,
        shape of an input Tensor,
        the first dimension is the batch dimension, which is allowed to be `None`
    num_filters: int, optional,
        number of filters, also the channel dimension
    kernel_size: int, or sequence of int, default 1,
        kernel size (filter size) of the layer, should be compatible with `input_shape`
    stride: int, or sequence of int, default 1,
        stride (down-sampling length) of the layer, should be compatible with `input_shape`
    padding: int, or sequence of int, default 0,
        padding length(s) of the layer, should be compatible with `input_shape`
    out_padding: int, or sequence of int, default 0,
        additional size added to one side of the output shape,
        used only for transpose convolution
    dilation: int, or sequence of int, default 1,
        dilation of the layer, should be compatible with `input_shape`
    channel_last: bool, default False,
        channel dimension is the last dimension,
        or the second dimension (the first is the batch dimension by convention)

    Returns
    -------
    output_shape: tuple,
        shape of the output Tensor

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
    )
    return output_shape


compute_sequential_output_shape_docstring = """

    Parameters
    ----------
    seq_len: int,
        length of the 1d sequence
    batch_size: int, optional,
        the batch size, can be None

    Returns
    -------
    output_shape: sequence,
        the output shape, given `seq_len` and `batch_size`

    """


def compute_sequential_output_shape(
    model: nn.Sequential,
    seq_len: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Sequence[Union[int, None]]:
    """

    compute the output shape of a sequential model

    Parameters
    ----------
    model: nn.Sequential,
        the sequential model
    seq_len: int,
        length of the 1d sequence
    batch_size: int, optional,
        the batch size, can be None

    Returns
    -------
    output_shape: sequence,
        the output shape, given `seq_len` and `batch_size`

    """
    assert issubclass(
        type(model), nn.Sequential
    ), f"model should be nn.Sequential, but got {type(model)}"
    _seq_len = seq_len
    for module in model:
        output_shape = module.compute_output_shape(_seq_len, batch_size)
        _, _, _seq_len = output_shape
    return output_shape


def compute_module_size(
    module: nn.Module, human: bool = False, dtype: str = "float32"
) -> Union[int, str]:
    """

    compute the size (number of parameters) of a module

    Parameters
    ----------
    module: Module,
        a torch Module
    human: bool, default False,
        return size in a way that is easy to read by a human,
        by appending a suffix corresponding to the unit (K, M, G, T, P)
    dtype: str, default "float32",
        data type of the module parameters, one of "float16", "float32", "float64"

    Returns
    -------
    n_params: int,
        size (number of parameters) of this torch module

    """
    module_parameters = filter(lambda p: p.requires_grad, module.parameters())
    n_params = sum([np.prod(p.size()) for p in module_parameters])
    if human:
        n_params = (
            n_params * {"float16": 2, "float32": 4, "float64": 8}[dtype.lower()] / 1024
        )
        div_count = 0
        while n_params >= 1024:
            n_params /= 1024
            div_count += 1
        # cvt_dict = {0:"K", 1:"M", 2:"G", 3:"T", 4:"P"}
        cvt_dict = {c: u for c, u in enumerate(list("KMGTP"))}
        n_params = f"""{n_params:.1f}{cvt_dict[div_count]}"""
    return n_params


def compute_receptive_field(
    kernel_sizes: Union[Sequence[int], int] = 1,
    strides: Union[Sequence[int], int] = 1,
    dilations: Union[Sequence[int], int] = 1,
    input_len: Optional[int] = None,
) -> Union[int, float]:
    r"""

    computes the (generic) receptive field of feature map of certain channel,
    from certain flow (if not merged, different flows, e.g. shortcut, must be computed separately),
    for convolutions, (non-global) poolings.
    "generic" refers to a general position, rather than specific positions,
    like on the edges, whose receptive field is definitely different

    .. math::
        Let the layers has kernel size, stride, dilation $(k_n, s_n, d_n)$ respectively.
        Let each feature map has receptive field length $r_n$,
        and difference of receptive fields of adjacent positions be $f_n$.
        By convention, $(r_0, f_0) = (1, 1)$. Then one has
        \begin{eqnarray}
        r_{n+1} & = & r_n + d_n(k_n-1)f_n, \\
        f_{n+1} & = & s_n f_n.
        \end{eqnarray}
        Hence
        \begin{eqnarray}
        f_{n} & = & \prod\limits_{i=0}^{n-1} s_i, \\
        r_{n} & = & 1 + \sum\limits_{i=0}^{n-1} d_i(k_i-1) \prod\limits_{i=0}^{j-1} s_j,
        \end{eqnarray}
        with empty products equaling 1 by convention.

    Parameters
    ----------
    kernel_sizes: int or sequence of int,
        the sequence of kernel size for all the layers in the flow
    strides: int or sequence of int, default 1,
        the sequence of strides for all the layers in the flow
    dilations: int or sequence of int, default 1,
        the sequence of strides for all the layers in the flow
    input_len: int, optional,
        length of the first feature map in the flow

    Returns
    -------
    receptive_field: int,
        (length) of the receptive field

    Examples
    --------
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2])
    90
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2],[2,1,2,4,1,8,8,8,1])
    484
    >>> compute_receptive_field([11,2,7,7,2,5,5,5,2],[1,2,1,1,2,1,1,1,2],[4,1,4,8,1,16,32,64,1])
    1984

    this is the receptive fields of the output feature maps
    of the 3 branches of the multi-scopic net, using its original hyper-parameters,
    (note the 3 max pooling layers)

    """
    _kernel_sizes = (
        [kernel_sizes] if isinstance(kernel_sizes, int) else list(kernel_sizes)
    )
    num_layers = len(_kernel_sizes)
    _strides = (
        list(repeat(strides, num_layers)) if isinstance(strides, int) else list(strides)
    )
    _dilations = (
        list(repeat(dilations, num_layers))
        if isinstance(dilations, int)
        else list(dilations)
    )
    assert num_layers == len(_strides) == len(_dilations)
    receptive_field = 1
    for idx, (k, d) in enumerate(zip(_kernel_sizes, _dilations)):
        s = np.prod(_strides[:idx]) if idx > 0 else 1
        receptive_field += d * (k - 1) * s
    if input_len is not None:
        receptive_field = min(receptive_field, input_len)
    return receptive_field


def default_collate_fn(batch: Sequence[Tuple[np.ndarray, ...]]) -> Tuple[Tensor, ...]:
    """

    collate functions for model training

    the data generator (`Dataset`) should generate (`__getitem__`) n-tuples `signals, labels, ...`

    Parameters
    ----------
    batch: sequence,
        sequence of n-tuples,
        in which the first element is the signal, the second is the label, ...

    Returns
    -------
    tuple of Tensor,
        the concatenated values to feed into neural networks

    """
    try:
        n_fields = len(batch[0])
    except Exception:
        raise ValueError("No data")
    ret = []
    for i in range(n_fields):
        values = [[item[i]] for item in batch]
        values = np.concatenate(values, axis=0).astype(DEFAULTS.np_dtype)
        values = torch.from_numpy(values)
        ret.append(values)
    return tuple(ret)


if torch.__version__ >= "1.15.0":

    def _true_divide(dividend, divisor):
        return torch.true_divide(dividend, divisor)

else:

    def _true_divide(dividend, divisor):
        return dividend / divisor


def _adjust_cnn_filter_lengths(
    config: dict,
    fs: int,
    ensure_odd: bool = True,
    pattern: str = "filter_length|filt_size",
) -> dict:
    """

    adjust the filter lengths (kernel sizes) in the config for convolutional neural networks,
    according to the new sampling frequency

    Parameters
    ----------
    config: dict,
        the config dictionary
    fs: int,
        the new sampling frequency
    ensure_odd: bool, default True,
        if True, the new filter lengths are ensured to be odd
    pattern: str, default "filter_length|filt_size",
        the pattern to search for in the config items related to filter lengths

    Returns
    -------
    config: dict,
        the adjusted config dictionary

    """
    assert "fs" in config
    config = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, dict):
            tmp_config = v
            tmp_config.update({"fs": config["fs"]})
            config[k] = _adjust_cnn_filter_lengths(tmp_config, fs, ensure_odd, pattern)
            config[k].pop("fs", None)
        elif re.findall(pattern, k):
            if isinstance(v, (Sequence, np.ndarray)):  # DO NOT use `Iterable`
                config[k] = [
                    _adjust_cnn_filter_lengths(
                        {"filter_length": fl, "fs": config["fs"]}, fs, ensure_odd
                    )["filter_length"]
                    for fl in v
                ]
            elif isinstance(v, Real):
                # DO NOT use `int`, which might not work for numpy array elements
                if v > 1:
                    config[k] = int(round(v * fs / config["fs"]))
                    if ensure_odd:
                        config[k] = config[k] - config[k] % 2 + 1
        elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            tmp_configs = [
                _adjust_cnn_filter_lengths(
                    {k: item, "fs": config["fs"]}, fs, ensure_odd, pattern
                )
                for item in v
            ]
            config[k] = [item[k] for item in tmp_configs]
    return config


def adjust_cnn_filter_lengths(
    config: dict,
    fs: int,
    ensure_odd: bool = True,
    pattern: str = "filter_length|filt_size",
) -> dict:
    """

    adjust the filter lengths in the config for convolutional neural networks,
    according to the new sampling frequency

    Parameters
    ----------
    config: dict,
        the config dictionary
    fs: int,
        the new sampling frequency
    ensure_odd: bool, default True,
        if True, the new filter lengths are ensured to be odd
    pattern: str, default "filter_length|filt_size",
        the pattern to search for in the config items related to filter lengths

    Returns
    -------
    config: dict,
        the adjusted config dictionary

    """
    config = _adjust_cnn_filter_lengths(config, fs, ensure_odd, pattern)
    config["fs"] = fs
    return config


class SizeMixin(object):
    """

    mixin class for size related methods

    """

    @property
    def module_size(self) -> int:
        return compute_module_size(
            self,
            dtype=self.dtype_,
        )

    @property
    def module_size_(self) -> str:
        return compute_module_size(
            self,
            human=True,
            dtype=self.dtype_,
        )

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
    """

    mixin class for loading from checkpoint class methods
    """

    @classmethod
    def from_checkpoint(
        cls, path: str, device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, dict]:
        """

        load a model from a checkpoint

        Parameters
        ----------
        path: str,
            path of the checkpoint
        device: torch.device, optional,
            map location of the model parameters,
            defaults "cuda" if available, otherwise "cpu"

        Returns
        -------
        model: Module,
            the model loaded from a checkpoint
        aux_config: dict,
            auxiliary configs that are needed for data preprocessing, etc.

        """
        _device = device or DEFAULTS.device
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert (
            aux_config is not None
        ), "input checkpoint has no sufficient data to recover a model"
        kwargs = dict(
            classes=aux_config["classes"],
            config=ckpt["model_config"],
        )
        if "n_leads" in aux_config:
            kwargs["n_leads"] = aux_config["n_leads"]
        model = cls(**kwargs)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, aux_config
