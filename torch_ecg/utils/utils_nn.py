"""
utilities for nn models
"""
from itertools import repeat
from math import floor
from typing import Union, Sequence, List, Tuple, Optional, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)


__all__ = [
    "extend_predictions",
    "compute_output_shape",
    "compute_conv_output_shape",
    "compute_maxpool_output_shape",
    "compute_avgpool_output_shape",
]


def extend_predictions(preds:Sequence, classes:List[str], extended_classes:List[str]) -> np.ndarray:
    """ finished, checked,

    extend the prediction arrays to prediction arrays in larger range of classes

    Parameters:
    -----------
    preds: sequence,
        sequence of predictions (scalar or binary),
        of shape (n_records, n_classes), or (n_classes,),
        where n_classes = `len(classes)`
    classes: list of str,
        classes of the predictions of `preds`
    extended_classes: list of str,
        a superset of `classes`

    Returns:
    --------
    extended_preds: ndarray,
        the extended array of predictions, with indices in `extended_classes`,
        of shape (n_records, n_classes), or (n_classes,)
    """
    _preds = np.atleast_2d(preds)
    assert _preds.shape[1] == len(classes), \
        f"`pred` indicates {_preds.shape[1]} classes, while `classes` has {len(classes)}"
    assert len(set(classes) - set(extended_classes)) == 0, \
        f"`extended_classes` is not a superset of `classes`, with {set(classes)-set(extended_classes)} in `classes` but not in `extended_classes`"

    extended_preds = np.zeros((_preds.shape[0], len(extended_classes)))

    for idx, c in enumerate(classes):
        new_idx = extended_classes.index(c)
        extended_preds[..., new_idx] = _preds[..., idx]

    if np.array(preds).ndim == 1:
        extended_preds = extended_preds[0]

    return extended_preds


# utils for computing output shape
def compute_output_shape(layer_type:str, input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, output_padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, checked,

    compute the output shape of a (transpose) convolution/maxpool/avgpool layer
    
    Parameters:
    -----------
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

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor

    References:
    -----------
    [1] https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    __TYPES__ = [
        'conv', 'convolution',
        'deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',
        'maxpool', 'maxpooling',
        'avgpool', 'avgpooling', 'averagepool', 'averagepooling',
    ]
    lt = "".join(layer_type.lower().split("_"))
    assert lt in __TYPES__
    if lt in ['conv', 'convolution',]:
        minus_term = lambda d, k: d * (k - 1) + 1
        out_channels = num_filters
    elif lt in ['maxpool', 'maxpooling',]:
        minus_term = lambda d, k: d * (k - 1) + 1
        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in ['avgpool', 'avgpooling', 'averagepool', 'averagepooling',]:
        minus_term = lambda d, k: k
        out_channels = input_shape[-1] if channel_last else input_shape[1]
    elif lt in ['deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',]:
        out_channels = num_filters
    dim = len(input_shape) - 2
    assert dim > 0, "input_shape should be a sequence of length at least 3, to be a valid (with batch and channel) shape of a non-degenerate Tensor"

    none_dim_msg = "only batch and channel dimension can be `None`"
    if channel_last:
        assert all([n is not None for n in input_shape[1:-1]]), none_dim_msg
    else:
        assert all([n is not None for n in input_shape[2:]]), none_dim_msg

    if isinstance(kernel_size, int):
        _kernel_size = list(repeat(kernel_size, dim))
    elif len(kernel_size) == dim:
        _kernel_size = kernel_size
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(kernel_size)} dimensions, both not including the channel dimension")
    
    if isinstance(stride, int):
        _stride = list(repeat(stride, dim))
    elif len(stride) == dim:
        _stride = stride
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(stride)} dimensions, both not including the channel dimension")

    if isinstance(padding, int):
        _padding = list(repeat(padding, dim))
    elif len(padding) == dim:
        _padding = padding
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(padding)} dimensions, both not including the channel dimension")

    if isinstance(output_padding, int):
        _output_padding = list(repeat(output_padding, dim))
    elif len(output_padding) == dim:
        _output_padding = output_padding
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(output_padding)} dimensions, both not including the channel dimension")

    if isinstance(dilation, int):
        _dilation = list(repeat(dilation, dim))
    elif len(dilation) == dim:
        _dilation = dilation
    else:
        raise ValueError(f"input has {dim} dimensions, while kernel has {len(dilation)} dimensions, both not including the channel dimension")
    
    if channel_last:
        _input_shape = input_shape[1:-1]
    else:
        _input_shape = input_shape[2:]
    
    if lt in ['deconv', 'deconvolution', 'transposeconv', 'transposeconvolution',]:
        output_shape = [
            (i-1) * s - 2 * p + d * (k-1) + o + 1 \
                for i, p, o, d, k, s in \
                    zip(_input_shape, _padding, _output_padding, _dilation, _kernel_size, _stride)
        ]    
    else:
        output_shape = [
            floor( ( ( i + 2*p - minus_term(d, k) ) / s ) + 1 ) \
                for i, p, d, k, s in \
                    zip(_input_shape, _padding, _dilation, _kernel_size, _stride)
        ]
    if channel_last:
        output_shape = tuple([input_shape[0]] + output_shape + [out_channels])
    else:
        output_shape = tuple([input_shape[0], out_channels] + output_shape)

    return output_shape


def compute_conv_output_shape(input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

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

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'conv',
        input_shape, num_filters, kernel_size, stride, padding, 0, dilation,
        channel_last,
    )
    return output_shape


def compute_maxpool_output_shape(input_shape:Sequence[Union[int, type(None)]], kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

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

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'maxpool',
        input_shape, 1, kernel_size, stride, padding, 0, dilation,
        channel_last,
    )
    return output_shape


def compute_avgpool_output_shape(input_shape:Sequence[Union[int, type(None)]], kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, cheched,

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

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'avgpool',
        input_shape, 1, kernel_size, stride, padding, 0, 1,
        channel_last,
    )
    return output_shape


def compute_deconv_output_shape(input_shape:Sequence[Union[int, type(None)]], num_filters:Optional[int]=None, kernel_size:Union[Sequence[int], int]=1, stride:Union[Sequence[int], int]=1, padding:Union[Sequence[int], int]=0, output_padding:Union[Sequence[int], int]=0, dilation:Union[Sequence[int], int]=1, channel_last:bool=False) -> Tuple[Union[int, type(None)]]:
    """ finished, checked,

    compute the output shape of a transpose convolution layer
    
    Parameters:
    -----------
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

    Returns:
    --------
    output_shape: tuple,
        shape of the output Tensor
    """
    output_shape = compute_output_shape(
        'deconv',
        input_shape, num_filters, kernel_size, stride, padding, output_padding, dilation,
        channel_last,
    )
    return output_shape
