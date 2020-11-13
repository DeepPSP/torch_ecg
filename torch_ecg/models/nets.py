"""
basic building blocks, for 1d signal (time series)
"""
import sys
from copy import deepcopy
from math import sqrt
from itertools import repeat
from typing import Union, Sequence, Tuple, List, Optional, NoReturn
from numbers import Real

from packaging import version
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from easydict import EasyDict as ED
from deprecated import deprecated

from torch_ecg.cfg import Cfg
from torch_ecg.utils.utils_nn import (
    compute_output_shape,
    compute_conv_output_shape,
    compute_maxpool_output_shape,
    compute_avgpool_output_shape,
)
from torch_ecg.utils.misc import dict_to_str

if Cfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = np.float64
else:
    _DTYPE = np.float32


__all__ = [
    "Mish", "Swish",
    "Initializers", "Activations",
    "Bn_Activation", "Conv_Bn_Activation",
    "MultiConv", "BranchedConv",
    "DownSample",
    "BidirectionalLSTM", "StackedLSTM",
    # "AML_Attention", "AML_GatedAttention",
    "GlobalContextBlock",
    "AttentionWithContext",
    "MultiHeadAttention", "SelfAttention",
    "AttentivePooling",
    "ZeroPadding",
    "SeqLin",
    "NonLocalBlock", "SEBlock", "GlobalContextBlock",
    "WeightedBCELoss", "BCEWithLogitsWithClassWeightLoss",
    "default_collate_fn",
]


if version.parse(torch.__version__) >= version.parse('1.5.0'):
    def _true_divide(dividend, divisor):
        return torch.true_divide(dividend, divisor)
else:
    def _true_divide(dividend, divisor):
        return dividend / divisor


# ---------------------------------------------
# activations
class Mish(nn.Module):
    """ The Mish activation """
    __name__ = "Mish"
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        return input * (torch.tanh(F.softplus(input)))


class Swish(nn.Module):
    """ The Swish activation """
    __name__ = "Swish"
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, input:Tensor) -> Tensor:
        """
        """
        return input * F.sigmoid(input)


# ---------------------------------------------
# initializers
Initializers = ED()
Initializers.he_normal = nn.init.kaiming_normal_
Initializers.kaiming_normal = nn.init.kaiming_normal_
Initializers.he_uniform = nn.init.kaiming_uniform_
Initializers.kaiming_uniform = nn.init.kaiming_uniform_
Initializers.xavier_normal = nn.init.xavier_normal_
Initializers.glorot_normal = nn.init.xavier_normal_
Initializers.xavier_uniform = nn.init.xavier_uniform_
Initializers.glorot_uniform = nn.init.xavier_uniform_
Initializers.normal = nn.init.normal_
Initializers.uniform = nn.init.uniform_
Initializers.orthogonal = nn.init.orthogonal_
Initializers.zeros = nn.init.zeros_
Initializers.ones = nn.init.ones_
Initializers.constant = nn.init.constant_


# ---------------------------------------------
# activations
Activations = ED()
Activations.mish = Mish
Activations.swish = Swish
Activations.relu = nn.ReLU
Activations.leaky = nn.LeakyReLU
Activations.leaky_relu = Activations.leaky
Activations.tanh = nn.Tanh
Activations.sigmoid = nn.Sigmoid
Activations.softmax = nn.Softmax
# Activations.linear = None


_DEFAULT_CONV_CONFIGS = ED(
    batch_norm=True,
    activation="relu",
    kw_activation={"inplace": True},
    kernel_initializer="he_normal",
    kw_initializer={}
)


# ---------------------------------------------
# basic building blocks of CNN
class Bn_Activation(nn.Sequential):
    """ finished, checked,

    batch normalization --> activation
    """
    __name__ = "Bn_Activation"

    def __init__(self, num_features:int, activation:Union[str,nn.Module], kw_activation:Optional[dict]=None, dropout:float=0.0) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        num_features: int,
            number of features (channels) of the input (and output)
        activation: str or Module,
            name of the activation or an activation `Module`
        kw_activation: dict, optional,
            key word arguments for `activation`
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer at the end of the block
        """
        super().__init__()
        self.__num_features = num_features
        self.__kw_activation = kw_activation or {}
        self.__dropout = dropout
        if callable(activation):
            act_layer = activation
            act_name = f"activation_{type(act_layer).__name__}"
        elif isinstance(activation, str) and activation.lower() in Activations.keys():
            act_layer = Activations[activation.lower()](**self.__kw_activation)
            act_name = f"activation_{activation.lower()}"

        self.add_module(
            "batch_norm",
            nn.BatchNorm1d(num_features),
        )
        self.add_module(
            act_name,
            act_layer,
        )
        if self.__dropout > 0:
            self.add_module(
                "dropout",
                nn.Dropout(self.__dropout),
            )
    
    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`
        
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `Bn_Activation` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__num_features, seq_len)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class Conv_Bn_Activation(nn.Sequential):
    """ finished, checked,

    1d convolution --> batch normalization (optional) -- > activation (optional),
    with "same" padding as default padding
    """
    __name__ = "Conv_Bn_Activation"

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:Optional[int]=None, dilation:int=1, groups:int=1, batch_norm:Union[bool,nn.Module]=True, activation:Optional[Union[str,nn.Module]]=None, kernel_initializer:Optional[Union[str,callable]]=None, bias:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input signal
        out_channels: int,
            number of channels produced by the convolution
        kernel_size: int,
            size (length) of the convolving kernel
        stride: int,
            stride (subsample length) of the convolution
        padding: int, optional,
            zero-padding added to both sides of the input
        dilation: int, default 1,
            spacing between the kernel points
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        batch_norm: bool or Module, default True,
            batch normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        activation: str or Module, optional,
            name or Module of the activation,
            if is str, can be one of
            "mish", "swish", "relu", "leaky", "leaky_relu", "linear",
            "linear" is equivalent to `activation=None`
        kernel_initializer: str or callable (function), optional,
            a function to initialize kernel weights of the convolution,
            or name or the initialzer, can be one of the keys of `Initializers`
        bias: bool, default True,
            if True, adds a learnable bias to the output

        NOTE that if `padding` is not specified (default None),
        then the actual padding used for the convolutional layer is automatically computed
        to fit the "same" padding (not actually "same" for even kernel sizes)
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__dilation = dilation
        if padding is None:
            # 'same' padding
            self.__padding = (self.__dilation * (self.__kernel_size - 1)) // 2
        elif isinstance(padding, int):
            self.__padding = padding
        self.__groups = groups
        self.__bias = bias
        self.__kw_activation = kwargs.get("kw_activation", {})
        self.__kw_initializer = kwargs.get("kw_initializer", {})

        conv_layer = nn.Conv1d(
            self.__in_channels, self.__out_channels,
            self.__kernel_size, self.__stride, self.__padding, self.__dilation, self.__groups,
            bias=self.__bias,
        )

        if kernel_initializer:
            if callable(kernel_initializer):
                kernel_initializer(conv_layer.weight)
            elif isinstance(kernel_initializer, str) and kernel_initializer.lower() in Initializers.keys():
                Initializers[kernel_initializer.lower()](conv_layer.weight, **self.__kw_initializer)
            else:  # TODO: add more initializers
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        self.add_module("conv1d", conv_layer)

        if batch_norm:
            bn_layer = nn.BatchNorm1d(out_channels) if isinstance(batch_norm, bool) else batch_norm(out_channels)
            self.add_module("batch_norm", bn_layer)

        if isinstance(activation, str):
            activation = activation.lower()

        if not activation:
            act_layer = None
            act_name = None
        elif callable(activation):
            act_layer = activation
            act_name = f"activation_{type(act_layer).__name__}"
        elif isinstance(activation, str) and activation.lower() in Activations.keys():
            act_layer = Activations[activation.lower()](**self.__kw_activation)
            act_name = f"activation_{activation.lower()}"
        else:
            print(f"activate error !!! {sys._getframe().f_code.co_filename} {sys._getframe().f_code.co_name} {sys._getframe().f_lineno}")
            act_layer = None
            act_name = None

        if act_layer:
            self.add_module(act_name, act_layer)

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `Conv_Bn_Activation` layer, given `seq_len` and `batch_size`
        """
        input_shape = [batch_size, self.__in_channels, seq_len]
        output_shape = compute_conv_output_shape(
            input_shape=input_shape,
            num_filters=self.__out_channels,
            kernel_size=self.__kernel_size,
            stride=self.__stride,
            dilation=self.__dilation,
            padding=self.__padding,
            channel_last=False,
        )
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class MultiConv(nn.Sequential):
    """ finished, checked,

    a sequence (stack) of `Conv_Bn_Activation` blocks,
    perhaps with `Dropout` between
    """
    __DEBUG__ = False
    __name__ = "MultiConv"
    
    def __init__(self, in_channels:int, out_channels:Sequence[int], filter_lengths:Union[Sequence[int],int], subsample_lengths:Union[Sequence[int],int]=1, dilations:Union[Sequence[int],int]=1, groups:int=1, dropouts:Union[Sequence[float], float]=0.0, out_activation:bool=True, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        dilations: int or sequence of int, default 1,
            spacing between the kernel points of (each) convolutional layer
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        out_activation: bool, default True,
            if True, the last mini-block of `Conv_Bn_Activation` will have activation as in `config`,
            otherwise None
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = list(out_channels)
        self.__num_convs = len(self.__out_channels)
        self.config = deepcopy(_DEFAULT_CONV_CONFIGS)
        self.config.update(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        if isinstance(filter_lengths, int):
            kernel_sizes = list(repeat(filter_lengths, self.__num_convs))
        else:
            kernel_sizes = list(filter_lengths)
        assert len(kernel_sizes) == self.__num_convs

        if isinstance(subsample_lengths, int):
            strides = list(repeat(subsample_lengths, self.__num_convs))
        else:
            strides = list(subsample_lengths)
        assert len(strides) == self.__num_convs

        if isinstance(dropouts, Real):
            _dropouts = list(repeat(dropouts, self.__num_convs))
        else:
            _dropouts = list(dropouts)
        assert len(_dropouts) == self.__num_convs

        if isinstance(dilations, int):
            _dilations = list(repeat(dilations, self.__num_convs))
        else:
            _dilations = list(dilations)
        assert len(_dilations) == self.__num_convs

        conv_in_channels = self.__in_channels
        for idx, (oc, ks, sd, dl, dp) in \
            enumerate(zip(self.__out_channels, kernel_sizes, strides, _dilations, _dropouts)):
            if idx < self.__num_convs - 1 or out_activation:
                activation = self.config.activation
            else:
                activation = None
            self.add_module(
                f"cba_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=oc,
                    kernel_size=ks,
                    stride=sd,
                    dilation=dl,
                    groups=groups,
                    batch_norm=self.config.batch_norm,
                    activation=activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                ),
            )
            conv_in_channels = oc
            if dp > 0:
                self.add_module(
                    f"dropout_{idx}",
                    nn.Dropout(dp),
                )
    
    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`
        input: of shape (batch_size, n_channels, seq_len)
        """
        out = super().forward(input)
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `MultiConv` layer, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            if hasattr(module, "__name__") and module.__name__ == Conv_Bn_Activation.__name__:
                output_shape = module.compute_output_shape(_seq_len, batch_size)
                _, _, _seq_len = output_shape
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class BranchedConv(nn.Module):
    """

    branched `MultiConv` blocks
    """
    __DEBUG__ = False
    __name__ = "BranchedConv"

    def __init__(self, in_channels:int, out_channels:Sequence[Sequence[int]], filter_lengths:Union[Sequence[Sequence[int]],Sequence[int],int], subsample_lengths:Union[Sequence[Sequence[int]],Sequence[int],int]=1, dilations:Union[Sequence[Sequence[int]],Sequence[int],int]=1, groups:int=1, dropouts:Union[Sequence[Sequence[float]], Sequence[float],float]=0.0, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of sequence of int,
            number of channels produced by the convolutional layers
        filter_lengths: int or sequence of int,
            length(s) of the filters (kernel size)
        subsample_lengths: int or sequence of int,
            subsample length(s) (stride(s)) of the convolutions
        dilations: int or sequence of int, default 1,
            spacing between the kernel points of (each) convolutional layer
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        dropouts: float or sequence of float, default 0.0,
            dropout ratio after each `Conv_Bn_Activation`
        config: dict,
            other parameters, including
            activation choices, weight initializer, batch normalization choices, etc.
            for the convolutional layers
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = list(out_channels)
        assert all([isinstance(item, (Sequence,np.ndarray)) for item in self.__out_channels])
        self.__num_branches = len(self.__out_channels)
        self.config = deepcopy(_DEFAULT_CONV_CONFIGS)
        self.config.update(config)
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        if isinstance(filter_lengths, int):
            kernel_sizes = list(repeat(filter_lengths, self.__num_branches))
        else:
            kernel_sizes = list(filter_lengths)
        assert len(kernel_sizes) == self.__num_branches

        if isinstance(subsample_lengths, int):
            strides = list(repeat(subsample_lengths, self.__num_branches))
        else:
            strides = list(subsample_lengths)
        assert len(strides) == self.__num_branches

        if isinstance(dropouts, Real):
            _dropouts = list(repeat(dropouts, self.__num_branches))
        else:
            _dropouts = list(dropouts)
        assert len(_dropouts) == self.__num_branches

        if isinstance(dilations, int):
            _dilations = list(repeat(dilations, self.__num_branches))
        else:
            _dilations = list(dilations)
        assert len(_dilations) == self.__num_branches

        self.branches = nn.ModuleDict()
        for idx, (oc, ks, sd, dl, dp) in \
            enumerate(zip(self.__out_channels, kernel_sizes, strides, _dilations, _dropouts)):
            self.branches[f"multi_conv_{idx}"] = \
                MultiConv(
                    in_channels=self.__in_channels,
                    out_channels=oc,
                    filter_lengths=ks,
                    subsample_lengths=sd,
                    dilations=dl,
                    groups=groups,
                    dropouts=dp,
                    **(self.config),
                )
    
    def forward(self, input:Tensor) -> List[Tensor]:
        """
        input: of shape (batch_size, n_channels, seq_len)
        """
        out = []
        for idx in range(self.__num_branches):
            out.append(self.branches[f"multi_conv_{idx}"](input))
        return out

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> List[Sequence[Union[int, type(None)]]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shapes: list of sequence,
            list of output shapes of each branch of this `BranchedConv` layer,
            given `seq_len` and `batch_size`
        """
        output_shapes = []
        for idx in range(self.__num_branches):
            branch_output_shape = \
                self.branches[f"multi_conv_{idx}"].compute_output_shape(seq_len, batch_size)
            output_shapes.append(branch_output_shape)
        return output_shapes

    @property
    def module_size(self) -> int:
        """
        """
        n_params = 0
        for idx in range(self.__num_branches): 
            module_parameters = \
                filter(lambda p: p.requires_grad, self.branches[f"multi_conv_{idx}"].parameters())
            n_params += sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class DownSample(nn.Sequential):
    """

    NOTE: this down sampling module allows changement of number of channels,
    via additional convolution, with some abuse of terminology

    the 'conv' mode is not simply down 'sampling' if `group` != `in_channels`
    """
    __name__ = "DownSample"
    __MODES__ = ['max', 'avg', 'conv', 'nearest', 'area', 'linear',]

    def __init__(self, down_scale:int, in_channels:int, out_channels:Optional[int]=None, groups:Optional[int]=None, padding:int=0, batch_norm:Union[bool,nn.Module]=False, mode:str='max') -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        down_scale: int,
            scale of down sampling
        in_channels: int,
            number of channels of the input
        out_channels: int, optional,
            number of channels of the output
        groups: int, optional,
            connection pattern (of channels) of the inputs and outputs
        padding: int, default 0,
            zero-padding added to both sides of the input
        batch_norm: bool or Module, default False,
            batch normalization,
            the Module itself or (if is bool) whether or not to use `nn.BatchNorm1d`
        mode: str, default 'max',
        """
        super().__init__()
        self.__mode = mode.lower()
        assert self.__mode in self.__MODES__
        self.__down_scale = down_scale
        self.__in_channels = in_channels
        self.__out_channels = out_channels or in_channels
        self.__groups = groups or self.__in_channels
        self.__padding = padding

        if self.__mode == 'max':
            if self.__in_channels == self.__out_channels:
                down_layer = nn.MaxPool1d(
                    kernel_size=self.__down_scale, padding=self.__padding
                )
            else:
                down_layer = nn.Sequential((
                    nn.MaxPool1d(
                        kernel_size=self.__down_scale, padding=self.__padding
                    ),
                    nn.Conv1d(
                        self.__in_channels, self.__out_channels, 
                        kernel_size=1, groups=self.__groups, bias=False
                    ),
                ))
        elif self.__mode == 'avg':
            if self.__in_channels == self.__out_channels:
                down_layer = nn.AvgPool1d(
                    kernel_size=self.__down_scale, padding=self.__padding
                )
            else:
                down_layer = nn.Sequential(
                    (
                        nn.AvgPool1d(
                            kernel_size=self.__down_scale, padding=self.__padding
                        ),
                        nn.Conv1d(
                            self.__in_channels,self.__out_channels,
                            kernel_size=1, groups=self.__groups, bias=False,
                        ),
                    )
                )
        elif self.__mode == 'conv':
            down_layer = nn.Conv1d(
                in_channels=self.__in_channels,
                out_channels=self.__out_channels,
                kernel_size=1,
                groups=self.__groups,
                bias=False,
                stride=self.__down_scale,
            )
        else:
            down_layer = None
        if down_layer:
            self.add_module(
                "down_sample",
                down_layer,
            )

        if batch_norm:
            bn_layer = nn.BatchNorm1d(self.__out_channels) if isinstance(batch_norm, bool) \
                else batch_norm(self.__out_channels)
            self.add_module(
                "batch_normalization",
                bn_layer,
            )

    def forward(self, input:Tensor) -> Tensor:
        """
        use the forward method of `nn.Sequential`

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        if self.__mode in ['max', 'avg', 'conv',]:
            output = super().forward(input)
        else:
            # align_corners = False if mode in ['nearest', 'area'] else True
            output = F.interpolate(
                input=input,
                scale_factor=1/self.__down_scale,
                mode=mode,
                # align_corners=align_corners,
            )
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `DownSample` layer, given `seq_len` and `batch_size`
        """
        if self.__mode == 'conv':
            out_seq_len = compute_conv_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode == 'max':
            out_seq_len = compute_maxpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__down_scale, stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        elif self.__mode in ['avg', 'nearest', 'area', 'linear',]:
            out_seq_len = compute_avgpool_output_shape(
                input_shape=(batch_size, self.__in_channels, seq_len),
                kernel_size=self.__down_scale, stride=self.__down_scale,
                padding=self.__padding,
            )[-1]
        output_shape = (batch_size, self.__out_channels, out_seq_len)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class BidirectionalLSTM(nn.Module):
    """
    from crnn_torch of references.ati_cnn
    """
    __name__ = "BidirectionalLSTM"

    def __init__(self, input_size:int, hidden_size:int, num_layers:int=1, bias:bool=True, dropout:float=0.0, return_sequences:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        input_size: int,
            the number of features in the input
        hidden_size: int,
            the number of features in the hidden state
        num_layers: int, default 1,
            number of lstm layers
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropout: float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
        return_sequences: bool, default True,
            if True, returns the the full output sequence,
            otherwise the last output in the output sequence
        kwargs: dict,
            other parameters,
        """
        super().__init__()
        self.__output_size = 2 * hidden_size
        self.return_sequence = return_sequences

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bias=bias,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        """
        output, _ = self.lstm(input)  #  seq_len, batch_size, 2 * hidden_size
        if not self.return_sequence:
            output = output[-1,...]
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `BidirectionalLSTM` layer, given `seq_len` and `batch_size`
        """
        output_shape = (seq_len, batch_size, self.__output_size)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class StackedLSTM(nn.Sequential):
    """ finished, checked (no bugs, but correctness is left further to check),

    stacked LSTM, which allows different hidden sizes for each LSTM layer

    NOTE:
    -----
    1. `batch_first` is fixed `False`
    2. currently, how to correctly pass the argument `hx` between LSTM layers is not known to me, hence should be careful (and not recommended, use `nn.LSTM` and set `num_layers` instead) to use
    """
    __DEBUG__ = False
    __name__ = "StackedLSTM"

    def __init__(self, input_size:int, hidden_sizes:Sequence[int], bias:Union[Sequence[bool], bool]=True, dropouts:Union[float,Sequence[float]]=0.0, bidirectional:bool=True, return_sequences:bool=True, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        input_size: int,
            the number of features in the input
        hidden_sizes: sequence of int,
            the number of features in the hidden state of each LSTM layer
        bias: bool, or sequence of bool, default True,
            use bias weights or not
        dropouts: float or sequence of float, default 0.0,
            if non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer EXCEPT the last layer, with dropout probability equal to this value
            or corresponding value in the sequence (except for the last LSTM layer)
        bidirectional: bool, default True,
            if True, each LSTM layer becomes bidirectional
        return_sequences: bool, default True,
            if True, returns the the full output sequence,
            otherwise the last output in the output sequence
        kwargs: dict,
            other parameters,
        """
        super().__init__()
        self.__hidden_sizes = hidden_sizes
        self.num_lstm_layers = len(hidden_sizes)
        l_bias = bias if isinstance(bias, Sequence) else list(repeat(bias, self.num_lstm_layers))
        self.__dropouts = dropouts if isinstance(dropouts, Sequence) else list(repeat(dropouts, self.num_lstm_layers))
        self.bidirectional = bidirectional
        self.batch_first = False
        self.return_sequences = return_sequences

        layer_name_prefix = "bidirectional_lstm" if bidirectional else "lstm"
        for idx, (hs, b) in enumerate(zip(hidden_sizes, l_bias)):
            if idx == 0:
                _input_size = input_size
            else:
                _input_size = hidden_sizes[idx-1]
                if self.bidirectional:
                    _input_size = 2*_input_size
            self.add_module(
                name=f"{layer_name_prefix}_{idx+1}",
                module=nn.LSTM(
                    input_size=_input_size,
                    hidden_size=hs,
                    num_layers=1,
                    bias=b,
                    batch_first=self.batch_first,
                    bidirectional=self.bidirectional,
                )
            )
            if self.__dropouts[idx] > 0 and idx < self.num_lstm_layers-1:
                self.add_module(
                    name=f"dropout_{idx+1}",
                    module=nn.Dropout(self.__dropouts[idx]),
                )
    
    def forward(self, input:Union[Tensor, PackedSequence], hx:Optional[Tuple[Tensor, Tensor]]=None) -> Union[Tensor, Tuple[Union[Tensor, PackedSequence], Tuple[Tensor, Tensor]]]:
        """
        keep up with `nn.LSTM.forward`, parameters ref. `nn.LSTM.forward`

        Parameters:
        -----------
        input: Tensor,
            of shape (seq_len, batch_size, n_channels)
        hx: 2-tuple of Tensor, optional,
        """
        n_layers = 0
        output, _hx = input, hx
        div = 2 if self.__dropout > 0 else 1
        for module in self:
            n_lstm, res = divmod(n_layers, div)
            if res == 1:
                output = module(output)
                # print(f"module = {type(module).__name__}")
            else:
                # print(f"n_layers = {n_layers}, input shape = {output.shape}")
                if n_lstm > 0:
                    _hx = None
                output, _hx = module(output, _hx)
                # print(f"module = {type(module).__name__}")
                # print(f"n_layers = {n_layers}, input shape = {output.shape}")
            n_layers += 1
        if self.return_sequences:
            final_output = output  # seq_len, batch_size, n_direction*hidden_size
        else:
            final_output = output[-1,...]  # batch_size, n_direction*hidden_size
        return final_output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `StackedLSTM` layer, given `seq_len` and `batch_size`
        """
        output_size = self.__hidden_sizes[-1]
        if self.bidirectional:
            output_size *= 2
        if self.return_sequences:
            output_shape = (seq_len, batch_size, output_size)
        else:
            output_shape = (batch_size, output_size)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


# ---------------------------------------------
# attention mechanisms, from various sources
@deprecated(reason="not checked yet")
class AML_Attention(nn.Module):
    """ NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    References:
    -----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L6
    """
    __name__ = "AML_Attention"

    def __init__(self, L:int, D:int, K:int):
        """ NOT checked,
        """
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, input):
        """
        """
        A = self.attention(input)  # NxK
        return A


@deprecated(reason="not checked yet")
class AML_GatedAttention(nn.Module):
    """ NOT checked,

    the feature extraction part is eliminated,
    with only the attention left,

    TODO: compare with `nn.MultiheadAttention`

    References:
    -----------
    [1] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py#L72
    """
    __name__ = "AML_GatedAttention"

    def __init__(self, L:int, D:int, K:int):
        """ NOT checked,
        """
        super().__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, input):
        """
        """
        A_V = self.attention_V(input)  # NxD
        A_U = self.attention_U(input)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        return A


class AttentionWithContext(nn.Module):
    """ finished, checked (might have bugs),

    from 0236 of CPSC2018 challenge
    """
    __DEBUG__ = False
    __name__ = "AttentionWithContext"

    def __init__(self, in_channels:int, bias:bool=True, initializer:str='glorot_uniform'):
        """ finished, checked (might have bugs),

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input signal
        bias: bool, default True,
            if True, adds a learnable bias to the output
        initializer: str, default 'glorot_uniform',
            weight initializer
        """
        super().__init__()
        self.supports_masking = True
        self.init = Initializers[initializer.lower()]
        self.bias = bias

        self.W = Parameter(torch.Tensor(in_channels, in_channels))
        if self.__DEBUG__:
            print(f"AttentionWithContext W.shape = {self.W.shape}")
        self.init(self.W)

        if self.bias:
            self.b = Parameter(torch.Tensor(in_channels))
            Initializers.zeros(self.b)
            if self.__DEBUG__:
                print(f"AttentionWithContext b.shape = {self.b.shape}")
            # Initializers['zeros'](self.b)
            self.u = Parameter(torch.Tensor(in_channels))
            Initializers.constant(self.u, 1/in_channels)
            if self.__DEBUG__:
                print(f"AttentionWithContext u.shape = {self.u.shape}")
            # self.init(self.u)
        else:
            self.register_parameter('b', None)
            self.register_parameter('u', None)

    def compute_mask(self, input:Tensor, input_mask:Optional[Tensor]=None):
        """

        Parameters:
        -----------
        to write
        """
        return None

    def forward(self, input:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, seq_len, n_channels)
        mask: Tensor, optional,
        """
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: input.shape = {input.shape}, W.shape = {self.W.shape}")

        # linear + activation
        # (batch_size, seq_len, n_channels) x (n_channels, n_channels)
        # -> (batch_size, seq_len, n_channels)
        uit = torch.tensordot(input, self.W, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: uit.shape = {uit.shape}")
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)

        # scores (weights)
        # (batch_size, seq_len, n_channels) x (n_channels,)
        # -> (batch_size, seq_len)
        ait = torch.tensordot(uit, self.u, dims=1)  # the same as torch.matmul
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: ait.shape = {ait.shape}")
        
        # softmax along seq_len
        # (batch_size, seq_len)
        a = torch.exp(ait)
        if mask is not None:
            a_masked = a * mask
        else:
            a_masked = a
        a_masked = _true_divide(a_masked, torch.sum(a_masked, dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: a_masked.shape = {a_masked.shape}")

        # weighted -> sum
        # (batch_size, seq_len, n_channels) x (batch_size, seq_len, 1)
        # -> (batch_size, seq_len, n_channels)
        weighted_input = input * a[..., np.newaxis]
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: weighted_input.shape = {weighted_input.shape}")
        output = torch.sum(weighted_input, dim=-1)
        if self.__DEBUG__:
            print(f"AttentionWithContext forward: output.shape = {output.shape}")
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `AttentionWithContext` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class _ScaledDotProductAttention(nn.Module):
    """
    References:
    -----------
    [1] https://github.com/CyberZHG/torch-multi-head-attention
    """
    __DEBUG__ = False
    __name__ = "_ScaledDotProductAttention"

    def forward(self, query:Tensor, key:Tensor, value:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        all tensors of shape (batch_size, seq_len, features)
        """
        # if self.__DEBUG__:
        #     print(f"query.shape = {query.shape}, key.shape = {key.shape}, value.shape = {value.shape}")
        dk = query.shape[-1]
        scores = query.matmul(key.transpose(-2, -1)) / sqrt(dk)  # -> (batch_size, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        output = attention.matmul(value)
        # if self.__DEBUG__:
        #     print(f"scores.shape = {scores.shape}, attention.shape = {attention.shape}, output.shape = {output.shape}")
        return output


class MultiHeadAttention(nn.Module):
    """

    Multi-head attention.

    References:
    -----------
    [1] https://github.com/CyberZHG/torch-multi-head-attention
    """
    __DEBUG__ = False
    __name__ = "MultiHeadAttention"

    def __init__(self,
                 in_features:int,
                 head_num:int,
                 bias:bool=True,
                 activation:Optional[Union[str,nn.Module]]="relu",
                 **kwargs):
        """ finished, checked,

        Parameters:
        -----------
        in_features: int,
            Size of each input sample
        head_num: int,
            Number of heads.
        bias: bool, default True,
            Whether to use the bias term.
        activation: str or Module,
            The activation after each linear transformation.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(f'`in_features`({in_features}) should be divisible by `head_num`({head_num})')
        self.in_features = in_features
        self.head_num = head_num
        self.activation = Activations[activation.lower()]() if isinstance(activation, str) else activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Optional[Tensor]=None) -> Tensor:
        """
        q, k, v are of shape (seq_len, batch_size, features)
        in order to keep accordance with `nn.MultiheadAttention`
        """
        # all (seq_len, batch_size, features) -> (batch_size, seq_len, features)
        q = self.linear_q(q.permute(1,0,2))
        k = self.linear_k(k.permute(1,0,2))
        v = self.linear_v(v.permute(1,0,2))
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = _ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        # shape back from (batch_size, seq_len, features) -> (seq_len, batch_size, features)
        y = y.permute(1,0,2)
        return y

    def _reshape_to_batches(self, x:Tensor) -> Tensor:
        """
        """
        batch_size, seq_len, in_features = x.shape
        sub_dim = in_features // self.head_num
        # if self.__DEBUG__:
        #     print(f"batch_size = {batch_size}, seq_len = {seq_len}, in_features = {in_features}, sub_dim = {sub_dim}")
        reshaped = x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                    .permute(0, 2, 1, 3)\
                    .reshape(batch_size * self.head_num, seq_len, sub_dim)
        return reshaped

    def _reshape_from_batches(self, x:Tensor) -> Tensor:
        """
        """
        batch_size, seq_len, in_features = x.shape
        batch_size //= self.head_num
        out_dim = in_features * self.head_num
        # if self.__DEBUG__:
        #     print(f"batch_size = {batch_size}, seq_len = {seq_len}, in_features = {in_features}, out_dim = {out_dim}")
        reshaped = x.reshape(batch_size, self.head_num, seq_len, in_features)\
                    .permute(0, 2, 1, 3)\
                    .reshape(batch_size, seq_len, out_dim)
        return reshaped

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `MHA` layer, given `seq_len` and `batch_size`
        """
        output_shape = (seq_len, batch_size, self.in_features*self.head_num)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class SelfAttention(nn.Module):
    """
    """
    __DEBUG__ = False
    __name__ = "SelfAttention"

    def __init__(self, in_features:int, head_num:int, dropout:float=0.0, bias:bool=True, activation:Optional[Union[str,nn.Module]]="relu", **kwargs):
        """ finished, checked,

        Parameters:
        -----------
        in_features: int,
            size of each input sample
        head_num: int,
            number of heads.
        dropout: float, default 0,
            dropout factor for out projection weight of MHA
        bias: bool, default True,
            whether to use the bias term.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(f'`in_features`({in_features}) should be divisible by `head_num`({head_num})')
        self.in_features = in_features
        self.head_num = head_num
        self.dropout = dropout
        self.bias = bias
        self.mha = nn.MultiheadAttention(
            in_features, head_num, dropout=dropout, bias=bias,
        )
        # self.mha = MultiHeadAttention(
        #     in_features, head_num, bias=bias,
        # )

    def forward(self, input:Tensor) -> Tensor:
        """
        input of shape (seq_len, batch_size, features)
        output of shape (seq_len, batch_size, features)
        """
        output, _ = self.mha(input, input, input)
        # output = self.mha(input, input, input)
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `SelfAttention` layer, given `seq_len` and `batch_size`
        """
        output_shape = (seq_len, batch_size, self.in_features)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class AttentivePooling(nn.Module):
    """
    """
    __DEBUG__ = False
    __name__ = "AttentivePooling"

    def __init__(self, in_channels:int, mid_channels:Optional[int]=None, activation:Optional[Union[str,nn.Module]]='tanh', dropout:float=0.2, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels of input Tensor
        mid_channels: int, optional,
            output channels of a intermediate linear layer
        activation: str or Module,
            name of the activation or an activation `Module`
        dropout: float, default 0.2,
            dropout ratio before computing attention scores
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = (mid_channels or self.__in_channels//2) or 1
        self.__dropout = dropout
        self.__kw_activation = kwargs.get("kw_activation", {})
        if callable(activation):
            self.activation = activation
        elif isinstance(activation, str) and activation.lower() in Activations.keys():
            self.activation = Activations[activation.lower()](**self.__kw_activation)

        self.dropout = nn.Dropout(self.__dropout, inplace=False)
        self.mid_linear = nn.Linear(self.__in_channels, self.__mid_channels)
        self.contraction = nn.Linear(self.__mid_channels, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, seq_len, n_channels)
        """
        scores = self.dropout(input)
        scores = self.mid_linear(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.activation(scores)  # -> (batch_size, seq_len, n_channels)
        scores = self.contraction(scores)  # -> (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # -> (batch_size, seq_len)
        scores = self.softmax(scores)  # -> (batch_size, seq_len)
        weighted_input = \
            input * (scores[..., np.newaxis]) # -> (batch_size, seq_len, n_channels)
        output = weighted_input.sum(1)  # -> (batch_size, n_channels)
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `AttentivePooling` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__in_channels)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class ZeroPadding(nn.Module):
    """
    zero padding for increasing channels,
    degenerates to `identity` if in and out channels are equal
    """
    __name__ = "ZeroPadding"
    __LOC__ = ["head", "tail",]

    def __init__(self, in_channels:int, out_channels:int, loc:str="head") -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels for the output
        loc: str, default "top", case insensitive,
            padding to the head or the tail channel
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__increase_channels = out_channels - in_channels
        assert self.__increase_channels >= 0
        self.__loc = loc.lower()
        assert self.__loc in self.__LOC__
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        batch_size, _, seq_len = input.shape
        if self.__increase_channels > 0:
            output = torch.zeros((batch_size, self.__increase_channels, seq_len))
            output = output.to(device=self.__device)
            if self.__loc == "head":
                output = torch.cat((output, input), dim=1)
            elif self.__loc == "tail":
                output = torch.cat((input, output), dim=1)
        else:
            output = input
        return output

    def compute_output_shape(self, seq_len:int, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `ZeroPadding` layer, given `seq_len` and `batch_size`
        """
        output_shape = (batch_size, self.__out_channels, seq_len)
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class SeqLin(nn.Sequential):
    """
    Sequential linear,
    might be useful in learning non-linear classifying hyper-surfaces
    """
    __DEBUG__ = False
    __name__ = "SeqLin"

    def __init__(self, in_channels:int, out_channels:Sequence[int], activation:str="relu", kernel_initializer:Optional[str]=None, bias:bool=True, dropouts:Union[float,Sequence[float]]=0.0, **kwargs) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        out_channels: sequence of int,
            number of ouput channels for each linear layer
        activation: str, default "relu",
            name of activation after each linear layer
        kernel_initializer: str, optional,
            name of kernel initializer for `weight` of each linear layer
        bias: bool, default True,
            if True, each linear layer will have a learnable bias vector
        dropouts: float or sequence of float, default 0,
            dropout ratio(s) (if > 0) after each (activation after each) linear layer
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__num_layers = len(self.__out_channels)
        self.__kw_activation = kwargs.get("kw_activation", {})
        self.__kw_initializer = kwargs.get("kw_initializer", {})
        if activation.lower() in Activations.keys():
            self.__activation = activation.lower()
        else:
            raise ValueError(f"activation `{activation}` not supported")
        if kernel_initializer:
            if kernel_initializer.lower() in Initializers.keys():
                self.__kernel_initializer = Initializers[kernel_initializer.lower()]
            else:
                raise ValueError(f"initializer `{kernel_initializer}` not supported")
        else:
            self.__kernel_initializer = None
        self.__bias = bias
        if isinstance(dropouts, Real):
            self.__dropouts = list(repeat(dropouts, self.__num_layers))
        else:
            self.__dropouts = dropouts
            assert len(self.__dropouts) == self.__num_layers, \
                f"`out_channels` indicates {self.__num_layers} linear layers, while `dropouts` indicates {len(self.__dropouts)}"
        self.__skip_last_activation = kwargs.get("skip_last_activation", False)
        
        lin_in_channels = self.__in_channels
        for idx in range(self.__num_layers):
            lin_layer = nn.Linear(
                in_features=lin_in_channels,
                out_features=self.__out_channels[idx],
                bias=self.__bias,
            )
            if self.__kernel_initializer:
                self.__kernel_initializer(lin_layer.weight, **self.__kw_initializer)
            self.add_module(
                f"lin_{idx}",
                lin_layer,
            )
            if idx < self.__num_layers-1 or not self.__skip_last_activation:
                self.add_module(
                    f"act_{idx}",
                    Activations[self.__activation](**self.__kw_activation),
                )
                if self.__dropouts[idx] > 0:
                    self.add_module(
                        f"dropout_{idx}",
                        nn.Dropout(self.__dropouts[idx]),
                    )
            lin_in_channels = self.__out_channels[idx]

    def forward(self, input:Tensor) -> Tensor:
        """
        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels)
        
        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels) or (batch_size, seq_len, n_channels),
            ndim in accordance with `input`
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None, input_seq:bool=True) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int, optional,
            length of the 1d sequence,
        batch_size: int, optional,
            the batch size, can be None
        input_seq: bool, default True,
            if True, the input is a sequence (Tensor of dim 3) of vectors of features,
            otherwise a vector of features (Tensor of dim 2)

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `SeqLin` layer, given `seq_len` and `batch_size`
        """
        if input_seq:
            output_shape = (batch_size, seq_len, self.__out_channels[-1])
        else:
            output_shape = (batch_size, self.__out_channels[-1])
        return output_shape

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class NonLocalBlock(nn.Module):
    """

    Non-local Neural Networks

    References:
    -----------
    [1] Wang, Xiaolong, et al. "Non-local neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
    [2] https://github.com/AlexHex7/Non-local_pytorch
    """
    __DEBUG__ = False
    __name__ = "NonLocalBlock"
    __MID_LAYERS__ = ["g", "theta", "phi", "W"]

    def __init__(self, in_channels:int, mid_channels:Optional[int]=None, filter_lengths:Union[ED,int]=1, subsample_length:int=1, **config) -> NoReturn:
        """ finished, checked,

        Paramters:
        ----------
        in_channels: int,
            number of channels in the input
        mid_channels: int, optional,
            number of output channels for the mid layers ("g", "phi", "theta")
        filter_lengths: dict or int, default 1,
            filter lengths (kernel sizes) for each convolutional layers ("g", "phi", 'theta', "W")
        subsample_length: int, default 1,
            subsample length (max pool size) of the "g" and "phi" layers
        config: dict,
            other parameters, including
            batch normalization choices, etc.
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = (mid_channels or self.__in_channels//2) or 1
        self.__out_channels = self.__in_channels
        if isinstance(filter_lengths, dict):
            assert [k.lower() for k in filter_lengths.keys()] == self.__MID_LAYERS__
            self.__kernel_sizes = ED({k.lower():v for k,v in filter_lengths.items()})
        else:
            self.__kernel_sizes = ED({k:filter_lengths for k in self.__MID_LAYERS__})
        self.__subsample_length = subsample_length
        self.config = ED(deepcopy(config))

        self.mid_layers = nn.ModuleDict()
        for k in ["g", "theta", "phi"]:
            self.mid_layers[k] = nn.Sequential()
            self.mid_layers[k].add_module(
                "conv",
                Conv_Bn_Activation(
                    in_channels=self.__in_channels,
                    out_channels=self.__mid_channels,
                    kernel_size=self.__kernel_sizes[k],
                    stride=1,
                    batch_norm=False,
                    activation=None,
                )
            )
            if self.__subsample_length > 1 and k != "theta":
                self.mid_layers[k].add_module(
                    "max_pool",
                    nn.MaxPool1d(kernel_size=self.__subsample_length)
                )

        self.W = Conv_Bn_Activation(
            in_channels=self.__mid_channels,
            out_channels=self.__out_channels,
            kernel_size=self.__kernel_sizes["W"],
            stride=1,
            batch_norm=self.config.batch_norm,
            activation=None,
        )

    def forward(self, x:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        x: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        y: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        g_x = self.mid_layers["g"](x)  # --> batch_size, n_channels, seq_len'
        g_x = g_x.permute(0, 2, 1)  # --> batch_size, seq_len', n_channels

        theta_x = self.mid_layers["theta"](x)  # --> batch_size, n_channels, seq_len
        theta_x = theta_x.permute(0, 2, 1)  # --> batch_size, seq_len, n_channels
        phi_x = self.mid_layers["phi"](x)  # --> batch_size, n_channels, seq_len'
        # (batch_size, seq_len, n_channels) x (batch_size, n_channels, seq_len')
        f = torch.matmul(theta_x, phi_x)  # --> batch_size, seq_len, seq_len'
        f = F.softmax(f, dim=-1)  # --> batch_size, seq_len, seq_len'

        # (batch_size, seq_len, seq_len') x (batch_size, seq_len', n_channels)
        y = torch.matmul(f, g_x)  # --> (batch_size, seq_len, n_channels)
        y = y.permute(0, 2, 1).contiguous()  # --> (batch_size, n_channels, seq_len)
        y = self.W(y)
        y += x
        return y

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `NonLocalBlock` layer, given `seq_len` and `batch_size`
        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class SEBlock(nn.Module):
    """ finished, checked,

    Squeeze-and-Excitation Block

    References:
    -----------
    [1] J. Hu, L. Shen, S. Albanie, G. Sun and E. Wu, "Squeeze-and-Excitation Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 8, pp. 2011-2023, 1 Aug. 2020, doi: 10.1109/TPAMI.2019.2913372.
    [2] J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, 2018, pp. 7132-7141, doi: 10.1109/CVPR.2018.00745.
    [3] https://github.com/hujie-frank/SENet
    [4] https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    __DEBUG__ = True
    __name__ = "SEBlock"
    __DEFAULT_CONFIG__ = ED(
        bias=False, activation="relu", kw_activation={"inplace": True}, dropouts=0.0
    )

    def __init__(self, in_channels:int, reduction:int=16, **config) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        reduction: int, default 16,
            reduction ratio of mid-channels to `in_channels`
        config: dict,
            other parameters, including
            activation choices, weight initializer, dropouts, etc.
            for the linear layers
        """
        super().__init__()
        self.__in_channels = in_channels
        self.__mid_channels = in_channels // reduction
        self.__out_channels = in_channels
        self.config = ED(deepcopy(self.__DEFAULT_CONFIG__))
        self.config.update(config)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            SeqLin(
                in_channels=self.__in_channels,
                out_channels=[self.__mid_channels, self.__out_channels],
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                bias=self.config.bias,
                dropouts=self.config.dropouts,
                skip_last_activation=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters:
        -----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)
        
        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        batch_size, n_channels, seq_len = input.shape
        y = self.avg_pool(input).squeeze(-1)  # --> batch_size, n_channels
        y = self.fc(y).unsqueeze(-1)  # --> batch_size, n_channels, 1
        # output = input * y.expand_as(input)  # equiv. to the following
        # (batch_size, n_channels, seq_len) x (batch_size, n_channels, 1)
        output = input * y  # --> (batch_size, n_channels, seq_len)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `SEBlock` layer, given `seq_len` and `batch_size`
        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


class GlobalContextBlock(nn.Module):
    """ finished, checked,

    Global Context Block

    References:
    -----------
    [1] Cao, Yue, et al. "Gcnet: Non-local networks meet squeeze-excitation networks and beyond." Proceedings of the IEEE International Conference on Computer Vision Workshops. 2019.
    [2] https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
    [3] entry 0436 of CPSC2019

    NOTE that in refs. [1,2], `mid-channels` is raised from `in_channels` by a factor of `ratio`,
    while in [3], it is reduced from `in_channels` (divided) by `ratio`
    """
    __DEBUG__ = True
    __name__ = "GlobalContextBlock"
    __POOLING_TYPES__ = ["attn", "avg",]
    __FUSION_TYPES__ = ["add", "mul",]

    def __init__(self, in_channels:int, ratio:int, reduction:bool=False, pooling_type:str="attn", fusion_types:Sequence[str]=["add",]) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        in_channels: int,
            number of channels in the input
        ratio: int,
            raise or reduction ratio of the mid-channels to `in_channels`
            in the "channel attention" sub-block
        reduction: bool, default False,
            if True, mid-channels would be `in_channels // ratio` (as in `SEBlock`),
            otherwise, mid-channels would be `in_channels * ratio`,
        pooling_type: str, default "attn",
            mode (or type) of subsampling (or pooling) of "spatial attention"
        fusion_types: sequence of str, default ["add",],
            types of fusion of context with the input
        """
        super().__init__()
        assert pooling_type in self.__POOLING_TYPES__
        assert all([f in self.__FUSION_TYPES__ for f in fusion_types])
        assert len(fusion_types) > 0, "at least one fusion should be used"
        self.__in_channels = in_channels
        self.__ratio = ratio
        if reduction:
            self.__mid_channels = in_channels // ratio
        else:
            self.__mid_channels = in_channels * ratio
        self.__pooling_type = pooling_type.lower()
        self.__fusion_types = [item.lower() for item in fusion_types]

        if self.__pooling_type == "attn":
            self.conv_mask = nn.Conv1d(self.__in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if "add" in self.__fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(self.__in_channels, self.__mid_channels, kernel_size=1),
                nn.LayerNorm([self.__mid_channels, 1]),
                Activations["relu"](inplace=True),
                nn.Conv1d(self.__mid_channels, self.__in_channels, kernel_size=1),
            )
        else:
            self.channel_add_conv = None
        if "mul" in self.__fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv1d(self.__in_channels, self.__mid_channels, kernel_size=1),
                nn.LayerNorm([self.__mid_channels, 1]),
                Activations["relu"](inplace=True),
                nn.Conv1d(self.__mid_channels, self.__in_channels, kernel_size=1),
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x:Tensor) -> Tensor:
        """ finished, checked,

        Paramters:
        ----------
        x: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        context: Tensor,
            of shape (batch_size, n_channels, 1)
        """
        if self.__pooling_type == "attn":
            input_x = x.unsqueeze(1)  # --> (batch_size, 1, n_channels, seq_len)
            context = self.conv_mask(x)  # --> (batch_size, 1, seq_len)
            context = self.softmax(context)  # --> (batch_size, 1, seq_len)
            context = context.unsqueeze(3)  # --> (batch_size, 1, seq_len, 1)
            # matmul: (batch_size, 1, n_channels, seq_len) x (batch_size, 1, seq_len, 1)
            context = torch.matmul(input_x, context)  # --> (batch_size, 1, n_channels, 1)
            context = context.squeeze(1)  # --> (batch_size, n_channels, 1)
        elif self.__pooling_type == "avg":
            context = self.avg_pool(x)  # --> (batch_size, n_channels, 1)
        return context

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,
        
        Paramters:
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns:
        --------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        context = self.spatial_pool(input)  # --> (batch_size, n_channels, 1)
        output = input
        if self.channel_mul_conv is not None:
            channel_mul_term = self.channel_mul_conv(context)  # --> (batch_size, n_channels, 1)
            channel_mul_term = torch.sigmoid(channel_mul_term)  # --> (batch_size, n_channels, 1)
            # (batch_size, n_channels, seq_len) x (batch_size, n_channels, 1)
            output = output * channel_mul_term  # --> (batch_size, n_channels, seq_len)
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)  # --> (batch_size, n_channels, 1)
            output = output + channel_add_term  # --> (batch_size, n_channels, seq_len)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, type(None)]]:
        """ finished, checked,

        Parameters:
        -----------
        seq_len: int, optional,
            length of the 1d sequence,
            if is None, then the input is composed of single feature vectors for each batch
        batch_size: int, optional,
            the batch size, can be None

        Returns:
        --------
        output_shape: sequence,
            the output shape of this `GlobalContextBlock` layer, given `seq_len` and `batch_size`
        """
        return (batch_size, self.__in_channels, seq_len)

    @property
    def module_size(self) -> int:
        """
        """
        module_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in module_parameters])
        return n_params


# custom losses
def weighted_binary_cross_entropy(sigmoid_x:Tensor, targets:Tensor, pos_weight:Tensor, weight:Optional[Tensor]=None, size_average:bool=True, reduce:bool=True) -> Tensor:
    """ finished, checked,

    Parameters:
    -----------
    sigmoid_x: Tensor,
        predicted probability of size [N,C], N sample and C Class.
        Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
    targets: Tensor,
        true value, one-hot-like vector of size [N,C]
    pos_weight: Tensor,
        Weight for postive sample
    weight: Tensor, optional,
    size_average: bool, default True,
    reduce: bool, default True,

    Reference (original source):
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss(nn.Module):
    """ finished, checked,

    Reference (original source):
    https://github.com/pytorch/pytorch/issues/5660#issuecomment-403770305
    """
    __name__ = "WeightedBCELoss"

    def __init__(self, pos_weight:Tensor=1, weight:Optional[Tensor]=None, PosWeightIsDynamic:bool=False, WeightIsDynamic:bool=False, size_average:bool=True, reduce:bool=True) -> NoReturn:
        """ Not checked,

        Parameters:
        -----------
        pos_weight: Tensor, default 1,
            Weight for postive samples. Size [1,C]
        weight: Tensor, optional,
            Weight for Each class. Size [1,C]
        PosWeightIsDynamic: bool, default False,
            If True, the pos_weight is computed on each batch.
            If `pos_weight` is None, then it remains None.
        WeightIsDynamic: bool, default False,
            If True, the weight is computed on each batch.
            If `weight` is None, then it remains None.
        size_average: bool, default True,
        reduce: bool, default True,
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """
        """
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts) / (positive_counts + 1e-5)

        return weighted_binary_cross_entropy(input, target,
                                             pos_weight=self.pos_weight,
                                             weight=self.weight,
                                             size_average=self.size_average,
                                             reduce=self.reduce)


class BCEWithLogitsWithClassWeightLoss(nn.BCEWithLogitsLoss):
    """ finished, checked,
    """
    __name__ = "BCEWithLogitsWithClassWeightsLoss"

    def __init__(self, class_weight:Tensor) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        class_weight: Tensor,
            class weight, of shape (1, n_classes)
        """
        super().__init__(reduction='none')
        self.class_weight = class_weight

    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """
        """
        loss = super().forward(input, target)
        loss = torch.mean(loss * self.class_weight)
        return loss


def default_collate_fn(batch:Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Tensor, Tensor]:
    """ finished, checked,

    collate functions for model training

    the data generator (`Dataset`) should generate (`__getitem__`) 2-tuples `signals, labels`

    Parameters:
    -----------
    batch: sequence,
        sequence of 2-tuples,
        in which the first element is the signal, the second is the label
    
    Returns:
    --------
    values: Tensor,
        the concatenated values as input for training
    labels: Tensor,
        the concatenated labels as ground truth for training
    """
    values = [[item[0]] for item in batch]
    labels = [[item[1]] for item in batch]
    values = np.concatenate(values, axis=0).astype(_DTYPE)
    values = torch.from_numpy(values)
    labels = np.concatenate(labels, axis=0).astype(_DTYPE)
    labels = torch.from_numpy(labels)
    return values, labels
