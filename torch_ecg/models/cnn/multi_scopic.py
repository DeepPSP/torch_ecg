"""
The core part of the SOTA model of CPSC2019,
branched, and has different scope (in terms of dilation) in each branch
"""

from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from typing import NoReturn, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn

from ...cfg import CFG, DEFAULTS
from ...models._nets import (  # noqa: F401
    Conv_Bn_Activation,
    DownSample,
)
from ...utils.misc import dict_to_str, list_sum, add_docstring
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)

if DEFAULTS.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "MultiScopicCNN",
    "MultiScopicBasicBlock",
    "MultiScopicBranch",
]


class MultiScopicBasicBlock(nn.Sequential, SizeMixin):
    """

    basic building block of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)

    (conv -> activation) * N --> bn --> down_sample

    """

    __DEBUG__ = False
    __name__ = "MultiScopicBasicBlock"

    def __init__(
        self,
        in_channels: int,
        scopes: Sequence[int],
        num_filters: Union[int, Sequence[int]],
        filter_lengths: Union[int, Sequence[int]],
        subsample_length: int,
        groups: int = 1,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        scopes: sequence of int,
            scopes of the convolutional layers, via `dilation`
        num_filters: int or sequence of int,
            number of filters of the convolutional layer(s)
        filter_lengths: int or sequence of int,
            filter length(s) (kernel size(s)) of the convolutional layer(s)
        subsample_length: int,
            subsample length (ratio) at the last layer of the block

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_convs = len(self.__scopes)
        if isinstance(num_filters, int):
            self.__out_channels = list(repeat(num_filters, self.__num_convs))
        else:
            self.__out_channels = num_filters
            assert len(self.__out_channels) == self.__num_convs, (
                f"`scopes` indicates {self.__num_convs} convolutional layers, "
                f"while `num_filters` indicates {len(self.__out_channels)}"
            )
        if isinstance(filter_lengths, int):
            self.__filter_lengths = list(repeat(filter_lengths, self.__num_convs))
        else:
            self.__filter_lengths = filter_lengths
            assert len(self.__filter_lengths) == self.__num_convs, (
                f"`scopes` indicates {self.__num_convs} convolutional layers, "
                f"while `filter_lengths` indicates {len(self.__filter_lengths)}"
            )
        self.__subsample_length = subsample_length
        self.__groups = groups
        self.config = CFG(deepcopy(config))

        conv_in_channels = self.__in_channels
        for idx in range(self.__num_convs):
            self.add_module(
                f"ca_{idx}",
                Conv_Bn_Activation(
                    in_channels=conv_in_channels,
                    out_channels=self.__out_channels[idx],
                    kernel_size=self.__filter_lengths[idx],
                    stride=1,
                    dilation=self.__scopes[idx],
                    groups=self.__groups,
                    norm=self.config.get("norm", self.config.get("batch_norm")),
                    # kw_bn=self.config.kw_bn,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    bias=self.config.bias,
                ),
            )
            conv_in_channels = self.__out_channels[idx]
        self.add_module("bn", nn.BatchNorm1d(self.__out_channels[-1]))
        self.add_module(
            "down",
            DownSample(
                down_scale=self.__subsample_length,
                in_channels=self.__out_channels[-1],
                groups=self.__groups,
                # padding=
                norm=False,
                mode=self.config.subsample_mode,
            ),
        )
        if self.config.dropout > 0:
            self.add_module("dropout", nn.Dropout(self.config.dropout, inplace=False))

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

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
        _seq_len = seq_len
        for idx, module in enumerate(self):
            if idx == self.__num_convs:  # bn layer
                continue
            elif self.config.dropout > 0 and idx == len(self) - 1:  # dropout layer
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    def _assign_weights_lead_wise(
        self, other: "MultiScopicBasicBlock", indices: Sequence[int]
    ) -> NoReturn:
        """ """
        assert not any([isinstance(m, nn.LayerNorm) for m in self]) and not any(
            [isinstance(m, nn.LayerNorm) for m in other]
        ), "Lead-wise assignment of weights is not supported for the existence of `LayerNorm` layers"
        for blk, o_blk in zip(self, other):
            if isinstance(blk, Conv_Bn_Activation):
                blk._assign_weights_lead_wise(o_blk, indices)
            elif isinstance(blk, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
                if hasattr(blk, "num_features"):  # batch norm and instance norm
                    out_channels = blk.num_features
                else:  # group norm
                    out_channels = blk.num_channels
                units = out_channels // self.groups
                out_indices = list_sum(
                    [[i * units + j for j in range(units)] for i in indices]
                )
                o_blk.weight.data = blk.weight.data[out_indices].clone()
                if blk.bias is not None:
                    o_blk.bias.data = blk.bias.data[out_indices].clone()

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def groups(self) -> int:
        return self.__groups


class MultiScopicBranch(nn.Sequential, SizeMixin):
    """

    branch path of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416)

    """

    __DEBUG__ = False
    __name__ = "MultiScopicBranch"

    def __init__(
        self,
        in_channels: int,
        scopes: Sequence[Sequence[int]],
        num_filters: Union[Sequence[int], Sequence[Sequence[int]]],
        filter_lengths: Union[Sequence[int], Sequence[Sequence[int]]],
        subsample_lengths: Union[int, Sequence[int]],
        groups: int = 1,
        **config,
    ) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of features (channels) of the input
        scopes: sequence of sequences of int,
            scopes (in terms of `dilation`) for the convolutional layers,
            each sequence of int is for one branch
        num_filters: sequence of int, or sequence of sequences of int,
            number of filters for the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same number of filters
        filter_lengths: sequence of int, or sequence of sequences of int,
            filter length (kernel size) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same filter length
        subsample_lengths: int, or sequence of int,
            subsample length (stride) of the convolutional layers,
            if is sequence of int,
            then convolutionaly layers in one branch will have the same subsample length
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other hyper-parameters, including
            dropout, activation choices, weight initializer, etc.

        """
        super().__init__()
        self.__in_channels = in_channels
        self.__scopes = scopes
        self.__num_blocks = len(self.__scopes)
        self.__num_filters = num_filters
        assert len(self.__num_filters) == self.__num_blocks, (
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, "
            f"while `num_filters` indicates {len(self.__num_filters)}"
        )
        self.__filter_lengths = filter_lengths
        assert len(self.__filter_lengths) == self.__num_blocks, (
            f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, "
            f"while `filter_lengths` indicates {len(self.__filter_lengths)}"
        )
        if isinstance(subsample_lengths, int):
            self.__subsample_lengths = list(
                repeat(subsample_lengths, self.__num_blocks)
            )
        else:
            self.__subsample_lengths = filter_lengths
            assert len(self.__subsample_lengths) == self.__num_blocks, (
                f"`scopes` indicates {self.__num_blocks} `MultiScopicBasicBlock`s, "
                f"while `subsample_lengths` indicates {len(self.__subsample_lengths)}"
            )
        self.__groups = groups
        self.config = CFG(deepcopy(config))

        block_in_channels = self.__in_channels
        for idx in range(self.__num_blocks):
            self.add_module(
                f"block_{idx}",
                MultiScopicBasicBlock(
                    in_channels=block_in_channels,
                    scopes=self.__scopes[idx],
                    num_filters=self.__num_filters[idx],
                    filter_lengths=self.__filter_lengths[idx],
                    subsample_length=self.__subsample_lengths[idx],
                    groups=self.__groups,
                    dropout=self.config.dropouts[idx],
                    **(self.config.block),
                ),
            )
            block_in_channels = self.__num_filters[idx]

    @add_docstring(compute_sequential_output_shape_docstring)
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """ """
        return compute_sequential_output_shape(self, seq_len, batch_size)

    def _assign_weights_lead_wise(
        self, other: "MultiScopicBranch", indices: Sequence[int]
    ) -> NoReturn:
        """ """
        for blk, o_blk in zip(self, other):
            blk._assign_weights_lead_wise(o_blk, indices)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class MultiScopicCNN(nn.Module, SizeMixin):
    """CNN part of the SOTA model from CPSC2019 challenge (entry 0416)"""

    __DEBUG__ = False
    __name__ = "MultiScopicCNN"

    def __init__(self, in_channels: int, **config) -> NoReturn:
        """

        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, ref. corresponding config file
            key word arguments that have to be set:
            scopes: sequence of sequences of sequences of int,
                scopes (in terms of dilation) of each convolution
            num_filters: sequence of sequences (of int or of sequences of int),
                number of filters of the convolutional layers,
                with granularity to each block of each branch,
                or to each convolution of each block of each branch
            filter_lengths: sequence of sequences (of int or of sequences of int),
                filter length(s) (kernel size(s)) of the convolutions,
                with granularity to each block of each branch,
                or to each convolution of each block of each branch
            subsample_lengths: sequence of int or sequence of sequences of int,
                subsampling length(s) (ratio(s)) of all blocks,
                with granularity to each branch or to each block of each branch,
                each subsamples after the last convolution of each block
            dropouts: sequence of int or sequence of sequences of int,
                dropout rates of all blocks,
                with granularity to each branch or to each block of each branch,
                each dropouts at the last of each block
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for the building blocks
            for a full list of configurable parameters, ref. corr. config file

        """
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        self.__scopes = self.config.scopes
        self.__num_branches = len(self.__scopes)

        if self.__DEBUG__:
            print(
                f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}"
            )

        self.branches = nn.ModuleDict()
        for idx in range(self.__num_branches):
            self.branches[f"branch_{idx}"] = MultiScopicBranch(
                in_channels=self.__in_channels,
                scopes=self.__scopes[idx],
                num_filters=self.config.num_filters[idx],
                filter_lengths=self.config.filter_lengths[idx],
                subsample_lengths=self.config.subsample_lengths[idx],
                groups=self.config.groups,
                dropouts=self.config.dropouts[idx],
                block=self.config.block,  # a dict
            )

    def forward(self, input: Tensor) -> Tensor:
        """

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)

        """
        branch_out = OrderedDict()
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            branch_out[key] = self.branches[key].forward(input)
        # NOTE: ideally, for lead-wise manner, output of each branch should be
        # split into `self.config.groups` groups,
        # channels from the same group from different branches should be first concatenated,
        # and then concatenate the concatenated groups.
        output = torch.cat(
            [branch_out[f"branch_{idx}"] for idx in range(self.__num_branches)],
            dim=1,  # along channels
        )
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """

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
        out_channels = 0
        for idx in range(self.__num_branches):
            key = f"branch_{idx}"
            _, _branch_oc, _seq_len = self.branches[key].compute_output_shape(
                seq_len, batch_size
            )
            out_channels += _branch_oc
        output_shape = (batch_size, out_channels, _seq_len)
        return output_shape

    def assign_weights_lead_wise(
        self, other: "MultiScopicCNN", indices: Sequence[int]
    ) -> NoReturn:
        """

        Assign weights to the `other` MultiScopicCNN module in the lead-wise manner

        Parameters
        ----------
        other: MultiScopicCNN,
            the other MultiScopicCNN module
        indices: sequence of int,
            indices of the MultiScopicCNN modules to be assigned weights

        Examples
        --------
        >>> from copy import deepcopy
        >>> import torch
        >>> from torch_ecg.models import ECG_CRNN
        >>> from torch_ecg.model_configs import ECG_CRNN_CONFIG
        >>> from torch_ecg.utils.misc import list_sum
        >>> # we create models using 12-lead ECGs and using reduced 6-lead ECGs
        >>> indices = [0, 1, 2, 3, 4, 10]  # chosen randomly, no special meaning
        >>> # chose the lead-wise models
        >>> lead_12_config = deepcopy(ECG_CRNN_CONFIG)
        >>> lead_12_config.cnn.name = "multi_scopic_leadwise"
        >>> lead_6_config = deepcopy(ECG_CRNN_CONFIG)
        >>> lead_6_config.cnn.name = "multi_scopic_leadwise"
        >>> # adjust groups and numbers of filters
        >>> lead_6_config.cnn.multi_scopic_leadwise.groups = 6
        >>> # numbers of filters should be proportional to numbers of groups
        >>> lead_6_config.cnn.multi_scopic_leadwise.num_filters = (
                np.array([[192, 384, 768], [192, 384, 768], [192, 384, 768]]) / 2
            ).astype(int).tolist()
        >>> # we assume that model12 is a trained model on 12-lead ECGs
        >>> model12 = ECG_CRNN(["AF", "PVC", "NSR"], 12, lead_12_config)
        >>> model6 = ECG_CRNN(["AF", "PVC", "NSR"], 6, lead_6_config)
        >>> model12.eval()
        >>> model6.eval()
        >>> # we create tensor12, tensor6 to check the correctness of the assignment of the weights
        >>> tensor12 = torch.zeros(1, 12, 200)  # batch, leads, seq_len
        >>> tensor6 = torch.rand(1, 6, 200)
        >>> # we make tensor12 has identical values as tensor6 at the given leads, and let the other leads have zero values
        >>> tensor12[:, indices, :] = tensor6
        >>> b = "branch_0"  # similarly for other branches
        >>> _, output_shape_12, _ = model12.cnn.branches[b].compute_output_shape()
        >>> _, output_shape_6, _ = model6.cnn.branches[b].compute_output_shape()
        >>> units = output_shape_12 // 12
        >>> out_indices = list_sum([[i * units + j for j in range(units)] for i in indices])
        >>> (model6.cnn.branches[b](tensor6) == model12.cnn.branches[b](tensor12)[:, out_indices, :]).all()
        tensor(False)  # different feature maps
        >>> # here, we assign weights from model12 that correspond to the given leads to model6
        >>> model12.cnn.assign_weights_lead_wise(model6.cnn, indices)
        >>> (model6.cnn.branches[b](tensor6) == model12.cnn.branches[b](tensor12)[:, out_indices, :]).all()
        tensor(True)  # identical feature maps

        """
        assert (
            self.in_channels == self.config.groups
        ), "the current model is not lead-wise"
        assert (
            other.in_channels == other.config.groups
        ), "the other model is not lead-wise"
        assert other.in_channels == len(
            indices
        ), "the length of indices should match the number of channels of the other model"
        assert (
            np.array(self.config.num_filters) * other.in_channels
            == np.array(other.config.num_filters) * self.in_channels
        ).all(), "the number of filters of the two models should be proportional to the number of channels of the other model"
        for idx in range(self.num_branches):
            for b, ob in zip(
                self.branches[f"branch_{idx}"], other.branches[f"branch_{idx}"]
            ):
                b._assign_weights_lead_wise(ob, indices)

    @property
    def in_channels(self) -> int:
        return self.__in_channels

    @property
    def num_branches(self) -> int:
        return self.__num_branches
