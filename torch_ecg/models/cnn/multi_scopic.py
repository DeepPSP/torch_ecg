"""
The core part of the SOTA model of CPSC2019,
branched, and has different scope (in terms of dilation) in each branch.
"""

import textwrap
from collections import OrderedDict
from copy import deepcopy
from itertools import repeat
from typing import Optional, Sequence, Union, List

import numpy as np
import torch
from torch import Tensor, nn

from ...cfg import CFG
from ...models._nets import Conv_Bn_Activation, DownSample
from ...utils.misc import list_sum, add_docstring, CitationMixin
from ...utils.utils_nn import (
    SizeMixin,
    compute_sequential_output_shape,
    compute_sequential_output_shape_docstring,
)


__all__ = [
    "MultiScopicCNN",
    "MultiScopicBasicBlock",
    "MultiScopicBranch",
]


if not hasattr(nn, "Dropout1d"):
    nn.Dropout1d = nn.Dropout  # added in pytorch 1.12


class MultiScopicBasicBlock(nn.Sequential, SizeMixin):
    """Basic building block of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416).

    (conv -> activation) * N --> bn --> down_sample

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor.
    scopes : Sequence[int]
        Scopes of the convolutional layers, via `dilation`.
    num_filters : int or Sequence[int]
        Number of filters of the convolutional layer(s).
    filter_lengths : int or Sequence[int]
        Filter length(s) (kernel size(s)) of the convolutional layer(s).
    subsample_length : int
        Subsample length (ratio) at the last layer of the block.

    """

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
    ) -> None:
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
        if isinstance(self.config.dropout, dict):
            if self.config.dropout.type == "1d" and self.config.dropout.p > 0:
                self.add_module(
                    "dropout",
                    nn.Dropout1d(
                        p=self.config.dropout.p,
                        inplace=self.config.dropout.get("inplace", False),
                    ),
                )
            elif self.config.dropout.type is None and self.config.dropout.p > 0:
                self.add_module(
                    "dropout",
                    nn.Dropout(
                        p=self.config.dropout.p,
                        inplace=self.config.dropout.get("inplace", False),
                    ),
                )
        elif self.config.dropout > 0:
            self.add_module("dropout", nn.Dropout(self.config.dropout, inplace=False))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        """
        output = super().forward(input)
        return output

    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the block.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            The batch size of the input tensor.

        Returns
        -------
        output_shape : sequence
            Output shape of the block.

        """
        _seq_len = seq_len
        for _, module in enumerate(self):
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.BatchNorm1d)):
                continue
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape

    def _assign_weights_lead_wise(
        self, other: "MultiScopicBasicBlock", indices: Sequence[int]
    ) -> None:
        """Assign weights lead-wise.

        This method is used to assign weights from a model with
        a superset of the current model's leads to the current model.

        Parameters
        ----------
        other : MultiScopicBasicBlock
            The model with a superset of the current model's leads.
        indices : Sequence[int]
            The indices of the leads of the current model in the
            superset model.

        Returns
        -------
        None

        """
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
    """Branch path of the CNN part of the SOTA model
    from CPSC2019 challenge (entry 0416).

    Parameters
    ----------
    in_channels : int
        Number of features (channels) of the input tensor.
    scopes : Sequence[Sequence[int]]
        Scopes (in terms of `dilation`) for the convolutional layers,
        each sequence of int is for one branch.
    num_filters : Sequence[int] or Sequence[Sequence[int]]
        Number of filters for the convolutional layers.
        If is sequence of int, then convolutionaly layers
        in one branch will have the same number of filters.
    filter_lengths : Sequence[int] or Sequence[Sequence[int]]
        Filter length (kernel size) of the convolutional layers.
        If is sequence of int, then convolutionaly layers
        in one branch will have the same filter length.
    subsample_lengths : int or Sequence[int]
        Subsample length (stride) of the convolutional layers.
        If is sequence of int, then convolutionaly layers
        in one branch will have the same subsample length.
    groups : int, default 1
        Connection pattern (of channels) of the inputs and outputs.
    config : dict
        Other hyper-parameters, including
        dropout, activation choices, weight initializer, etc.

    """

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
    ) -> None:
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

    @add_docstring(
        textwrap.indent(compute_sequential_output_shape_docstring, " " * 4),
        mode="append",
    )
    def compute_output_shape(
        self, seq_len: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Sequence[Union[int, None]]:
        """Compute the output shape of the branch."""
        return compute_sequential_output_shape(self, seq_len, batch_size)

    def _assign_weights_lead_wise(
        self, other: "MultiScopicBranch", indices: Sequence[int]
    ) -> None:
        """Assign weights lead-wise.

        This method is used to assign weights from a model with
        a superset of the current model's leads to the current model.

        Parameters
        ----------
        other : MultiScopicBranch
            The model with a superset of the current model's leads.
        indices : Sequence[int]
            The indices of the leads of the current model in the
            superset model.

        Returns
        -------
        None

        """
        for blk, o_blk in zip(self, other):
            blk._assign_weights_lead_wise(o_blk, indices)

    @property
    def in_channels(self) -> int:
        return self.__in_channels


class MultiScopicCNN(nn.Module, SizeMixin, CitationMixin):
    """CNN part of the SOTA model from CPSC2019 challenge (entry 0416).

    This architecture is a multi-branch CNN with multi-scopic convolutions,
    proposed by the winning team of the CPSC2019 challenge and described in
    :cite:p:`cai2020rpeak_seq_lab_net`. The multi-scopic convolutions are
    implemented via different dilations. Similar architectures can be found
    in the model DeepLabv3 :cite:p:`chen2017_deeplabv3`.

    Parameters
    ----------
    in_channels : int
        Number of channels (leads) in the input signal tensor.
    config : dict
        Other hyper-parameters of the Module, ref. corr. config file.
        Key word arguments that must be set:

            - scopes: sequence of sequences of sequences of :obj:`int`,
              scopes (in terms of dilation) of each convolution.
            - num_filters: sequence of sequences (of :obj:`int` or of sequences of :obj:`int`),
              number of filters of the convolutional layers,
              with granularity to each block of each branch,
              or to each convolution of each block of each branch.
            - filter_lengths: sequence of sequences (of :obj:`int` or of sequences of :obj:`int`),
              filter length(s) (kernel size(s)) of the convolutions,
              with granularity to each block of each branch,
              or to each convolution of each block of each branch.
            - subsample_lengths: sequence of :obj:`int` or sequence of sequences of :obj:`int`,
              subsampling length(s) (ratio(s)) of all blocks,
              with granularity to each branch or to each block of each branch,
              each subsamples after the last convolution of each block.
            - dropouts: sequence of :obj:`int` or sequence of sequences of :obj:`int`,
              dropout rates of all blocks,
              with granularity to each branch or to each block of each branch,
              each dropouts at the last of each block.
            - groups: :obj:`int`,
              connection pattern (of channels) of the inputs and outputs.
            - block: :obj:`dict`,
              other parameters that can be set for the building blocks.

        For a full list of configurable parameters, ref. corr. config file.

    .. bibliography::
        :filter: docname in docnames

    """

    __name__ = "MultiScopicCNN"

    def __init__(self, in_channels: int, **config) -> None:
        super().__init__()
        self.__in_channels = in_channels
        self.config = CFG(deepcopy(config))
        self.__scopes = self.config.scopes
        self.__num_branches = len(self.__scopes)

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
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

        Returns
        -------
        output : torch.Tensor
            Output tensor,
            of shape ``(batch_size, n_channels, seq_len)``.

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
        """Compute the output shape of the Module.

        Parameters
        ----------
        seq_len : int, optional
            Length of the input tensor.
        batch_size : int, optional
            The batch size of the input tensor.

        Returns
        -------
        output_shape : sequence
            The output shape of the Module.

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
    ) -> None:
        """Assign weights to the `other` :class:`MultiScopicCNN`
        module in the lead-wise manner

        Parameters
        ----------
        other : MultiScopicCNN
            The other :class:`MultiScopicCNN` module.
        indices : Sequence[int]
            Indices of the :class:`MultiScopicCNN` modules to assign weights.

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
        >>> tensor6 = torch.randn(1, 6, 200)
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

    @property
    def doi(self) -> List[str]:
        return list(set(self.config.get("doi", []) + ["10.1109/access.2020.2997473"]))
