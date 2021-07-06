"""
test of the classes from models._nets.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from torch_ecg.models._nets import (
    Mish, Swish, Hardswish,
    Initializers, Activations,
    Bn_Activation, Conv_Bn_Activation, CBA,
    MultiConv, BranchedConv,
    SeparableConv,
    DownSample,
    BidirectionalLSTM, StackedLSTM,
    AttentionWithContext,
    MultiHeadAttention, SelfAttention,
    AttentivePooling,
    ZeroPadding,
    SeqLin, MLP,
    NonLocalBlock, SEBlock, GlobalContextBlock,
    CRF, ExtendedCRF,
    WeightedBCELoss, BCEWithLogitsWithClassWeightLoss,
)


mish = Mish()
swish = Swish()
hard_swish = Hardswish()

cba = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12*4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
)
cba_alpha = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12*4,
    kernel_size=5,
    stride=1,
    activation="hardswish",
    groups=12,
    width_multiplier=1.5,
    depth_multiplier=2,
    conv_type="separable",
)
cab = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12*4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
    ordering="cab",
)
bac = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12*4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
    ordering="bac",
)

# TODO: add more test of different modules


if __name__ == "__main__":
    test_input = torch.rand((1,12,5000))

    out = cba(test_input)
    print(f"out shape of cba = {out.shape}")
    out = cba_alpha(test_input)
    print(f"out shape of cba_alpha = {out.shape}")
    out = cab(test_input)
    print(f"out shape of cab = {out.shape}")
    out = bac(test_input)
    print(f"out shape of bac = {out.shape}")
