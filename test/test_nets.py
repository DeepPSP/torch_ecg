"""
test of the classes from models._nets.py
"""

import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.models._nets import CBA  # noqa: F401; noqa: F401
from torch_ecg.models._nets import CRF  # noqa: F401
from torch_ecg.models._nets import MLP  # noqa: F401
from torch_ecg.models._nets import Activations  # noqa: F401
from torch_ecg.models._nets import AttentionWithContext  # noqa: F401
from torch_ecg.models._nets import AttentivePooling  # noqa: F401
from torch_ecg.models._nets import BidirectionalLSTM  # noqa: F401
from torch_ecg.models._nets import Bn_Activation  # noqa: F401
from torch_ecg.models._nets import BranchedConv  # noqa: F401
from torch_ecg.models._nets import Conv_Bn_Activation  # noqa: F401
from torch_ecg.models._nets import DownSample  # noqa: F401
from torch_ecg.models._nets import ExtendedCRF  # noqa: F401
from torch_ecg.models._nets import GlobalContextBlock  # noqa: F401
from torch_ecg.models._nets import Hardswish  # noqa: F401
from torch_ecg.models._nets import Initializers  # noqa: F401
from torch_ecg.models._nets import Mish  # noqa: F401
from torch_ecg.models._nets import MultiConv  # noqa: F401
from torch_ecg.models._nets import MultiHeadAttention  # noqa: F401
from torch_ecg.models._nets import NonLocalBlock  # noqa: F401
from torch_ecg.models._nets import SEBlock  # noqa: F401
from torch_ecg.models._nets import SelfAttention  # noqa: F401
from torch_ecg.models._nets import SeparableConv  # noqa: F401
from torch_ecg.models._nets import SeqLin  # noqa: F401
from torch_ecg.models._nets import StackedLSTM  # noqa: F401
from torch_ecg.models._nets import Swish  # noqa: F401
from torch_ecg.models._nets import ZeroPadding  # noqa: F401

mish = Mish()
swish = Swish()
hard_swish = Hardswish()

cba = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12 * 4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
)
cba_alpha = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12 * 4,
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
    out_channels=12 * 4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
    ordering="cab",
)
bac = Conv_Bn_Activation(
    in_channels=12,
    out_channels=12 * 4,
    kernel_size=5,
    stride=1,
    activation="relu6",
    groups=12,
    ordering="bac",
)

# TODO: add more test of different modules


if __name__ == "__main__":
    test_input = torch.rand((1, 12, 5000))

    out = cba(test_input)
    print(f"out shape of cba = {out.shape}")
    out = cba_alpha(test_input)
    print(f"out shape of cba_alpha = {out.shape}")
    out = cab(test_input)
    print(f"out shape of cab = {out.shape}")
    out = bac(test_input)
    print(f"out shape of bac = {out.shape}")
