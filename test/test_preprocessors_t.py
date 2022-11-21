"""
"""

import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.cfg import CFG
from torch_ecg.preprocessors import (  # noqa: F401
    PreprocManager,
    BandPass,
    BaselineRemove,
    Normalize,
    MinMaxNormalize,
    NaiveNormalize,
    ZScoreNormalize,
    Resample,
)  # noqa: F401


class DummyPreProcessor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return sig


def test_preproc_manager() -> None:
    ppm = PreprocManager(random=False)
    assert not ppm.random
    assert ppm.empty
    ppm.add_(DummyPreProcessor())
    assert not ppm.empty
    ppm.add_(BandPass(fs=500))
    ppm.add_(BaselineRemove(fs=500))
    ppm.add_(Normalize(method="min-max"))
    ppm.add_(Resample(fs=300, dst_fs=500), pos=0)

    sig = torch.rand(2, 12, 8000)
    sig = ppm(sig)

    del ppm, sig

    config = CFG(
        random=True,
        bandpass={"fs": 500},
        normalize={"method": "min-max"},
    )
    ppm = PreprocManager.from_config(config)
    assert ppm.random

    sig = torch.rand(2, 12, 8000)
    sig = ppm(sig)
