"""
"""

from numbers import Real
from typing import Tuple

import numpy as np
import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from torch_ecg.cfg import CFG
from torch_ecg._preprocessors import (  # noqa: F401
    PreProcessor,
    PreprocManager,
    BandPass,
    BaselineRemove,
    Normalize,
    MinMaxNormalize,
    NaiveNormalize,
    ZScoreNormalize,
    Resample,
)  # noqa: F401


class DummyPreProcessor(PreProcessor):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, sig: np.ndarray, fs: Real) -> Tuple[np.ndarray, int]:
        return sig, fs


def test_preproc_manager():
    ppm = PreprocManager()
    assert ppm.empty
    ppm.add_(BandPass(0.5, 40))
    assert not ppm.empty
    ppm.add_(Resample(100), pos=0)
    ppm.add_(ZScoreNormalize())
    ppm.add_(BaselineRemove(), pos=1)
    assert len(ppm.preprocessors) == 4
    del ppm.preprocessors[-1]
    assert len(ppm.preprocessors) == 3
    ppm.add_(NaiveNormalize())
    ppm.add_(DummyPreProcessor(), pos=0)

    sig = torch.rand(12, 80000).numpy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs

    config = CFG(
        random=False,
        resample={"fs": 500},
        bandpass={"filter_type": "fir"},
        normalize={"method": "min-max"},
    )
    ppm = PreprocManager.from_config(config)

    sig = torch.rand(12, 80000).numpy()
    sig, fs = ppm(sig, 200)

    del ppm, sig, fs
