"""
"""

# import torch

try:
    import torch_ecg  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

# from torch_ecg._preprocessors import (
#     BandPass,
#     BaselineRemove,
#     MinMaxNormalize,
#     NaiveNormalize,
#     Normalize,
#     Resample,
#     ZScoreNormalize,
# )
