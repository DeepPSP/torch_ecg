"""
"""

from pathlib import Path

from torch_ecg.databases import CPSC2018


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"
###############################################################################


reader = CPSC2018(_CWD)
