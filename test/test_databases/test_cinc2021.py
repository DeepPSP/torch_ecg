"""
"""

from pathlib import Path

from torch_ecg.databases import CINC2021


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cinc2021"
###############################################################################


reader = CINC2021(_CWD)
