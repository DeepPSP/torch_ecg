"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import CPSC2020


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cpsc2020"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = CPSC2020(_CWD)
