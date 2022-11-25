"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import LTAFDB


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "ltafdb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = LTAFDB(_CWD)
