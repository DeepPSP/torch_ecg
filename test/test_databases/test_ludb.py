"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import LUDB


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "ludb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = LUDB(_CWD)
reader.download()
