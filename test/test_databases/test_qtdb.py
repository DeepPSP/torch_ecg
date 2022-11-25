"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import QTDB


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "qtdb"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = QTDB(_CWD)
reader.download()
