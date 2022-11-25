"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import CPSC2019


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "cpsc2019"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = CPSC2019(_CWD)
reader.download()


class TestCPSC2019:
    def test_len(self):
        assert len(reader) == 2000

    def test_load_data(self):
        pass

    def test_load_ann(self):
        pass
