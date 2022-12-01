"""
TestSHHS: NOT accomplished

subsampling: NOT tested
"""

import shutil
from pathlib import Path

from torch_ecg.databases import SHHS, DataBaseInfo


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "shhs"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = SHHS(_CWD)


class TestSHHS:
    def test_len(self):
        pass

    def test_load_data(self):
        pass

    def test_load_ann(self):
        pass

    def test_meta_data(self):
        # TODO: add more....
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        pass
