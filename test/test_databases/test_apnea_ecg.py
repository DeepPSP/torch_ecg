"""
"""

import shutil
from pathlib import Path

from torch_ecg.databases import ApneaECG


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "tmp" / "test-db" / "apnea-ecg"
try:
    shutil.rmtree(_CWD)
except FileNotFoundError:
    pass
_CWD.mkdir(parents=True, exist_ok=True)
###############################################################################


reader = ApneaECG(_CWD)
reader.download()


class TestApneaECG:
    def test_len(self):
        assert len(reader) == 200

    def test_load_data(self):
        pass

    def test_load_ann(self):
        pass
