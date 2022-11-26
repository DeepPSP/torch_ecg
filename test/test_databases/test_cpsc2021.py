"""
"""

from copy import deepcopy
from pathlib import Path

from torch_ecg.databases import CPSC2021
from torch_ecg.databases.datasets import CPSC2021Dataset, CPSC2021TrainCfg


###############################################################################
# set paths
_CWD = Path(__file__).absolute().parents[2] / "sample-data" / "cpsc2021"
###############################################################################


reader = CPSC2021(_CWD)


class TestCPSC2021:
    def test_len(self):
        assert len(reader) == 18

    def test_load_data(self):
        pass

    def test_load_ann(self):
        pass

    def test_meta_data(self):
        pass

    def test_plot(self):
        pass


config = deepcopy(CPSC2021TrainCfg)
config.db_dir = _CWD

ds = CPSC2021Dataset(config, task="main", training=False, lazy=True)
