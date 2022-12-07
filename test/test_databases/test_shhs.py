"""
TestSHHS: NOT accomplished

subsampling: NOT tested
"""

from pathlib import Path

from torch_ecg.databases import SHHS, DataBaseInfo


###############################################################################
# set paths
# 9 files are downloaded in the following directory using `nsrr`
# ref. the action file .github/workflows/run-pytest.yml
_CWD = Path("~/tmp/nsrr-data/shhs").expanduser().resolve()
###############################################################################


reader = SHHS(_CWD)


class TestSHHS:
    def test_len(self):
        assert len(reader) == 9

    def test_load_data(self):
        pass

    def test_load_ann(self):
        pass

    def test_meta_data(self):
        # TODO: add more....
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        pass
