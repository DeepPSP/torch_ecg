"""
TestQTDB: accomplished

subsampling: NOT tested
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from torch_ecg.databases import QTDB, DataBaseInfo
from torch_ecg.utils.download import PHYSIONET_DB_VERSION_PATTERN


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


class TestQTDB:
    def test_len(self):
        assert len(reader) == 105

    def test_load_data(self):
        data = reader.load_data(0)
        assert data.ndim == 2
        assert data.shape[0] == len(reader.get_lead_names(0))
        data_1 = reader.load_data(
            0, leads=0, data_format="flat", sampto=1000, units="uV"
        )
        assert np.allclose(data[0][:1000], data_1 / 1000)

    def test_load_ann(self):
        ann = reader.load_ann(0)
        ann_1 = reader.load_ann(0, ignore_beat_types=False)
        assert len(ann) <= len(ann_1)
        ann = reader.load_ann(0, sampfrom=1000, sampto=2000, extension="pu1")
        ann_1 = reader.load_ann(
            0, sampfrom=1000, sampto=2000, extension="pu1", keep_original=True
        )
        assert len(ann) == len(ann_1)

        with pytest.raises(
            AssertionError,
            match="extension should be one of `q1c`, `q2c`, `pu1`, `pu2`",
        ):
            reader.load_ann(0, extension="q1")
        with pytest.raises(
            AssertionError, match="`sampto` should be greater than `sampfrom`"
        ):
            reader.load_ann(0, sampfrom=2000, sampto=1000)

    def test_load_wave_ann(self):
        # alias of `load_ann`
        assert len(reader.load_wave_ann(0)) == len(reader.load_ann(0))

    def test_load_wave_masks(self):
        # Not implemented yet
        with pytest.raises(
            NotImplementedError,
            match="A large proportion of the wave delineation annotations lack onset indices",
        ):
            reader.load_wave_masks(0)

    def test_load_rhythm_ann(self):
        # Not implemented yet
        with pytest.raises(
            NotImplementedError,
            match="Only a small part of the recordings have rhythm annotations",
        ):
            reader.load_rhythm_ann(0)

    def test_load_beat_ann(self):
        beat_ann = reader.load_beat_ann(0)
        beat_ann_1 = reader.load_beat_ann(0, beat_types=reader.beat_types[:2])
        assert len(beat_ann) >= len(beat_ann_1)
        beat_ann = reader.load_beat_ann(
            0, sampfrom=1000, sampto=2000, beat_format="dict"
        )
        assert isinstance(beat_ann, dict)
        assert all([isinstance(beat_ann[k], np.ndarray) for k in beat_ann])
        beat_ann_1 = reader.load_beat_ann(
            0, sampfrom=1000, sampto=2000, beat_format="dict", keep_original=True
        )
        assert beat_ann.keys() == beat_ann_1.keys()
        assert all([np.allclose(beat_ann[k], beat_ann_1[k] - 1000) for k in beat_ann])

        with pytest.raises(
            AssertionError,
            match="`beat_format` must be one of \\['beat', 'dict'\\], but got `list`",
        ):
            reader.load_beat_ann(0, beat_format="list")

    def test_load_rpeak_indices(self):
        rpeak_indices = reader.load_rpeak_indices(0, sampfrom=1000, sampto=2000)
        rpeak_indices_1 = reader.load_rpeak_indices(
            0, sampfrom=1000, sampto=2000, keep_original=True
        )
        assert np.allclose(rpeak_indices, rpeak_indices_1 - 1000)

        with pytest.raises(
            AssertionError,
            match="`extension` must be one of \\['atr', 'man'\\], but got `qrs`",
        ):
            reader.load_rpeak_indices(0, extension="qrs")
        with pytest.raises(
            FileNotFoundError,
            match="""annotation file `sel30` does not exist, try setting `extension = "man"`""",
        ):
            reader.load_rpeak_indices("sel30")

    def test_meta_data(self):
        assert isinstance(reader.version, str) and re.match(
            PHYSIONET_DB_VERSION_PATTERN, reader.version
        )
        assert isinstance(reader.webpage, str) and len(reader.webpage) > 0
        assert reader.get_citation() is None  # printed
        assert isinstance(reader.database_info, DataBaseInfo)

    def test_plot(self):
        # `plot` not implemented yet
        with pytest.raises(NotImplementedError):
            reader.plot(0)
