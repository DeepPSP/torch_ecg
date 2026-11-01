"""Test the wfdb pandas-3 compatibility shim."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases._wfdb_compat import apply_wfdb_pandas3_compat

_CWD = Path(__file__).resolve().parents[2]


def _make_annotation():
    """Build a minimal wfdb Annotation carrying ``symbol`` (no files needed)."""
    wfdb = pytest.importorskip("wfdb")
    return wfdb.Annotation(
        record_name="test",
        extension="atr",
        sample=np.array([0, 100, 200], dtype=int),
        symbol=["N", "N", "A"],
        aux_note=[None, None, None],
        label_store=None,
    )


class TestWfdbPandas3Compat:
    def test_get_contained_labels(self):
        """get_contained_labels works on symbol-carrying annotations.

        Under pandas < 3, or under pandas >= 3 once the shim (or the
        upstream fix, MIT-LCP/wfdb-python#575) is applied, this returns
        the label map with the symbol column kept; under pandas >= 3
        with wfdb <= 4.3.1 and no shim it raises TypeError.
        """
        ann = _make_annotation()
        labels = ann.get_contained_labels(inplace=False)
        assert isinstance(labels, pd.DataFrame)
        # the symbol column must survive as a column (the array-index
        # path on pandas < 3 keeps it; drop=False in the shim preserves
        # that behavior)
        assert "symbol" in labels.columns
        # both N and A occur in the annotation
        assert len(labels) == 2
        assert labels["n_occurrences"].sum() == 3

    def test_patch_is_idempotent_and_conditional(self):
        """Applying the compat twice is safe, and the guard skips
        fixed/irrelevant environments without raising."""
        apply_wfdb_pandas3_compat()
        apply_wfdb_pandas3_compat()
        # re-run the functional assertion after the double application
        ann = _make_annotation()
        labels = ann.get_contained_labels(inplace=False)
        assert "symbol" in labels.columns
