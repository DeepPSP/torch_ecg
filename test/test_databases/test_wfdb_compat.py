"""Test the wfdb pandas-3 compatibility shim."""

import numpy as np
import pandas as pd
import pytest

from torch_ecg.databases._wfdb_compat import apply_wfdb_pandas3_compat


def _make_annotation(**overrides):
    """Build a minimal wfdb Annotation without touching any files."""
    wfdb = pytest.importorskip("wfdb")
    fields = dict(
        record_name="test",
        extension="atr",
        sample=np.array([0, 100, 200], dtype=int),
        symbol=["N", "N", "A"],
        aux_note=[None, None, None],
        label_store=None,
    )
    fields.update(overrides)
    return wfdb.Annotation(**fields)


class TestWfdbPandas3Compat:
    def test_symbol_branch(self):
        """The symbol branch: works under pandas < 3 natively, and under
        pandas >= 3 via the shim (or the upstream fix,
        MIT-LCP/wfdb-python#575); the symbol column must survive as a
        column (drop=False preserves the pandas<3 output schema).
        """
        ann = _make_annotation()
        labels = ann.get_contained_labels(inplace=False)
        assert isinstance(labels, pd.DataFrame)
        assert "symbol" in labels.columns
        assert len(labels) == 2  # N and A
        assert labels["n_occurrences"].sum() == 3

    def test_label_store_branch(self):
        """The label_store branch (no temporary index is set)."""
        ann = _make_annotation(symbol=None, label_store=[1, 1, 1])
        labels = ann.get_contained_labels(inplace=False)
        assert len(labels) == 1
        assert labels["n_occurrences"].iloc[0] == 3

    def test_description_branch(self):
        """The description branch (string column, indexed by name)."""
        ann = _make_annotation(symbol=None, description=["Normal beat"] * 3)
        labels = ann.get_contained_labels(inplace=False)
        assert "description" in labels.columns
        assert labels["n_occurrences"].sum() == 3

    def test_custom_labels_triplets(self):
        """Custom labels supplied as a list of triplets."""
        ann = _make_annotation(
            custom_labels=[(40, "X1", "Custom one"), (41, "X2", "Custom two")],
            label_store=[40, 40, 41],
        )
        labels = ann.get_contained_labels(inplace=False)
        assert "X1" in labels["symbol"].tolist()
        assert labels.loc[40, "n_occurrences"] == 2

    def test_custom_labels_dataframe(self):
        """Custom labels supplied as a DataFrame (exercises its
        label_store re-indexing)."""
        custom = pd.DataFrame(
            {
                "label_store": [40, 41],
                "symbol": ["X1", "X2"],
                "description": ["Custom one", "Custom two"],
            }
        )
        ann = _make_annotation(custom_labels=custom, label_store=[40, 40, 41])
        labels = ann.get_contained_labels(inplace=False)
        assert labels.loc[40, "n_occurrences"] == 2
        assert labels.loc[41, "n_occurrences"] == 1

    def test_no_labels_raises(self):
        """An annotation without any label field raises."""
        ann = _make_annotation(symbol=None, description=None, label_store=None)
        with pytest.raises(Exception):
            ann.get_contained_labels(inplace=False)

    def test_inplace_true(self):
        """inplace=True sets the attribute and returns None."""
        ann = _make_annotation()
        ret = ann.get_contained_labels(inplace=True)
        assert ret is None
        assert isinstance(ann.contained_labels, pd.DataFrame)
        assert "symbol" in ann.contained_labels.columns

    def test_patch_is_idempotent(self):
        """Applying the compat twice is safe, and the guard skips
        fixed/irrelevant environments without raising."""
        apply_wfdb_pandas3_compat()
        apply_wfdb_pandas3_compat()
        # re-run a functional assertion after the double application
        ann = _make_annotation()
        labels = ann.get_contained_labels(inplace=False)
        assert "symbol" in labels.columns
