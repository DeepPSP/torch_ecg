"""Compatibility shims for third-party dependencies of the database readers.

Currently one shim is provided:

- :func:`apply_wfdb_pandas3_compat` -- repairs
  ``wfdb.Annotation.get_contained_labels`` under pandas >= 3.
"""

import inspect
import logging

__all__ = [
    "apply_wfdb_pandas3_compat",
]


_logger = logging.getLogger(__name__)


def apply_wfdb_pandas3_compat() -> None:
    """Repair ``Annotation.get_contained_labels`` for pandas >= 3.

    Under pandas >= 3 (Arrow-backed strings by default), wfdb <= 4.3.1
    raises ``TypeError`` inside ``get_contained_labels`` when the
    annotation carries ``symbol`` or ``description``::

        label_map.set_index(label_map["symbol"].values, inplace=True)
        # TypeError: ... Received column of type ArrowStringArray

    ``torch_ecg`` itself never calls this method (its readers do not
    pass ``summarize_labels=True`` to ``rdann``), but the method is a
    public API of the ``Annotation`` objects our readers return, so
    end users can reach it.  Apply the same two-line fix as
    `MIT-LCP/wfdb-python#575 <https://github.com/MIT-LCP/wfdb-python/pull/575>`__
    (pass the column *name* to ``set_index`` instead of the Arrow
    string array) until a wfdb release carries it.

    The patch is applied only when both hold:

    - pandas >= 3 is installed;
    - the installed wfdb still contains the buggy pattern (checked
      against the method source, so any upstream release containing
      the fix disables this shim automatically, regardless of
      version numbering).
    """

    try:
        import numpy as np
        import pandas as pd
        import wfdb
        from wfdb.io.annotation import Annotation, ann_label_table, label_triplets_to_df
    except ImportError:
        return

    if int(pd.__version__.split(".")[0]) < 3:
        return

    source = inspect.getsource(Annotation.get_contained_labels)
    if 'label_map["symbol"].values' not in source and "'symbol'].values" not in source:
        # the installed wfdb is already fixed (or restructured); nothing to do
        return

    def get_contained_labels(self, inplace: bool = True):
        """Patched ``Annotation.get_contained_labels`` (pandas >= 3).

        Identical to the wfdb implementation, except that string
        columns are passed to ``DataFrame.set_index`` by name instead
        of via ``.values`` (which yields an unindexable
        ``ArrowStringArray`` under pandas >= 3).  See
        MIT-LCP/wfdb-python#575.
        """
        if self.custom_labels is not None:
            self.check_field("custom_labels")

        # Create the label map
        label_map = ann_label_table.copy()

        # Convert the tuple triplets into a pandas dataframe if needed
        if isinstance(self.custom_labels, (list, tuple)):
            custom_labels = label_triplets_to_df(self.custom_labels)
        elif isinstance(self.custom_labels, pd.DataFrame):
            # Set the index just in case it doesn't already match the label_store
            # (`label_store` is an integer column; `.values` is safe here)
            self.custom_labels.set_index(self.custom_labels["label_store"].values, inplace=True)
            custom_labels = self.custom_labels
        else:
            custom_labels = None

        # Merge the standard WFDB labels with the custom labels.
        # custom labels values overwrite standard WFDB if overlap.
        if custom_labels is not None:
            for i in custom_labels.index:
                label_map.loc[i] = custom_labels.loc[i]

        # Get the labels using one of the features
        if self.label_store is not None:
            index_vals = set(self.label_store)
            reset_index = False
            counts = np.unique(self.label_store, return_counts=True)
        elif self.symbol is not None:
            index_vals = set(self.symbol)
            # `drop=False` keeps the column in the output, matching the
            # behavior of the original array-index (pandas < 3) path
            label_map.set_index("symbol", drop=False, inplace=True)
            reset_index = True
            counts = np.unique(self.symbol, return_counts=True)
        elif self.description is not None:
            index_vals = set(self.description)
            label_map.set_index("description", drop=False, inplace=True)
            reset_index = True
            counts = np.unique(self.description, return_counts=True)
        else:
            raise Exception("No annotation labels contained in object")

        contained_labels = label_map.loc[list(index_vals), :]

        # Add the counts
        for i in range(len(counts[0])):
            contained_labels.loc[counts[0][i], "n_occurrences"] = counts[1][i]
        contained_labels["n_occurrences"] = pd.to_numeric(contained_labels["n_occurrences"], downcast="integer")

        if reset_index:
            # `label_store` is an integer column; `.values` is safe here
            contained_labels.set_index(contained_labels["label_store"].values, inplace=True)

        if inplace:
            self.contained_labels = contained_labels
            return
        else:
            return contained_labels

    Annotation.get_contained_labels = get_contained_labels
    _logger.debug(
        "patched wfdb (%s) Annotation.get_contained_labels for pandas %s " "(upstream fix: MIT-LCP/wfdb-python#575)",
        wfdb.__version__,
        pd.__version__,
    )
