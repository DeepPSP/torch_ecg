"""The AHA statement tables must not depend on the pandas downcasting option.

The AHA aux-data tables are parsed at import time with ``ffill`` on
object-dtype columns.  On pandas < 3.1 the module enables
``future.no_silent_downcasting`` around the parsing and restores the user's
value afterwards; on pandas >= 3.1 the option itself is deprecated and the
module leaves it untouched.  That version gate is safe only if the parsed
tables are the same either way, which is what these tests pin down.

The tables are module-level state, so every regime below is imported in a
fresh subprocess.
"""

import os
import pickle
import subprocess
import sys
from pathlib import Path

from pandas.testing import assert_frame_equal

_CWD = Path(__file__).absolute().parents[1]

# see test_home_cache.py for why the parent's import path is passed through
_PYTHONPATH = os.pathsep.join(p for p in sys.path if p)

_OPTION = "future.no_silent_downcasting"

# parses the tables under one regime and pickles them together with the
# value of the option before and after the import
#   plain   : untouched import
#   user_on : the option is turned on by the user before the import
#   forced  : pandas reports a 2.x version, so the module takes its
#             pandas-2 branch whatever version is installed
_PARSE = """
import pickle, sys, warnings

out, regime = sys.argv[1], sys.argv[2]
warnings.simplefilter("ignore")  # the option is deprecated on pandas >= 3.1

import pandas as pd

before = pd.get_option("future.no_silent_downcasting")
if regime == "user_on":
    pd.set_option("future.no_silent_downcasting", True)
elif regime == "forced":
    pd.__version__ = "2.3.3"

import torch_ecg.databases.aux_data.aha as aha

with open(out, "wb") as f:
    pickle.dump(
        {
            "tables": {k: v for k, v in vars(aha).items() if k.startswith("df_")},
            "before": before,
            "after": pd.get_option("future.no_silent_downcasting"),
        },
        f,
    )
"""


def _parse(regime, tmp_path):
    out = tmp_path / f"{regime}.pkl"
    result = subprocess.run(
        [sys.executable, "-c", _PARSE, str(out), regime],
        capture_output=True,
        text=True,
        cwd=str(_CWD),
        env={**os.environ, "PYTHONPATH": _PYTHONPATH},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return pickle.loads(out.read_bytes())


def test_tables_independent_of_downcasting_option(tmp_path):
    """The parsed tables are identical with the option off, on, and set by
    the module's pandas-2 branch (values and dtypes)."""
    regimes = {regime: _parse(regime, tmp_path) for regime in ("plain", "user_on", "forced")}
    tables = regimes["plain"]["tables"]
    assert len(tables) == 7
    for name, df in tables.items():
        for regime in ("user_on", "forced"):
            assert_frame_equal(regimes[regime]["tables"][name], df, check_dtype=True)


def test_user_option_survives_import(tmp_path):
    """Whatever the user set before importing is still in effect afterwards,
    whichever branch the module takes."""
    for regime in ("plain", "user_on", "forced"):
        result = _parse(regime, tmp_path)
        expected = True if regime == "user_on" else result["before"]
        assert result["after"] == expected, regime


def test_forward_filled_grouping_columns(tmp_path):
    """The forward fill does what the option could have disturbed: the
    grouping columns are complete and consistent."""
    tables = _parse("plain", tmp_path)["tables"]

    df_primary = tables["df_primary_statements"]
    assert not df_primary[["CategoryCode", "Category"]].isna().any().any()
    # every category keeps exactly one code, including the rows written as
    # `,,code,description` in the table literal
    assert (df_primary.groupby("Category")["CategoryCode"].nunique() == 1).all()
    assert df_primary["Code"].is_unique

    assert not tables["df_secondary_statements"][["Group"]].isna().any().any()
    assert not tables["df_modifiers"][["Category"]].isna().any().any()
    assert not tables["df_comparison_statements"][["Code", "Statement", "Criteria"]].isna().any().any()


def test_import_does_not_warn_about_downcasting_option():
    """No deprecation warning about the option is emitted at import (the
    original symptom on pandas >= 3.1)."""
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import torch_ecg"],
        capture_output=True,
        text=True,
        cwd=str(_CWD),
        env={**os.environ, "PYTHONPATH": _PYTHONPATH},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert _OPTION not in result.stderr
