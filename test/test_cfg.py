"""
"""

import numpy as np
import pytest
import torch

from torch_ecg.cfg import CFG, DEFAULTS, DTYPE


def test_cfg():
    c = CFG(hehe={"a": 1, "b": 2})
    c.update(hehe={"a": [-1]})
    assert str(c) == repr(c) == r"{'hehe': {'a': [-1], 'b': 2}}"
    c.update(hehe={"c": -10})
    assert str(c) == repr(c) == r"{'hehe': {'a': [-1], 'b': 2, 'c': -10}}"
    assert c.hehe.pop("a") == [-1]
    assert str(c) == repr(c) == r"{'hehe': {'b': 2, 'c': -10}}"

    c = CFG({1: 2, 3: 4})
    assert str(c) == repr(c) == r"{1: 2, 3: 4}"

    with pytest.raises(TypeError, match="expected at most 1 arguments, got 2"):
        CFG(1, 2)


def test_dtype():
    dtp = DTYPE("float32")
    assert str(dtp) == repr(dtp) == "DTYPE(STR='float32', NP=dtype('float32'), TORCH=torch.float32, INT=32)"

    with pytest.raises(TypeError, match="data type 'hehe' not understood"):
        DTYPE("hehe")

    with pytest.raises(AssertionError, match="inconsistent dtype"):
        DTYPE("float32", INT=64)


def test_defaults():
    assert DEFAULTS.DTYPE == DTYPE("float32")
    assert DEFAULTS.dtype == torch.float32
    DEFAULTS.change_dtype(torch.float16)
    assert DEFAULTS.dtype == torch.float16
    DEFAULTS.change_dtype(np.float32)
    assert DEFAULTS.dtype == torch.float32

    with pytest.raises(TypeError, match="`dtype` must be a str or np.dtype or torch.dtype"):
        DEFAULTS.change_dtype(32)
    with pytest.raises(AssertionError, match="`dtype` must be one of "):
        DEFAULTS.change_dtype("float128")

    assert DEFAULTS.SEED == 42
    DEFAULTS.set_seed(100)
    assert DEFAULTS.SEED == 100
