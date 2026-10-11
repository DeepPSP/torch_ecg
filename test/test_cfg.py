""" """

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


def test_cfg_reserved_keys():
    """Keys that would shadow the dict API must be rejected loudly."""
    for key in ["keys", "values", "items", "get", "pop", "update", "clear", "copy", "setdefault"]:
        with pytest.raises(ValueError, match="reserved"):
            CFG({key: 1})
        c = CFG()
        with pytest.raises(ValueError, match="reserved"):
            setattr(c, key, 1)
        with pytest.raises(ValueError, match="reserved"):
            c[key] = 1
        with pytest.raises(ValueError, match="reserved"):
            c.update({key: 1})


def test_cfg_dict_typed_field():
    """Regression: a dict-typed field used to be converted with
    `self.__class__`, routing it through the subclass constructor and
    crashing CFG subclasses with stricter `__init__` (e.g. the Output
    family documented this in its `_KNOWN_ISSUES`)."""
    from torch_ecg.components.outputs import ClassificationOutput

    output = ClassificationOutput(classes=["AF", "N", "SPB"], pred=np.ones((1, 3)), prob=np.ones((1, 3)), d={"d": 1})
    # both interfaces must work for the dict-typed field
    assert output.d == {"d": 1}
    assert output["d"] == {"d": 1}


def test_cfg_update_deepcopies_replacement():
    """Regression: when `update` replaces a non-mapping value (e.g. None)
    with a mapping, the source must not be shared with the target."""
    source = {"monitor": {"val": [1, 2]}}
    c = CFG(monitor=None)
    c.update(source)
    source["monitor"]["val"].append(3)
    assert c.monitor.val == [1, 2]

    source = {"monitor": {"val": [1, 2]}}
    c2 = CFG(monitor=None)
    c2.update(**source)
    source["monitor"]["val"].append(4)
    assert c2.monitor.val == [1, 2]


def test_cfg_repr_keeps_callable_placeholder():
    def foo():
        pass

    c = CFG(fn=foo, n=1)
    assert "<callable: foo>" in repr(c)
    assert "<callable: foo>" in str(c)
    assert "'n': 1" in repr(c)


def test_cfg_deepcopy():
    import pickle
    from copy import deepcopy

    c = CFG(a=CFG(b=[1, {"c": 2}]), n=None)
    # update REPLACES list values (no merging for sequences)
    c.update(a=CFG(b=[1, {"d": 3}]))
    assert c.a.b == [1, {"d": 3}]
    c2 = deepcopy(c)
    assert c2 == c
    assert c2.a.b[1].d == 3
    c2.a.b.append(4)
    assert c.a.b == [1, {"d": 3}]
    # attribute view preserved on the copy
    assert c2.a.b[1].d == 3

    # deepcopy of a subclass with a strict __init__ must not crash
    from torch_ecg.components.outputs import ClassificationOutput

    output = ClassificationOutput(classes=["AF"], pred=np.ones((1, 1)), prob=np.ones((1, 1)))
    output2 = deepcopy(output)
    assert output2.classes == ["AF"]

    # pickle round-trip (configs are pickled inside saved checkpoints)
    blob = pickle.dumps(c)
    c3 = pickle.loads(blob)
    assert c3 == c
    assert c3.a.b[1].d == 3
