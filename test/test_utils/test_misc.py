"""
"""

import datetime
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

from torch_ecg.utils.misc import (  # noqa: F401
    get_record_list_recursive,
    get_record_list_recursive2,
    get_record_list_recursive3,
    dict_to_str,
    str2bool,
    diff_with_step,
    ms2samples,
    samples2ms,
    plot_single_lead,
    init_logger,
    get_date_str,
    list_sum,
    read_log_txt,
    read_event_scalars,
    dicts_equal,
    ReprMixin,
    CitationMixin,
    MovingAverage,
    nildent,
    add_docstring,
    remove_parameters_returns_from_docstring,
    timeout,
    Timer,
    get_kwargs,
    get_required_args,
    add_kwargs,
)  # noqa: F401
from torch_ecg.cfg import DEFAULTS, _DATA_CACHE


_SAMPLE_DATA_DIR = Path(__file__).parents[2].resolve() / "sample-data"


# create `_DATA_CACHE / "database_citation.csv"`
# to test backward compatibility of `CitationMixin`
(_DATA_CACHE / "database_citation.csv").touch(exist_ok=True)


class SomeClass(ReprMixin, CitationMixin):
    def __init__(self, aaa, bb, c):
        self.aaa = aaa
        self.bb = bb
        self.c = c

    def extra_repr_keys(self):
        return ["aaa", "bb"]

    @property
    def doi(self):
        return "10.1088/1361-6579/ac9451"


class AnotherClass(ReprMixin, CitationMixin):
    def __init__(self, aaa, bb, c):
        self.aaa = aaa
        self.bb = bb
        self.c = c

    def extra_repr_keys(self):
        return ["aaa", "bb"]

    @property
    def doi(self):
        return ["10.48550/ARXIV.2204.04420", "10.1088/1361-6579/ac9451"]


def test_get_record_list_recursive():
    path = _SAMPLE_DATA_DIR / "cinc2021"
    record_list = get_record_list_recursive(path, rec_ext="mat")
    record_list_1 = get_record_list_recursive(path, rec_ext=".hea")
    assert set(record_list_1) == set(record_list)
    record_list_1 = get_record_list_recursive(path, rec_ext="mat", relative=False)
    assert all([Path(p).is_absolute() for p in record_list_1])
    assert all([p.startswith(str(path)) for p in record_list_1])


def test_get_record_list_recursive2():
    path = _SAMPLE_DATA_DIR / "cinc2021"
    record_list = get_record_list_recursive(path, rec_ext="mat")
    with pytest.warns(DeprecationWarning):
        record_list_1 = get_record_list_recursive2(path, rec_pattern="[A-Z]*.mat")
    assert len(record_list_1) == len(record_list)


def test_get_record_list_recursive3():
    path = _SAMPLE_DATA_DIR / "cinc2021"
    rec_prefix = {
        "A": "A",
        "B": "Q",
        "C": "I",
        "D": "S",
        "E": "HR",
        "F": "E",
        "G": "JS",
    }
    rec_patterns_with_ext = {
        tranche: f"^{rec_prefix[tranche]}(?:\\d+)\\.mat$" for tranche in list("ABCDEFG")
    }
    record_list = get_record_list_recursive3(path, rec_patterns_with_ext)
    assert isinstance(record_list, dict)
    assert record_list.keys() == rec_patterns_with_ext.keys()
    assert all([isinstance(v, list) for v in record_list.values()])
    for tranche in list("ABCD"):
        assert len(record_list[tranche]) == 0
    assert len(record_list["E"]) == 10
    assert len(record_list["F"]) == 20
    assert len(record_list["G"]) == 20


def test_dict_to_str():
    d = {"a": 1, "b": [1, 2, 3], "c": {"d": 1, "e": 2}}
    s = dict_to_str(d)
    assert isinstance(s, str)


def test_str2bool():
    assert str2bool("True") is True
    assert str2bool("False") is False
    assert str2bool("true") is True
    assert str2bool("false") is False
    assert str2bool("1") is True
    assert str2bool("0") is False
    assert str2bool("yes") is True
    assert str2bool("no") is False
    assert str2bool("y") is True
    assert str2bool("n") is False
    with pytest.raises(ValueError, match="Boolean value expected"):
        str2bool("abc")
    with pytest.raises(ValueError, match="Boolean value expected"):
        str2bool("2")


def test_diff_with_step():
    data = np.arange(100)
    assert np.allclose(diff_with_step(data, 1), np.diff(data))
    assert (diff_with_step(data, 2) == 2).all()
    assert (diff_with_step(data, 3) == 3).all()


def test_ms2samples():
    n_samples = ms2samples(1200, 100)
    assert n_samples == 120
    n_samples = ms2samples(1210, 100)
    assert n_samples == 121
    n_samples = ms2samples(1212, 100)
    assert n_samples == 121
    n_samples = ms2samples(1219, 100)
    assert n_samples == 121


def test_samples2ms():
    t_ms = samples2ms(120, 100)
    assert isinstance(t_ms, float) and t_ms == 1200
    t_ms = samples2ms(121, 100)
    assert isinstance(t_ms, float) and t_ms == 1210


def test_plot_single_lead():
    fs = 500
    n_samples = 5000
    plot_single_lead(
        t=np.arange(n_samples) / fs,
        sig=500 * DEFAULTS.RNG.normal(size=(n_samples,)),
        ticks_granularity=2,
    )


def test_init_logger():
    pass


def test_get_date_str():
    assert (
        datetime.datetime.strptime(get_date_str(), "%m-%d_%H-%M")
        < datetime.datetime.now()
    )


def test_list_sum():
    lst = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert list_sum(lst) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    lst = []
    assert list_sum(lst) == []
    lst = [[{"a": 1}, {"b": 2}], ["xxx"]]
    assert list_sum(lst) == [{"a": 1}, {"b": 2}, "xxx"]


def test_read_log_txt():
    pass


def test_read_event_scalars():
    pass


def test_dicts_equal():
    d1 = {"a": pd.DataFrame([{"hehe": 1, "haha": 2}])[["haha", "hehe"]]}
    d2 = {"a": pd.DataFrame([{"hehe": 1, "haha": 2}])[["hehe", "haha"]]}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True

    d1 = {"a": torch.tensor([1, 2, 3])}
    d2 = {"a": torch.tensor([1, 2, 3])}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True

    d1 = {"a": np.array([1, 2, 3])}
    d2 = {"a": np.array([1, 2, 3])}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True

    d1 = {"a": [1, 2, 3]}
    d2 = {"a": np.array([1, 2, 3])}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True
    assert dicts_equal(d1, d2, allow_array_diff_types=False) is False
    assert dicts_equal(d2, d1, allow_array_diff_types=False) is False

    d1 = {"a": (1, 2, 3)}
    d2 = {"a": np.array([1, 2, 3])}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True
    assert dicts_equal(d1, d2, allow_array_diff_types=False) is False
    assert dicts_equal(d2, d1, allow_array_diff_types=False) is False

    d1 = {"a": torch.tensor([1, 2, 3])}
    d2 = {"a": np.array([1, 2, 3])}
    assert dicts_equal(d1, d2) is True
    assert dicts_equal(d2, d1) is True
    assert dicts_equal(d1, d2, allow_array_diff_types=False) is False
    assert dicts_equal(d2, d1, allow_array_diff_types=False) is False

    d1 = {"a": torch.tensor([1, 2, 3])}
    d2 = {"a": pd.Series([1, 2, 3])}
    assert dicts_equal(d1, d2) is False
    assert dicts_equal(d2, d1) is False
    assert dicts_equal(d1, d2, allow_array_diff_types=False) is False
    assert dicts_equal(d2, d1, allow_array_diff_types=False) is False

    d1 = {"a": torch.tensor([1, 2, 3])}
    d2 = {"a": pd.DataFrame([1, 2, 3])}
    assert dicts_equal(d1, d2) is False
    assert dicts_equal(d2, d1) is False
    assert dicts_equal(d1, d2, allow_array_diff_types=False) is False
    assert dicts_equal(d2, d1, allow_array_diff_types=False) is False


def test_ReprMixin():
    some_class = SomeClass(1, 2, 3)
    string = textwrap.dedent(
        """
        SomeClass(
            aaa = 1,
            bb  = 2
        )
        """
    ).strip("\n")
    assert str(some_class) == repr(some_class)
    assert str(some_class) == string

    another_class = AnotherClass(1, 2, 3)
    assert str(another_class) == repr(another_class)


def test_CitationMixin():
    some_class = SomeClass(1, 2, 3)
    citation = some_class.get_citation(lookup=True, print_result=False)
    assert isinstance(citation, str) and len(citation) > 0
    citation = some_class.get_citation(lookup=False, print_result=False)
    assert citation == some_class.doi
    assert some_class.get_citation(print_result=True) is None

    another_class = AnotherClass(1, 2, 3)
    citation = another_class.get_citation(lookup=True, print_result=False)
    assert isinstance(citation, str) and len(citation) > 0
    citation = another_class.get_citation(lookup=False, print_result=False)
    assert citation == "\n".join(another_class.doi)
    assert another_class.get_citation(print_result=True) is None


def test_MovingAverage():
    ma = MovingAverage(verbose=2)
    data = DEFAULTS.RNG.normal(size=(100,))
    new_data = ma(data, method="sma", window=7, center=True)
    assert new_data.shape == data.shape
    new_data = ma(data, method="ema", weight=0.7)
    assert new_data.shape == data.shape
    new_data = ma(data, method="cma")
    assert new_data.shape == data.shape
    new_data = ma(data, method="wma", window=7)

    with pytest.raises(
        NotImplementedError, match="method `xxx` is not implemented yet"
    ):
        ma(data, method="xxx")

    with pytest.warns(
        RuntimeWarning,
        match="the following arguments are not used: `.+` for simple moving average",
    ):
        ma(data, method="sma", weight=0.7)
    with pytest.warns(
        RuntimeWarning,
        match="the following arguments are not used: `.+` for exponential moving average",
    ):
        ma(data, method="ema", window=7)
    with pytest.warns(
        RuntimeWarning,
        match="the following arguments are not used: `.+` for cumulative moving average",
    ):
        ma(data, method="cma", window=7)
    with pytest.warns(
        RuntimeWarning,
        match="the following arguments are not used: `.+` for weighted moving average",
    ):
        ma(data, method="wma", center=True)


def test_nildent():
    string = """
    this is the first line,
      this is the second line,
              this is the third line,
 this is the fourth line.
    """
    assert (
        nildent(string)
        == "\nthis is the first line,\nthis is the second line,\nthis is the third line,\nthis is the fourth line.\n"
    )


def test_add_docstring():
    @add_docstring("This is a new docstring.")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "This is a new docstring."
    assert func(1, 2) == 3

    @add_docstring("Leading docstring.", mode="prepend")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "Leading docstring.\nThis is a docstring."
    assert func(1, 2) == 3

    @add_docstring("Trailing docstring.", mode="append")
    def func(a, b):
        """This is a docstring."""
        return a + b

    assert func.__doc__ == "This is a docstring.\nTrailing docstring."
    assert func(1, 2) == 3


def test_remove_parameters_returns_from_docstring():
    new_docstring = remove_parameters_returns_from_docstring(
        remove_parameters_returns_from_docstring.__doc__,
        parameters=["returns_indicator", "parameters_indicator"],
        returns="new_doc",
    )
    assert (
        new_docstring
        == """
    remove parameters and/or returns from docstring,
    which is of the format of numpydoc

    Parameters
    ----------
    doc: str,
        docstring to be processed
    parameters: str or list of str, default None,
        parameters to be removed
    returns: str or list of str, default None,
        returned values to be removed

    Returns
    -------
    """
    )


def test_timeout():
    with timeout(1):
        time.sleep(0.5)

    with pytest.raises(TimeoutError, match="block timedout after `1` seconds"):
        with timeout(1):
            time.sleep(2)

    with pytest.raises(ValueError, match="`duration` must be non-negative"):
        with timeout(-1):
            pass


def test_Timer():
    timer = Timer()

    with timer:
        time.sleep(0.5)
        timer.add_timer("xxx")
        # do something
        time.sleep(0.5)
        timer.add_timer("yyy")
        # do some other thing
        time.sleep(0.5)
        timer.stop_timer("yyy")
        # do some other thing
        time.sleep(0.5)
        timer.stop_timer("xxx")


def test_get_kwargs():
    def func1(a, b, c, d=2, e=3, f=4):
        pass

    def func2(a, b, c=1, d=2, *, e=3, f=4):
        pass

    class CLS1:
        def __init__(self, a, b, c, d=2, e=3, f=4):
            pass

    class CLS2:
        def __init__(self, a, b, c=1, d=2, *, e=3, f=4):
            pass

    kw = get_kwargs(func1, kwonly=False)
    assert kw == {"d": 2, "e": 3, "f": 4}
    func1(1, 2, 3)
    kw = get_kwargs(func1, kwonly=True)
    assert kw == {}
    kw = get_kwargs(func2, kwonly=False)
    assert kw == {"c": 1, "d": 2, "e": 3, "f": 4}
    func2(1, 2)
    kw = get_kwargs(func2, kwonly=True)
    assert kw == {"e": 3, "f": 4}

    kw = get_kwargs(CLS1, kwonly=False)
    assert kw == {"d": 2, "e": 3, "f": 4}
    CLS1(1, 2, 3)
    kw = get_kwargs(CLS1, kwonly=True)
    assert kw == {}
    kw = get_kwargs(CLS2, kwonly=False)
    assert kw == {"c": 1, "d": 2, "e": 3, "f": 4}
    CLS2(1, 2)
    kw = get_kwargs(CLS2, kwonly=True)
    assert kw == {"e": 3, "f": 4}


def test_get_required_args():
    def func1(a, b, c, d=2, e=3, f=4):
        pass

    def func2(a, b, c=1, d=2, *, e=3, f=4):
        pass

    class CLS1:
        def __init__(self, a, b, c, d=2, e=3, f=4):
            pass

    class CLS2:
        def __init__(self, a, b, c=1, d=2, *, e=3, f=4):
            pass

    kw = get_required_args(func1)
    assert kw == ["a", "b", "c"]
    func1(1, 2, 3)
    kw = get_required_args(func2)
    assert kw == ["a", "b"]
    func2(1, 2)

    kw = get_required_args(CLS1)
    assert kw == ["a", "b", "c"]
    CLS1(1, 2, 3)
    kw = get_required_args(CLS2)
    assert kw == ["a", "b"]
    CLS2(1, 2)


def test_add_kwargs():
    def func(a, b=1):
        return a + b

    new_func = add_kwargs(func, xxx="yyy", zzz=None)

    assert new_func(2) == new_func(2, xxx="a", zzz=100) == 3
