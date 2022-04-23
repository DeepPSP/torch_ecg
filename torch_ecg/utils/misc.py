"""
"""

import datetime
import logging
import os
import re
import sys
import signal
import time
import warnings
import inspect
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce, wraps
from glob import glob
from numbers import Number, Real
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Sequence,
    Union,
    Tuple,
)

import numpy as np
import pandas as pd

from ..cfg import DEFAULTS

__all__ = [
    "get_record_list_recursive",
    "get_record_list_recursive2",
    "get_record_list_recursive3",
    "dict_to_str",
    "str2bool",
    "diff_with_step",
    "ms2samples",
    "samples2ms",
    "plot_single_lead",
    "init_logger",
    "get_date_str",
    "list_sum",
    "read_log_txt",
    "read_event_scalars",
    "dicts_equal",
    "default_class_repr",
    "ReprMixin",
    "MovingAverage",
    "nildent",
    "isclass",
    "add_docstring",
    "deprecate_kwargs",
    "timeout",
    "Timer",
]


def get_record_list_recursive(db_dir: Union[str, Path], rec_ext: str) -> List[str]:
    """

    get the list of records in `db_dir` recursively,
    for example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1"; "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system

    Parameters
    ----------
    db_dir: str or Path,
        the parent (root) path of the whole database
    rec_ext: str,
        extension of the record files

    Returns
    -------
    res: list of str,
        list of records, in lexicographical order

    """
    res = []
    roots = [str(db_dir)]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            res += [item for item in tmp if os.path.isfile(item)]
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [
        os.path.splitext(item)[0].replace(str(db_dir), "").strip(os.sep)
        for item in res
        if item.endswith(rec_ext)
    ]
    res = sorted(res)

    return res


def get_record_list_recursive2(db_dir: Union[str, Path], rec_pattern: str) -> List[str]:
    """

    get the list of records in `db_dir` recursively,
    for example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1"; "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system

    Parameters
    ----------
    db_dir: str or Path,
        the parent (root) path of the whole database
    rec_pattern: str,
        pattern of the record filenames, e.g. "A*.mat"

    Returns
    -------
    res: list of str,
        list of records, in lexicographical order

    """
    res = []
    roots = [str(db_dir)]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            # res += [item for item in tmp if os.path.isfile(item)]
            res += glob(os.path.join(r, rec_pattern), recursive=False)
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [
        os.path.splitext(item)[0].replace(str(db_dir), "").strip(os.sep) for item in res
    ]
    res = sorted(res)

    return res


def get_record_list_recursive3(
    db_dir: Union[str, Path], rec_patterns: Union[str, Dict[str, str]]
) -> Union[List[str], Dict[str, List[str]]]:
    r"""

    get the list of records in `db_dir` recursively,
    for example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1"; "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system

    Parameters
    ----------
    db_dir: str or Path,
        the parent (root) path of the whole database
    rec_patterns: str or dict,
        pattern of the record filenames, e.g. "A(?:\d+).mat",
        or patterns of several subsets, e.g. `{"A": "A(?:\d+).mat"}`

    Returns
    -------
    res: list of str,
        list of records, in lexicographical order

    """
    if isinstance(rec_patterns, str):
        res = []
    elif isinstance(rec_patterns, dict):
        res = {k: [] for k in rec_patterns.keys()}
    roots = [str(db_dir)]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            # tmp = [os.path.join(r, item) for item in os.listdir(r)]
            tmp = os.listdir(r)
            if isinstance(rec_patterns, str):
                res += [
                    os.path.join(r, item)
                    for item in filter(re.compile(rec_patterns).search, tmp)
                ]
            elif isinstance(rec_patterns, dict):
                for k in rec_patterns.keys():
                    res[k] += [
                        os.path.join(r, item)
                        for item in filter(re.compile(rec_patterns[k]).search, tmp)
                    ]
            new_roots += [
                os.path.join(r, item)
                for item in tmp
                if os.path.isdir(os.path.join(r, item))
            ]
        roots = deepcopy(new_roots)
    if isinstance(rec_patterns, str):
        res = [
            os.path.splitext(item)[0].replace(str(db_dir), "").strip(os.sep)
            for item in res
        ]
        res = sorted(res)
    elif isinstance(rec_patterns, dict):
        for k in rec_patterns.keys():
            res[k] = [
                os.path.splitext(item)[0].replace(str(db_dir), "").strip(os.sep)
                for item in res[k]
            ]
            res[k] = sorted(res[k])
    return res


def dict_to_str(
    d: Union[dict, list, tuple], current_depth: int = 1, indent_spaces: int = 4
) -> str:
    """

    convert a (possibly) nested dict into a `str` of json-like formatted form,
    this nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters
    ----------
    d: dict, or list, or tuple,
        a (possibly) nested `dict`, or a list of `dict`
    current_depth: int, default 1,
        depth of `d` in the (possible) parent `dict` or `list`
    indent_spaces: int, default 4,
        the indent spaces of each depth

    Returns
    -------
    s: str,
        the formatted string

    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = r"{}" if isinstance(d, dict) else "[]"
        return s
    # flat_types = (Number, bool, str,)
    flat_types = (
        Number,
        bool,
    )
    flat_sep = ", "
    s = "\n"
    unit_indent = " " * indent_spaces
    prefix = unit_indent * current_depth
    if isinstance(d, (list, tuple)):
        if all([isinstance(v, flat_types) for v in d]):
            len_per_line = 110
            current_len = len(prefix) + 1  # + 1 for a comma
            val = []
            for idx, v in enumerate(d):
                add_v = f"\042{v}\042" if isinstance(v, str) else str(v)
                add_len = len(add_v) + len(flat_sep)
                if current_len + add_len > len_per_line:
                    val = ", ".join([item for item in val])
                    s += f"{prefix}{val},\n"
                    val = [add_v]
                    current_len = len(prefix) + 1 + len(add_v)
                else:
                    val.append(add_v)
                    current_len += add_len
            if len(val) > 0:
                val = ", ".join([item for item in val])
                s += f"{prefix}{val}\n"
        else:
            for idx, v in enumerate(d):
                if isinstance(v, (dict, list, tuple)):
                    s += f"{prefix}{dict_to_str(v, current_depth+1)}"
                else:
                    val = f"\042{v}\042" if isinstance(v, str) else v
                    s += f"{prefix}{val}"
                if idx < len(d) - 1:
                    s += ",\n"
                else:
                    s += "\n"
    elif isinstance(d, dict):
        for idx, (k, v) in enumerate(d.items()):
            key = f"\042{k}\042" if isinstance(k, str) else k
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{key}: {dict_to_str(v, current_depth+1)}"
            else:
                val = f"\042{v}\042" if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}"
            if idx < len(d) - 1:
                s += ",\n"
            else:
                s += "\n"
    s += unit_indent * (current_depth - 1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def str2bool(v: Union[str, bool]) -> bool:
    """

    converts a "boolean" value possibly in the format of str to bool

    Parameters
    ----------
    v: str or bool,
        the "boolean" value

    Returns
    -------
    b: bool,
        `v` in the format of bool

    References
    ----------
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    """
    if isinstance(v, bool):
        b = v
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        b = True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        b = False
    else:
        raise ValueError("Boolean value expected.")
    return b


def diff_with_step(a: np.ndarray, step: int = 1, **kwargs) -> np.ndarray:
    """

    compute a[n+step] - a[n] for all valid n

    Parameters
    ----------
    a: ndarray,
        the input data
    step: int, default 1,
        the step to compute the difference
    kwargs: dict,

    Returns
    -------
    d: ndarray:
        the difference array

    """
    if step >= len(a):
        raise ValueError(
            f"step ({step}) should be less than the length ({len(a)}) of `a`"
        )
    d = a[step:] - a[:-step]
    return d


def ms2samples(t: Real, fs: Real) -> int:
    """

    convert time `t` with units in ms to number of samples

    Parameters
    ----------
    t: real number,
        time with units in ms
    fs: real number,
        sampling frequency of a signal

    Returns
    -------
    n_samples: int,
        number of samples corresponding to time `t`

    """
    n_samples = t * fs // 1000
    return n_samples


def samples2ms(n_samples: int, fs: Real) -> Real:
    """

    inverse function of `ms2samples`

    Parameters
    ----------
    n_samples: int,
        number of sample points
    fs: real number,
        sampling frequency of a signal

    Returns
    -------
    t: real number,
        time duration correponding to `n_samples`

    """
    t = n_samples * 1000 / fs
    return t


def plot_single_lead(
    t: np.ndarray,
    sig: np.ndarray,
    ax: Optional[Any] = None,
    ticks_granularity: int = 0,
    **kwargs,
) -> NoReturn:
    """finished, NOT checked,

    Parameters
    ----------
    t: ndarray,
        the array of time of the signal
    sig: ndarray,
        the signal itself
    ax: Artist, optional,
        the `Artist` to plot on
    ticks_granularity: int, default 0,
        the granularity to plot axis ticks, the higher the more,
        0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)

    """
    if "plt" not in dir():
        import matplotlib.pyplot as plt
    palette = {
        "p_waves": "green",
        "qrs": "red",
        "t_waves": "pink",
    }
    plot_alpha = 0.4
    y_range = np.max(np.abs(sig)) + 100
    if ax is None:
        fig_sz_w = int(round(4.8 * (t[-1] - t[0])))
        fig_sz_h = 6 * y_range / 1500
        fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
    label = kwargs.get("label", None)
    if label:
        ax.plot(t, sig, label=kwargs.get("label"))
    else:
        ax.plot(t, sig)
    ax.axhline(y=0, linestyle="-", linewidth="1.0", color="red")
    # NOTE that `Locator` has default `MAXTICKS` equal to 1000
    if ticks_granularity >= 1:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")
    if ticks_granularity >= 2:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
        ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

    waves = kwargs.get("waves", {"p_waves": [], "qrs": [], "t_waves": []})
    for w, l_itv in waves.items():
        for itv in l_itv:
            ax.axvspan(itv[0], itv[1], color=palette[w], alpha=plot_alpha)
    if label:
        ax.legend(loc="upper left")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-y_range, y_range)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [μV]")


def init_logger(
    log_dir: Union[str, Path],
    log_file: Optional[str] = None,
    log_name: Optional[str] = None,
    mode: str = "a",
    verbose: int = 0,
) -> logging.Logger:
    """

    Parameters
    ----------
    log_dir: str or Path,
        directory of the log file
    log_file: str, optional,
        name of the log file
    log_name: str, optional,
        name of the logger
    mode: str, default "a",
        mode of writing the log file, can be one of "a", "w"
    verbose: int, default 0,
        log verbosity

    Returns
    -------
    logger: Logger

    """
    if log_file is None:
        log_file = f"log_{get_date_str()}.txt"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_file
    print(f"log file path: {str(log_file)}")

    logger = logging.getLogger(
        log_name or DEFAULTS.prefix
    )  # "ECG" to prevent from using the root logger

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(str(log_file))

    if verbose >= 2:
        print("levels of c_handler and f_handler are set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        print("level of c_handler is set INFO, level of f_handler is set DEBUG")
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        print("level of c_handler is set WARNING, level of f_handler is set INFO")
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def get_date_str(fmt: Optional[str] = None):
    """

    Parameters
    ----------
    fmt: str, optional,
        format of the string of date

    Returns
    -------
    date_str: str,
        current time in the `str` format

    """
    now = datetime.datetime.now()
    date_str = now.strftime(fmt or "%m-%d_%H-%M")
    return date_str


def list_sum(lst: Sequence[list]) -> list:
    """

    Parameters
    ----------
    lst: sequence of list,
        the sequence of lists to obtain the summation

    Returns
    -------
    l_sum: list,
        sum of `lst`,
        i.e. if lst = [list1, list2, ...], then l_sum = list1 + list2 + ...

    """
    l_sum = reduce(lambda a, b: a + b, lst, [])
    return l_sum


def read_log_txt(
    fp: str,
    epoch_startswith: str = "Train epoch_",
    scalar_startswith: Union[str, Iterable[str]] = "train/|test/",
) -> pd.DataFrame:
    """

    read from log txt file, in case tensorboard not working

    Parameters
    ----------
    fp: str,
        path to the log txt file
    epoch_startswith: str,
        indicator of the start of the start of an epoch
    scalar_startswith: str or iterable of str,
        indicators of the scalar recordings,
        if is str, should be indicators separated by "|"

    Returns
    -------
    summary: DataFrame,
        scalars summary, in the format of a pandas DataFrame

    """
    with open(fp, "r") as f:
        content = f.read().splitlines()
    if isinstance(scalar_startswith, str):
        field_pattern = f"^({scalar_startswith})"
    else:
        field_pattern = f"""^({"|".join(scalar_startswith)})"""
    summary = []
    new_line = None
    for line in content:
        if line.startswith(epoch_startswith):
            if new_line:
                summary.append(new_line)
            epoch = re.findall("[\\d]+", line)[0]
            new_line = {"epoch": epoch}
        if re.findall(field_pattern, line):
            field, val = line.split(":")
            field = field.strip()
            val = float(val.strip())
            new_line[field] = val
    summary.append(new_line)
    summary = pd.DataFrame(summary)
    return summary


def read_event_scalars(
    fp: str, keys: Optional[Union[str, Iterable[str]]] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """

    read scalars from event file, in case tensorboard not working

    Parameters
    ----------
    fp: str,
        path to the event file
    keys: str or iterable of str, optional,
        field names of the scalars to read,
        if is None, scalars of all fields will be read

    Returns
    -------
    summary: DataFrame or dict of DataFrame
        the wall_time, step, value of the scalars

    """
    try:
        from tensorflow.python.summary.event_accumulator import EventAccumulator
    except Exception:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    event_acc = EventAccumulator(fp)
    event_acc.Reload()
    if keys:
        if isinstance(keys, str):
            _keys = [keys]
        else:
            _keys = keys
    else:
        _keys = event_acc.scalars.Keys()
    summary = {}
    for k in _keys:
        df = pd.DataFrame(
            [
                [item.wall_time, item.step, item.value]
                for item in event_acc.scalars.Items(k)
            ]
        )
        df.columns = ["wall_time", "step", "value"]
        summary[k] = df
    if isinstance(keys, str):
        summary = summary[k]
    return summary


def dicts_equal(d1: dict, d2: dict) -> bool:
    """

    Parameters
    ----------
    d1, d2: dict,
        the two dicts to compare equality

    Returns
    -------
    bool, True if `d1` equals `d2`

    NOTE
    ----
    the existence of numpy array, torch Tensor, pandas DataFrame and Series would probably
    cause errors when directly use the default `__eq__` method of dict,
    for example `{"a": np.array([1,2])} == {"a": np.array([1,2])}` would raise the following
    ```python
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ```

    Example
    -------
    >>> d1 = {"a": pd.DataFrame([{"hehe":1,"haha":2}])[["haha","hehe"]]}
    >>> d2 = {"a": pd.DataFrame([{"hehe":1,"haha":2}])[["hehe","haha"]]}
    >>> dicts_equal(d1, d2)
    True

    """
    import torch

    if len(d1) != len(d2):
        return False
    for k, v in d1.items():
        if k not in d2 or not isinstance(d2[k], type(v)):
            return False
        if isinstance(v, dict):
            if not dicts_equal(v, d2[k]):
                return False
        elif isinstance(v, np.ndarray):
            if v.shape != d2[k].shape or not (v == d2[k]).all():
                return False
        elif isinstance(v, torch.Tensor):
            if v.shape != d2[k].shape or not (v == d2[k]).all().item():
                return False
        elif isinstance(v, pd.DataFrame):
            if v.shape != d2[k].shape or set(v.columns) != set(d2[k].columns):
                # consider: should one check index be equal?
                return False
            # for c in v.columns:
            #     if not (v[c] == d2[k][c]).all():
            #         return False
            if not (v.values == d2[k][v.columns].values).all():
                return False
        elif isinstance(v, pd.Series):
            if v.shape != d2[k].shape or v.name != d2[k].name:
                return False
            if not (v == d2[k]).all():
                return False
        # TODO: consider whether there are any other dtypes that should be treated similarly
        else:  # other dtypes whose equality can be checked directly
            if v != d2[k]:
                return False
    return True


def default_class_repr(c: object, align: str = "center", depth: int = 1) -> str:
    """

    Parameters
    ----------
    c: object,
        the object to be represented
    align: str, default "center",
        the alignment of the class arguments
    depth: int, default 1,
        the depth of the class arguments to display

    Returns
    -------
    str,
        the representation of the class

    """
    indent = 4 * depth * " "
    closing_indent = 4 * (depth - 1) * " "
    if not hasattr(c, "extra_repr_keys"):
        return repr(c)
    elif len(c.extra_repr_keys()) > 0:
        max_len = max([len(k) for k in c.extra_repr_keys()])
        extra_str = (
            "(\n"
            + ",\n".join(
                [
                    f"""{indent}{k.ljust(max_len, " ") if align.lower() in ["center", "c"] else k} = {default_class_repr(eval(f"c.{k}"),align,depth+1)}"""
                    for k in c.__dir__()
                    if k in c.extra_repr_keys()
                ]
            )
            + f"{closing_indent}\n)"
        )
    else:
        extra_str = ""
    return f"{c.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """
    Mixin for enhanced __repr__ and __str__ methods.

    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """ """
        return []


class MovingAverage(object):
    """to be improved,

    moving average

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Moving_average

    """

    def __init__(self, data: Optional[Sequence] = None, **kwargs: Any) -> NoReturn:
        """
        Parameters
        ----------
        data: array_like,
            the series data to compute its moving average
        kwargs: auxilliary key word arguments

        """
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data)
        self.verbose = kwargs.get("verbose", 0)

    def __call__(
        self, data: Optional[Sequence] = None, method: str = "ema", **kwargs: Any
    ) -> np.ndarray:
        """
        Parameters
        ----------
        method: str,
            method for computing moving average, can be one of
            - "sma", "simple", "simple moving average"
            - "ema", "ewma", "exponential", "exponential weighted", "exponential moving average", "exponential weighted moving average"
            - "cma", "cumulative", "cumulative moving average"
            - "wma", "weighted", "weighted moving average"

        """
        m = method.lower().replace("_", " ")
        if m in ["sma", "simple", "simple moving average"]:
            func = self._sma
        elif m in [
            "ema",
            "ewma",
            "exponential",
            "exponential weighted",
            "exponential moving average",
            "exponential weighted moving average",
        ]:
            func = self._ema
        elif m in ["cma", "cumulative", "cumulative moving average"]:
            func = self._cma
        elif m in ["wma", "weighted", "weighted moving average"]:
            func = self._wma
        else:
            raise NotImplementedError
        if data is not None:
            self.data = np.array(data)
        return func(**kwargs)

    def _sma(self, window: int = 5, center: bool = False, **kwargs: Any) -> np.ndarray:
        """
        simple moving average

        Parameters
        ----------
        window: int, default 5,
            window length of the moving average
        center: bool, default False,
            if True, when computing the output value at each point, the window will be centered at that point;
            otherwise the previous `window` points of the current point will be used

        """
        smoothed = []
        if center:
            hw = window // 2
            window = hw * 2 + 1
        for n in range(window):
            smoothed.append(np.mean(self.data[: n + 1]))
        prev = smoothed[-1]
        for n, d in enumerate(self.data[window:]):
            s = prev + (d - self.data[n]) / window
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        if center:
            smoothed[hw:-hw] = smoothed[window - 1 :]
            for n in range(hw):
                smoothed[n] = np.mean(self.data[: n + hw + 1])
                smoothed[-n - 1] = np.mean(self.data[-n - hw - 1 :])
        return smoothed

    def _ema(self, weight: float = 0.6, **kwargs: Any) -> np.ndarray:
        """
        exponential moving average,
        which is also the function used in Tensorboard Scalar panel,
        whose parameter `smoothing` is the `weight` here

        Parameters
        ----------
        weight: float, default 0.6,
            weight of the previous data point

        """
        smoothed = []
        prev = self.data[0]
        for d in self.data:
            s = prev * weight + (1 - weight) * d
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _cma(self, **kwargs) -> np.ndarray:
        """
        cumulative moving average

        """
        smoothed = []
        prev = 0
        for n, d in enumerate(self.data):
            s = prev + (d - prev) / (n + 1)
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _wma(self, window: int = 5, **kwargs: Any) -> np.ndarray:
        """
        weighted moving average

        Parameters
        ----------
        window: int, default 5,
            window length of the moving average

        """
        conv = np.arange(1, window + 1)[::-1]
        deno = np.sum(conv)
        smoothed = np.convolve(conv, self.data, mode="same") / deno
        return smoothed


def nildent(text: str) -> str:
    """

    kill all leading white spaces in each line of `text`,
    while keeping all lines (including empty)

    Parameters
    ----------
    text: str,
        text to be processed

    Returns
    -------
    new_text: str,
        processed text

    """
    new_text = "\n".join([line.lstrip() for line in text.splitlines()]) + (
        "\n" if text.endswith("\n") else ""
    )
    return new_text


def isclass(obj: Any) -> bool:
    """

    Parameters
    ----------
    obj: any object,
        any object, including class, instance of class, etc

    Returns
    -------
    bool:
        True if `obj` is a class, False otherwise

    """
    try:
        return issubclass(obj, object)
    except TypeError:
        return False


def add_docstring(doc: str, mode: str = "replace") -> Callable:
    """
    decorator to add docstring to a function

    Parameters
    ----------
    doc: str,
        the docstring to be added
    mode: str, default "replace",
        the mode of the docstring,
        can be "replace", "append" or "prepend",
        case insensitive

    """

    def decorator(func: Callable) -> Callable:
        """ """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
            """ """
            return func(*args, **kwargs)

        pattern = "(\\s^\n){1,}"
        if mode.lower() == "replace":
            wrapper.__doc__ = doc
        elif mode.lower() == "append":
            tmp = re.sub(pattern, "", wrapper.__doc__)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            wrapper.__doc__ += new_lines + doc
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", wrapper.__doc__)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            wrapper.__doc__ = doc + new_lines + wrapper.__doc__
        else:
            raise ValueError(f"mode {mode} is not supported")
        return wrapper

    return decorator


def deprecate_kwargs(l_kwargs: Sequence[Sequence[str]]):
    """

    decorator to deprecate old kwargs in a function

    Parameters
    ----------
    l_kwargs: Sequence[Sequence[str]],
        a list of kwargs to be deprecated,
        each element is a sequence of length 2,
        of the form (new_kwarg, old_kwarg)

    """

    def decorator(func: Callable) -> Callable:
        """ """

        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable:
            """ """
            old_kwargs = deepcopy(kwargs)
            for new_kw, old_kw in l_kwargs:
                if new_kw in kwargs:
                    old_kwargs.pop(new_kw, None)
                    old_kwargs[old_kw] = kwargs[new_kw]
                elif old_kw in kwargs:
                    warnings.warn(
                        f"key word argument \042{old_kw}\042 is deprecated, use \042{new_kw}\042 instead"
                    )
            return func(*args, **old_kwargs)

        func_params = list(inspect.signature(func).parameters.values())
        func_param_names = list(inspect.signature(func).parameters.keys())
        for new_kw, old_kw in l_kwargs:
            idx = func_param_names.index(old_kw)
            func_params[idx] = func_params[idx].replace(name=new_kw)
            wrapper.__doc__ = func.__doc__.replace(old_kw, new_kw)
        wrapper.__signature__ = inspect.Signature(parameters=func_params)
        return wrapper

    return decorator


@contextmanager
def timeout(duration: float):
    """
    A context manager that raises a `TimeoutError` after a specified time.

    Parameters
    ----------
    duration: float,
        the time duration in seconds,
        should be non-negative,
        0 for no timeout

    References
    ----------
    https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if np.isinf(duration):
        duration = 0
    elif duration < 0:
        raise ValueError("duration must be non-negative")
    elif duration > 0:  # granularity is 1 second, so round up
        duration = max(1, int(duration))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


class Timer(ReprMixin):
    """

    Context manager to time the execution of a block of code.

    Usage
    -----
    >>> with Timer("task name", verbose=2) as timer:
    >>>     do_something()
    >>>     timer.add_time("subtask 1", level=2)
    >>>     do_subtask_1()
    >>>     timer.stop_timer("subtask 1")
    >>>     timer.add_time("subtask 2", level=2)
    >>>     do_subtask_2()
    >>>     timer.stop_timer("subtask 2")
    >>>     do_something_else()

    """

    __name__ = "Timer"

    def __init__(self, name: Optional[str] = None, verbose: int = 0) -> NoReturn:
        """

        Parameters
        ----------
        name: str, optional
            the name of the timer, defaults to "default timer"
        verbose: int, default 0
            the verbosity level of the timer,

        """
        self.name = name or "default timer"
        self.verbose = verbose
        self.timers = {self.name: 0.0}
        self.ends = {self.name: 0.0}
        self.levels = {self.name: 1}

    def __enter__(self) -> "Timer":
        self.timers = {self.name: time.perf_counter()}
        self.ends = {self.name: 0.0}
        self.levels = {self.name: 1}
        return self

    def __exit__(self, *args) -> NoReturn:
        for k in self.timers:
            self.stop_timer(k)
            self.timers[k] = self.ends[k] - self.timers[k]

    def add_timer(self, name: str, level: int = 1) -> NoReturn:
        """
        add a new timer for some subtask

        Parameters
        ----------
        name: str,
            the name of the timer to be added
        level: int, default 1
            the verbosity level of the timer,

        """
        self.timers[name] = time.perf_counter()
        self.ends[name] = 0
        self.levels[name] = level

    def stop_timer(self, name: str) -> NoReturn:
        """
        stop a timer

        Parameters
        ----------
        name: str,
            the name of the timer to be stopped

        """
        if self.ends[name] == 0:
            self.ends[name] = time.perf_counter()
            if self.verbose >= self.levels[name]:
                time_cost, unit = self._simplify_time_expr(
                    self.ends[name] - self.timers[name]
                )
                print(f"{name} took {time_cost:.4f} {unit}")

    def _simplify_time_expr(self, time_cost: float) -> Tuple[float, str]:
        """
        simplify the time expression

        Parameters
        ----------
        time_cost: float,
            the time cost, with units in seconds

        Returns
        -------
        time_cost: float,
            the time cost,
        unit: str,
            the unit of the time cost

        """
        if time_cost <= 0.1:
            return 1000 * time_cost, "ms"
        return time_cost, "s"

    def extra_repr_keys(self) -> List[str]:
        return ["name", "verbose"]
