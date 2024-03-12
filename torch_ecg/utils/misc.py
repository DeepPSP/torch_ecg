"""
"""

import datetime
import inspect
import logging
import os
import re
import signal
import sys
import time
import types
import warnings
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce, wraps
from glob import glob
from numbers import Number, Real
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from bib_lookup import CitationMixin as _CitationMixin
from deprecated import deprecated

from ..cfg import _DATA_CACHE, DEFAULTS

__all__ = [
    "get_record_list_recursive",
    "get_record_list_recursive2",
    "get_record_list_recursive3",
    "dict_to_str",
    "str2bool",
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
    "CitationMixin",
    "MovingAverage",
    "nildent",
    "add_docstring",
    "remove_parameters_returns_from_docstring",
    "timeout",
    "Timer",
    "get_kwargs",
    "get_required_args",
    "add_kwargs",
    "make_serializable",
    "np_topk",
]


def get_record_list_recursive(db_dir: Union[str, bytes, os.PathLike], rec_ext: str, relative: bool = True) -> List[str]:
    """Get the list of records in a recursive manner.

    For example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1";
    "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system.

    Parameters
    ----------
    db_dir : `path-like`
        The parent (root) path of to search for records.
    rec_ext : str
        Extension of the record files.
    relative : bool, default True
        Whether to return the relative path of the records.

    Returns
    -------
    List[str]
        The list of records, in lexicographical order.

    """
    if not rec_ext.startswith("."):
        res = Path(db_dir).rglob(f"*.{rec_ext}")
    else:
        res = Path(db_dir).rglob(f"*{rec_ext}")
    res = [str((item.relative_to(db_dir) if relative else item).with_suffix("")) for item in res if str(item).endswith(rec_ext)]
    res = sorted(res)

    return res


@deprecated(reason="use `get_record_list_recursive3` instead")
def get_record_list_recursive2(db_dir: Union[str, bytes, os.PathLike], rec_pattern: str) -> List[str]:
    """Get the list of records in a recursive manner.

    For example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1";
    "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system.

    Parameters
    ----------
    db_dir : `path-like`
        The parent (root) path of to search for records.
    rec_pattern : str
        Pattern of the record filenames, e.g. ``"A*.mat"``.

    Returns
    -------
    List[str]
        The list of records, in lexicographical order.

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
    res = [os.path.splitext(item)[0].replace(str(db_dir), "").strip(os.sep) for item in res]
    res = sorted(res)

    return res


def get_record_list_recursive3(
    db_dir: Union[str, bytes, os.PathLike],
    rec_patterns: Union[str, Dict[str, str]],
    relative: bool = True,
) -> Union[List[str], Dict[str, List[str]]]:
    """Get the list of records in a recursive manner.

    For example, there are two folders "patient1", "patient2" in `db_dir`,
    and there are records "A0001", "A0002", ... in "patient1";
    "B0001", "B0002", ... in "patient2",
    then the output would be "patient1{sep}A0001", ..., "patient2{sep}B0001", ...,
    sep is determined by the system.

    Parameters
    ----------
    db_dir : `path-like`
        The parent (root) path of to search for records.
    rec_patterns : str or dict
        Pattern of the record filenames, e.g. ``"A(?:\\d+).mat"``,
        or patterns of several subsets, e.g. ``{"A": "A(?:\\d+).mat"}``
    relative : bool, default True
        Whether to return the relative path of the records.

    Returns
    -------
    List[str] or dict
        The list of records, in lexicographical order.

    """
    if isinstance(rec_patterns, str):
        res = []
    elif isinstance(rec_patterns, dict):
        res = {k: [] for k in rec_patterns.keys()}
    _db_dir = Path(db_dir).resolve()  # make absolute
    roots = [_db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = os.listdir(r)
            if isinstance(rec_patterns, str):
                res += [r / item for item in filter(re.compile(rec_patterns).search, tmp)]
            elif isinstance(rec_patterns, dict):
                for k in rec_patterns.keys():
                    res[k] += [r / item for item in filter(re.compile(rec_patterns[k]).search, tmp)]
            new_roots += [r / item for item in tmp if (r / item).is_dir()]
        roots = deepcopy(new_roots)
    if isinstance(rec_patterns, str):
        res = [str((item.relative_to(_db_dir) if relative else item).with_suffix("")) for item in res]
        res = sorted(res)
    elif isinstance(rec_patterns, dict):
        for k in rec_patterns.keys():
            res[k] = [str((item.relative_to(_db_dir) if relative else item).with_suffix("")) for item in res[k]]
            res[k] = sorted(res[k])
    return res


def dict_to_str(d: Union[dict, list, tuple], current_depth: int = 1, indent_spaces: int = 4) -> str:
    """Convert a (possibly) nested dict into a `str` of json-like formatted form.

    This nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters
    ----------
    d : dict or list or tuple
        A (possibly) nested :class:`dict`, or a list of :class:`dict`.
    current_depth : int, default 1
        Depth of `d` in the (possible) parent :class:`dict` or :class:`list`.
    indent_spaces : int, default 4
        The indent spaces of each depth.

    Returns
    -------
    str
        The formatted string.

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
    """Converts a "boolean" value possibly
    in the format of :class:`str` to :class:`bool`.

    Modified from [#str2bool]_.

    Parameters
    ----------
    v : str or bool
        The "boolean" value.

    Returns
    -------
    bool
        `v` in the format of a bool.

    References
    ----------
    .. [#str2bool] https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

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


@deprecated("Use `np.diff` instead.")
def diff_with_step(a: np.ndarray, step: int = 1) -> np.ndarray:
    """Compute ``a[n+step] - a[n]`` for all valid `n`.

    Parameters
    ----------
    a : numpy.ndarray
        The input data.
    step : int, default 1
        The step size to compute the difference.

    Returns
    -------
    numpy.ndarray
        The difference array.

    """
    if step >= len(a):
        raise ValueError(f"`step` ({step}) should be less than the length ({len(a)}) of `a`")
    d = a[step:] - a[:-step]
    return d


def ms2samples(t: Real, fs: Real) -> int:
    """Convert time duration in ms to number of samples.

    Parameters
    ----------
    t : numbers.Real
        Time duration in ms.
    fs : numbers.Real
        Sampling frequency.

    Returns
    -------
    n_samples : int
        Number of samples converted from `t`,
        with sampling frequency `fs`.

    """
    n_samples = t * fs // 1000
    return n_samples


def samples2ms(n_samples: int, fs: Real) -> Real:
    """Convert number of samples to time duration in ms.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    fs : numbers.Real
        Sampling frequency.

    Returns
    -------
    t : numbers.Real
        Time duration in ms converted from `n_samples`,
        with sampling frequency `fs`.

    """
    t = n_samples * 1000 / fs
    return t


def plot_single_lead(
    t: np.ndarray,
    sig: np.ndarray,
    ax: Optional[Any] = None,
    ticks_granularity: int = 0,
    **kwargs,
) -> None:
    """Plot single lead ECG signal.

    Parameters
    ----------
    t : numpy.ndarray
        The array of time points.
    sig : numpy.ndarray
        The signal itself.
    ax : matplotlib.axes.Axes, default None
        The axes to plot on.
    ticks_granularity : int, default 0
        Granularity to plot axis ticks, the higher the more ticks.
        0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)

    Returns
    -------
    None

    """
    if "plt" not in dir():
        import matplotlib.pyplot as plt
    palette = {
        "p_waves": "cyan",
        "qrs": "green",
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
    log_dir: Optional[Union[str, Path, bool]] = None,
    log_file: Optional[str] = None,
    log_name: Optional[str] = None,
    suffix: Optional[str] = None,
    mode: str = "a",
    verbose: int = 0,
) -> logging.Logger:
    """Initialize a logger.

    Parameters
    ----------
    log_dir : `path-like` or bool, optional
        Directory of the log file,
        default to `DEFAULTS.log_dir`.
        If is `False`, then no log file will be created.
    log_file : str, optional
        Name of the log file,
        default to ``{DEFAULTS.prefix}-log-{get_date_str()}.txt``.
    log_name : str, optional
        Name of the logger.
    suffix : str, optional
        Suffix of the logger name.
        Ignored if `log_name` is not `None`.
    mode : {"a", "w"}, default "a"
        Mode to open the log file.
    verbose : int, default 0
        Verbosity level for the logger.

    Returns
    -------
    logger : logging.Logger
        The logger.

    """
    if log_dir is False:
        log_file = None
    else:
        if log_file is None:
            log_file = f"{DEFAULTS.prefix}-log-{get_date_str()}.txt"
        log_dir = Path(log_dir).expanduser().resolve() if log_dir is not None else DEFAULTS.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / log_file
        print(f"log file path: {str(log_file)}")

    log_name = (log_name or DEFAULTS.prefix) + (f"-{suffix}" if suffix else "")
    # if a logger with the same name already exists, remove it
    if log_name in logging.root.manager.loggerDict:
        logging.getLogger(log_name).handlers = []
    logger = logging.getLogger(log_name)  # to prevent from using the root logger

    c_handler = logging.StreamHandler(sys.stdout)
    if log_file is not None:
        f_handler = logging.FileHandler(str(log_file))

    if verbose >= 2:
        # print("level of `c_handler` is set DEBUG")
        c_handler.setLevel(logging.DEBUG)
        if log_file is not None:
            # print("level of `f_handler` is set DEBUG")
            f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        # print("level of `c_handler` is set INFO")
        c_handler.setLevel(logging.INFO)
        if log_file is not None:
            # print("level of `f_handler` is set DEBUG")
            f_handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        # print("level of `c_handler` is set WARNING")
        c_handler.setLevel(logging.WARNING)
        if log_file is not None:
            # print("level of `f_handler` is set INFO")
            f_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if log_file is not None:
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger


def get_date_str(fmt: Optional[str] = None):
    """Get the current time in the :class:`str` format.

    Parameters
    ----------
    fmt : str, optional
        Format of the string of date,
        default to ``"%m-%d_%H-%M"``.

    Returns
    -------
    str
        Current time in the :class:`str` format.

    """
    now = datetime.datetime.now()
    date_str = now.strftime(fmt or "%m-%d_%H-%M")
    return date_str


def list_sum(lst: Sequence[list]) -> list:
    """Sum a sequence of lists.

    Parameters
    ----------
    lst : Sequence[list]
        The sequence of lists to obtain the summation.

    Returns
    -------
    list
        sum of `lst`,
        i.e. if ``lst = [list1, list2, ...]``,
        then ``l_sum = list1 + list2 + ...``.

    """
    l_sum = reduce(lambda a, b: a + b, lst, [])
    return l_sum


def read_log_txt(
    fp: Union[str, bytes, os.PathLike],
    epoch_startswith: str = "Train epoch_",
    scalar_startswith: Union[str, Iterable[str]] = "train/|test/",
) -> pd.DataFrame:
    """Read from log txt file, in case tensorboard not working.

    Parameters
    ----------
    fp : `path-like`
        Path to the log txt file.
    epoch_startswith : str, default "Train epoch_"
        Indicator of the start of the start of an epoch
    scalar_startswith : str or Iterable[str], default "train/|test/"
        Indicators of the scalar recordings.
        If is :class:`str`, should be indicators separated by ``"|"``.

    Returns
    -------
    summary : pandas.DataFrame
        Scalars summary, in the format of a :class:`~pandas.DataFrame`.

    """
    content = Path(fp).read_text().splitlines()
    if isinstance(scalar_startswith, str):
        field_pattern = f"({scalar_startswith})"
    else:
        field_pattern = f"""({"|".join(scalar_startswith)})"""
    summary = []
    new_line = None
    for line in content:
        if re.findall(f"{epoch_startswith}([\\d]+)", line):
            if new_line:
                summary.append(new_line)
            epoch = re.findall(f"{epoch_startswith}([\\d]+)", line)[0]
            new_line = {"epoch": epoch}
        if re.findall(field_pattern, line):
            field, val = line.split(":")[-2:]
            field = field.strip()
            val = float(val.strip())
            new_line[field] = val
    summary.append(new_line)
    summary = pd.DataFrame(summary)
    return summary


def read_event_scalars(
    fp: Union[str, bytes, os.PathLike], keys: Optional[Union[str, Iterable[str]]] = None
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Read scalars from event file, in case tensorboard not working.

    Parameters
    ----------
    fp : `path-like`
        Path to the event file.
    keys : str or Iterable[str], optional
        Field names of the scalars to read.
        If is None, scalars of all fields will be read.

    Returns
    -------
    summary : pandas.DataFrame or dict of pandas.DataFrame
        The wall_time, step, value of the scalars.

    """
    try:
        from tensorflow.python.summary.event_accumulator import EventAccumulator
    except Exception:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception:
            raise ImportError("cannot import `EventAccumulator` from `tensorflow` or `tensorboard`")
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
        df = pd.DataFrame([[item.wall_time, item.step, item.value] for item in event_acc.scalars.Items(k)])
        df.columns = ["wall_time", "step", "value"]
        summary[k] = df
    if isinstance(keys, str):
        summary = summary[k]
    return summary


def dicts_equal(d1: dict, d2: dict, allow_array_diff_types: bool = True) -> bool:
    """Determine if two dicts are equal.

    Parameters
    ----------
    d1, d2 : dict
        The two dicts to compare equality.
    allow_array_diff_types : bool, default True
        Whether allow the equality of two arrays with different types,
        including `list`, `tuple`, `numpy.ndarray`, `torch.Tensor`,
        **NOT** including `pandas.DataFrame`, `pandas.Series`.

    Returns
    -------
    bool
        True if `d1` equals `d2`, False otherwise.

    NOTE
    ----
    The existence of :class:`~numpy.ndarray`, :class:`~torch.Tensor`,
    :class:`~pandas.DataFrame` and :class:`~pandas.Series` would probably
    cause errors when directly use the default ``__eq__`` method of :class:`dict`
    For example:

    .. code-block:: python

        >>> {"a": np.array([1,2])} == {"a": np.array([1,2])}
        ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

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
        if k not in d2:
            return False
        if not allow_array_diff_types and not isinstance(d2[k], type(v)):
            return False
        if allow_array_diff_types and isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
            if not isinstance(d2[k], (list, tuple, np.ndarray, torch.Tensor)):
                return False
            if not np.array_equal(v, d2[k]):
                return False
        elif allow_array_diff_types and not isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
            if not isinstance(d2[k], type(v)):
                return False
        if isinstance(v, dict):
            if not dicts_equal(v, d2[k]):
                return False
        elif isinstance(v, (list, tuple, np.ndarray, torch.Tensor)):
            return np.array_equal(v, d2[k])
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


def add_docstring(doc: str, mode: str = "replace") -> Callable:
    """Decorator to add docstring to a function or a class.

    Parameters
    ----------
    doc : str
        The docstring to be added.
    mode : {"replace", "append", "prepend"}, optional
        The mode of the adding to the original docstring,
        by default "replace", case insensitive.

    """

    def decorator(func_or_cls: Callable) -> Callable:
        if func_or_cls.__doc__ is None:
            func_or_cls.__doc__ = ""
        pattern = "(\\s^\n){1,}"
        if mode.lower() == "replace":
            func_or_cls.__doc__ = doc
        elif mode.lower() == "append":
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", doc)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ += new_lines + doc
        elif mode.lower() == "prepend":
            tmp = re.sub(pattern, "", doc)
            new_lines = 1 - (len(tmp) - len(tmp.rstrip("\n")))
            tmp = re.sub(pattern, "", func_or_cls.__doc__)
            new_lines -= len(tmp) - len(tmp.lstrip("\n"))
            new_lines = max(0, new_lines) * "\n"
            func_or_cls.__doc__ = doc + new_lines + func_or_cls.__doc__
        else:
            raise ValueError(f"mode `{mode}` is not supported")
        return func_or_cls

    return decorator


def default_class_repr(c: object, align: str = "center", depth: int = 1) -> str:
    """Default class representation.

    Parameters
    ----------
    c : object
        The object to be represented.
    align : str, default "center"
        Alignment of the class arguments.
    depth : int, default 1
        Depth of the class arguments to be displayed.

    Returns
    -------
    str
        The representation of the class.

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
    """Mixin class for enhanced
    :meth:`__repr__` and :meth:`__str__` methods.
    """

    def __repr__(self) -> str:
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self) -> List[str]:
        """Extra keys for :meth:`__repr__` and :meth:`__str__`."""
        return []


class CitationMixin(_CitationMixin):
    """Mixin class for getting citations from DOIs."""

    # backwar compatibility
    if (_DATA_CACHE / "database_citation.csv").exists():
        try:
            df_old = pd.read_csv(_DATA_CACHE / "database_citation.csv")
        except pd.errors.EmptyDataError:
            df_old = pd.DataFrame(columns=["doi", "citation"])
        if set(df_old.columns) != set(["doi", "citation"]):
            df_old = pd.DataFrame(columns=["doi", "citation"])
        df_old = df_old[["doi", "citation"]]
        if _CitationMixin.citation_cache.exists():
            df = pd.read_csv(_CitationMixin.citation_cache)
        else:
            df = pd.DataFrame(columns=["doi", "citation"])
        # merge the old and new tables and drop duplicates
        df = pd.concat([df, df_old], axis=0, ignore_index=True)
        df = df.drop_duplicates(subset="doi", keep="first")
        df = df.reset_index(drop=True)
        df.to_csv(_CitationMixin.citation_cache, index=False)
        del df_old, df
        # delete the old cache
        (_DATA_CACHE / "database_citation.csv").unlink()

    def get_citation(
        self,
        lookup: bool = True,
        format: Optional[str] = None,
        style: Optional[str] = None,
        timeout: Optional[float] = None,
        print_result: bool = True,
    ) -> Union[str, type(None)]:
        """Get bib citation from DOIs.

        Overrides the default method to make the `print_result` argument
        have default value ``True``.

        Parameters
        ----------
        lookup : bool, default True
            Whether to look up the citation from the cache.
        format : str, optional
            The format of the citation. If not specified, the citation
            will be returned in the default format (bibtex).
        style : str, optional
            The style of the citation. If not specified, the citation
            will be returned in the default style (apa).
            Valid only when `format` is ``"text"``.
        timeout : float, optional
            The timeout for the request.
        print_result : bool, default True
            Whether to print the citation.

        Returns
        -------
        str or None
            bib citation(s) from the DOI(s),
            or None if `print_result` is True.

        """
        return super().get_citation(
            lookup=lookup,
            format=format,
            style=style,
            timeout=timeout,
            print_result=print_result,
        )


class MovingAverage(object):
    """Class for computing moving average.

    For more information, see [#ma_wiki]_.

    Parameters
    ----------
    data : array_like, optional
        The series data to compute its moving average.
    kwargs : dict, optional
        Auxilliary keyword arguments

    References
    ----------
    .. [#ma_wiki] https://en.wikipedia.org/wiki/Moving_average

    """

    def __init__(self, data: Optional[Sequence] = None, **kwargs: Any) -> None:
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data)
        self.verbose = kwargs.get("verbose", 0)

    def __call__(self, data: Optional[Sequence] = None, method: str = "ema", **kwargs: Any) -> np.ndarray:
        """Compute moving average.

        Parameters
        ----------
        data : array_like, optional
            The series data to compute its moving average.
        method : str
            method for computing moving average, can be one of

                - "sma", "simple", "simple moving average";
                - "ema", "ewma", "exponential", "exponential weighted",
                  "exponential moving average", "exponential weighted moving average";
                - "cma", "cumulative", "cumulative moving average";
                - "wma", "weighted", "weighted moving average".

        kwargs : dict, optional
            Keyword arguments for the specific moving average method.

        Returns
        -------
        ma : numpy.ndarray
            The moving average of the input data.

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
            raise NotImplementedError(f"method `{method}` is not implemented yet")
        if data is not None:
            self.data = np.array(data)
        return func(**kwargs)

    def _sma(self, window: int = 5, center: bool = False, **kwargs: Any) -> np.ndarray:
        """Simple moving average.

        Parameters
        ----------
        window : int, default 5
            Window length of the moving average
        center : bool, default False
            If True, when computing the output value at each point,
            the window will be centered at that point;
            otherwise the previous `window` points of the current point will be used.

        Returns
        -------
        numpy.ndarray
            The simple moving average of the input data.

        """
        if len(kwargs) > 0:
            warnings.warn(
                f"the following arguments are not used: `{kwargs}` for simple moving average",
                RuntimeWarning,
            )
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
        """Exponential moving average

        This is also the function used in Tensorboard Scalar panel,
        whose parameter `smoothing` is the `weight` here.

        Parameters
        ----------
        weight : float, default 0.6
            Weight of the previous data point.

        Returns
        -------
        numpy.ndarray
            The exponential moving average of the input data.

        """
        if len(kwargs) > 0:
            warnings.warn(
                f"the following arguments are not used: `{kwargs}` for exponential moving average",
                RuntimeWarning,
            )
        smoothed = []
        prev = self.data[0]
        for d in self.data:
            s = prev * weight + (1 - weight) * d
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _cma(self, **kwargs) -> np.ndarray:
        """Cumulative moving average.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            The cumulative moving average of the input data.

        """
        if len(kwargs) > 0:
            warnings.warn(
                f"the following arguments are not used: `{kwargs}` for cumulative moving average",
                RuntimeWarning,
            )
        smoothed = []
        prev = 0
        for n, d in enumerate(self.data):
            s = prev + (d - prev) / (n + 1)
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _wma(self, window: int = 5, **kwargs: Any) -> np.ndarray:
        """Weighted moving average.

        Parameters
        ----------
        window : int, default 5
            Window length of the moving average.

        Returns
        -------
        numpy.ndarray
            The weighted moving average of the input data.

        """
        if len(kwargs) > 0:
            warnings.warn(
                f"the following arguments are not used: `{kwargs}` for weighted moving average",
                RuntimeWarning,
            )
        conv = np.arange(1, window + 1)[::-1]
        deno = np.sum(conv)
        smoothed = np.convolve(conv, self.data, mode="same") / deno
        return smoothed


def nildent(text: str) -> str:
    """
    Kill all leading white spaces in each line of `text`,
    while keeping all lines (including empty)

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text.

    """
    new_text = "\n".join([line.lstrip() for line in text.splitlines()]) + ("\n" if text.endswith("\n") else "")
    return new_text


def remove_parameters_returns_from_docstring(
    doc: str,
    parameters: Optional[Union[str, List[str]]] = None,
    returns: Optional[Union[str, List[str]]] = None,
    parameters_indicator: str = "Parameters",
    returns_indicator: str = "Returns",
) -> str:
    """Remove parameters and/or returns from docstring,
    which is of the format of `numpydoc`.

    Parameters
    ----------
    doc : str
        Docstring to be processed.
    parameters : str or List[str], optional
        Parameters to be removed.
    returns : str or List[str], optional
        Returned values to be removed.
    parameters_indicator : str, default "Parameters"
        Indicator of the parameters section.
    returns_indicator : str, default "Returns"
        Indicator of the returns section.

    Returns
    -------
    str
        The processed docstring.

    TODO
    ----
    When one section is empty, remove the whole section,
    or add a line of `None` to the section.

    """
    if parameters is None:
        parameters = []
    elif isinstance(parameters, str):
        parameters = [parameters]
    if returns is None:
        returns = []
    elif isinstance(returns, str):
        returns = [returns]

    new_doc = doc.splitlines()
    parameters_indent = None
    returns_indent = None
    start_idx = None
    parameters_starts = False
    returns_starts = False
    indices2remove = []
    for idx, line in enumerate(new_doc):
        if (
            line.strip().startswith(parameters_indicator)
            and idx < len(new_doc) - 1
            and new_doc[idx + 1].strip() == "-" * len(parameters_indicator)
        ):
            parameters_indent = " " * line.index(parameters_indicator)
            parameters_starts = True
            returns_starts = False
        if (
            line.strip().startswith(returns_indicator)
            and idx < len(new_doc) - 1
            and new_doc[idx + 1].strip() == "-" * len(returns_indicator)
        ):
            returns_indent = " " * line.index(returns_indicator)
            returns_starts = True
            parameters_starts = False
        if start_idx is not None and len(line.strip()) == 0:
            indices2remove.extend(list(range(start_idx, idx)))
            start_idx = None
        if parameters_starts and len(line.lstrip()) == len(line) - len(parameters_indent):
            if any([line.lstrip().startswith(p) for p in parameters]):
                if start_idx is not None:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = idx
            elif start_idx is not None:
                if line.lstrip().startswith(returns_indicator) and len(new_doc[idx - 1].strip()) == 0:
                    indices2remove.extend(list(range(start_idx, idx - 1)))
                else:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = None
        if returns_starts and len(line.lstrip()) == len(line) - len(returns_indent):
            if any([line.lstrip().startswith(p) for p in returns]):
                if start_idx is not None:
                    indices2remove.extend(list(range(start_idx, idx)))
                start_idx = idx
            elif start_idx is not None:
                indices2remove.extend(list(range(start_idx, idx)))
                start_idx = None
    if start_idx is not None:
        indices2remove.extend(list(range(start_idx, len(new_doc))))
        new_doc.extend(["\n", parameters_indicator or returns_indicator])
    new_doc = "\n".join([line for idx, line in enumerate(new_doc) if idx not in indices2remove])
    return new_doc


@contextmanager
def timeout(duration: float):
    """A context manager that raises a
    :class:`TimeoutError` after a specified time.

    Modified from [#timeout]_.

    Parameters
    ----------
    duration : float
        The time duration in seconds,
        should be non-negative, 0 for no timeout.

    References
    ----------
    .. [#timeout] https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if np.isinf(duration):
        duration = 0
    elif duration < 0:
        raise ValueError("`duration` must be non-negative")
    elif duration > 0:  # granularity is 1 second, so round up
        duration = max(1, int(duration))

    def timeout_handler(signum, frame):
        raise TimeoutError(f"block timedout after `{duration}` seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


class Timer(ReprMixin):
    """Context manager to time the execution of a block of code.

    Parameters
    ----------
    name : str, optional
        Name of the timer, defaults to "default timer".
    verbose : int, default 0
        Verbosity level of the timer.

    Example
    -------
    >>> with Timer("task name", verbose=2) as timer:
    ...     do_something()
    ...     timer.add_time("subtask 1", level=2)
    ...     do_subtask_1()
    ...     timer.stop_timer("subtask 1")
    ...     timer.add_time("subtask 2", level=2)
    ...     do_subtask_2()
    ...     timer.stop_timer("subtask 2")
    ...     do_something_else()

    """

    __name__ = "Timer"

    def __init__(self, name: Optional[str] = None, verbose: int = 0) -> None:
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

    def __exit__(self, *args) -> None:
        for k in self.timers:
            self.stop_timer(k)
            self.timers[k] = self.ends[k] - self.timers[k]

    def add_timer(self, name: str, level: int = 1) -> None:
        """Add a new timer for some sub-task.

        Parameters
        ----------
        name : str
            Name of the timer to be added.
        level : int, default 1
            Verbosity level of the timer.

        Returns
        -------
        None

        """
        self.timers[name] = time.perf_counter()
        self.ends[name] = 0
        self.levels[name] = level

    def stop_timer(self, name: str) -> None:
        """Stop a timer.

        Parameters
        ----------
        name : str
            Name of the timer to be stopped.

        Returns
        -------
        None

        """
        if self.ends[name] == 0:
            self.ends[name] = time.perf_counter()
            if self.verbose >= self.levels[name]:
                time_cost, unit = self._simplify_time_expr(self.ends[name] - self.timers[name])
                print(f"{name} took {time_cost:.4f} {unit}")

    def _simplify_time_expr(self, time_cost: float) -> Tuple[float, str]:
        """Simplify the time expression.

        Parameters
        ----------
        time_cost : float
            The time cost, with units in seconds.

        Returns
        -------
        time_cost : float
            The time cost.
        unit : str
            Unit of the time cost.

        """
        if time_cost <= 0.1:
            return 1000 * time_cost, "ms"
        return time_cost, "s"

    def extra_repr_keys(self) -> List[str]:
        return ["name", "verbose"]


def get_kwargs(func_or_cls: callable, kwonly: bool = False) -> Dict[str, Any]:
    """Get the kwargs of a function or class.

    Parameters
    ----------
    func_or_cls : callable
        The function or class to get the kwargs of.
    kwonly : bool, default False
        Whether to get the kwonly kwargs of the function or class.

    Returns
    -------
    kwargs : Dict[str, Any]
        The kwargs of the function or class.

    """
    fas = inspect.getfullargspec(func_or_cls)
    kwargs = {}
    if fas.kwonlydefaults is not None:
        kwargs = deepcopy(fas.kwonlydefaults)
    if not kwonly and fas.defaults is not None:
        kwargs.update({k: v for k, v in zip(fas.args[-len(fas.defaults) :], fas.defaults)})
    if len(kwargs) == 0:
        # perhaps `inspect.getfullargspec` does not work
        # we should use `inspect.signature` instead
        # for example, the model init functions defined in
        # https://github.com/pytorch/vision/blob/release/0.13/torchvision/models/resnet.py
        # TODO: discard old code, and use only this block
        signature = inspect.signature(func_or_cls)
        valid_kinds = [inspect.Parameter.KEYWORD_ONLY]
        if not kwonly:
            valid_kinds.append(inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for k, v in signature.parameters.items():
            if v.default is not inspect.Parameter.empty and v.kind in valid_kinds:
                kwargs[k] = v.default
    return kwargs


def get_required_args(func_or_cls: callable) -> List[str]:
    """Get the required positional arguments of a function or class.

    Parameters
    ----------
    func_or_cls : callable
        The function or class to get the required arguments of.

    Returns
    -------
    required_args : List[str]
        Names of required arguments of the function or class.

    """
    signature = inspect.signature(func_or_cls)
    required_args = [
        k
        for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty
        and v.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
    ]
    return required_args


def add_kwargs(func: callable, **kwargs: Any) -> callable:
    """Add keyword arguments to a function.

    This function is used to add keyword arguments to a function
    in order to make it compatible with other functions。

    Parameters
    ----------
    func : callable
        The function to be decorated.
    kwargs : dict
        The keyword arguments to be added.

    Returns
    -------
    callable
        The decorated function, with the keyword arguments added.

    """
    old_kwargs = get_kwargs(func)
    func_signature = inspect.signature(func)
    func_parameters = func_signature.parameters.copy()  # ordered dict

    full_kwargs = deepcopy(old_kwargs)
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.KEYWORD_ONLY:
            kind = inspect.Parameter.KEYWORD_ONLY
            break

    for k, v in kwargs.items():
        if k in old_kwargs:
            raise ValueError(f"keyword argument `{k}` already exists!")
        full_kwargs[k] = v
        func_parameters[k] = inspect.Parameter(k, kind, default=v)

    # move the VAR_POSITIONAL and VAR_KEYWORD in `func_parameters` to the end
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.VAR_POSITIONAL:
            func_parameters.move_to_end(k)
            break
    for k, v in func_parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            func_parameters.move_to_end(k)
            break

    if isinstance(func, types.MethodType):
        # can not assign `__signature__` to a bound method directly
        func.__func__.__signature__ = func_signature.replace(parameters=func_parameters.values())
    else:
        func.__signature__ = func_signature.replace(parameters=func_parameters.values())

    # docstring is automatically copied by `functools.wraps`

    @wraps(func)
    def wrapper(*args: Any, **kwargs_: Any) -> Any:
        assert set(kwargs_).issubset(full_kwargs), (
            "got unexpected keyword arguments: " f"{list(set(kwargs_).difference(full_kwargs))}"
        )
        filtered_kwargs = {k: v for k, v in kwargs_.items() if k in old_kwargs}
        return func(*args, **filtered_kwargs)

    return wrapper


def make_serializable(x: Union[np.ndarray, np.generic, dict, list, tuple]) -> Union[list, dict, Number]:
    """Make an object serializable.

    This function is used to convert all numpy arrays to list in an object,
    and also convert numpy data types to python data types in the object,
    so that it can be serialized by :mod:`json`.

    Parameters
    ----------
    x : Union[numpy.ndarray, numpy.generic, dict, list, tuple]
        Input data, which can be numpy array (or numpy data type),
        or dict, list, tuple containing numpy arrays (or numpy data type).

    Returns
    -------
    Union[list, dict, numbers.Number]
        Converted data.

    Examples
    --------
    >>> import numpy as np
    >>> from fl_sim.utils.misc import make_serializable
    >>> x = np.array([1, 2, 3])
    >>> make_serializable(x)
    [1, 2, 3]
    >>> x = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    >>> make_serializable(x)
    {'a': [1, 2, 3], 'b': [4, 5, 6]}
    >>> x = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    >>> make_serializable(x)
    [[1, 2, 3], [4, 5, 6]]
    >>> x = (np.array([1, 2, 3]), np.array([4, 5, 6]).mean())
    >>> obj = make_serializable(x)
    >>> obj
    [[1, 2, 3], 5.0]
    >>> type(obj[1]), type(x[1])
    (float, numpy.float64)

    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, (list, tuple)):
        # to avoid cases where the list contains numpy data types
        return [make_serializable(v) for v in x]
    elif isinstance(x, dict):
        for k, v in x.items():
            x[k] = make_serializable(v)
    elif isinstance(x, np.generic):
        return x.item()
    # the other types will be returned directly
    return x


def select_k(
    arr: np.ndarray, k: Union[int, List[int], np.ndarray], dim: int = -1, largest: bool = True, sorted: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Select elements from an array along a specified axis of specific rankings.

    Parameters
    ----------
    arr : array_like
        Input array.
    k : int or array_like
        Number of elements to retrieve. If k is a 1D array, it represents the specific rankings to retrieve.
        NOTE that the rankings are 0-based, for example the rankings of the top 3 elements are [0, 1, 2].
    dim : int, default -1
        Axis along which to operate. Default is -1 (the last axis).
    largest : bool, default True
        If True, find the largest elements, else find the smallest elements.
    sorted : bool, default True
        If True, the result is sorted. If False, the result is not sorted.

    Returns
    -------
    values : numpy.ndarray
        The selected values along each axis.
    indices : numpy.ndarray
        The indices of the selected values along each axis.

    .. note::

        For integer k, this function has the same functionality as :func:`torch.topk`.

    """
    arr = np.asarray(arr).copy()  # copy to avoid modifying the input array
    if isinstance(k, (list, np.ndarray)):
        k = np.asarray(k)
    else:
        k = np.arange(k)
    assert -arr.ndim <= dim < arr.ndim, "dim out of bounds"
    dim = dim % arr.ndim  # convert negative dim to positive
    assert k.ndim == 1, f"k must be 1-dimensional, but got {k.ndim} dimensions"
    assert len(np.unique(k)) == len(k), "k must be unique"
    assert len(k) > 0, "k must be a non-empty array, or a positive integer"
    assert 0 <= k.min() <= k.max() <= arr.shape[dim], "k out of bounds"

    if largest:
        arr = -arr
    if sorted:
        indices = np.take(np.argsort(arr, axis=dim), k, axis=dim)
    else:
        indices = np.take(np.argpartition(arr, kth=k, axis=dim), k, axis=dim)
    values = np.take_along_axis(arr, indices, axis=dim)

    if largest:
        values = -values

    return values, indices


def np_topk(arr: np.ndarray, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Find the k largest elements of an array along a specified axis.

    Parameters
    ----------
    arr : array_like
        Input array.
    k : int
        Number of elements to retrieve.
    dim : int, default -1
        Axis along which to operate. Default is -1 (the last axis).
    largest : bool, default True
        If True, find the largest elements, else find the smallest elements.
    sorted : bool, default True
        If True, the result is sorted. If False, the result is not sorted.

    Returns
    -------
    values : numpy.ndarray
        The k largest values along each axis.
    indices : numpy.ndarray
        The indices of the k largest values along each axis.

    .. note::

        This function has the same functionality as :func:`torch.topk`,
        but is implemented using only numpy.

    """
    # arr = np.asarray(arr).copy()  # copy to avoid modifying the input array
    # assert -arr.ndim <= dim < arr.ndim, "dim out of bounds"
    # dim = dim % arr.ndim  # convert negative dim to positive
    # assert 0 < k <= arr.shape[dim], "k out of bounds"

    # if largest:
    #     arr = -arr
    # if sorted:
    #     indices = np.take(np.argsort(arr, axis=dim), np.arange(k), axis=dim)
    # else:
    #     indices = np.take(np.argpartition(arr, kth=k, axis=dim), np.arange(k), axis=dim)
    # values = np.take_along_axis(arr, indices, axis=dim)

    # if largest:
    #     values = -values

    # return values, indices
    assert isinstance(k, int) and k > 0, "k must be a positive integer"
    return select_k(arr, k, dim, largest, sorted)
