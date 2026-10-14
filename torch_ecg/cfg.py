""" """

import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import numpy as np
import torch

__all__ = [
    "CFG",
    "DEFAULTS",
]


_PACKAGE_CACHE = Path("~").expanduser() / ".cache" / "torch_ecg"
if os.access(_PACKAGE_CACHE.parent, os.W_OK) and not _PACKAGE_CACHE.exists():
    _PACKAGE_CACHE.mkdir(parents=True, exist_ok=True)
_DATA_CACHE = _PACKAGE_CACHE / "data"
if os.access(_DATA_CACHE.parent, os.W_OK) and not _DATA_CACHE.exists():
    _DATA_CACHE.mkdir(parents=True, exist_ok=True)


# keys that would shadow `dict`/`dict`-like API when written onto a `CFG`
# via either the attribute or the item interface
_RESERVED_KEYS = frozenset(
    {
        "clear",
        "copy",
        "fromkeys",
        "get",
        "items",
        "keys",
        "pop",
        "popitem",
        "setdefault",
        "update",
        "values",
    }
)


class CFG(dict):
    """This class is created in order to renew the :meth:`update` method,
    to fit the hierarchical structure of configurations.

    Examples
    --------
    >>> c = CFG(hehe={"a": 1, "b": 2})
    >>> c.update(hehe={"a": [-1]})
    >>> c
    {'hehe': {'a': [-1], 'b': 2}}
    >>> c.update(hehe={"c": -10})
    >>> c
    {'hehe': {'a': [-1], 'b': 2, 'c': -10}}
    >>> c.hehe.pop("a")
    [-1]
    >>> c
    {'hehe': {'b': 2, 'c': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> None:
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        elif len(args) == 1:
            d = args[0]
            assert isinstance(d, MutableMapping)
        else:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            # try:
            #     setattr(self, k, v)
            # except Exception:
            #     dict.__setitem__(self, k, v)
            if isinstance(k, str) and k.isidentifier():
                setattr(self, k, v)
            else:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if not (k.startswith("__") and k.endswith("__")) and k not in exclude_fields:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RESERVED_KEYS:
            raise ValueError(
                f"key `{name}` is reserved by the dict API of `CFG` and would shadow it; " f"please use a different key"
            )
        # NOTE nested values are always converted with the base `CFG` class,
        # NOT with `self.__class__`: subclasses may override `__init__` with
        # stricter contracts (e.g. `BaseOutput` asserts its required fields),
        # and routing a plain nested dict through the subclass constructor
        # used to crash those subclasses (see the former `_KNOWN_ISSUES` in
        # components/outputs.py)
        if isinstance(value, (list, tuple)):
            value = [CFG(x) if isinstance(x, dict) and not isinstance(x, CFG) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, CFG):
            value = CFG(value)

        if isinstance(name, str) and name.isidentifier():
            super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name: str) -> Any:
        """Called only when regular attribute lookup fails.

        Implements the lazy ``device`` of ``DEFAULTS``: probing the CUDA
        driver at import time costs time in every Python 3.14+
        multiprocessing worker (forkserver/spawn re-import the package)
        and freezes the device before ``CUDA_VISIBLE_DEVICES`` could be
        set; the first ``DEFAULTS.device`` access detects and caches it.
        """
        if name == "device":
            value = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            # cache through both views (attribute and dict item), like a
            # regular `__setattr__` would; primitives avoid re-entering
            # the conversion logic above
            object.__setattr__(self, name, value)
            dict.__setitem__(self, name, value)
            return value
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def update(self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any) -> None:  # type: ignore
        """The new hierarchical update method.

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            The new configuration, by default None.
            New values recursively override old ones.
        **kwargs : dict, optional
            Key value pairs, by default None.

        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                if isinstance(self[k], MutableMapping):
                    self[k].update(_new_cfg[k])
                else:  # for example, self[k] is `None`
                    # deepcopy, so that the updated config does not share
                    # (mutable) values with the config it was updated from
                    self[k] = deepcopy(_new_cfg[k])
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except TypeError:
                    # non-str keys cannot go through `setattr`
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """The updated pop method.

        Parameters
        ----------
        key : str
            The key to pop.
        default : Any, optional
            The default value, by default None.

        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)

    def __repr__(self) -> str:
        return repr(
            {k: (v if not callable(v) else f"<callable: {getattr(v, '__name__', type(v).__name__)}>") for k, v in self.items()}
        )

    def __str__(self) -> str:
        return str(
            {k: (v if not callable(v) else f"<callable: {getattr(v, '__name__', type(v).__name__)}>") for k, v in self.items()}
        )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "CFG":
        # bypass `__init__` (subclasses may have stricter constructors,
        # e.g. `BaseOutput` asserts required fields) and rebuild both views
        # (dict items + mirrored attributes) explicitly; `id(self)` is
        # registered in `memo` first, so cyclic references are safe
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.items():
            dict.__setitem__(new, deepcopy(k, memo), deepcopy(v, memo))
        for k, v in self.__dict__.items():
            object.__setattr__(new, k, deepcopy(v, memo))
        return new


@dataclass
class DTYPE:
    """
    A dataclass to store the dtype information.

    Attributes
    ----------
    STR : str
        The string representation of the dtype.
    NP : np.dtype
        The numpy dtype.
    TORCH : torch.dtype
        The torch dtype.
    INT : int
        The int representation of the dtype, mainly used for `wfdb.rdrecord`.

    Examples
    --------
    >>> dtype = DTYPE("int16")
    >>> dtype
    DTYPE(STR='int16', NP=dtype('int16'), TORCH=torch.int16, INT=16)

    """

    STR: str
    NP: np.dtype = None  # type: ignore
    TORCH: torch.dtype = None  # type: ignore
    INT: int = None  # int representation of the dtype, mainly used for `wfdb.rdrecord`  # type: ignore

    def __post_init__(self) -> None:
        """check consistency"""
        if self.NP is None:
            self.NP = np.dtype(self.STR)
        if self.TORCH is None:
            self.TORCH = getattr(torch, self.STR)
        if self.INT is None:
            self.INT = int(re.search("\\d+", self.STR).group(0))  # type: ignore
        assert all(
            [
                self.NP == getattr(np, self.STR),
                self.TORCH == getattr(torch, self.STR),
                self.INT == int(re.search("\\d+", self.STR).group(0)),  # type: ignore
            ]
        ), "inconsistent dtype"

    @property
    def PRECISION(self) -> int:
        return self.INT


FLOAT16 = DTYPE("float16")
FLOAT32 = DTYPE("float32")
FLOAT64 = DTYPE("float64")
INT8 = DTYPE("int8")
INT16 = DTYPE("int16")
INT32 = DTYPE("int32")
INT64 = DTYPE("int64")


DEFAULTS = CFG()

DEFAULTS.log_dir = _PACKAGE_CACHE / "log"
DEFAULTS.checkpoints = _PACKAGE_CACHE / "checkpoints"
DEFAULTS.model_dir = _PACKAGE_CACHE / "saved_models"
DEFAULTS.working_dir = _PACKAGE_CACHE / "working_dir"
DEFAULTS.prefix = "TorchECG"

DEFAULTS.DTYPE = FLOAT32
DEFAULTS.DTYPE.TORCH = torch.float32  # torch.float64, torch.float16
DEFAULTS.str_dtype = str(DEFAULTS.DTYPE.TORCH).replace("torch.", "")
DEFAULTS.np_dtype = np.dtype(DEFAULTS.str_dtype)
DEFAULTS.dtype = DEFAULTS.DTYPE.TORCH

# `device` is NOT set eagerly here on purpose: see `CFG.__getattr__` for the
# lazy detection, which keeps `import torch_ecg` free of CUDA driver probes.
DEFAULTS.eps = 1e-7

DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)
DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)


def set_seed(seed: int) -> None:
    """
    Set the seed of all random number generators:
    the :class:`numpy.random.Generator` ``DEFAULTS.RNG``,
    the global ``random`` and ``numpy.random`` states,
    and the ``torch`` (CPU and CUDA) seeds.

    Parameters
    ----------
    seed : int
        The seed to be set.

    """

    DEFAULTS.SEED = seed
    DEFAULTS.RNG = np.random.default_rng(seed=seed)
    DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
    DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)

    random.seed(seed)
    # `DEFAULTS.RNG` is the canonical RNG of this package; the global
    # `numpy.random` state is seeded as well only because third-party
    # libraries and legacy code (e.g. scipy functions with
    # `random_state=None`) draw from the global state and know nothing
    # about `DEFAULTS.RNG`. Seeding it costs nothing and does not affect
    # the stream of `DEFAULTS.RNG` (the two are independent).
    # Legacy `np.random.seed` only accepts values in [0, 2**32 - 1], while
    # `seed` may be a full 64-bit value (e.g. derived from
    # `torch.initial_seed()`), hence the normalization below.
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


DEFAULTS.set_seed = set_seed


def change_dtype(dtype: Union[str, np.dtype, torch.dtype]) -> None:
    """
    Change the dtype of the defaults.

    Parameters
    ----------
    dtype: str or numpy.dtype or torch.dtype,
        The dtype to be set.

    """
    # fmt: off
    _dtypes = [
        "float32", "float64", "float16",
        "int32", "int64", "int16", "int8", "uint8", "long",
    ]
    # fmt: on
    if isinstance(dtype, torch.dtype):
        _dtype = str(dtype).replace("torch.", "")
    elif isinstance(dtype, np.dtype):
        _dtype = str(dtype)
    elif isinstance(dtype, str):
        _dtype = dtype
    else:  # for example, dtype=np.float64
        try:
            _dtype = dtype.__name__
        except AttributeError:
            raise TypeError(f"`dtype` must be a str or np.dtype or torch.dtype, got {type(dtype)}")
    assert _dtype in _dtypes, f"`dtype` must be one of {_dtypes}, got {_dtype}"
    DEFAULTS.DTYPE = DTYPE(_dtype)
    DEFAULTS.dtype = DEFAULTS.DTYPE.TORCH


DEFAULTS.change_dtype = change_dtype
