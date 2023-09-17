"""
"""

import random
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, MutableMapping, Optional, Union

import numpy as np
import torch

__all__ = [
    "CFG",
    "DEFAULTS",
]


_PACKAGE_CACHE = Path("~").expanduser() / ".cache" / "torch_ecg"
_PACKAGE_CACHE.mkdir(parents=True, exist_ok=True)
_DATA_CACHE = _PACKAGE_CACHE / "data"
_DATA_CACHE.mkdir(parents=True, exist_ok=True)


class CFG(dict):
    """
    This class is created in order to renew the :meth:`update` method,
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
            try:
                setattr(self, k, v)
            except Exception:
                dict.__setitem__(self, k, v)
        # Class attributes
        exclude_fields = ["update", "pop"]
        for k in self.__class__.__dict__:
            if not (k.startswith("__") and k.endswith("__")) and k not in exclude_fields:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, new_cfg: Optional[MutableMapping] = None, **kwargs: Any) -> None:
        """
        The new hierarchical update method.

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            The new configuration, by default None.
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
                    self[k] = _new_cfg[k]  # deepcopy?
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """
        The updated pop method.

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
        return repr({k: v for k, v in self.items() if not callable(v)})

    def __str__(self) -> str:
        return str({k: v for k, v in self.items() if not callable(v)})


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
    NP: np.dtype = None
    TORCH: torch.dtype = None
    INT: int = None  # int representation of the dtype, mainly used for `wfdb.rdrecord`

    def __post_init__(self) -> None:
        """check consistency"""
        if self.NP is None:
            self.NP = np.dtype(self.STR)
        if self.TORCH is None:
            self.TORCH = eval(f"torch.{self.STR}")
        if self.INT is None:
            self.INT = int(re.search("\\d+", self.STR).group(0))
        assert all(
            [
                self.NP == getattr(np, self.STR),
                self.TORCH == getattr(torch, self.STR),
                self.INT == int(re.search("\\d+", self.STR).group(0)),
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

DEFAULTS.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DEFAULTS.eps = 1e-7

DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)
DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)


def set_seed(seed: int) -> None:
    """
    Set the seed of the random number generator.

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
