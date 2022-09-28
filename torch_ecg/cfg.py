"""
"""

import random
from functools import partial
from pathlib import Path
from typing import MutableMapping, Optional, Any, Union

import numpy as np
import torch

__all__ = [
    "CFG",
    "DEFAULTS",
]


_PROJECT_ROOT = Path(__file__).parent.resolve()
_PROJECT_CACHE = Path("~").expanduser() / ".cache" / "torch_ecg"
_PROJECT_CACHE.mkdir(parents=True, exist_ok=True)
_DATA_CACHE = _PROJECT_CACHE / "data"
_DATA_CACHE.mkdir(parents=True, exist_ok=True)


class CFG(dict):
    """
    this class is created in order to renew the `update` method,
    to fit the hierarchical structure of configurations

    Examples
    --------
    >>> c = CFG(hehe={"a":1,"b":2})
    >>> c.update(hehe={"a":-1})
    >>> c
    {'hehe': {'a': -1, 'b': 2}}
    >>> c.__update__(hehe={"a":-10})
    >>> c
    {'hehe': {'a': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> None:
        """ """
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
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in exclude_fields
            ):
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
        the new hierarchical update method

        Parameters
        ----------
        new_cfg : MutableMapping, optional
            the new configuration, by default None
        kwargs : Any, optional
            key value pairs, by default None

        """
        _new_cfg = new_cfg or CFG()
        if len(kwargs) > 0:  # avoid RecursionError
            _new_cfg.update(kwargs)
        for k in _new_cfg:
            # if _new_cfg[k].__class__.__name__ in ["dict", "EasyDict", "CFG"] and k in self:
            if isinstance(_new_cfg[k], MutableMapping) and k in self:
                self[k].update(_new_cfg[k])
            else:
                try:
                    setattr(self, k, _new_cfg[k])
                except Exception:
                    dict.__setitem__(self, k, _new_cfg[k])

    def pop(self, key: str, default: Optional[Any] = None) -> Any:
        """
        the updated pop method

        Parameters
        ----------
        key : str
            the key to pop
        default : Any, optional
            the default value, by default None

        """
        if key in self:
            delattr(self, key)
        return super().pop(key, default)


DEFAULTS = CFG()

DEFAULTS.log_dir = _PROJECT_CACHE / "log"
DEFAULTS.checkpoints = _PROJECT_CACHE / "checkpoints"
DEFAULTS.model_dir = _PROJECT_CACHE / "saved_models"
DEFAULTS.prefix = "TorchECG"

DEFAULTS.torch_dtype = torch.float32  # torch.float64, torch.float16
DEFAULTS.str_dtype = str(DEFAULTS.torch_dtype).replace("torch.", "")
DEFAULTS.np_dtype = np.dtype(DEFAULTS.str_dtype)
DEFAULTS.dtype = DEFAULTS.torch_dtype

DEFAULTS.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

DEFAULTS.eps = 1e-7

DEFAULTS.SEED = 42
DEFAULTS.RNG = np.random.default_rng(seed=DEFAULTS.SEED)
DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)


def set_seed(seed: int) -> None:
    """
    set the seed of the random number generator

    Parameters
    ----------
    seed: int,
        the seed to be set

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
    change the dtype of the defaults

    Parameters
    ----------
    dtype: str or np.dtype or torch.dtype,
        the dtype to be set

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
            raise TypeError(
                f"dtype must be a str or np.dtype or torch.dtype, got {type(dtype)}"
            )
    assert _dtype in _dtypes, f"dtype must be one of {_dtypes}, got {_dtype}"
    DEFAULTS.str_dtype = _dtype
    DEFAULTS.np_dtype = np.dtype(_dtype)
    DEFAULTS.torch_dtype = eval(f"torch.{_dtype}")
    DEFAULTS.dtype = DEFAULTS.torch_dtype


DEFAULTS.change_dtype = change_dtype
