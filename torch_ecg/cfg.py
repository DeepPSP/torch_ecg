"""
"""

from pathlib import Path
from typing import Optional, MutableMapping, NoReturn

import numpy as np
import torch
from easydict import EasyDict as ED


__all__ = [
    "CFG",
    "DEFAULTS",
]


_PROJECT_ROOT = Path(__file__).parent.absolute()
_PROJECT_CACHE = Path("~").expanduser() / ".cache" / "torch_ecg"
_PROJECT_CACHE.mkdir(parents=True, exist_ok=True)


class CFG(ED):
    """
    this class is created in order to renew the `update` method,
    to fit the hierarchical structure of configurations

    for example:
    >>> c = CFG(hehe={"a":1,"b":2})
    >>> c.update(hehe={"a":-1})
    >>> c
    {'hehe': {'a': -1, 'b': 2}}
    >>> c.__update__(hehe={"a":-10})
    >>> c
    {'hehe': {'a': -10}}

    """

    __name__ = "CFG"

    def __init__(self, *args, **kwargs) -> NoReturn:
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
            except:
                dict.__setitem__(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__:
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                "update",
                "pop",
            ):
                setattr(self, k, getattr(self, k))

    def __update__(
        self, new_cfg: Optional[MutableMapping] = None, **kwargs
    ) -> NoReturn:
        """
        the original normal update method

        """
        super().update(new_cfg, **kwargs)

    def update(self, new_cfg: Optional[MutableMapping] = None, **kwargs) -> NoReturn:
        """
        the new hierarchical update method

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
                except:
                    dict.__setitem__(self, k, _new_cfg[k])


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
