"""
"""

import inspect
from copy import deepcopy
from typing import Callable, Dict, Any


__all__ = [
    "get_kwargs",
]


def get_kwargs(func_or_cls: Callable, kwonly: bool = False) -> Dict[str, Any]:
    """
    get the kwargs of a function or class

    Parameters
    ----------
    func_or_cls: Callable,
        the function or class to get the kwargs of
    kwonly: bool, default False,
        whether to get the kwonly kwargs of the function or class

    Returns
    -------
    kwargs: Dict[str, Any],
        the kwargs of the function or class

    """
    fas = inspect.getfullargspec(func_or_cls)
    kwargs = {}
    if fas.kwonlydefaults is not None:
        kwargs = deepcopy(fas.kwonlydefaults)
    if kwonly:
        return kwargs
    if fas.defaults is not None:
        kwargs.update(
            {k: v for k, v in zip(fas.args[-len(fas.defaults) :], fas.defaults)}
        )
    return kwargs
