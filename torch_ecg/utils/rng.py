"""Utilities for random number generation and reproducibility."""

from functools import partial
from typing import Any

import numpy as np
import torch

from ..cfg import DEFAULTS

__all__ = [
    "seed_everything",
    "worker_init_fn",
]


def seed_everything(seed: int) -> None:
    """Seed all random number generators used by ``torch_ecg``
    (and by the libraries it relies on).

    This is an alias of :data:`DEFAULTS.set_seed
    <torch_ecg.cfg.DEFAULTS>`: it (re)creates ``DEFAULTS.RNG``
    (:class:`numpy.random.Generator`), and seeds the global ``random``
    and ``numpy.random`` states and the ``torch`` (CPU and CUDA)
    generators.

    Parameters
    ----------
    seed : int
        The seed to be set.

    Examples
    --------
    .. code-block:: python

        from torch_ecg.utils import seed_everything

        seed_everything(42)

    """
    DEFAULTS.set_seed(seed)


def worker_init_fn(worker_id: int) -> None:
    """``worker_init_fn`` for :class:`torch.utils.data.DataLoader`
    which reseeds ``DEFAULTS.RNG`` in each worker process.

    PyTorch already reseeds the global ``random``, ``torch`` and
    ``numpy.random`` states in each worker (with ``base_seed +
    worker_id``, where ``base_seed`` derives from the generator of the
    data loader), but it knows nothing about :class:`numpy.random.Generator`
    instances held by the user. Via ``fork`` (the default start method on
    Linux), each worker inherits a copy of the parent's ``DEFAULTS.RNG``,
    so all workers would produce **identical** random streams, e.g. for
    augmentations driven by ``DEFAULTS.RNG``.

    This function derives a distinct, reproducible seed from the
    worker-specific torch seed (:func:`torch.initial_seed`) and reseeds
    ``DEFAULTS.RNG`` (and the ``RNG_sample`` / ``RNG_randint`` partials)
    accordingly.

    Usage
    -----
    .. code-block:: python

        from torch.utils.data import DataLoader
        from torch_ecg.utils import seed_everything, worker_init_fn

        seed_everything(42)
        dl = DataLoader(
            ds,
            batch_size=32,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )

    Parameters
    ----------
    worker_id : int
        Id of the worker process, passed by :class:`torch.utils.data.DataLoader`.

    """
    # inside a worker process, `torch.initial_seed()` returns the
    # worker-specific seed (base_seed + worker_id)
    worker_seed = torch.initial_seed() % 2**32
    DEFAULTS.RNG = np.random.default_rng(worker_seed)
    DEFAULTS.RNG_sample = partial(DEFAULTS.RNG.choice, replace=False, shuffle=False)
    DEFAULTS.RNG_randint = partial(DEFAULTS.RNG.integers, endpoint=True)


def get_rng() -> Any:
    """Get the current ``DEFAULTS.RNG``, e.g. for passing around
    or for inspecting its bit generator state.

    Returns
    -------
    numpy.random.Generator
        The current default random number generator.

    """
    return DEFAULTS.RNG
