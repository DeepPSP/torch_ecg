"""
Registries for components (optimizers, schedulers, losses, etc.).
"""

from ..utils.registry import Registry

__all__ = [
    "OPTIMIZERS",
    "SCHEDULERS",
    "LOSSES",
]

OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
LOSSES = Registry("losses")
