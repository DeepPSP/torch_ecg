"""
Registry for augmenters.
"""

from ..utils.registry import Registry

__all__ = [
    "AUGMENTERS",
]

AUGMENTERS = Registry("augmenters")
