"""
Registry for preprocessors.
"""

from ..utils.registry import Registry

__all__ = [
    "PREPROCESSORS",
]

PREPROCESSORS = Registry("preprocessors")
