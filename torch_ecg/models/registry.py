"""
Registries for models and backbones.
"""

from ..utils.registry import Registry

__all__ = [
    "BACKBONES",
    "MODELS",
]

BACKBONES = Registry("backbones")
MODELS = Registry("models")
