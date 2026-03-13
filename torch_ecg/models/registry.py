"""
Registries for models and backbones.
"""

from ..utils.registry import Registry

__all__ = [
    "BACKBONES",
    "MODELS",
    "ATTN_LAYERS",
]

BACKBONES = Registry("backbones")
MODELS = Registry("models")
ATTN_LAYERS = Registry("attn_layers")
