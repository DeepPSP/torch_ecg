"""
"""

from easydict import EasyDict as ED


__all__ = ["Cfg"]


Cfg = ED()

Cfg.torch_dtype = "float"  # "double"
