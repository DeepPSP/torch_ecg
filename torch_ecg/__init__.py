"""
"""
import os, sys

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_BASE_DIR)  # e.g. "site-packages"
# _IN_SYS_PATH = [p for p in [_BASE_DIR, _PARENT_DIR] if p in sys.path]
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
# if _BASE_DIR not in sys.path:
#     sys.path.insert(0, _BASE_DIR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
