"""Regression tests: importing the package must never fail because no cache
directory can be determined.

The failure mode was reported from minimal container environments: ``HOME``
unset for a UID without a ``/etc/passwd`` entry makes
``Path("~").expanduser()`` raise ``RuntimeError`` at import time
(``torch_ecg/cfg.py``), before any user code can react.

All checks run in fresh subprocesses because ``torch_ecg.cfg`` caches
module-level state.  Checks that depend on POSIX ``HOME``/passwd semantics
are skipped on other platforms; the environment-variable overrides are
platform-neutral.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_CWD = Path(__file__).absolute().parents[1]

_VARS = ("HOME", "XDG_CACHE_HOME", "TORCH_ECG_CACHE_DIR")

_SKIP_NON_POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX HOME/passwd semantics")

# the environment under test changes HOME, which can silently disable the
# interpreter's user site-packages (e.g. numpy installed under ~/.local on
# developer machines); pass the parent's import path through so package
# availability is decoupled from the cache-discovery variables under test
_PYTHONPATH = os.pathsep.join(p for p in sys.path if p)

# prints the resolved package cache; asserts the public contract that
# consumers rely on (path-like constants after successful import)
_CHECK_IMPORT = """
from pathlib import Path
import torch_ecg
from torch_ecg.cfg import DEFAULTS, _DATA_CACHE, _PACKAGE_CACHE
assert isinstance(_PACKAGE_CACHE, Path), _PACKAGE_CACHE
assert isinstance(_DATA_CACHE, Path), _DATA_CACHE
assert isinstance(DEFAULTS.log_dir, Path), DEFAULTS.log_dir
print(_PACKAGE_CACHE)
"""

# HOME unset AND no passwd entry: patch pwd.getpwuid to raise KeyError,
# faithfully reproducing an unregistered UID on any machine
_CHECK_NO_PASSWD = (
    """
import pwd

def _no_passwd_entry(uid):
    raise KeyError(f"getpwuid(): uid not found: {uid}")

pwd.getpwuid = _no_passwd_entry
"""
    + _CHECK_IMPORT
    + """
import tempfile
assert Path(tempfile.gettempdir()) in _PACKAGE_CACHE.parents, _PACKAGE_CACHE
"""
)

# no candidate succeeds AND the temporary-directory machinery itself fails:
# mkdtemp and gettempdir are patched to raise, but only AFTER the third-party
# import chain (torch -> dill calls gettempdir at import) has settled and
# right before torch_ecg is imported; the sentinel path must then be
# constructed without any further filesystem access and the import must
# still succeed
_CHECK_NO_TEMP = (
    """
import pwd

def _no_passwd_entry(uid):
    raise KeyError(f"getpwuid(): uid not found: {uid}")

pwd.getpwuid = _no_passwd_entry

import torch  # settles the third-party import chain before the patches

import tempfile

def _no_temp_dir(*args, **kwargs):
    raise OSError("no writable temporary directory")

tempfile.mkdtemp = _no_temp_dir
tempfile.gettempdir = _no_temp_dir
"""
    + _CHECK_IMPORT
    + """
import os
assert _PACKAGE_CACHE == Path(os.sep) / f"torch_ecg-cache-disabled-{os.getuid()}", _PACKAGE_CACHE
"""
)


def _check_passwd_home(home):
    """HOME unset, passwd entry faked to `home`: the passwd home is used,
    without touching the real user home."""
    return """
import pwd
from types import SimpleNamespace

def _fake_passwd_entry(uid):
    return SimpleNamespace(pw_dir={home!r})

pwd.getpwuid = _fake_passwd_entry
""".format(home=home) + _CHECK_IMPORT


_TEMP_WARNING = "No permanent cache directory is writable"
_DISABLED_WARNING = "the package cache is disabled"


def _run_import(env_updates=None, unset=(), script=_CHECK_IMPORT, cwd=None):
    """Import torch_ecg in a fresh subprocess under a controlled environment."""
    env = {k: v for k, v in os.environ.items() if k not in _VARS}
    env["PYTHONPATH"] = _PYTHONPATH
    if env_updates:
        env.update(env_updates)
    for k in unset:
        env.pop(k, None)
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or str(_CWD),
        timeout=300,
    )


@_SKIP_NON_POSIX
def test_no_home_no_passwd_entry():
    """HOME unset for a UID without a passwd entry: import succeeds via the
    temporary-directory fallback."""
    result = _run_import(script=_CHECK_NO_PASSWD)
    assert result.returncode == 0, result.stderr
    assert _TEMP_WARNING in result.stderr
    assert result.stderr.count(_TEMP_WARNING) == 1
    assert _DISABLED_WARNING not in result.stderr


@_SKIP_NON_POSIX
def test_no_writable_temp_dir():
    """No candidate succeeds and the temporary-directory machinery itself
    raises OSError: import still succeeds with the disabled-cache sentinel,
    whose construction performs no further filesystem access."""
    result = _run_import(script=_CHECK_NO_TEMP)
    assert result.returncode == 0, result.stderr
    assert _DISABLED_WARNING in result.stderr
    assert result.stderr.count(_DISABLED_WARNING) == 1
    assert _TEMP_WARNING not in result.stderr


@_SKIP_NON_POSIX
def test_home_unset_with_passwd_entry(tmp_path):
    """HOME unset, the UID has a passwd entry: the passwd home is used.
    The passwd entry is faked to point into ``tmp_path``, so the real user
    home is never read or written."""
    result = _run_import(script=_check_passwd_home(str(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / ".cache" / "torch_ecg")
    assert _TEMP_WARNING not in result.stderr


@_SKIP_NON_POSIX
def test_home_nonexistent_path(tmp_path):
    """HOME pointing to a nonexistent directory whose parent is writable:
    the cache chain is created and used (no fallback needed)."""
    home = tmp_path / "no-such-dir"
    result = _run_import({"HOME": str(home)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(home / ".cache" / "torch_ecg")
    assert _TEMP_WARNING not in result.stderr


@_SKIP_NON_POSIX
def test_home_read_only(tmp_path):
    """HOME pointing to a read-only directory: falls through to the
    temporary-directory fallback."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("read-only HOME is not effective for root")
    ro_home = tmp_path / "ro-home"
    ro_home.mkdir()
    ro_home.chmod(0o555)
    try:
        result = _run_import({"HOME": str(ro_home)})
        assert result.returncode == 0, result.stderr
        assert _TEMP_WARNING in result.stderr
    finally:
        ro_home.chmod(0o755)


@_SKIP_NON_POSIX
def test_home_proc():
    """HOME=/proc (a read-only pseudo-filesystem): import succeeds."""
    result = _run_import({"HOME": "/proc"})
    assert result.returncode == 0, result.stderr
    assert _TEMP_WARNING in result.stderr


def test_env_cache_dir(tmp_path):
    """TORCH_ECG_CACHE_DIR wins over all defaults and is used as-is."""
    cache_dir = tmp_path / "custom-cache"
    result = _run_import({"TORCH_ECG_CACHE_DIR": str(cache_dir)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(cache_dir)
    assert _TEMP_WARNING not in result.stderr


def test_xdg_cache_home(tmp_path):
    """XDG_CACHE_HOME/torch_ecg is used when TORCH_ECG_CACHE_DIR is unset."""
    result = _run_import({"XDG_CACHE_HOME": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "torch_ecg")
    assert _TEMP_WARNING not in result.stderr


def test_env_cache_dir_beats_xdg(tmp_path):
    """TORCH_ECG_CACHE_DIR takes precedence over XDG_CACHE_HOME."""
    cache_dir = tmp_path / "custom-cache"
    result = _run_import({"TORCH_ECG_CACHE_DIR": str(cache_dir), "XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(cache_dir)


@_SKIP_NON_POSIX
def test_normal_home(tmp_path):
    """A writable HOME resolves to <home>/.cache/torch_ecg (pre-fix behavior
    preserved)."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_import({"HOME": str(home)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(home / ".cache" / "torch_ecg")
    assert (home / ".cache" / "torch_ecg").is_dir()
    assert (home / ".cache" / "torch_ecg" / "data").is_dir()
    assert _TEMP_WARNING not in result.stderr
