"""
"""

import shutil
from pathlib import Path

import pytest

from torch_ecg.utils.download import http_get


_TMP_DIR = Path(__file__).resolve().parents[2] / "tmp" / "test_download"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_http_get():
    url = "https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=1"
    http_get(
        url, _TMP_DIR / "action-test-zip-extract", extract=True, filename="test.zip"
    )
    shutil.rmtree(_TMP_DIR / "action-test-zip-extract")

    url = (
        "https://github.com/DeepPSP/cinc2021/blob/master/results/"
        "20211121-12leads/TorchECG_11-20_21-52_ECG_CRNN_CINC2021_adamw_amsgrad_"
        "LR_0.0001_BS_64_resnet_nature_comm_bottle_neck_se.txt"
    )
    with pytest.warns(
        RuntimeWarning,
        match=(
            "filename is given, and it is not a `zip` file or a compressed `tar` file\\. "
            "Automatic decompression is turned off\\."
        ),
    ):
        http_get(url, _TMP_DIR, extract=True, filename="test.txt")
    with pytest.raises(AssertionError, match="file already exists"):
        http_get(url, _TMP_DIR, extract=True, filename="test.txt")
    (_TMP_DIR / "test.txt").unlink()

    with pytest.warns(
        RuntimeWarning,
        match=(
            "URL must be pointing to a `zip` file or a compressed `tar` file\\. "
            "Automatic decompression is turned off\\. "
            "The user is responsible for decompressing the file manually\\."
        ),
    ):
        http_get(url, _TMP_DIR, extract=True)
    Path(_TMP_DIR / Path(url).name).unlink()
