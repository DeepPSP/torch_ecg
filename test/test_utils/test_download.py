"""
"""

import shutil
from pathlib import Path

import pytest

from torch_ecg.utils.download import (
    _download_from_aws_s3_using_boto3,
    _download_from_google_drive,
    http_get,
    is_compressed_file,
    url_is_reachable,
)

_TMP_DIR = Path(__file__).resolve().parents[2] / "tmp" / "test_download"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_http_get():
    # normally, direct downloading from dropbox with `dl=0` will not download the file
    # http_get internally replaces `dl=0` with `dl=1` to force download
    url = "https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=0"
    http_get(url, _TMP_DIR / "action-test-zip-extract", extract=True, filename="test.zip")
    shutil.rmtree(_TMP_DIR / "action-test-zip-extract")
    http_get(url, _TMP_DIR / "action-test-zip-extract", extract="auto", filename="test.zip")
    shutil.rmtree(_TMP_DIR / "action-test-zip-extract")
    http_get(url, _TMP_DIR / "action-test-zip-extract", extract="auto")
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
    with pytest.raises(FileExistsError, match="file already exists"):
        http_get(url, _TMP_DIR, extract=True, filename="test.txt")
    (_TMP_DIR / "test.txt").unlink()
    http_get(url, _TMP_DIR, extract="auto", filename="test.txt")
    (_TMP_DIR / "test.txt").unlink()
    http_get(url, _TMP_DIR, extract="auto")

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

    # test downloading from Google Drive
    file_id = "1Yys567-MZIMf3eXGJd8bGrsWIvDatbsZ"
    url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    url_no_scheme = f"drive.google.com/file/d/{file_id}/view?usp=sharing"
    url_xxx_schme = f"xxx://drive.google.com/file/d/{file_id}/view?usp=sharing"
    with pytest.raises(AssertionError, match="filename can not be inferred from Google Drive URL"):
        http_get(url_no_scheme, _TMP_DIR)
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        http_get(url_xxx_schme, _TMP_DIR, extract=False, filename="torch-ecg-paper.bib")
    http_get(url, _TMP_DIR, filename="torch-ecg-paper.bib", extract=False)
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()
    _download_from_google_drive(file_id, _TMP_DIR / "torch-ecg-paper.bib")
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()
    _download_from_google_drive(url_no_scheme, _TMP_DIR / "torch-ecg-paper.bib")
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()

    # test downloading from AWS S3 (by default using AWS CLI)
    (_TMP_DIR / "ludb").mkdir(exist_ok=True)
    http_get("s3://physionet-open/ludb/1.0.1/", _TMP_DIR / "ludb")

    # test downloading from AWS S3 (by using boto3)
    shutil.rmtree(_TMP_DIR / "ludb")
    (_TMP_DIR / "ludb").mkdir(exist_ok=True)
    _download_from_aws_s3_using_boto3("s3://physionet-open/ludb/1.0.1/", _TMP_DIR / "ludb")


def test_url_is_reachable():
    assert url_is_reachable("https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=1")
    assert not url_is_reachable("https://www.some-unknown-domain.com/unknown-path/unknown-file.zip")


def test_is_compressed_file():
    assert not is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.txt")
    assert not is_compressed_file(_TMP_DIR / "action-test-zip-extract")
    assert not is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test")
    assert not is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.pth.tar")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.gz")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tgz")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.bz2")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tbz2")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.xz")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.txz")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.zip")
    assert is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.7z")
