""" """

import shutil
import subprocess
import types
import urllib.parse
from pathlib import Path

import pytest

import torch_ecg.utils.download as dl

_TMP_DIR = Path(__file__).resolve().parents[2] / "tmp" / "test_download"
_TMP_DIR.mkdir(parents=True, exist_ok=True)


def test_http_get():
    # normally, direct downloading from dropbox with `dl=0` will not download the file
    # http_get internally replaces `dl=0` with `dl=1` to force download
    url = "https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=0"
    dl.http_get(url, _TMP_DIR / "action-test-zip-extract", extract=True, filename="test.zip")
    shutil.rmtree(_TMP_DIR / "action-test-zip-extract")
    dl.http_get(url, _TMP_DIR / "action-test-zip-extract", extract="auto", filename="test.zip")
    shutil.rmtree(_TMP_DIR / "action-test-zip-extract")
    dl.http_get(url, _TMP_DIR / "action-test-zip-extract", extract="auto")
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
        dl.http_get(url, _TMP_DIR, extract=True, filename="test.txt")
    with pytest.raises(FileExistsError, match="file already exists"):
        dl.http_get(url, _TMP_DIR, extract=True, filename="test.txt")
    (_TMP_DIR / "test.txt").unlink()
    dl.http_get(url, _TMP_DIR, extract="auto", filename="test.txt")
    (_TMP_DIR / "test.txt").unlink()
    dl.http_get(url, _TMP_DIR, extract="auto")

    with pytest.warns(
        RuntimeWarning,
        match=(
            "URL must be pointing to a `zip` file or a compressed `tar` file\\. "
            "Automatic decompression is turned off\\. "
            "The user is responsible for decompressing the file manually\\."
        ),
    ):
        dl.http_get(url, _TMP_DIR, extract=True)
    Path(_TMP_DIR / Path(url).name).unlink()

    # test downloading from Google Drive
    file_id = "1Yys567-MZIMf3eXGJd8bGrsWIvDatbsZ"
    url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    url_no_scheme = f"drive.google.com/file/d/{file_id}/view?usp=sharing"
    url_xxx_schme = f"xxx://drive.google.com/file/d/{file_id}/view?usp=sharing"
    with pytest.raises(AssertionError, match="filename can not be inferred from Google Drive URL"):
        dl.http_get(url_no_scheme, _TMP_DIR)
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        dl.http_get(url_xxx_schme, _TMP_DIR, extract=False, filename="torch-ecg-paper.bib")
    dl.http_get(url, _TMP_DIR, filename="torch-ecg-paper.bib", extract=False)
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()
    dl._download_from_google_drive(file_id, _TMP_DIR / "torch-ecg-paper.bib")
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()
    dl._download_from_google_drive(url_no_scheme, _TMP_DIR / "torch-ecg-paper.bib")
    (_TMP_DIR / "torch-ecg-paper.bib").unlink()

    # test downloading from AWS S3 (by default using AWS CLI)
    (_TMP_DIR / "ludb").mkdir(exist_ok=True)
    dl.http_get("s3://physionet-open/ludb/1.0.1/", _TMP_DIR / "ludb")

    # test downloading from AWS S3 (by using boto3)
    shutil.rmtree(_TMP_DIR / "ludb")
    (_TMP_DIR / "ludb").mkdir(exist_ok=True)
    dl._download_from_aws_s3_using_boto3("s3://physionet-open/ludb/1.0.1/", _TMP_DIR / "ludb")

    with pytest.raises(ValueError, match="Invalid S3 URL"):
        dl.http_get("s3://xxx", _TMP_DIR / "ludb")


def test_url_is_reachable():
    assert dl.url_is_reachable("https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=1")
    assert not dl.url_is_reachable("https://www.some-unknown-domain.com/unknown-path/unknown-file.zip")


def test_is_compressed_file():
    # check local files
    assert not dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.txt")
    assert not dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract")
    assert not dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test")
    assert not dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.pth.tar")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.gz")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tgz")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.bz2")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tbz2")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.tar.xz")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.txz")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.zip")
    assert dl.is_compressed_file(_TMP_DIR / "action-test-zip-extract" / "test.7z")

    # check remote files (by URL)
    assert dl.is_compressed_file(urllib.parse.urlparse("https://www.dropbox.com/s/oz0n1j3o1m31cbh/action_test.zip?dl=0").path)


class FakeResponseOK:
    """A fake streaming HTTP response that will raise during iter_content."""

    def __init__(self, raise_in_iter=False, raise_text=False):
        self.status_code = 200
        self.headers = {"Content-Length": "10"}
        self._raise_in_iter = raise_in_iter
        self._raise_text = raise_text
        self._text_value = "FAKE_CONTENT_ABCDEFG" * 5

    def iter_content(self, chunk_size=1024):
        if self._raise_in_iter:
            # simulate mid-download exception
            raise RuntimeError("iter boom")
        yield b"abc"
        yield b"def"

    @property
    def text(self):
        if self._raise_text:
            raise ValueError("text boom")
        return self._text_value


class FakeSession:
    def __init__(self, response: FakeResponseOK):
        self._resp = response

    def get(self, *a, **kw):
        return self._resp


@pytest.mark.parametrize("raise_text", [False, True], ids=["text_ok", "text_raises"])
def test_http_get_iter_exception_triggers_runtime(monkeypatch, tmp_path, raise_text):
    """Cover the outer except and the inner try/except that reads req.text."""
    resp = FakeResponseOK(raise_in_iter=True, raise_text=raise_text)

    def fake_retry_session():
        return FakeSession(resp)

    monkeypatch.setattr(dl, "_requests_retry_session", lambda: types.SimpleNamespace(get=fake_retry_session().get))

    target_dir = tmp_path / "download"
    with pytest.raises(RuntimeError) as exc:
        dl.http_get("https://example.com/file.dat", target_dir, extract=False, filename="f.bin")

    msg = str(exc.value)
    if raise_text:
        # inner text extraction failed => snippet empty
        assert "body[:300]=''" in msg or "body[:300]=''"[:10]
    else:
        assert "FAKE_CONTENT" in msg


def test_http_get_status_403(monkeypatch, tmp_path):
    """Force status 403 path (your code raises generic Exception then caught by outer except)."""

    class Resp403:
        status_code = 403
        headers = {}
        text = "Forbidden"

        def iter_content(self, chunk_size=1024):
            yield b""

    def fake_session():
        return types.SimpleNamespace(get=lambda *a, **k: Resp403())

    monkeypatch.setattr(dl, "_requests_retry_session", lambda: fake_session())

    with pytest.raises(RuntimeError) as exc:
        dl.http_get("https://example.com/forbidden.bin", tmp_path, extract=False, filename="f.bin")
    assert "Failed to download" in str(exc.value)
    assert "status=403" in str(exc.value)
    assert "Forbidden" in str(exc.value)


def test_download_from_aws_s3_using_boto3_empty_bucket(monkeypatch, tmp_path):
    """Cover object_count == 0 -> ValueError."""

    class FakePaginator:
        def paginate(self, **kwargs):
            # return an iterator with no 'Contents'
            return iter([{"Other": 1}, {"Meta": 2}])

    class FakeBoto3Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def close(self):
            pass

    monkeypatch.setattr(dl, "boto3", types.SimpleNamespace(client=lambda *a, **k: FakeBoto3Client()))

    with pytest.raises(ValueError, match="No objects found"):
        dl._download_from_aws_s3_using_boto3("s3://fake-bucket/prefix/", tmp_path / "out")


def test_download_from_aws_s3_using_awscli_subprocess_fail(monkeypatch, tmp_path):
    """Force awscli sync subprocess to exit with non-zero code -> CalledProcessError."""
    # Pretend aws exists
    monkeypatch.setattr(dl.shutil, "which", lambda name: "/usr/bin/aws")
    # Control object count
    monkeypatch.setattr(dl, "count_aws_s3_bucket", lambda bucket, prefix: 3)

    class FakeStdout:
        def __init__(self, lines):
            self._lines = lines  # list[str]
            self._idx = 0
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx < len(self._lines):
                line = self._lines[self._idx]
                self._idx += 1
                return line
            raise StopIteration

        def close(self):
            self.closed = True

    class FakePopen:
        def __init__(self, *a, **k):
            self.stdout = FakeStdout(
                [
                    "download: s3://bucket/file1 to file1\n",
                    "download: s3://bucket/file2 to file2\n",
                    "some other line\n",
                ]
            )
            self._returncode = None
            self._waited = False
            self.args = a
            self.kwargs = k

        def poll(self):
            if self.stdout._idx < len(self.stdout._lines):
                return None
            if self._returncode is None:
                self._returncode = 2
            return self._returncode

        def wait(self):
            self.stdout._idx = len(self.stdout._lines)
            if self._returncode is None:
                self._returncode = 2
            self._waited = True
            return self._returncode

        def communicate(self):
            combined = "".join(self.stdout._lines)
            return (combined, "boom\n")

    monkeypatch.setattr(dl.subprocess, "Popen", lambda *a, **k: FakePopen(*a, **k))

    with pytest.raises(subprocess.CalledProcessError) as exc:
        dl._download_from_aws_s3_using_awscli("s3://bucket/prefix/", tmp_path / "out", show_progress=False)

    assert "download: s3://bucket/file1" in exc.value.output
    assert exc.value.returncode != 0


def test_url_is_reachable_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(dl.requests, "head", boom)
    assert dl.url_is_reachable("https://whatever") is False
