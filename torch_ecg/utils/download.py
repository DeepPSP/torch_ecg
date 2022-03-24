"""
utilities for downloading and extracting compressed data files

For most PhysioNet databases, the WFDB package already has a method `dl_database`
for downloading the data files.
"""

import os, tempfile, zipfile, tarfile
from pathlib import Path
from typing import NoReturn, Union, Optional

import requests, tqdm


__all__ = ["http_get",]


def http_get(url:str, dst_dir:Union[str,Path], proxies:Optional[dict]=None, extract:bool=True) -> NoReturn:
    """Get contents of a URL and save to a file.

    Parameters
    ----------
    url: str,
        URL to download.
    dst_dir: str or Path,
        Directory to save the file.
    proxies: dict, optional,
        Dictionary of proxy settings.
    extract: bool, default True,
        Whether to extract the downloaded file.

    References
    ----------
    1. https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    print(f"Downloading {url}.")
    parent_dir = Path(dst_dir).parent
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=parent_dir,
        suffix=_suffix(url),
        delete=False
    )
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403 or req.status_code == 404:
        raise Exception(f"Could not reach {url}.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            downloaded_file.write(chunk)
    progress.close()
    downloaded_file.close()
    if extract:
        if _suffix(url) == ".zip":
            _unzip_file(str(downloaded_file.name), str(dst_dir))
        else:  # tar files
            _untar_file(str(downloaded_file.name), str(dst_dir))
    else:
        shutil.copyfile(downloaded_file.name, Path(dst_dir) / Path(url).name)
    os.remove(downloaded_file.name)


def _stem(path:Union[str,Path]) -> str:
    """
    get filename without extension, especially for .tar.xx files

    Parameters
    ----------
    path: str or Path,
        path to the file
    
    Returns
    -------
    str,
        filename without extension
    """
    ret = Path(path).stem
    for _ in range(3):
        ret = Path(ret).stem
    return ret


def _suffix(path:Union[str,Path]) -> str:
    """
    get file extension, including all suffixes

    Parameters
    ----------
    path: str or Path,
        path to the file

    Returns
    -------
    str,
        full file extension
    """
    return "".join(Path(path).suffixes)


def _unzip_file(path_to_zip_file:Union[str, Path], dst_dir:Union[str, Path]) -> NoReturn:
    """
    Unzips a .zip file to folder path.
    
    Parameters
    ----------
    path_to_zip_file: str or Path,
        path to the .zip file
    dst_dir: str or Path,
        path to the destination folder
    """
    print(f"Extracting file {path_to_zip_file} to {dst_dir}.")
    with zipfile.ZipFile(str(path_to_zip_file)) as zip_ref:
        zip_ref.extractall(str(dst_dir))


def _untar_file(path_to_tar_file:Union[str, Path], dst_dir:Union[str, Path]) -> NoReturn:
    """
    Decompress a .tar.xx file to folder path.
    
    Parameters
    ----------
    path_to_tar_file: str or Path,
        path to the .tar.xx file
    dst_dir: str or Path,
        path to the destination folder
    """
    print(f"Extracting file {path_to_tar_file} to {dst_dir}.")
    mode = Path(path_to_tar_file).suffix.replace(".", "r:").replace("tar", "")
    # print(f"mode: {mode}")
    with tarfile.open(str(path_to_tar_file), mode) as tar_ref:
        tar_ref.extractall(str(dst_dir))
