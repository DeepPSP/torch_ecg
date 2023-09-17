"""
"""

from pathlib import Path

import setuptools

from torch_ecg import __version__

cwd = Path(__file__).absolute().parent

long_description = (cwd / "README.md").read_text(encoding="utf-8")

install_requires = (cwd / "requirements.txt").read_text(encoding="utf-8").splitlines()

extras = {}
extras["test"] = (cwd / "test/requirements.txt").read_text(encoding="utf-8").splitlines()
extras["docs"] = (cwd / "docs/requirements.txt").read_text(encoding="utf-8").splitlines()
extras["dev"] = extras["docs"] + extras["test"]


setuptools.setup(
    name="torch_ecg",
    version=__version__,
    author="DeepPSP",
    author_email="wenh06@gmail.com",
    license="MIT",
    description="A Deep Learning Framework for ECG Processing Tasks Based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepPSP/torch_ecg",
    # project_urls={},
    packages=setuptools.find_packages(
        exclude=[
            "references*",
            "docs*",
            "benchmarks*",
            "test*",
            "sample*",
        ]
    ),
    package_data={"torch_ecg.databases.aux_data": ["*.gz"]},
    # entry_points=,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras,
)
