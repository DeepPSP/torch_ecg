"""
"""

from pathlib import Path

import setuptools

from torch_ecg import __version__

cwd = Path(__file__).absolute().parent

long_description = (cwd / "README.md").read_text(encoding="utf-8")

extras = {}
extras["test"] = [
    "black==22.3.0",
    "flake8",
    "pytest",
    "pytest-xdist",
]
extras["docs"] = [
    "recommonmark",
    "nbsphinx",
    "sphinx-autobuild",
    "sphinx-rtd-theme",
    "sphinx-markdown-tables",
    "sphinx-copybutton",
]
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
            "tests*",
            "sample*",
        ]
    ),
    package_data={"torch_ecg.databases.aux_data": ["*.tar.gz"]},
    # entry_points=,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").readlines(),
    extras_require=extras,
)
