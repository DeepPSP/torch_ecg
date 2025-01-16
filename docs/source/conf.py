# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import sys
from pathlib import Path

import pydata_sphinx_theme  # noqa: F401
import sphinx_book_theme
import sphinx_rtd_theme
import sphinx_theme

# import recommonmark  # noqa: F401
# from recommonmark.transform import AutoStructify


project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "torch_ecg"
docs_root = Path(__file__).resolve().parents[0]

sys.path.insert(0, str(project_root))


# -- Project information -----------------------------------------------------

project = "torch-ecg"
copyright = "2021, WEN Hao, KANG Jingsu"
author = "WEN Hao, KANG Jingsu"

# The full version, including alpha/beta/rc tags
version = Path(src_root / "version.py").read_text().split("=")[1].strip()[1:-1]
release = version

today_fmt = "%Y/%m/%d"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    # "recommonmark",
    # 'sphinx.ext.autosectionlabel',
    "sphinx_multiversion",
    "sphinx_emoji_favicon",
    # "sphinx_toolbox.collapse",  # replaced by dropdown of sphinx_design
    # "numpydoc",
    "sphinxcontrib.tikz",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = [str(docs_root / "references.bib")]
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "wfdb": ("https://wfdb.readthedocs.io/en/latest/", None),
    # "biosppy": ("https://biosppy.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

autodoc_default_options = {
    "show-inheritance": True,
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "DeepPSP",  # Username
    "github_repo": "torch_ecg",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "docs/source/",  # Path in the checkout to the docs root
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# html_sidebars = {"*": ["versions.html"]}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Napoleon settings
napoleon_custom_sections = [
    "ABOUT",
    "ISSUES",
    "Usage",
    "Citation",
    "TODO",
    "Version history",
    "Pipeline",
]
# napoleon_custom_section_rename = False # True is default for backwards compatibility.


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
_theme_name = "pydata_sphinx_theme"  # "stanford_theme" "sphinx_book_theme", etc.

if _theme_name == "stanford_theme":
    html_theme = "stanford_theme"
    html_theme_path = [sphinx_theme.get_html_theme_path("stanford-theme")]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_rtd_theme":
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_book_theme":
    html_theme = "sphinx_book_theme"
    html_theme_path = [sphinx_book_theme.get_html_theme_path()]
    html_theme_options = {
        "repository_url": "https://github.com/DeepPSP/torch_ecg",
        "use_repository_button": True,
        "use_issues_button": True,
        "use_edit_page_button": True,
        "use_download_button": True,
        "use_fullscreen_button": True,
        "path_to_docs": "docs/source",
        "repository_branch": "master",
    }
elif _theme_name == "pydata_sphinx_theme":
    html_theme = "pydata_sphinx_theme"
    html_logo = "_static/images/deep-psp-logo.png"
    html_sidebars = {"index": ["search-button-field"], "**": ["search-button-field", "sidebar-nav-bs"]}
    html_theme_options = {
        "collapse_navigation": False,
        "header_links_before_dropdown": 6,
        # "display_version": True,
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/DeepPSP/torch_ecg",
                "icon": "fa-brands fa-github",
            },
        ],
        "logo": {
            "text": "torch-ecg",
        },
        "navbar_start": ["navbar-logo"],
        "navbar_end": ["theme-switcher", "navbar-icon-links"],
        "navbar_persistent": [],
        "secondary_sidebar_items": ["page-toc"],
    }
else:
    raise ValueError(f"Unknown theme name: {_theme_name}")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

master_doc = "index"

emoji_favicon = ":cœur_battant:"


def setup(app):
    # app.add_config_value(
    #     "recommonmark_config",
    #     {
    #         "url_resolver": lambda url: github_doc_root + url,  # noqa: F821
    #         "auto_toc_tree_section": "Contents",
    #     },
    #     True,
    # )
    # app.add_transform(AutoStructify)
    app.add_css_file("css/custom.css")


latex_documents = [
    (
        master_doc,
        f"{project}.tex",
        f"{project} Documentation",
        author,
        "manual",
    ),
]

# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#latex-backend-fails-with-citations-in-figure-captions
latex_elements = {
    "preamble": r"""
        % make phantomsection empty inside figures
        \usepackage{etoolbox}
        \AtBeginEnvironment{figure}{\renewcommand{\phantomsection}{}}
    """
}

man_pages = [(master_doc, project, f"{project} Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        project,
        f"{project} Documentation",
        author,
        project,
        "ECG Deep Learning Framework Implemented using PyTorch.",
        "Miscellaneous",
    ),
]

numfig = False

linkcheck_ignore = [
    r"https://doi.org/*",  # 418 Client Error
]
