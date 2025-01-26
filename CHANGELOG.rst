Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.1.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

`Unreleased <https://github.com/DeepPSP/torch_ecg/compare/v0.0.30...HEAD>`__
----------------------------------------------------------------------------

Added
~~~~~

-  Functions for downloading PhysioNet data from AWS S3. It is now made
   the default way to download data from PhysioNet.
-  Add ``easydict`` as a dependency for backward compatibility (loading
   old models using safe-mode ``torch.load`` with ``weights_only=True``.
   Extra dependencies are added with
   ``torch.serialization.add_safe_globals``).

Changed
~~~~~~~

-  Test files (in the `sample-data <sample-data>`__ directory) are
   updated.
-  ``requires-python`` is updated from ``>=3.7`` to ``>=3.9``.
-  Add keyword argument ``weights_only`` to ``from_checkpoint`` and
   ``from_remote`` methods of the models (indeed the ``CkptMixin``
   class). The default value is ``"auto"``, which means the behavior is
   the same as before. It checks if ``torch.serialization`` has
   ``add_safe_globals`` attribute. If it does, it will use safe-mode
   ``torch.load`` with ``weights_only=True``. Otherwise, it will use
   ``torch.load`` with ``weights_only=False``.

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

-  Restrictions on the version of ``wfdb`` and ``numpy`` packages are
   removed.

Fixed
~~~~~

-  Fix IO issues with several PhysioNet databases.

Security
~~~~~~~~

-  Models are now loaded using safe-mode ``torch.load`` with
   ``weights_only=True`` by default.

`0.0.30 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.29...v0.0.30>`__ - 2024-10-10
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.29 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.28...v0.0.29>`__ - 2024-07-21
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.28 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.27...v0.0.28>`__ - 2024-04-02
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.27 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.26...v0.0.27>`__ - 2023-03-14
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.26 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.25...v0.0.26>`__ - 2022-12-25
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.25 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.24...v0.0.25>`__ - 2022-10-08
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.24 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.23...v0.0.24>`__ - 2022-08-13
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.23 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.22...v0.0.23>`__ - 2022-08-09
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.22 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.21...v0.0.22>`__ - 2022-08-05
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.21 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.20...v0.0.21>`__ - 2022-08-01
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.20 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.19...v0.0.20>`__ - 2022-06-15
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.19 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.18...v0.0.19>`__ - 2022-06-09
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.18 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.17...v0.0.18>`__ - 2022-06-05
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.17 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.16...v0.0.17>`__ - 2022-05-03
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.16 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.15...v0.0.16>`__ - 2022-04-28
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.15 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.14...v0.0.15>`__ - 2022-04-14
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.14 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.13...v0.0.14>`__ - 2022-04-10
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.13 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.12...v0.0.13>`__ - 2022-04-09
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.12 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.11...v0.0.12>`__ - 2022-04-05
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.11 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.10...v0.0.11>`__ - 2022-04-03
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.10 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.9...v0.0.10>`__ - 2022-04-01
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.9 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.8...v0.0.9>`__ - 2023-03-30
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.8 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.7...v0.0.8>`__ - 2022-03-29
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.7 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.6...v0.0.7>`__ - 2022-03-28
----------------------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

Security
~~~~~~~~

`0.0.6 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.5...v0.0.6>`__ - 2022-03-28
----------------------------------------------------------------------------------------

Added
~~~~~

- Methods ``__len__`` and ``__getitem__`` for the base class ``_DataBase``.

Changed
~~~~~~~

- The base class of ``CPSC2021`` is changed from ``CPSCDataBase`` to
  ``PhysioNetDataBase``.
- Function ``compute_output_shape`` is enhanced to support different
  paddings in two ends of the input signal.
- ``README`` is updated.
- Docstrings of many classes and functions are updated.
- ``black`` is used for code formatting.

`0.0.5 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.4...v0.0.5>`__ - 2022-03-27
----------------------------------------------------------------------------------------

Added
~~~~~

- Cached list of PhysioNet databases as a data file
  stored in the package.
- ``requests`` as a dependency in the ``requirements.txt`` file.

Changed
~~~~~~~

- An optional argument ``btype`` is added to the function
  ``butter_bandpass_filter`` to specify the type of the filter:
  ``"lohi"``, ``"hilo"``.
- A ``compressed`` argument is added to the ``download`` method of the
  ``PhysioNetDataBase`` class to specify whether to download the
  compressed version of the database.

Fixed
~~~~~

- Fix bugs in the function ``preprocess_multi_lead_signal``.

`0.0.4 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.2...v0.0.4>`__ - 2022-03-26
----------------------------------------------------------------------------------------

Added
~~~~~

- ``ReprMixin`` class for better representation of the classes (e.g., models,
  preprocessors, database readers, etc.).
- Added model_dir to default config.
- ``Dataset`` classes for generating input data for the models:
   - ``CINC2020``
   - ``CINC2021``
   - ``CPSC2019``
   - ``CPSC2021``
   - ``LUDB``
- ``sample-data`` directory for storing sample data for testing.
- ``url`` property to the database classes.
- Utility functions for the computation of metrics.
- ``BeatAnn`` class for better annotation of ECG beats.
- Download utility functions.
- Output classes for the models. The output classes are used to store the
  output of the models and provide methods for post-processing.

Changed
~~~~~~~

- Manipulation of custom preprocessor classes is enhanced.
- ``SizeMixin`` class is improved for better computation of the sizes of the models.
- Replace ``os`` with ``pathlib``, which is more flexible for path operations.
- Several database reader classes are updated: mitdb, ltafdb.
- Improve ``PhysioNetDataBase`` by using wfdb built-in methods of
  getting database version string and downloading the database.
- Update the ``README`` file.

Removed
~~~~~~~

- Unnecessary imports are removed.

Fixed
~~~~~

- Fix bugs in the ``flush`` method of the ``TxtLogger``.

0.0.3 - 2022-03-24 [YANKED]
-----------------------------

This release was yanked.

`0.0.2 <https://github.com/DeepPSP/torch_ecg/releases/tag/v0.0.2>`__ - 2022-03-04
----------------------------------------------------------------------------------------

Added
~~~~~

- Preprocessor classes for ECG data preprocessing.
- Augmenter classes for ECG data augmentation.
- Database reader classes for reading ECG data from different sources.
- Model classes for ECG signal analysis, including classification,
  segmentation (R-peak detection, wave delineation, etc.).
- Several benchmark studies for ECG signal analysis tasks:
   - CinC2020, multi-label classification.
   - CinC2021, multi-label classification.
   - CPSC2019, QRS detection.
   - CPSC2020, single-label classification.
   - CPSC2021, single-label classification.
   - LUDB, wave delineation.
- Documentation for the project.
- CodeQL action for security analysis.
- Unit tests for the project.

0.0.1 - 2022-03-03 [YANKED]
-----------------------------

This release was yanked.
