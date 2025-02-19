Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a
Changelog <https://keepachangelog.com/en/1.1.0/>`__, and this project
adheres to `Semantic
Versioning <https://semver.org/spec/v2.0.0.html>`__.

`Unreleased <https://github.com/DeepPSP/torch_ecg/compare/v0.0.31...HEAD>`__
----------------------------------------------------------------------------

Added
~~~~~

Changed
~~~~~~~

- Make the function `remove_spikes_naive` in `torch_ecg.utils.utils_signal`
  support 2D and 3D input signals.

Deprecated
~~~~~~~~~~

Removed
~~~~~~~

Fixed
~~~~~

- Correctly update the `_df_metadata` attribute of the `PTBXL` database reader
  classes after filtering records.
- Enhance the `save` method of the `torch_ecg.utils.utils_nn.CkptMixin` class:
  non-safe items in the configs are removed before saving the model.

Security
~~~~~~~~

`0.0.31 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.30...v0.0.31>`__ - 2025-01-28
----------------------------------------------------------------------------------------

Added
~~~~~

- Add functions for downloading PhysioNet data from AWS S3. It is now made
  the default way to download data from PhysioNet.
- Add ``easydict`` as a dependency for backward compatibility (loading
  old models using safe-mode ``torch.load`` with ``weights_only=True``.
  Extra dependencies are added with
  ``torch.serialization.add_safe_globals``).

Changed
~~~~~~~

- Test files (in the ``sample-data`` directory) are updated.
- Add keyword argument ``weights_only`` to ``from_checkpoint`` and
  ``from_remote`` methods of the models (indeed the ``CkptMixin``
  class). The default value is ``"auto"``, which means the behavior is
  the same as before. It checks if ``torch.serialization`` has
  ``add_safe_globals`` attribute. If it does, it will use safe-mode
  ``torch.load`` with ``weights_only=True``. Otherwise, it will use
  ``torch.load`` with ``weights_only=False``.

Deprecated
~~~~~~~~~~

- Support for Python 3.7, 3.8 is deprecated. The minimum supported Python
  version is now 3.9. In the ``pyproject.toml`` file, the field
  ``requires-python`` is updated from ``>=3.7`` to ``>=3.9``.

Removed
~~~~~~~

- Restrictions on the version of ``wfdb`` and ``numpy`` packages are
  removed.

Fixed
~~~~~

- Fix IO issues with several PhysioNet databases.

Security
~~~~~~~~

- Models are now loaded using safe-mode ``torch.load`` with
  ``weights_only=True`` by default.

`0.0.30 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.29...v0.0.30>`__ - 2024-10-10
----------------------------------------------------------------------------------------

Added
~~~~~

- Add support for AWS S3 in the download utility function ``http_get``
  in the ``torch_ecg.utils.download`` module. PhysioNet now provides
  data download links from AWS S3, and the download utility function
  can now handle these links if AWS CLI is installed. This feature is
  implemented but not put into use yet.

Changed
~~~~~~~

- Change the default value of the ``method`` argument of the
  ``torch_ecg.utils.utils_signal.resample_irregular_timeseries`` function
  from ``"spline"`` to ``"interp1d"``. The former is probably not
  correctly implemented.
- Update the logger classes: add checking of the write access of the
  ``log_dir``. If the directory is not writable, the default log dir
  ``~/.cache/torch_ecg/logs`` is used (ref. ``torch_ecg.cfg.DEFAULTS.log_dir``).
- Update the selection mechanism of the final model for the trainer
  classes: if no monitor is specified, the last model is selected by
  default (previously, no model was selected and saved).
- The main part of the ``_setup_criterion`` method of the ``BaseTrainer``
  class is moved to the function ``setup_criterion`` in the
  ``torch_ecg.models.loss`` module. The method is simplified and
  enhanced.

Deprecated
~~~~~~~~~~

- Script ``setup.py`` is deprecated. The package building system is
  switched to ``hatch``.

Removed
~~~~~~~

- Remove redundancy in base trainer classes: identical ``if`` blocks
  are removed from the ``_setup_criterion`` method of the ``BaseTrainer``
  class.

Fixed
~~~~~

- Fix potential error in getting model name in the trainer classes.
- Fix bugs in the ``CINC2020`` and ``CINC2021`` database reader classes
  for parsing the header files.

`0.0.29 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.28...v0.0.29>`__ - 2024-07-21
----------------------------------------------------------------------------------------

Added
~~~~~

- Add keyword argument ``with_suffix`` to function
  ``torch_ecg.utils.misc.get_record_list_recursive3``.
- Add function ``_download_from_google_drive`` to the
  ``torch_ecg.utils.download`` module for downloading files from Google
  Drive.
- Add ``gdown`` as a dependency in the ``requirements.txt`` file.
- Add database reader class ``PTBXLPlus`` for the PTB-XL+ database in
  ``torch_ecg.databases.physionet_databases``.
- Add github-release job to the publish action for creating a release
  on GitHub automatically.

Changed
~~~~~~~

- Improve the main training loop method of the base trainer class
  ``torch_ecg.components.trainers.BaseTrainer``.
- Allow passing additional keyword arguments to pass to ``requests.head``
  in the ``url_is_reachable`` function of the ``torch_ecg.utils.download``
  module (via adding the ``**kwargs`` argument).
- Restrict version of ``numpy`` to be ``<=2.0.0`` in the
  ``requirements.txt`` file. ``numpy`` version ``2.0.0`` is a breaking
  update, and a large proportion of the dependencies of this project
  are not compatible with it yet.
- Enhance the ``cls_to_bin`` function and rename it to ``one_hot_encode``
  in the ``torch_ecg.utils.utils_data`` module.

Fixed
~~~~~

- Enhance compatibility for different ``pandas`` versions.
- Fix errors for taking length of an empty database reader class.

Security
~~~~~~~~

- Fix code scanning alert - Incomplete regular expression for hostnames
  `#21 <https://github.com/DeepPSP/torch_ecg/pull/21>`__.
- Fix code scanning alert - Incomplete URL substring sanitization
  `#23 <https://github.com/DeepPSP/torch_ecg/pull/23>`__.

`0.0.28 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.27...v0.0.28>`__ - 2024-04-02
----------------------------------------------------------------------------------------

Added
~~~~~

- Add CD workflow for the publish action with GitHub Action.
- Add an optional argument ``return_fs`` for the ``load_data``
  method for the database reader classes. If ``True``, the sampling
  frequency of the record is returned along with the data as a tuple.
  To keep the behavior consistent, the default value is ``False``.
- Add an optional parameter ``fs`` for the function ``compute_receptive_field``
  in the ``torch_ecg.utils.utils_nn`` module. If ``fs`` is provided, the
  receptive field is computed based on the sampling frequency.
- Add method ``compute_receptive_field`` for several convolutional neural
  network models (layers) in the ``torch_ecg.models._nets`` module.
- Add helper function ``make_serializable`` in the ``torch_ecg.utils.misc``
  module for making an object serializable (with the ``json`` package).
  It will convert all ``numpy`` arrays to ``list`` in an object, and
  also convert ``numpy`` data types to python data types in the object
  recursively.
- Add helper function ``url_is_reachable`` in the ``torch_ecg.utils.download``
  module for checking if a URL is reachable.
- Add database reader class ``PTBXL`` for the PTB-XL database in
  ``torch_ecg.databases.physionet_databases``.
- Add class method ``from_remote`` for ``CkptMixin`` classes. It is used
  to load a model from a remote location (e.g., a URL) directly.
- Add ``sphinx-emoji-favicon`` as a dependency for generating the favicon
  for the documentation.
- Add utility function ``ecg_plot`` from
  `ecg-image-kit <https://github.com/alphanumericslab/ecg-image-kit/.>`__.
- Add ``pyarrow`` as a dependency in the ``requirements.txt`` file.
- Add benchmark study ``train_crnn_cinc2023`` for the CinC2023 challenge.

Changed
~~~~~~~

- Change the default value ``reset_index`` of the utility function
  ``torch_ecg.utils.utils_data.stratified_train_test_split`` from
  ``True`` to ``False``.
- Enhance the decorator ``torch_ecg.utils.misc.add_kwargs`` so that
  the signature of the decorated function is also updated.
- Update the documentation: use ``sphinx_toolbox.collapse`` and
  ``sphinxcontrib.bibtex``; add citation info in the index page.
- Make ``Dataset`` classes accept slice index for the ``__getitem__``
  method.

Deprecated
~~~~~~~~~~

- Support for Python 3.6 is deprecated. The minimum supported Python
  version is updated to 3.7.

Removed
~~~~~~~

- Remove broken links in the docstrings of the database reader classes.
- Remove unused scripts ``formatting.sh`` and ``push2pypi.sh``.

Fixed
~~~~~

- Fix errors in the decorator ``torch_ecg.utils.misc.add_kwargs``
  when a bound method is decorated.
- Fix bugs related to data overflow for preprocessor classes that
  work with ``numpy`` arrays as reported in issue
  `#12 <https://github.com/DeepPSP/torch_ecg/issues/12>`__.
- Fix bugs in augmentor class ``StretchCompress`` in the
  ``torch_ecg.augmenters`` module.
- Fix dtype error when calling ``compute_class_weight`` from
  ``sklearn.utils``.
- Fix the issue when handling nan values in in computing metrics.
- Fix errors for the ``ApneaECG`` database reader class when passing
  a path that does not exist or a path that contains no records at
  initialization.

`0.0.27 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.26...v0.0.27>`__ - 2023-03-14
----------------------------------------------------------------------------------------

Added
~~~~~

- Add default configs for blocks of the ``ResNet`` model in the
  ``torch_ecg.models.cnn`` module.
- Add ``RegNet`` model in the ``torch_ecg.models.cnn`` module.
- Add ``CutMix`` augmentor in the ``torch_ecg.augmenters`` module.
- Add support for ``torch.nn.Dropout1d`` in the models.
- Add ``.readthedocs.yml`` to the project. The documentation is
  now hosted on Read the Docs besides GitHub Pages.

Changed
~~~~~~~

- Move ``torch_ecg.utils.preproc`` to ``torch_ecg.utils._preproc``.
- Allow ``embed_dim`` of ``SelfAttention`` layer not divisible by
  ``num_heads`` via adding a linear projection layer before the
  multi-head attention layer.
- Documentation is largely improved.

Deprecated
~~~~~~~~~~

- Drop compability for older versions of ``torch`` (1.5 and below).

Removed
~~~~~~~

- Remove ``protobuf`` from the ``requirements.txt`` file.
- Clear unused methods in the ``CINC2020`` and ``CINC2021`` database
  reader classes.
- Clear unused layers in the ``torch_ecg.models._nets`` module.
- Remove the ``torch_ecg.utils._pantompkins`` module. It contains
  the implementation of the Pan-Tompkins algorithm for QRS detection,
  modified from old versions of the ``wfdb`` package. It is moved to
  the ``legacy`` folder of the project.
- Remove ``WandbLogger`` class from the ``torch_ecg.components.loggers``
  module.

Fixed
~~~~~

- Fix bugs when passing ``units=None`` for the ``load_data`` method
  of the PhysioNet database reader classes.

`0.0.26 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.25...v0.0.26>`__ - 2022-12-25
----------------------------------------------------------------------------------------

Added
~~~~~

- Add a default ``load_data`` method for physionet databases reader
  classes in the base class ``PhysioNetDataBase``. In most cases,
  in the inherited classes, one does not need to implement the
  ``load_data`` method, as the default method is sufficient. This
  method is a slight improvement over ``wfdb.rdrecord``.
- Add decorator ``add_kwargs`` in the ``torch_ecg.utils.misc`` module
  for adding keyword arguments to a function or method.
- Add functions ``list_databases``, ``list_datasets`` in the
  ``torch_ecg.datasets`` module for listing available databases reader
  classes and ``Dataset`` classes.
- Add ``save`` method for the ``CkptMixin`` class. It is used to save
  the model to a file.
- Add ``_normalize_leads`` a method of the base ``_DataBase`` class
  in the ``torch_ecg.databases.base`` module. It is used to normalize
  the names of the ECG leads.
- Add subsampling functionality for database reader classes.
- Add benchmark study ``train_mtl_cinc2022`` for the CinC2022 challenge.
- Add ``CITATIONS.bib`` file for storing BibTeX entries of the
  papers related to the project.
- Add 10 sample data from the CPSC2018 database for testing in the
  ``sample-data`` directory.

Changed
~~~~~~~

- Use ``CitationMixin`` from the ``bib-lookup`` package as the base
  class for the ``DataBaseInfo`` class in ``torch_ecg.databases.base``.
- Use ``CitationMixin`` as one of the base classes for the models
  in ``torch_ecg.models``.
- Allow dummy (empty) preprocessor managers, a warning instead of an
  error is raised in such cases.
- Enhance error message for the computation of metrics.
- Add keyword argument ``requires_grad`` and ``include_buffers`` to
  the ``torch_ecg.utils.utils_nn.compute_module_size`` function.
  The ``dtype`` argument is removed as the data type of the model
  is now inferred from the model itself.
- Improve several database reader classes: ``CPSC2018``, ``CPSC2021``,
  ``CINC2017``, ``ApneaECG``, ``MITDB``, ``SPH``.
- Add asymmetric zero pad for convolution layers, so that when
  ``stride = 1`` and ``kernel_size`` is even, strict ``"same"``
  padding is conducted.
- Use loggers instead of ``print`` in database reader classes.
- Integrate code coverage into the CI workflow. The coverage report
  is generated and uploaded to Codecov.
- More unit tests are added, and the existing ones are updated.
  Code coverage is largely improved.

Deprecated
~~~~~~~~~~

- Drop compatibility for ``tqdm`` < 4.29.1

Removed
~~~~~~~

- Remove unused rpeaks detection methods in the ``torch_ecg.utils.rpeaks``
  module.
- Remove ``_normalize_leads`` method in ``LUDB`` database reader class.
- Remove unused functions in the file of the ``CPSC2020`` database reader
  class.

Fixed
~~~~~

- Fix bugs in the config class ``torch_ecg.cfg.CFG``.
- Fix errors in the ``plot`` method of ``CINC2020`` and ``CINC2021``
  database reader classes.

Security
~~~~~~~~

- `CVE-2007-4559 <https://github.com/advisories/GHSA-gw9q-c7gh-j9vm>`__
  patch: Fix a potential security vulnerability in the
  ``torch_ecg.utils.download.http_get`` function.

`0.0.25 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.23...v0.0.25>`__ - 2022-10-08
----------------------------------------------------------------------------------------

Added
~~~~~

- Add docstring utility function ``remove_parameters_returns_from_docstring``
  in ``torch_ecg.utils.misc``.
- Add abstract property ``database_info`` to the base class ``_DataBase`` in
  ``torch_ecg.databases.base`` so that when implementing a new database reader
  class that inherits from the base class, its ``DataBaseInfo`` must be
  implemented and assigned to the property.
- Add method ``get_citation`` to the base abstract class ``_DataBase`` in
  ``torch_ecg.databases.base`` which enhances the process for getting citations
  for the databases.
- Add database reader class ``CACHET_CADB`` for the CACHET-CADB database in
  ``torch_ecg.databases.other_databases``.
- Add ``download`` method for the base abstract class ``CPSCDataBase`` in
  ``torch_ecg.databases.base``.

Changed
~~~~~~~

- Improve the warning message for passing an non-existing path when
  initializing a database reader class.
- Change the default behavior of the ``download`` method for
  ``PhysioNetDataBase`` class: default to download the compressed
  version of the database.
- Update the ``README`` file in the ``torch_ecg/databases`` directory.

Fixed
~~~~~

- Use ``register_buffer`` in custom loss classes for constant tensors
  to avoid potential device mismatch issues.
- Rename and update the data file ``physionet_dbs.csv.tar.gz`` to
  ``physionet_dbs.csv.gz`` to comply with the changement of the
  ``pandas.read_csv`` function from version 1.4.x to 1.5.x.
- Fix the incorrect usage of ``NoReturn`` type hints. It is replaced
  with ``None`` to indicate that the function/method does not return
  anything.

0.0.24 - 2022-08-13 [YANKED]
-----------------------------

This release was yanked.

`0.0.23 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.22...v0.0.23>`__ - 2022-08-09
----------------------------------------------------------------------------------------

Added
~~~~~

- Add ``collate_fn`` as an optional argument for ``BaseTrainer`` class
  in ``torch_ecg.components.trainers``.

Changed
~~~~~~~

- Let ``db_dir`` attribute of the database reader classes be absolute
  when instantiated, to avoid potential ``pathlib`` errors.
- Update utility function `torch_ecg.utils.utils_nn.adjust_cnn_filter_lengths``:
  avoid assigning unnecessary fs to dict-type config items; change default
  value of the ``pattern`` argument from ``"filter_length|filt_size"`` to
  ``"filter_length|filter_size"`` to avoid unintended changement of configs
  for ``BlurPool`` (in ``torch_ecg.models._nets``).
- Enhance error message for ``BlurPool`` in ``torch_ecg.models._nets``.

`0.0.22 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.21...v0.0.22>`__ - 2022-08-05
----------------------------------------------------------------------------------------

Changed
~~~~~~~

- Make utility function ``torch_ecg.utils.utils_data.default_collate_fn``
  support ``dict`` type batched data.
- Update docstrings of several metrics utility functions in
  ``torch_ecg.utils.utils_metrics``.

`0.0.21 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.20...v0.0.21>`__ - 2022-08-01
----------------------------------------------------------------------------------------

Added
~~~~~

- Add utility function ``get_kwargs`` in ``torch_ecg.utils.misc`` for
  getting (keyword) arguments from a function/method.
- Add AHA diagnosis statements in ``torch_ecg.databases.aux_data``.
- Add argument ``reset_index`` to the utility function
  ``torch_ecg.utils.utils_data.stratified_train_test_split``.
- Add ``typing-extensions`` as a dependency in the ``requirements.txt``
  file.
- Add database reader class ``QTDB`` for the QTDB database in
  ``torch_ecg.databases.physionet_databases``.

Changed
~~~~~~~

- Enhance data handling (typicall when using the ``load_data`` method of
  the database reader classes) with precise dtypes via
  ``torch_ecg.cfg.DEFAUTLS``.
- Update the setup of optimizer for the base trainer class
  ``torch_ecg.components.trainers.BaseTrainer``.
- Update the ``DataBaseInfo`` class for the ``SPH`` database.
- Update the ``README`` file in the ``torch_ecg/databases`` directory.
- Update plotted figures of the benchmark studies.
- Rename ``SequenceLabelingOutput`` to ``SequenceLabellingOutput``
  (typo fixed) in the ``torch_ecg.components.outputs`` module.
- Enhance docstring of ``LUDB`` database reader class via updating its
  ``DataBaseInfo`` class.
- Append the ``_ls_rec`` method as the last step in the ``download``
  method of the database reader classes.
- Change ``torch_ecg.utils.utils_data.ECGWaveForm`` from a ``namedtuple``
  to a ``dataclass``.

Removed
~~~~~~~

- ``bib_lookup.py`` is removed from the project. It is now delivered in
  an isolated package ``bib_lookup`` published on PyPI, and added as a
  dependency in the ``requirements.txt`` file.
- Remove unnecessary script ``exec_git.py``.
- Remove ``joblib`` in the ``requirements.txt`` file.

`0.0.20 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.19...v0.0.20>`__ - 2022-06-15
----------------------------------------------------------------------------------------

Added
~~~~~

- Add database reader class ``SPH`` for the SPH database in
  ``torch_ecg.databases.other_databases``.
- Add ``dataclass`` ``DataBaseInfo`` for storing information of a
  database. It has attributes ``title``, ``about``, ``note``,
  ``usage``, ``issues``, ``reference``, etc., and has a method
  ``format_database_docstring`` for formatting the docstring of a
  database reader class. The generated docstring can be assigned to
  corresponding database reader class via the ``add_docstring``
  decorator (in ``torch_ecg.utils.misc``).
- Add default cache directory ``~/.cache/torch_ecg`` for storing
  downloaded data files, model weight files, etc.
- Add helper function ``is_compressed_file`` for checking if a file is
  compressed in ``torch_ecg.utils.download``.

`0.0.19 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.18...v0.0.19>`__ - 2022-06-09
----------------------------------------------------------------------------------------

Added
~~~~~

- Add argument ``relative`` to the utility function ``get_record_list_recursive3``.
- Add attribute ``_df_records`` to the database reader classes. The attribute
  stores the DataFrame of the records of the database, containing paths to the
  records and other information (labels, demographics, etc.).

Fixed
~~~~~

- Fix bugs in the download utility function ``http_get``.
- Fix bugs in the database reader classe ``CPSC2021``.

`0.0.18 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.16...v0.0.18>`__ - 2022-06-05
----------------------------------------------------------------------------------------

Added
~~~~~

- Add property ``in_channels`` for the models.The number of input channels
  is stored as a private attribute ``_in_channels``, and the property
  ``in_channels`` makes it easier to access the value.
- Add warning message to the ``download`` method of the ``CPSC2019`` database
  reader class.
- Add ``get_absolute_path`` method for the database reader classes to
  uniformly handle the path operations.

Changed
~~~~~~~

- All all absolute imports are replaced with relative imports.
- Update citation and images for several benchmark studies
- Update the ``downlaod`` link for the ``CPSC2019`` database reader class
  (ref. property ``torch_ecg.databases.CPSC2019.url``).

Removed
~~~~~~~

- Remove the ``torch_ecg.utils.misc.deprecate_kwargs`` decorator. It is
  delivered in an isolated package ``deprecate_kwargs`` published on PyPI,
  and added as a dependency in the ``requirements.txt`` file.

Fixed
~~~~~

- Fix errors in the ``_ls_rec`` method of the ``CPSC2019`` database reader
  class.
- Fix bugs in the ``torch_ecg.utils.misc.deprecate_kwargs`` decorator.
- Fix the issue that ``tensorboardX`` is incompatible with the latest version
  of ``protobuf``.

0.0.17 - 2022-05-03 [YANKED]
-----------------------------

This release was yanked.

`0.0.16 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.15...v0.0.16>`__ - 2022-04-28
----------------------------------------------------------------------------------------

Added
~~~~~

- Add method ``_categorize_records`` for the ``MITDB`` database reader class,
  categorize records by specific attributes. Related helper properties
  ``beat_types_records`` and ``rhythm_types_records`` are added.
- Add method ``_aggregate_stats`` for the ``MITDB`` database reader class.
  Related helper properties ``df_stats`` and ``db_stats`` are added.
- Add  function ``cls_to_bin`` for converting categorical (typically multi-label)
  class labels to binary class labels (2D array with 0/1 values).
- Add context manager ``torch_ecg.utils.misc.timeout`` for setting a timeout for
  a block of code.
- Add context manager ``torch_ecg.utils.misc.Timer`` to time the execution of
  a block of code.
- Add module ``torch_ecg.components.inputs`` for input data classes.
- Add class ``Spectrogram`` (in ``torch_ecg.utils``) for generating spectrogram
  input data. This class is modified from the ``torchaudio.transforms.Spectrogram``.
- Add decorator ``torch_ecg.utils.misc.deprecate_kwargs`` for deprecating keyword
  arguments of a function/method.
- Top-level module ``torch_ecg.ssl`` for self-supervised learning methods and
  models is introduced, but not implemented yet.
- Add helper function ``torch_ecg.utils.utils_nn.compute_sequential_output_shape``
  to simplify the computation of output shape of sequential models.
- ``mobilenet_v3`` model is added to the ``torch_ecg.models`` module. It is
  now available as a cnn backbone choice for the ``ECG_CRNN`` model (and for other
  downstream task models).

Changed
~~~~~~~

- Use ``numpy``'s default ``rng`` for random number generation in place
  of ``np.random`` and Python built-in ``random`` module.
- Update the ``README`` file.
- Move the function ``generate_weight_mask`` from ``CPSC2021`` dataset
  to ``torch_ecg.utils.utils_data``.
- Database reader ``MITDB`` is enhanced: add properties ``df_stats_expanded``;
  add arguments ``beat_types`` and ``rhythm_types`` to the data and annotation
  loading methods.
- Downloading function ``http_get`` is enhanced to support downloading
  normal files other than compressed files.
- Update ``__init__`` file of the ``torch_ecg.utils`` module.
- Database reader class ``CinC2017`` is updated: add property ``_validation_set``.
- The ``ECG_UNET`` model is simplified by removing the unnecessary zero padding
  along the channel axis.
- Update the ``README`` file.

Deprecated
~~~~~~~~~~

- Keyword argument ``batch_norm`` in model building blocks (ref. ``torch_ecg.models``)
  is deprecated. Use ``norm`` instead.

Removed
~~~~~~~

- Redundant functions in ``torch_ecg.utils.utils_interval`` are removed:
  ``diff_with_step``, ``mask_to_intervals``.

Fixed
~~~~~

- Remove redudant code for the ``ECG_UNET`` model which might cause error in
  computing output shapes.

`0.0.15 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.14...v0.0.15>`__ - 2022-04-14
----------------------------------------------------------------------------------------

Changed
~~~~~~~

- Use ``pathlib.Path.parents`` instead of sequence of ``pathlib.Path..parent``
  to get the parent directory of a file path.
- Type hints and docstrings of some database reader classes are enhanced:
  ``ApneaECG``, ``CINC2020``, ``CINC2021``.
- Update the ``README`` file: add citation information for the package.

`0.0.14 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.13...v0.0.14>`__ - 2022-04-10
----------------------------------------------------------------------------------------

Added
~~~~~

- Implements the lead-wise mechanism (as a method ``_assign_weights_lead_wise``)
  for the ``Conv_Bn_Activation`` layer in the ``torch_ecg.models._nets`` module.
- Implements ``assign_weights_lead_wise`` for model ``MultiScopicCNN``
  (in ``torch_ecg.models``).
- Zenodo configuration file ``.zenodo.json`` is added.

Changed
~~~~~~~

- Update the ``README`` file: add ``:point_right: [Back to TOC](#torch_ecg)``
  to the end of long sections.

Fixed
~~~~~

- Fix errors in the computation of classification metrics.

`0.0.13 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.12...v0.0.13>`__ - 2022-04-09
----------------------------------------------------------------------------------------

Added
~~~~~

- Add metrics computation class ``WaveDelineationMetrics`` for evaluating the
  performance of ECG wave delineation models.
- Add methods for computing the metrics to the output classes (in the module
  ``torch_ecg.components.outputs``).
- Add script ``push2pypi.sh`` for pushing the package to PyPI.
- Add attribute ``global_pool_size`` to the configuration of the classification
  models (``torch_ecg.models.ECG_CRNN``).

Changed
~~~~~~~

- ``flake8`` check ignore list is updated.
- ``README`` is updated.

Removed
~~~~~~~

- Usage of ``easydict`` is removed. Now we use ``torch_ecg.cfg.CFG`` for
  configuration.

Fixed
~~~~~

- Computation of the metric of `mean_error` for ECG wave delineation is corrected.
- Fix bugs in ``SpaceToDepth`` layer (``torch_ecg.models.resnet``).

`0.0.12 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.11...v0.0.12>`__ - 2022-04-05
----------------------------------------------------------------------------------------

Changed
~~~~~~~

- Some out-of-date ``sample-data`` files are updated, unnecessary files
  are removed.
- Passing a path that does not exist to a database reader class now raises
  no error, but a warning is issued instead.
- Include ``isort`` and ``flake8`` in the code formatting and linting steps.
  Code are reformatted and linted.

`0.0.11 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.10...v0.0.11>`__ - 2022-04-03
----------------------------------------------------------------------------------------

Changed
~~~~~~~

- Docstrings are cleaned up.
- Unit tests are updated.

`0.0.10 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.9...v0.0.10>`__ - 2022-04-01
----------------------------------------------------------------------------------------

Added
~~~~~

- Add ``BibLookup`` class for looking up BibTeX entries from DOIs
  of papers related to datasets and models.
- Add ``RPeaksDetectionMetrics`` class to the ``torch_ecg.components.metrics``
  module for evaluating the performance of R-peaks detection models.
- Add CI workflow for running tests via GitHub Actions.

Changed
~~~~~~~

- The loading methods (``load_data``, ``load_ann``, etc.) of the database
  reader classes are enhanced to accept ``int`` type record name argument
  (``rec``), which redirects to the record with the corresponding index
  in the ``all_records`` attribute of the database reader class.

`0.0.9 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.8...v0.0.9>`__ - 2023-03-30
----------------------------------------------------------------------------------------

Added
~~~~~

- Add decorator ``add_docstring`` for adding/modifying docstrings of functions
  and classes.
- Add method ``append`` for the ``BaseOutput`` class.
- Add several metrics computation functions in ``torch_ecg/utils/utils_metrics.py``:
   - ``confusion_matrix``
   - ``ovr_confusion_matrix``
   - ``auc``
   - ``accuracy``
   - ``f_measure``
   - ``QRS_score``
- Add top-level module ``torch_ecg.components``.
- Add classes for metrics computation to the ``torch_ecg.components.metrics`` module.

Changed
~~~~~~~

- ``Dataset`` classes and corresponding config classes are added to the
  ``__init__.py`` file of the ``torch_ecg.databases.dataset`` module
  so that they can be imported directly from the module.
- Logger classes, output classes, and trainer classes are moved to the new
  module ``torch_ecg.components``.
- Callbacks in ``BaseTrainer`` are enhanced, allowing empty monitor, and allowing
  non-positive number of checkpoints to be saved (i.e., no checkpoint is saved).

`0.0.8 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.7...v0.0.8>`__ - 2022-03-29
----------------------------------------------------------------------------------------

Fixed
~~~~~

- Bugs in extracting compressed files in the ``http_get`` function
  of the ``utils.download`` module.

Security
~~~~~~~~

`0.0.7 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.6...v0.0.7>`__ - 2022-03-28
----------------------------------------------------------------------------------------

Fixed
~~~~~

- Import errors for early versions of pytorch.
- Cached table of PhysioNet databases is added as ``package_data`` in
  ``setup.py`` to avoid the error of missing the table file when
  installing the package.

Security
~~~~~~~~

`0.0.6 <https://github.com/DeepPSP/torch_ecg/compare/v0.0.5...v0.0.6>`__ - 2022-03-28
----------------------------------------------------------------------------------------

Added
~~~~~

- Add methods ``__len__`` and ``__getitem__`` for the base class
  ``torch_ecg.databases.base._DataBase``.

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

- Add cached table of PhysioNet databases as a data file
  stored in the package.
- Add ``requests`` as a dependency in the ``requirements.txt`` file.

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

- Add ``ReprMixin`` class for better representation of the classes
  (e.g., models, preprocessors, database readers, etc.).
- Added model_dir to default config.
- Add ``Dataset`` classes for generating input data for the models:
   - ``CINC2020``
   - ``CINC2021``
   - ``CPSC2019``
   - ``CPSC2021``
   - ``LUDB``
- Add ``sample-data`` directory for storing sample data for testing.
- Add ``url`` property to the database classes.
- Add utility functions for the computation of metrics.
- Add ``BeatAnn`` class for better annotation of ECG beats.
- Add download utility functions.
- Add ``Output`` classes for the models. The output classes are used to
  store the output of the models and provide methods for post-processing.

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

- Add ``Preprocessor`` classes for ECG data preprocessing (ref.
  ``torch_ecg.preprocessors``).
- Add ``Augmenter`` classes for ECG data augmentation (ref.
  ``torch_ecg.augmenters``).
- Add database reader classes for reading ECG data from different
  sources (ref. ``torch_ecg.databases``).
- Add model classes for ECG signal analysis, including classification,
  segmentation (R-peak detection, wave delineation, etc., ref.
  ``torch_ecg.models``).
- Add several benchmark studies for ECG signal analysis tasks:

   - CinC2020, multi-label classification.
   - CinC2021, multi-label classification.
   - CPSC2019, QRS detection.
   - CPSC2020, single-label classification.
   - CPSC2021, single-label classification.
   - LUDB, wave delineation.

  ref. the ``benchmarks`` directory of the project.
- Add documentation for the project (ref. ``docs`` directory).
- Add CodeQL action for security analysis (ref. ``.github/workflows``).
- Add unit tests for the project (ref. ``test`` directory).

0.0.1 - 2022-03-03 [YANKED]
-----------------------------

This release was yanked.
