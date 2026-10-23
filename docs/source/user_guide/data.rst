.. _ug_data:

Reading databases
=================

Readers
-------

Every supported database has a reader class in :mod:`torch_ecg.databases`.
A reader handles local storage, record listing, and signal/annotation
loading; it never assumes the data is already present:

.. code-block:: python

    from torch_ecg.databases import CPSC2018

    reader = CPSC2018("~/data/cpsc2018")
    print(len(reader))          # 45 records
    print(reader[0])            # 'A0001' -- records are addressed by name or index
    data = reader.load_data(0)  # float32 ndarray, shape (n_leads=12, n_samples)
    ann = reader.load_ann(0)    # database-specific annotation structure

Readers expose a uniform surface (:meth:`~torch_ecg.databases.base._DataBase.get_absolute_path`,
``fs``, ``all_leads``, database metadata via ``database_info``) so that
downstream code can be database-agnostic.  Use
:func:`~torch_ecg.databases.list_databases` to enumerate the available
readers.

Downloading
-----------

PhysioNet-hosted databases download from the ``s3://physionet-open``
bucket by default -- faster and much more stable than the website:

.. code-block:: python

    reader.download()          # s3://physionet-open/<db>/<version>/ via the `aws` CLI,
                               # or via the pure-Python boto3 backend if the CLI is absent

.. note::

    If the PhysioNet website is unreachable, the reader infers the
    latest database version from the S3 bucket layout instead of
    silently falling back to an old version.

Subsampling
-----------

Any reader accepts a ``subsample`` ratio (0 < ratio <= 1) for quick
experiments; with a fixed ``seed`` the subset is deterministic.

Datasets
--------

Datasets pair a reader with a training configuration and yield tensors.
Start from the library-provided train config for the database, adapt a
few fields, and keep everything else at its tested default:

.. code-block:: python

    from copy import deepcopy
    from torch.utils.data import DataLoader
    from torch_ecg.databases.datasets.cpsc2021 import CPSC2021Dataset, CPSC2021TrainCfg
    from torch_ecg.utils.utils_nn import default_collate_fn as collate_fn

    config = deepcopy(CPSC2021TrainCfg)
    config.db_dir = "~/data/cpsc2021"
    config.log_dir = ...        # benchmark-local dirs, see the examples
    config.model_dir = ...
    config.checkpoints = ...

    ds = CPSC2021Dataset(config, task="qrs_detection", training=True, lazy=False)
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)

The dataset classes pre-load and slice the signals (``lazy=False``) and
apply the configured leads/classes/segment length; ``collate_fn``
handles the (signal, label) pairs.

One-time offline preparation
----------------------------

Some databases derive per-segment files once, up front.  The pattern
(idempotent -- re-runs are skipped):

.. code-block:: python

    ds = CPSC2021Dataset(config, task="qrs_detection", training=True, lazy=True)
    ds.persistence()            # bandpass + slicing + RR sequences, one time only
    del ds

    ds = CPSC2021Dataset(config, task="qrs_detection", training=True, lazy=False)

Train/validation splits follow the config's ratio and are deterministic
given the seed.  See the ``train_hybrid_cpsc2021`` example for the full
flow.
