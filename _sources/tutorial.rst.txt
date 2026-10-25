.. _tutorial:

Tutorial
========

A five-minute tour of the main building blocks.  The complete, runnable
versions of these patterns live in the :ref:`examples <examples>`.

1. Read a database
------------------

Every supported database has a reader class in :mod:`torch_ecg.databases`
that handles local caching, record listing and signal/annotation loading:

.. code-block:: python

    from torch_ecg.databases import LUDB

    reader = LUDB("~/data/ludb")      # local storage path
    reader.download()                  # from the physionet-open S3 bucket
    data = reader.load_data(0)         # (n_leads, n_samples) array
    ann = reader.load_ann(0)           # waveform annotations

2. Wrap it in a dataset
-----------------------

Datasets pair readers with a training configuration (leads, classes,
segment length, ...) and yield tensors:

.. code-block:: python

    from torch_ecg.databases.datasets import LUDBDataset, LUDBTrainCfg

    train_cfg = LUDBTrainCfg.copy()
    train_cfg.db_dir = "~/data/ludb"
    train_cfg.n_epochs = 10
    ds_train = LUDBDataset(train_cfg, training=True, lazy=False)

3. Pick a model
---------------

Models take a config from :mod:`torch_ecg.model_configs` (or your own
CFG) plus the number of classes / leads:

.. code-block:: python

    from torch_ecg.models import ECG_CRNN
    from torch_ecg.model_configs import ECG_CRNN_CONFIG

    config = ECG_CRNN_CONFIG.copy()
    config.cnn.name = "resnet_block_leadwise"
    model = ECG_CRNN(config, n_leads=12, n_classes=26)

4. Train and reload
-------------------

Subclass :class:`~torch_ecg.components.trainer.BaseTrainer` (or reuse one
of the benchmark trainers) and call :code:`train()`.  Checkpoints written
by the trainer contain the full model config, so a trained model is a
one-liner to reload:

.. code-block:: python

    model, train_config = ECG_CRNN.from_checkpoint("path/to/checkpoint_dir")
